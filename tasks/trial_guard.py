"""Reject research eligibility text without discarding a patient's whole note.

Quote-only checks cannot recognize a diagnosis cropped from a numbered list.
The note runner therefore also locates each grounded quote in its source
evidence at label-build time. Eligibility sections extend to the next clinical
heading; trial statements outside such sections are scoped to the clause that
names the trial, so "Biopsy showed small cell carcinoma; enrolled on NCT0123"
keeps its diagnosis. If identical text also appears outside those spans, it
remains usable evidence.
This context is recomputed on --rebuild-labels-only, never supplied by the LLM.
"""

import re

from preprocessing.grounding import _fold_punctuation, quote_core
TRIAL_GUARD_VERSION = "trial-context-v2"

# Research-specific cues. met_veto.BOILERPLATE also matches words with everyday
# clinical uses ("compared with the prior study of 3/2021", "per CAP protocol",
# "consented to biopsy", "tumor registry"), which vetoed real diagnoses.
TRIAL_CUE = re.compile(
    r"\b(?:(?:inclusion|exclusion)(?:\s+criteri\w*)?|eligib\w*|ineligible|"
    r"enroll\w*|trials?|nct\d*|phase\s+(?:i{1,3}|iv|[1-4])|randomiz\w*|"
    r"(?:research|study|trial)\s+protocol|protocol\s+(?:#\s*|no\.?\s*)?\d[\w-]*|"
    r"on\s+(?:the\s+|a\s+)?(?:study|protocol)|"
    r"study\s+(?:population|arm|drug|treatment|participants?|subjects?|enrollment|"
    r"visit|eligibility)|(?:clinical|research|investigational)\s+study|"
    r"investigational\s+(?:agent|drug|product)|informed\s+consent|"
    r"consent(?:ed|ing)?\s+(?:to|for)\s+(?:the\s+|a\s+)?(?:study|trial|protocol|research)|"
    r"screening\s+(?:log|criteria|visit)|cohort\s+[a-z0-9]|"
    r"subjects?\s+must|patients?\s+must)\b",
    re.IGNORECASE,
)

# These requirements often omit the words trial/study, especially in a list.
_REQUIREMENT = re.compile(
    r"\b(?:patients?|subjects?|participants?|candidates?|men|women)\s+"
    r"(?:(?:must|shall|should)\s+(?:have|be|demonstrate|meet)|"
    r"(?:are|will\s+be)\s+(?:required|eligible))\b|"
    r"\b(?:required|permitted|allowed)\s+for\s+(?:entry|participation)\b",
    re.IGNORECASE,
)
_ELIGIBILITY_LABEL = (
    r"(?:(?:key|study|trial|protocol)\s+)?"
    r"(?:(?:inclusion|exclusion|eligibility)(?:\s+(?:criteria|requirements))?|"
    r"(?:study|trial|protocol)\s+(?:population|criteria|requirements|title))"
)
_ELIGIBILITY_SUFFIX = (
    r"(?:[ \t]*\([^\n():]{1,80}\))?"
    r"(?:[ \t]+(?:include(?:s)?|are(?:[ \t]+as[ \t]+follows)?))?"
)
_SECTION_START = re.compile(
    rf"\b{_ELIGIBILITY_LABEL}{_ELIGIBILITY_SUFFIX}[ \t]*:|"
    rf"^[ \t]*(?:\.\.\.)?(?:\#{{1,6}}[ \t]+|[-*+][ \t]+)?"
    rf"(?:\d+(?:\.\d+)*[.)]?[ \t]+)?"
    rf"{_ELIGIBILITY_LABEL}{_ELIGIBILITY_SUFFIX}[ \t]*(?:\.{{1,3}})?[ \t]*(?=\r?$)",
    re.IGNORECASE | re.MULTILINE,
)
_CLINICAL_LABEL = (
    r"(?:assessment(?:\s*(?:and|&|/)\s*plan)?|impression|"
    r"(?:final|pathologic)\s+diagnosis|"
    r"a\s*/\s*p|a\s*&\s*p|plan|diagnos[ie]s|findings|addendum|comments?|"
    r"problem\s+list|"
    r"(?:oncologic(?:al)?|oncology|cancer|treatment|patient|clinical|"
    r"past\s+medical)\s+history|"
    r"history\s+of\s+present\s+illness|hpi|interval\s+history|"
    r"physical\s+exam(?:ination)?|pathology\s+results|"
    r"imaging\s+results|laboratory\s+results|clinical\s+course|"
    r"patient[ -]specific\s+findings)"
)
_CLINICAL_HEADING = re.compile(
    # A label embedded in a criterion ("1. Disease assessment: ...") is not a
    # new clinical section. Inline headings must begin a separate sentence;
    # the period after a numbered list item does not create that boundary.
    rf"(?:^[ \t]*(?:\.\.\.)?(?:\#{{1,6}}[ \t]+)?|(?<!\d)[.!?][ \t]+)"
    rf"{_CLINICAL_LABEL}[ \t]*(?::|(?:\.\.\.)?[ \t]*(?=\r?$))",
    re.IGNORECASE | re.MULTILINE,
)
# A sentence that opens by describing this patient; a trial clause later in
# it does not retract that description. Plural "patients with" stays
# unprotected because it is how study populations are written.
_PATIENT_ASSERTION = re.compile(
    rf"^\s*(?:{_CLINICAL_LABEL}[ \t]*:|"
    r"(?:he|she|(?:the\s+|this\s+)?patient|pt\.?)\s+"
    r"(?:has|had|is|was|with|w/|presents?\s+with|presented\s+with|developed)\b|"
    r"(?:an?\s+)?\d{1,3}[ \t-]*(?:y(?:ears?)?[ \t.-]*o(?:ld)?\.?|yo|y/o)?[ \t-]*"
    r"(?:m|f|man|male|gentleman|woman|female)\b)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^\n]+?(?:(?<=[.!?])(?=\s|$)|(?=\n|$))")
# Independent clauses within a sentence; a trial cue never crosses these.
_CLAUSE = re.compile(r"[^;]+")
# Within a clause, a trial phrase extends back only to the nearest comma,
# colon, or parenthesis: "Patient with SCPC, now enrolled on a trial ...".
_SOFT_BREAK = re.compile(r"[,:(]")


def _trial_cue(text):
    return TRIAL_CUE.search(text) or _REQUIREMENT.search(text)


def trial_boilerplate_reason(quote):
    """Return an audit reason for trial/requirement language in a quote."""
    hit = _trial_cue(quote_core(quote))
    return f"boilerplate:{hit.group(0).lower()}" if hit else None


class TrialContext:
    """Source-note spans that describe study requirements, built once per note."""

    def __init__(self, note_text):
        self.text = _fold_punctuation(str(note_text or ""))
        self.spans = []
        clinical_headers = [(m.start(), m.end()) for m in _CLINICAL_HEADING.finditer(self.text)]
        section_headers = [(m.start(), m.end()) for m in _SECTION_START.finditer(self.text)]
        # A requirement introducing a list acts as a heading even without an
        # explicit "Eligibility" label: "Participants must have one of: ...".
        for line in re.finditer(r"[^\n]+", self.text):
            if line.group().rstrip().endswith(":") and _REQUIREMENT.search(line.group()):
                section_headers.append((line.start(), line.end()))
        # Collectors retain both opening and closing headings even when a
        # trigger window starts far inside a list. Folding punctuation above
        # substitutes single characters, so these offsets also index the source.
        self.context_anchors = sorted(set(section_headers + clinical_headers)) if section_headers else []
        for start, _ in section_headers:
            end = next((p for p, _ in clinical_headers if p > start), len(self.text))
            self.spans.append((start, end, "boilerplate:eligibility_section"))
        for sentence in _SENTENCE.finditer(self.text):
            opens_with_patient = _PATIENT_ASSERTION.match(sentence.group()) is not None
            for clause in _CLAUSE.finditer(sentence.group()):
                cue = _trial_cue(clause.group())
                if cue is None:
                    continue
                # Text after the cue is the trial's own description ("a trial
                # of small cell carcinoma"), so it is always blocked. Text
                # before it is blocked back to the last soft break, unless the
                # sentence opens with an assertion about this patient
                # ("Assessment: SCPC and enrolled on a trial"). Eligibility
                # section spans still take precedence over this local exception.
                if opens_with_patient:
                    offset = cue.start()
                else:
                    breaks = [m.end() for m in _SOFT_BREAK.finditer(clause.group(), 0, cue.start())]
                    offset = breaks[-1] if breaks else 0
                start = sentence.start() + clause.start() + offset
                self.spans.append(
                    (start, sentence.start() + clause.end(), "boilerplate:source_context")
                )

    def reason(self, quote):
        """Reject only when every occurrence is inside research boilerplate."""
        direct = trial_boilerplate_reason(quote)
        if direct:
            return direct
        if not self.spans:
            return None
        core = quote_core(quote)
        if not core:
            return None
        # Same harmless whitespace/punctuation tolerance as quote grounding,
        # while retaining newlines and offsets for the section boundaries.
        pattern = re.compile(r"\s+".join(re.escape(w) for w in core.split()), re.IGNORECASE)
        reason = None
        for occurrence in pattern.finditer(self.text):
            blocked = next(
                (why for start, end, why in self.spans
                 if occurrence.start() < end and occurrence.end() > start),
                None,
            )
            if blocked is None:
                return None
            reason = reason or blocked
        return reason
