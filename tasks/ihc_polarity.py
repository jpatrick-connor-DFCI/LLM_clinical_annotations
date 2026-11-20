"""Marker-agnostic IHC polarity reading, shared by the NE and AR quote gates.

Panels mix results ("positive for synaptophysin and negative for
chromogranin"), so polarity is read per marker from the cue nearest to it. A
cue written "<polarity> for" binds forward, so it only counts for markers that
follow it. A positive cue is flipped to negative when the marker's own clause
negates it ("no synaptophysin positivity").

The NE gate re-exports these patterns. Changing them changes both gates: bump NE_GATE_VERSION and the
AR gate's AR_GATE_VERSION together.
"""

import re

IHC_POSITIVE = re.compile(
    r"(?:\b(?:positive|positivity|pos|reactive|reactivity|immunoreactiv\w*|"
    r"express\w*|stain(?:s|ed|ing)?\s+(?:positive|strongly|diffusely|focally)|"
    r"label(?:s|ed|ing)?|highlight\w*|retained)\b|\(\s*\+\s*\)|(?<=\w)\+|\s\+(?=\s|,|;|$))",
    flags=re.IGNORECASE,
)
IHC_NEGATIVE = re.compile(
    r"(?:\b(?:negative|negativity|neg|nonreactive|non-reactive|unreactive|"
    r"no\s+(?:staining|expression|reactivity)|not\s+(?:expressed|reactive|"
    r"stained|staining|positive)|lack\w*|absent|loss\s+of)\b|\(\s*-\s*\)|"
    r"(?<=\w)-(?=\s|,|;|$))",
    flags=re.IGNORECASE,
)
_BINDS_FORWARD = re.compile(r"\s+for\b", flags=re.IGNORECASE)
IHC_ASSERTION_NEGATION = re.compile(
    r"\b(?:no|not|never|without|absence|lack\w*|neither)\b", flags=re.IGNORECASE
)
IHC_CLAUSE_BREAK = re.compile(r",|\b(?:but|whereas|however)\b", flags=re.IGNORECASE)
IHC_RESULT_NOUN = re.compile(r"\b(?:staining|expression|reactivity)\b", flags=re.IGNORECASE)
IHC_POST_NEGATION = re.compile(
    r"^\s*(?:(?:staining|expression|reactivity)\s+)?"
    r"(?:(?:is|was|are|were|has\s+been|have\s+been)\s+)?"
    r"(?:(?:not|never)\s+(?:identified|seen|present|detected|observed|"
    r"appreciated|demonstrated|expressed)|absent|negative)\b",
    flags=re.IGNORECASE,
)
IHC_POLARITY_PRE_CHARS = 60
IHC_POLARITY_POST_CHARS = 30
SENTENCE_END = re.compile(r"(?<!\bvs)(?<!\bdr)(?<!\bno)[.;!?](?=\s|$)", flags=re.IGNORECASE)


# Any marker name: bounds one marker's clause in a panel.
PANEL_MARKER = re.compile(
    r"\b(?:androgen\s+receptors?|ar|psa|psap|nkx[\s-]?3[.\s-]?1|psma|hoxb[\s-]?13|"
    r"prostatic\s+acid\s+phosphatase|synaptophysin|chromogranin(?:\s+a)?|neuron[\s-]+specific\s+enolase|nse|cd\s?56|insm-?1|"
    r"ki-?67|ck\s?\d+\w*|cytokeratin\w*|gata-?3|ttf-?1|p63|p40|erg|pten|p53|rb1?|"
    r"cdx-?2|napsin|uroplakin|s100|sox10|thyroglobulin|pax-?8)\b",
    flags=re.IGNORECASE,
)
CLAUSE_END = re.compile(r",|;|\b(?:and|or|while|whereas|but|however)\b", flags=re.IGNORECASE)
MARKER_LIST_SEPARATOR = re.compile(r"[\s,/&]*(?:(?:and|or)\s*)?", re.IGNORECASE)
CLAUSE_AFTER_CHARS = 40

def flatten(quote):
    return re.sub(r"\s+", " ", str(quote or "")).strip()


def same_sentence_before(text, start, window):
    before = text[max(0, start - window):start]
    ends = list(SENTENCE_END.finditer(before))
    return before[ends[-1].end():] if ends else before


def same_sentence_after(text, end, window):
    after = text[end:end + window]
    boundary = SENTENCE_END.search(after)
    return after[:boundary.start()] if boundary else after


def positive_cue_is_negated(text, marker, cue):
    """Read negation on this marker's result, without borrowing other panel results."""
    before = same_sentence_before(
        text, min(marker.start(), cue.start()), IHC_POLARITY_PRE_CHARS
    )
    before = IHC_CLAUSE_BREAK.split(before)[-1]
    # A conjunction after a completed result starts another result clause:
    # "no synaptophysin positivity and chromogranin positive". A marker list
    # such as "no synaptophysin and chromogranin positivity" keeps its negation.
    for conjunction in reversed(list(re.finditer(r"\b(?:and|or)\b", before, re.IGNORECASE))):
        preceding = before[:conjunction.start()]
        if (IHC_POSITIVE.search(preceding) or IHC_NEGATIVE.search(preceding)
                or IHC_RESULT_NOUN.search(preceding)):
            before = before[conjunction.end():]
            break
    if IHC_ASSERTION_NEGATION.search(before):
        return True
    after = same_sentence_after(
        text, max(marker.end(), cue.end()), IHC_POLARITY_POST_CHARS
    )
    return IHC_POST_NEGATION.match(after) is not None


def marker_context(text, marker):
    """Isolate a result clause, retaining coordinated lists with a shared result.

    ``AR and NKX3.1 negative`` shares a result; ``AR strong, NKX3.1
    negative`` does not. Keep the actual marker in the local text so polarity
    distances and forward-binding cues still work.
    """
    start, end = marker.span()
    others = list(PANEL_MARKER.finditer(text))
    for other in reversed([m for m in others if m.end() <= start]):
        if not MARKER_LIST_SEPARATOR.fullmatch(text[other.end():start]):
            break
        start = other.start()
    for other in [m for m in others if m.start() >= end]:
        if not MARKER_LIST_SEPARATOR.fullmatch(text[end:other.start()]):
            break
        end = other.end()

    before = same_sentence_before(text, start, IHC_POLARITY_PRE_CHARS)
    cut = 0
    for pattern in (CLAUSE_END, PANEL_MARKER):
        hits = list(pattern.finditer(before))
        if hits:
            cut = max(cut, hits[-1].end())
    lo = start - len(before) + cut
    after = same_sentence_after(text, end, CLAUSE_AFTER_CHARS)
    stops = [m.start() for p in (CLAUSE_END, PANEL_MARKER) for m in [p.search(after)] if m]
    hi = end + (min(stops) if stops else len(after))
    local = text[lo:hi]
    local_marker = re.compile(re.escape(marker.group(0))).match(local, marker.start() - lo)
    return local, local_marker


def nearest_polarity(text, marker, *, tie="negative"):
    """Return "positive" / "negative" / None for the cue nearest one marker match.

    `tie` decides equally distant cues of opposite polarity: "negative" (the NE
    gate's conservative reading when positive is the target) or None (the AR
    gate, where negative is the target, reads a tie as unresolved).
    """
    text, marker = marker_context(text, marker)
    # marker_context already bounds the clause. Include a shared result after
    # the last marker in a coordinated list, even for the first marker.
    lo, hi = 0, len(text)
    best = None
    tied = False
    for polarity, pattern in (("negative", IHC_NEGATIVE), ("positive", IHC_POSITIVE)):
        for cue in pattern.finditer(text, lo, min(hi, len(text))):
            if cue.end() <= marker.start():
                distance = marker.start() - cue.end()
            elif cue.start() >= marker.end():
                if _BINDS_FORWARD.match(text, cue.end()):
                    continue
                distance = cue.start() - marker.end()
            else:
                continue
            result = polarity
            if polarity == "positive" and positive_cue_is_negated(text, marker, cue):
                result = "negative"
            if best is None or distance < best[0]:
                best, tied = (distance, result), False
            elif distance == best[0] and result != best[1]:
                if tie == "negative":
                    best = (distance, "negative")
                else:
                    tied = True
    if best is None or tied:
        return None
    return best[1]
