"""Trigger regexes and snippet-window construction around matches."""

import re

from preprocessing.config import SNIPPET_GAP_CHARS


# Triggers — combined NEPC + AVPC + biomarker terms
TRIGGER_REGEX = {
    "nepc": (
        r"\b(?:"
        r"neuroendocrine|neuro-endocrine|nepc|t-nepc|"
        r"small[\s-]?cell(?:\s+carcinoma)?|scpc|scnc|oat[\s-]?cell|"
        r"small[- ]cell\s+neuroendocrine\s+carcinoma|"
        r"histolog(?:ic|ical)\s+transform(?:ation|ed|ing)|"
        r"transform(?:ation|ed|ing)(?:\s+(?:to|into))?|"
        r"transdifferentiat(?:e|ed|ion|ing)|dedifferentiat(?:e|ed|ion|ing)|"
        r"lineage\s+plasticity|treatment[\s-]?emergent\s+neuroendocrine|"
        r"synaptophysin|chromogranin(?:\s+a)?|cd56|neuron[- ]specific\s+enolase|nse"
        r")\b"
    ),
    "avpc": (
        r"\b(?:"
        r"aggressive[\s-]?variant|avpc|anaplastic|variant\s+crpc|androgen[- ]indifferent|"
        r"visceral\s+met(?:astases|astasis|astatic)?|"
        r"liver\s+met(?:astases|astasis|astatic)?|hepatic\s+met(?:astases|astasis|astatic)?|"
        r"lung\s+met(?:astases|astasis|astatic)?|pulmonary\s+met(?:astases|astasis|astatic)?|"
        r"adrenal\s+met(?:astases|astasis|astatic)?|brain\s+met(?:astases|astasis|astatic)?|"
        r"pleural\s+met(?:astases|astasis|astatic)?|peritoneal\s+met(?:astases|astasis|astatic)?|"
        r"lytic\s+(?:bone|lesion)|predominantly\s+lytic|osseous\s+lytic|destructive\s+bone\s+lesion|"
        r"bulky\s+(?:lymphadenopathy|adenopathy|nodal|nodes?|pelvic\s+mass|prostate\s+mass)|"
        r"large\s+(?:pelvic|prostatic)\s+mass|"
        r"low\s+psa|disproportionately\s+low\s+psa|psa\s+discordant|"
        r"(?:high[- ]volume|extensive|diffuse|innumerable)\s+(?:bone|osseous)\s+met(?:astases|astatic)?|"
        r"bombesin|grp|cea|ldh|hypercalc(?:emia|aemia)|"
        r"castration[- ]resistant|androgen[- ]independent|rapidly?\s+progress(?:ion|ive)|"
        r"refractory\s+to\s+adt|despite\s+adt"
        r")\b"
    ),
    "biomarker": (
        r"\b(?:"
        r"brca1|brca2|atm|cdk12|palb2|"
        r"hrd|hrr|ddr|homologous\s+recombination|dna\s+damage\s+repair|"
        r"msi[- ]h(?:igh)?|mmr|mismatch\s+repair|msh2|msh6|mlh1|pms2|"
        r"tumor\s+mutational\s+burden|tmb"
        r")\b"
    ),
    "non_prostate_primary": (
        r"\b(?:"
        # Lung
        r"nsclc|sclc|non[- ]small[- ]cell\s+lung|lung\s+adenocarcinoma|lung\s+(?:cancer|carcinoma)|"
        # GI
        r"colorectal|colon\s+(?:cancer|carcinoma)|rectal\s+(?:cancer|carcinoma)|"
        r"pancreatic\s+(?:cancer|carcinoma|adenocarcinoma)|gastric\s+(?:cancer|carcinoma)|"
        r"esophageal\s+(?:cancer|carcinoma)|hepatocellular\s+carcinoma|hcc|"
        # GU (non-prostate)
        r"urothelial\s+(?:cancer|carcinoma)|bladder\s+(?:cancer|carcinoma)|"
        r"renal\s+cell\s+carcinoma|rcc|kidney\s+(?:cancer|carcinoma)|"
        # Heme
        r"lymphoma|leukemia|multiple\s+myeloma|"
        # Other solid
        r"melanoma|glioblastoma|head\s+and\s+neck\s+(?:cancer|carcinoma|squamous)|"
        r"breast\s+(?:cancer|carcinoma)|"
        # Multi-primary phrases
        r"second\s+primary|synchronous\s+primary|metachronous\s+primary|"
        r"history\s+of\s+(?:lung|colon|colorectal|breast|bladder|kidney|renal|pancreatic|gastric|esophageal|melanoma|lymphoma|leukemia)"
        r")\b"
    ),
}


def merge_windows(windows, gap_chars=SNIPPET_GAP_CHARS):
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap_chars:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def find_trigger_matches(text, trigger_regex=TRIGGER_REGEX):
    matches = []
    for label, pattern in trigger_regex.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((label, match.start(), match.end()))
    return sorted(matches, key=lambda item: (item[1], item[2]))


def build_snippet(text, matches, *, context_chars=750, max_chars=300_000):
    """Build a snippet string around trigger matches.

    Callers pass sizing explicitly, typically from a `SnippetProfile`
    (`preprocessing.config.SNIPPET_PROFILES`), e.g.
    `build_snippet(text, matches, context_chars=profile.context_chars, max_chars=profile.max_chars)`.
    """
    if not matches:
        return ""
    windows = [(max(0, s - context_chars), min(len(text), e + context_chars)) for _, s, e in matches]
    parts = []
    for start, end in merge_windows(windows):
        snippet = text[start:end].strip()
        if not snippet:
            continue
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        parts.append(snippet)
    out = "\n\n...\n\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out
