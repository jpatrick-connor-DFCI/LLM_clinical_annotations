"""Hybrid carry-forward patient history threaded across a patient's map chunks.

Each map chunk is a stateless LLM call, so it cannot see what earlier chunks of
the same patient already established. This module builds a deterministic
digest of prior chunks' validated findings plus a short LLM-written narrative,
so a later chunk gets disease-state context without ever being able to
manufacture a quote: grounding still checks every quote against the chunk it
was reported in, never against this carried context.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.longitudinal import flatten_ws  # noqa: E402

HISTORY_VERSION = "nepc-history-v1"
MAX_NARRATIVE_CHARS = 2000
MAX_DIGEST_FACTS = 60
MAX_FACTS_PER_GROUP = 3

_FACT_CARRY_FIELDS = (
    "candidate_criterion",
    "fact_type",
    "fact_value",
    "fact_date",
    "source_note_date",
)


def normalize_history_summary(value):
    """Normalize an LLM-written narrative: flatten whitespace, cap length."""
    if not isinstance(value, str):
        return None
    text = flatten_ws(value)
    if not text:
        return None
    if len(text) <= MAX_NARRATIVE_CHARS:
        return text
    truncated = text[:MAX_NARRATIVE_CHARS]
    boundary = truncated.rfind(" ")
    if boundary > 0:
        truncated = truncated[:boundary]
    return truncated.rstrip()


def _criteria_established(prior_results):
    earliest = {}
    for result in prior_results.values():
        for finding in result.get("criteria_found", []) or []:
            criterion = finding.get("criterion")
            if not criterion:
                continue
            date = finding.get("diagnosis_date")
            existing = earliest.get(criterion)
            if existing is None or (date or "9999-99-99") < (existing or "9999-99-99"):
                earliest[criterion] = date
    return [
        {"criterion": criterion, "diagnosis_date": earliest[criterion]}
        for criterion in sorted(earliest)
    ]


def _fact_sort_key(item):
    """Total order over facts: earliest first, then a full tiebreak.

    Sorting on the date alone leaves ties broken by input order (Python's sort
    is stable), which would let dict/list ordering decide which facts survive
    MAX_FACTS_PER_GROUP and MAX_DIGEST_FACTS. Same reasoning as the `snippet`
    tiebreaker in preprocessing.longitudinal.group_patient_snippets: this feeds
    a fingerprinted, resumable pipeline, so the order must not depend on how
    the caller happened to accumulate the input.
    """
    return (
        item.get("fact_date") or item.get("source_note_date") or "9999-99-99",
        str(item.get("fact_value") or ""),
        str(item.get("source_note_date") or ""),
    )


def _established_facts(prior_results):
    seen = set()
    groups = {}
    for result in prior_results.values():
        for item in result.get("evidence_items", []) or []:
            candidate_criterion = item.get("candidate_criterion")
            fact_type = item.get("fact_type")
            fact_value = item.get("fact_value")
            if not candidate_criterion or not fact_type or not fact_value:
                continue
            dedup_key = (
                candidate_criterion,
                fact_type,
                str(fact_value).casefold(),
                item.get("fact_date"),
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            group_key = (candidate_criterion, fact_type)
            groups.setdefault(group_key, []).append(item)

    kept = []
    for group_key in sorted(groups):
        members = sorted(groups[group_key], key=_fact_sort_key)
        kept.extend(members[:MAX_FACTS_PER_GROUP])

    kept.sort(
        key=lambda item: (
            item.get("candidate_criterion") or "",
            item.get("fact_type") or "",
            *_fact_sort_key(item),
        )
    )
    facts = [
        {field: item.get(field) for field in _FACT_CARRY_FIELDS}
        for item in kept[:MAX_DIGEST_FACTS]
    ]
    return facts


def build_history_context(prior_results, narrative):
    """Return a deterministic `prior_history` payload, or `None` for the first chunk."""
    criteria = _criteria_established(prior_results or {})
    facts = _established_facts(prior_results or {})
    if not criteria and not facts and not narrative:
        # Earlier chunks ran but established nothing -- sending an empty object
        # would announce a history that carries no information.
        return None
    return {
        "criteria_established": criteria,
        "established_facts": facts,
        "narrative": narrative,
    }
