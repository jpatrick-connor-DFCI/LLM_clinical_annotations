"""Shared helpers for grounding LLM quotes back to supplied note snippets."""

import re


def _flatten(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _fold_punctuation(text):
    for source, target in (
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
        ("–", "-"),
        ("—", "-"),
        ("−", "-"),
        ("\xa0", " "),
    ):
        text = text.replace(source, target)
    return text


def quote_core(value):
    """Normalize harmless model-added quote decoration for substring matching."""
    text = _fold_punctuation(_flatten(value))
    for left, right in (("\"", "\""), ("'", "'")):
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
            break
    while text.startswith(("...", "…")):
        text = text[3:] if text.startswith("...") else text[1:]
        text = text.lstrip()
    while text.endswith(("...", "…")):
        text = text[:-3] if text.endswith("...") else text[:-1]
        text = text.rstrip()
    return text


def find_quote_support(quote, snippets, *, claimed_date=None):
    """Return the supporting snippet for a grounded quote, or ``None``.

    If ``claimed_date`` is supplied, the quote must occur in a snippet carrying
    exactly that note date. This prevents a model from grounding a quote in one
    note while assigning provenance from another.
    """
    normalized_quote = quote_core(quote).casefold()
    if len(normalized_quote) < 8:
        return None

    matches = []
    for snippet in snippets:
        note_text = _fold_punctuation(_flatten(snippet.get("snippet"))).casefold()
        if normalized_quote in note_text:
            matches.append(snippet)
    if claimed_date is not None:
        matches = [item for item in matches if item.get("note_date") == claimed_date]
    if not matches:
        return None
    return min(matches, key=lambda item: item.get("note_date") or "9999-99-99")
