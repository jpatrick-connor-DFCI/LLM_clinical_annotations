"""Partial review: positives stand, negatives stay unresolved (review-policy-v2)."""

import polars as pl
import pytest

from tasks import note_extraction
from tasks.arlpc import run_arlpc_extraction as arlpc
from tasks.avpc import run_avpc_extraction as avpc
from tasks.avpc_adjacent import run_avpc_adjacent_extraction as adjacent
from tasks.nepc import run_nepc_extraction as nepc

INCOMPLETE = ["failed", "not_run", "partial:1/2", "partial:4/5"]
COMPLETE = ["ok", "no_trigger_notes"]


@pytest.mark.parametrize("status", INCOMPLETE)
def test_incomplete_statuses(status):
    assert note_extraction.review_incomplete(status)


@pytest.mark.parametrize("status", COMPLETE)
def test_complete_statuses(status):
    assert not note_extraction.review_incomplete(status)


def _labels(module, status):
    empty = pl.DataFrame(schema=module.SPEC.findings_schema)
    rows = module.build_labels(empty, {1: status})
    return rows[0] if isinstance(rows, list) else rows.row(0, named=True)


@pytest.mark.parametrize("status", INCOMPLETE)
def test_incomplete_review_leaves_negatives_unset(status):
    assert _labels(nepc, status)["nepc_status"] is None
    assert _labels(nepc, status)["has_stated_nepc_dx"] is None
    assert _labels(avpc, status)["is_avpc"] is None
    assert _labels(adjacent, status)["is_avpc_adjacent"] is None
    assert _labels(arlpc, status)["arlpc_strict"] is None


@pytest.mark.parametrize("status", COMPLETE)
def test_complete_review_resolves_negatives(status):
    assert _labels(nepc, status)["nepc_status"] == "non_nepc"
    assert _labels(avpc, status)["is_avpc"] is False
    assert _labels(adjacent, status)["is_avpc_adjacent"] is False
    assert _labels(arlpc, status)["arlpc_strict"] is False


def test_policy_version_in_label_versions():
    for module in (nepc, avpc, adjacent, arlpc):
        assert module.SPEC.label_versions["review_policy_version"] == note_extraction.REVIEW_POLICY_VERSION
