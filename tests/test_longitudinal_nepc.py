import json
from argparse import Namespace

import polars as pl
import pytest

from preprocessing.longitudinal import group_patient_snippets
from preprocessing.longitudinal import file_sha256
from preprocessing.triggers import TRIGGER_REGEX, find_trigger_matches
from tasks.longitudinal_NEPC import build_nepc_timeline as nepc
from tasks.longitudinal_NEPC.prompts import (
    NEPC_SYNTHESIS_PROMPT,
    NEPC_SYSTEM_PROMPT,
)


def _chunks():
    return {
        0: [
            {
                "note_date": "2021-06-03",
                "note_type": "Labs",
                "snippet": "At progression PSA was 7.2 ng/mL.",
            }
        ],
        2: [
            {
                "note_date": "2021-06-05",
                "note_type": "Imaging",
                "snippet": "Bone scan demonstrates at least 24 osseous metastases.",
            }
        ],
    }


def _map_result(criterion, quote, source_date, *, fact_type="finding", fact_value="present"):
    return {
        "criteria_found": [],
        "evidence_items": [
            {
                "candidate_criterion": criterion,
                "fact_type": fact_type,
                "fact_value": fact_value,
                "fact_date": None,
                "source_note_date": source_date,
                "modality": "clinical",
                "quote": quote,
                "confidence": "high",
            }
        ],
    }


def test_prompt_uses_canonical_aparicio_thresholds():
    combined = f"{NEPC_SYSTEM_PROMPT}\n{NEPC_SYNTHESIS_PROMPT}"
    assert "EXCLUSIVELY visceral" in combined
    assert "Liver and other visceral organs qualify" in combined
    assert "Gleason score >=8" in combined
    assert "PSA <=10 ng/mL" in combined
    assert "high-volume (>=20) bone metastases" in combined
    assert "LDH >=2 times" in combined
    assert "<=6 months" in combined


@pytest.mark.parametrize(
    "text",
    [
        "PSA: 7.2 ng/mL",
        "There is a 6.3 cm pelvic mass.",
        "Pelvic mass measures 6.3 cm.",
        "Bone scan shows 24 osseous lesions.",
        "Androgen deprivation therapy began in January.",
        "Hepatic metastases are present.",
    ],
)
def test_avpc_retrieval_covers_canonical_atomic_evidence(text):
    assert find_trigger_matches(text, {"avpc": TRIGGER_REGEX["avpc"]})


def test_extraction_fingerprint_changes_with_model():
    first = nepc.extraction_run_config("scan", "vertex_ai", "model-a")
    second = nepc.extraction_run_config("scan", "vertex_ai", "model-b")
    assert first != second


def test_custom_evidence_uses_adjacent_metadata(tmp_path):
    evidence = tmp_path / "custom.tsv"
    assert nepc._meta_path_for_evidence(evidence) == tmp_path / "custom.meta.json"


def test_sparse_chunk_indices_are_preserved_and_fractional_indices_rejected():
    evidence = pl.DataFrame(
        {
            "DFCI_MRN": ["123", "123", "123"],
            "chunk_index": ["0", "1.5", "2"],
            "note_date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "note_type": ["Clinical", "Clinical", "Imaging"],
            "snippet": ["small cell carcinoma", "bad", "liver metastasis"],
        }
    )
    chunks = nepc._load_patient_chunks(evidence)
    assert set(chunks[123]) == {0, 2}


def _finding(criterion, quote, source_date, diagnosis_date, *, vmp="none"):
    return {
        "criterion": criterion,
        "diagnosis_date": diagnosis_date,
        "source_note_date": source_date,
        "modality": "clinical",
        "visceral_met_pattern": vmp,
        "quote": quote,
        "confidence": "high",
    }


def test_map_validation_rejects_invented_quote_and_future_date():
    chunks = _chunks()
    result = {
        "criteria_found": [
            _finding("C5", "not in the note", "2021-06-03", "2022-01-01")
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    # A bad finding is dropped and counted, not fatal to the chunk.
    assert error is None
    assert normalized["criteria_found"] == []
    assert set(normalized["rejected"]) <= {
        "diagnosis_date_after_source_note",
        "quote_or_source_not_in_evidence",
    }
    assert sum(normalized["rejected"].values()) == 1


def test_map_validation_keeps_good_findings_alongside_bad_ones():
    """A hallucinated quote must not cost the chunk's other valid findings."""
    chunks = _chunks()
    good_quote = "At progression PSA was 7.2 ng/mL."
    result = {
        "criteria_found": [
            _finding("C5", "not in the note", "2021-06-03", "2021-06-03"),
            _finding("C5", good_quote, "2021-06-03", "2021-06-03"),
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert [f["criterion"] for f in normalized["criteria_found"]] == ["C5"]
    assert normalized["rejected"] == {"quote_or_source_not_in_evidence": 1}


def test_map_validation_rejects_c2_without_established_exclusivity():
    """The prompt forbids reporting C2 unless exclusivity is shown; enforce it."""
    chunks = {
        0: [
            {
                "note_date": "2021-06-05",
                "note_type": "Imaging",
                "snippet": "Imaging shows liver metastases and no bone metastases.",
            }
        ]
    }
    result = {
        "criteria_found": [
            _finding(
                "C2",
                "Imaging shows liver metastases and no bone metastases.",
                "2021-06-05",
                "2021-06-05",
                vmp="none",
            )
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert normalized["criteria_found"] == []
    assert normalized["rejected"] == {"c2_without_established_exclusivity": 1}


def test_map_validation_fatal_only_on_structural_errors():
    normalized, error = nepc.validate_map_result({"evidence_items": []}, _chunks())
    assert normalized is None
    assert error == "missing_criteria_found"


@pytest.mark.parametrize(
    "quote",
    [
        '"At progression PSA was 7.2 ng/mL."',
        "...At progression PSA was 7.2 ng/mL...",
        "“At progression PSA was 7.2 ng/mL.”",
        "at progression psa was 7.2 ng/ml.",
    ],
)
def test_quote_matching_tolerates_normal_llm_quoting(quote):
    """Wrapping quotes, ellipses, curly punctuation and case must not drop a finding."""
    chunks = _chunks()
    result = {
        "criteria_found": [_finding("C5", quote, "2021-06-03", "2021-06-03")],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert [f["criterion"] for f in normalized["criteria_found"]] == ["C5"]


def test_wrong_source_note_date_is_corrected_to_grounded_chunk():
    """A grounded quote is reassigned to its actual note before date validation."""
    chunks = _chunks()
    result = {
        "criteria_found": [
            # Quote lives in chunk 2 (note 2021-06-05) but is attributed to 2021-06-03.
            _finding(
                "C5",
                "Bone scan demonstrates at least 24 osseous metastases.",
                "2021-06-03",
                "2021-06-03",
            )
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert [f["criterion"] for f in normalized["criteria_found"]] == ["C5"]
    assert normalized["criteria_found"][0]["_support_chunk_index"] == 2
    assert normalized["criteria_found"][0]["source_note_date"] == "2021-06-05"
    assert normalized["criteria_found"][0]["_claimed_source_note_date"] == "2021-06-03"


def test_actual_source_date_prevents_impossible_future_diagnosis():
    chunks = _chunks()
    result = {
        "criteria_found": [
            _finding(
                "C5",
                "Bone scan demonstrates at least 24 osseous metastases.",
                "2025-01-01",
                "2024-01-01",
            )
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert normalized["criteria_found"] == []
    assert normalized["rejected"] == {"diagnosis_date_after_source_note": 1}


def test_semantically_unrelated_quote_cannot_support_criterion():
    chunks = _chunks()
    result = {
        "criteria_found": [
            _finding(
                "C1",
                "At progression PSA was 7.2 ng/mL.",
                "2021-06-03",
                "2021-06-03",
            )
        ],
        "evidence_items": [],
    }
    normalized, error = nepc.validate_map_result(result, chunks)
    assert error is None
    assert normalized["criteria_found"] == []
    assert normalized["rejected"] == {"quote_does_not_support_criterion:C1": 1}


@pytest.mark.parametrize(
    ("criterion", "fact_type", "quote", "source_date"),
    [
        (
            "C2",
            "bone_metastasis_status",
            "Bone scan demonstrates at least 24 osseous metastases.",
            "2021-06-05",
        ),
        ("C4", "gleason_score", "At progression PSA was 7.2 ng/mL.", "2021-06-03"),
    ],
)
def test_atomic_fact_semantics_are_validated_by_fact_type(
    criterion, fact_type, quote, source_date
):
    result = _map_result(
        criterion,
        quote,
        source_date,
        fact_type=fact_type,
        fact_value="present",
    )
    normalized, error = nepc.validate_map_result(result, _chunks())
    assert error is None
    if fact_type == "bone_metastasis_status":
        assert len(normalized["evidence_items"]) == 1
        assert not normalized["rejected"]
    else:
        assert normalized["evidence_items"] == []
        assert normalized["rejected"] == {
            "quote_does_not_support_fact_type:gleason_score": 1
        }


def test_synthesis_keeps_earliest_duplicate_criterion():
    chunks = {
        0: [
            {
                "note_date": "2023-01-01",
                "note_type": "Pathology",
                "snippet": "Small-cell carcinoma diagnosed.",
            }
        ],
        1: [
            {
                "note_date": "2020-01-01",
                "note_type": "Pathology",
                "snippet": "Small-cell carcinoma confirmed.",
            }
        ],
    }
    result = {
        "criteria_found": [
            _finding(
                "C1",
                "Small-cell carcinoma diagnosed.",
                "2023-01-01",
                "2023-01-01",
            ),
            _finding(
                "C1",
                "Small-cell carcinoma confirmed.",
                "2020-01-01",
                "2020-01-01",
            ),
        ]
    }
    normalized, error = nepc.validate_synthesis_result(result, chunks)
    assert error is None
    assert normalized["criteria_found"][0]["diagnosis_date"] == "2020-01-01"
    assert normalized["rejected"] == {"duplicate_criterion:C1": 1}


def test_patient_synthesis_receives_all_sparse_chunk_maps():
    class Provider:
        def __init__(self):
            self.payload = None

        def call_with_retry(self, client, model, messages, max_retries):
            self.payload = json.loads(messages[1]["content"])
            return json.dumps(
                {
                    "criteria_found": [
                        {
                            "criterion": "C5",
                            "diagnosis_date": "2021-06-05",
                            "source_note_date": "2021-06-05",
                            "modality": "imaging",
                            "visceral_met_pattern": "none",
                            "quote": "Bone scan demonstrates at least 24 osseous metastases.",
                            "confidence": "high",
                        }
                    ]
                }
            ), None

    provider = Provider()
    chunk_maps = {
        0: _map_result(
            "C5",
            "At progression PSA was 7.2 ng/mL.",
            "2021-06-03",
            fact_type="psa_value",
            fact_value="7.2 ng/mL",
        ),
        2: _map_result(
            "C5",
            "Bone scan demonstrates at least 24 osseous metastases.",
            "2021-06-05",
            fact_type="bone_metastasis_count",
            fact_value="24",
        ),
    }
    result, error = nepc.synthesize_patient(
        provider, object(), "model", 1, 123, chunk_maps, _chunks()
    )
    assert error is None
    assert result["criteria_found"][0]["criterion"] == "C5"
    assert [item["chunk_index"] for item in provider.payload["chunk_maps"]] == [0, 2]


def test_raw_is_rebuilt_from_latest_successful_patient_synthesis(tmp_path):
    processed = tmp_path / "processed.tsv"
    raw = tmp_path / "raw.tsv"
    result = {
        "criteria_found": [
            {
                "criterion": "C5",
                "diagnosis_date": "2021-06-05",
                "source_note_date": "2021-06-05",
                "modality": "imaging",
                "visceral_met_pattern": "none",
                "quote": "Bone scan demonstrates at least 24 osseous metastases.",
                "confidence": "high",
                "_support_chunk_index": 2,
            }
        ]
    }
    nepc._write_rows_atomic(
        processed,
        [
            {
                "DFCI_MRN": 123,
                "num_chunks": 2,
                "num_chunks_ok": 2,
                "num_criteria": 1,
                "num_dropped": 0,
                "status": "ok",
                "run_config": "run",
                "result_json": json.dumps(result),
            }
        ],
        nepc.PROCESSED_COLUMNS,
    )
    assert nepc.rebuild_raw_from_processed(processed, raw, "run", {123: _chunks()}) == 1
    output = pl.read_csv(raw, separator="\t", infer_schema_length=0)
    assert output.height == 1
    assert output["criterion"].to_list() == ["C5"]
    assert output["chunk_index"].to_list() == ["2"]


def test_same_day_events_share_complete_cumulative_count(tmp_path):
    raw = tmp_path / "raw.tsv"
    timeline = tmp_path / "timeline.tsv"
    rows = []
    for criterion in ("C1", "C3"):
        rows.append(
            {
                "DFCI_MRN": 123,
                "chunk_index": 0,
                "source_note_date": "2021-01-01",
                "criterion": criterion,
                "diagnosis_date": "2021-01-01",
                "modality": "pathology",
                "visceral_met_pattern": None,
                "quote": "support",
                "confidence": "high",
            }
        )
    nepc._write_rows_atomic(raw, rows, nepc.RAW_COLUMNS)
    assert nepc.build_timeline(raw, timeline) == 2
    output = pl.read_csv(timeline, separator="\t")
    assert output["num_criteria_to_date"].to_list() == [2, 2]


def test_payload_budget_smaller_than_snippet_cap_is_rejected():
    notes = pl.DataFrame(
        {
            "DFCI_MRN": [123],
            "EVENT_DATE": ["2021-01-01"],
            "NOTE_TYPE": ["Clinical"],
            "CLINICAL_TEXT": ["small cell carcinoma"],
        }
    )
    with pytest.raises(ValueError, match="payload_max_chars"):
        group_patient_snippets(
            notes,
            {"nepc": r"small cell"},
            snippet_max_chars=100,
            payload_max_chars=99,
            max_workers=1,
        )


def test_end_to_end_resume_and_model_guard(tmp_path, monkeypatch):
    evidence = tmp_path / "avpc_nepc_evidence.tsv"
    meta = tmp_path / "avpc_nepc_evidence.meta.json"
    pl.DataFrame(
        {
            "DFCI_MRN": [123, 123],
            "chunk_index": [0, 2],
            "note_date": ["2021-06-03", "2021-06-05"],
            "note_type": ["Labs", "Imaging"],
            "snippet": [
                "At progression PSA was 7.2 ng/mL.",
                "Bone scan demonstrates at least 24 osseous metastases.",
            ],
        }
    ).write_csv(evidence, separator="\t")
    meta.write_text(
        json.dumps(
            {
                "scan_config": "scan-1",
                "evidence_sha256": file_sha256(evidence),
            }
        ),
        encoding="utf-8",
    )

    class Provider:
        default_model = "model-a"

        def __init__(self):
            self.calls = 0

        def build_client(self):
            return object()

        def call_with_retry(self, client, model, messages, max_retries):
            self.calls += 1
            payload = json.loads(messages[1]["content"])
            if "chunk_maps" in payload:
                return json.dumps(
                    {
                        "criteria_found": [
                            {
                                "criterion": "C5",
                                "diagnosis_date": "2021-06-05",
                                "source_note_date": "2021-06-05",
                                "modality": "imaging",
                                "visceral_met_pattern": "none",
                                "quote": "Bone scan demonstrates at least 24 osseous metastases.",
                                "confidence": "high",
                            }
                        ]
                    }
                ), None
            note = payload["notes"][0]
            if payload["chunk_index"] == 0:
                item = {
                    "candidate_criterion": "C5",
                    "fact_type": "psa_value",
                    "fact_value": "7.2 ng/mL",
                    "fact_date": None,
                    "source_note_date": note["note_date"],
                    "modality": "labs",
                    "quote": note["note_text"],
                    "confidence": "high",
                }
            else:
                item = {
                    "candidate_criterion": "C5",
                    "fact_type": "bone_metastasis_count",
                    "fact_value": "24",
                    "fact_date": None,
                    "source_note_date": note["note_date"],
                    "modality": "imaging",
                    "quote": note["note_text"],
                    "confidence": "high",
                }
            return json.dumps({"criteria_found": [], "evidence_items": [item]}), None

    provider = Provider()
    monkeypatch.setattr(nepc, "get_provider", lambda name: provider)

    args = Namespace(
        output_dir=tmp_path,
        evidence_path=evidence,
        evidence_meta_path=None,
        mrn_file=None,
        mrns=None,
        provider="vertex_ai",
        model=None,
        max_workers=2,
        max_retries=1,
        limit_patients=None,
        overwrite=False,
    )
    nepc.run(args)
    assert provider.calls == 3  # two maps + one patient synthesis
    assert pl.read_csv(tmp_path / "avpc_nepc_extractions_raw.tsv", separator="\t").height == 1
    processed = pl.read_csv(
        tmp_path / "avpc_nepc_processed_patients.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    assert processed["num_criteria"].to_list() == ["1"]

    nepc.run(args)
    assert provider.calls == 3

    changed = Namespace(**{**vars(args), "model": "model-b"})
    with pytest.raises(ValueError, match="Provider, model, prompt"):
        nepc.run(changed)

    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence content"):
        nepc.run(args)


def test_dropped_findings_are_counted_in_the_processed_log(tmp_path, monkeypatch):
    """A rejected finding must be visible as num_dropped, not silently vanish."""
    evidence = tmp_path / "avpc_nepc_evidence.tsv"
    meta = tmp_path / "avpc_nepc_evidence.meta.json"
    pl.DataFrame(
        {
            "DFCI_MRN": [123],
            "chunk_index": [0],
            "note_date": ["2021-06-03"],
            "note_type": ["Labs"],
            "snippet": ["At progression PSA was 7.2 ng/mL."],
        }
    ).write_csv(evidence, separator="\t")
    meta.write_text(
        json.dumps({"scan_config": "scan-1", "evidence_sha256": file_sha256(evidence)}),
        encoding="utf-8",
    )

    class Provider:
        default_model = "model-a"

        def build_client(self):
            return object()

        def call_with_retry(self, client, model, messages, max_retries):
            payload = json.loads(messages[1]["content"])
            grounded = "At progression PSA was 7.2 ng/mL."
            if "chunk_maps" in payload:
                # One valid finding plus one hallucinated quote.
                return json.dumps(
                    {
                        "criteria_found": [
                            _finding("C5", grounded, "2021-06-03", "2021-06-03"),
                            _finding("C3", "invented text", "2021-06-03", "2021-06-03"),
                        ]
                    }
                ), None
            return json.dumps(
                {
                    "criteria_found": [],
                    "evidence_items": [
                        {
                            "candidate_criterion": "C5",
                            "fact_type": "psa_value",
                            "fact_value": "7.2 ng/mL",
                            "fact_date": None,
                            "source_note_date": "2021-06-03",
                            "modality": "labs",
                            "quote": grounded,
                            "confidence": "high",
                        }
                    ],
                }
            ), None

    monkeypatch.setattr(nepc, "get_provider", lambda name: Provider())
    nepc.run(
        Namespace(
            output_dir=tmp_path,
            evidence_path=evidence,
            evidence_meta_path=None,
            mrn_file=None,
            mrns=None,
            provider="vertex_ai",
            model=None,
            max_workers=1,
            max_retries=1,
            limit_patients=None,
            overwrite=False,
        )
    )
    processed = pl.read_csv(
        tmp_path / "avpc_nepc_processed_patients.tsv", separator="\t", infer_schema_length=0
    )
    assert processed["num_criteria"].to_list() == ["1"]
    assert processed["num_dropped"].to_list() == ["1"]
    assert processed["status"].to_list() == ["ok_with_rejections"]

    rejected = pl.read_csv(
        tmp_path / "avpc_nepc_rejected_findings.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    assert rejected.height == 1
    assert rejected["stage"].to_list() == ["synthesis"]
    assert rejected["reason"].to_list() == ["quote_or_source_not_in_evidence"]
