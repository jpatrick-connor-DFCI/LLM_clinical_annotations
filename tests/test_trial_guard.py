import json
from types import SimpleNamespace

import polars as pl
import pytest

from preprocessing.evidence import note_task_paths
from preprocessing.longitudinal import file_sha256, write_scan_config_meta
from preprocessing.parquet_io import read_metadata
from tasks import note_extraction
from tasks.avpc import run_avpc_extraction as avpc
from tasks.nepc import run_nepc_extraction as nepc
from tasks.trial_guard import TRIAL_GUARD_VERSION, TrialContext, trial_boilerplate_reason


@pytest.mark.parametrize("quote", [
    "Inclusion criteria: Gleason score 9.",
    "Exclusion criteria: LDH twice the upper limit of normal.",
    "Participants must have small cell carcinoma of the prostate.",
    "Subjects shall have a prostate mass measuring 6 cm.",
    "Men must have castration-resistant prostate cancer.",
    "Positive synaptophysin staining is required for entry.",
    "A phase II study of metastatic neuroendocrine prostate cancer.",
])
def test_quote_rejects_study_population_and_requirements(quote):
    assert trial_boilerplate_reason(quote).startswith("boilerplate:")


@pytest.mark.parametrize("header", [
    "INCLUSION CRITERIA", "Exclusion criteria:", "Eligibility:",
    "Key inclusion criteria:", "3.1 Eligibility criteria", "## Study population",
    "Eligibility criteria include:", "INCLUSION CRITERIA (all required):",
    "- Inclusion criteria", "Eligibility criteria.", "...INCLUSION CRITERIA",
    "...INCLUSION CRITERIA...",
    "Participants must have one of the following:",
])
def test_cropped_quote_in_long_eligibility_list_is_rejected(header):
    quote = "Small cell carcinoma of the prostate."
    text = f"{header}\n1. Age at least 18.\n" + "Performance status requirements.\n" * 30
    text += f"2. Histological diagnosis: {quote}\n"
    assert TrialContext(text).reason(quote) == "boilerplate:eligibility_section"


def test_cropped_quote_from_trial_title_is_rejected():
    text = "Discussed a phase II trial of Small cell carcinoma of the prostate."
    assert TrialContext(text).reason("Small cell carcinoma of the prostate.") == "boilerplate:source_context"


@pytest.mark.parametrize("item", ["1. Disease assessment:", "1. Assessment:", "2) Assessment:"])
def test_clinical_word_in_criterion_does_not_end_eligibility_section(item):
    quote = "Small cell carcinoma of the prostate."
    text = f"Inclusion criteria:\n{item} {quote}"
    assert TrialContext(text).reason(quote) == "boilerplate:eligibility_section"


@pytest.mark.parametrize("heading", [
    "Assessment:", "ASSESSMENT AND PLAN", "Final diagnosis:", "...Assessment:", "...ASSESSMENT...",
])
def test_clinical_section_after_eligibility_is_retained(heading):
    quote = "Small cell carcinoma of the prostate."
    text = "Inclusion criteria:\n1. Age at least 18.\n2. Measurable disease.\n"
    text += f"{heading}\n{quote}"
    assert TrialContext(text).reason(quote) is None


def test_unrelated_trial_discussion_does_not_veto_patient_assessment():
    quote = "Small cell carcinoma of the prostate."
    text = f"Assessment: {quote} Discussed trial enrollment."
    assert TrialContext(text).reason(quote) is None


@pytest.mark.parametrize("prefix", ["Assessment:", "The patient has"])
def test_patient_assertion_before_enrollment_in_same_sentence_is_retained(prefix):
    quote = "Small cell carcinoma of the prostate"
    text = f"{prefix} {quote}, currently enrolled on a clinical trial."
    assert TrialContext(text).reason(quote) is None


def test_trial_population_in_assessment_is_still_rejected():
    quote = "small cell carcinoma of the prostate"
    text = f"Assessment: considering a trial for {quote}."
    assert TrialContext(text).reason(quote) == "boilerplate:source_context"


def test_identical_quote_can_be_grounded_outside_eligibility_section():
    quote = "Small cell carcinoma of the prostate."
    text = f"Eligibility criteria:\n{quote}\nAssessment:\n{quote}"
    assert TrialContext(text).reason(quote) is None


def test_source_context_uses_same_whitespace_and_punctuation_tolerance_as_grounding():
    text = "Inclusion criteria:\nSmall–cell carcinoma\n of the prostate."
    assert TrialContext(text).reason('"Small-cell carcinoma of the prostate."') == "boilerplate:eligibility_section"


@pytest.mark.parametrize("module", [nepc, avpc], ids=["nepc", "avpc"])
def test_pipeline_and_rebuild_use_source_context_without_losing_other_findings(tmp_path, module):
    spec = module.SPEC
    if module is nepc:
        model_findings = [{
            "evidence_type": "nepc_diagnosis",
            "quote": "Small cell carcinoma of the prostate.",
        }]
        label_column, negative, positive = "nepc_status", "non_nepc", "nepc"
    else:
        model_findings = [
            {"item": "metastatic_disease", "quote": "Metastatic prostate cancer to liver."},
            {"item": "castration_resistance", "quote": "Castration-resistant prostate cancer."},
            {"item": "gleason_high_grade", "gleason_total": 9, "quote": "Gleason score 9."},
            {"item": "bulky_prostate_pelvic_mass", "size_cm": 6, "quote": "Prostate mass measures 6 cm."},
        ]
        label_column, negative, positive = "is_avpc", False, True
    quotes = "\n".join(f["quote"] for f in model_findings)
    trial_text = "Inclusion criteria:\n" + "Performance status requirements.\n" * 20 + quotes
    # Patient 2 independently has each finding in an assessment. The identical
    # criteria list elsewhere in that note must not erase the clinical evidence.
    clinical_text = trial_text + "\nAssessment:\n" + quotes
    paths = note_task_paths(tmp_path, spec.prefix)
    pl.DataFrame({
        "DFCI_MRN": [1, 2], "note_uid": ["Clinician:1", "Clinician:2"],
        "note_date": ["2021-01-01"] * 2, "note_type": ["Clinician"] * 2,
        "snippet": [trial_text, clinical_text],
    }).write_parquet(paths["evidence"])
    write_scan_config_meta(
        paths["evidence_meta"], "test-scan", evidence_schema_version=spec.evidence_schema_version,
        evidence_sha256=file_sha256(paths["evidence"]), cohort_mrns=[1, 2],
    )

    class Provider:
        default_model = "test-model"

        def build_client(self):
            return None

        def call_with_retry(self, *args):
            return json.dumps({"findings": model_findings}), None

    args = SimpleNamespace(
        output_dir=tmp_path, evidence_path=None, mrns=None, mrn_file=None,
        provider="dfci_gpt", model=None, max_workers=1, max_retries=1, limit_notes=None,
        overwrite=False, retry_failures=False, rebuild_labels_only=False,
    )
    labels = note_extraction.run(spec, args, lambda _: Provider())
    by_mrn = {r["DFCI_MRN"]: r for r in labels.iter_rows(named=True)}
    assert by_mrn[1][label_column] == negative
    assert by_mrn[2][label_column] == positive
    assert all("boilerplate:eligibility_section" in r for r in by_mrn[1]["gate_rejections"])
    assert by_mrn[2]["gate_rejections"] == []

    findings = pl.read_parquet(paths["findings"])
    assert findings.height == 2 * len(model_findings)
    assert "_trial_context_reason" not in findings.columns
    # Simulate labels built before source-context gating was implemented. The
    # stored extraction remains usable; rebuilding must apply the new guard.
    ungated = note_extraction.write_labels(spec, paths, findings, {1: "ok", 2: "ok"}, "old")
    assert ungated[label_column].to_list() == [positive, positive]
    before = file_sha256(paths["findings"])
    args.rebuild_labels_only = True

    def no_provider(_):
        pytest.fail("label rebuild must not use an LLM provider")

    rebuilt = note_extraction.run(spec, args, no_provider)
    assert rebuilt.equals(labels)
    assert file_sha256(paths["findings"]) == before
    assert read_metadata(paths["labels_meta"])["trial_guard_version"] == TRIAL_GUARD_VERSION


@pytest.mark.parametrize("text,quote", [
    ("Patient with small cell carcinoma of the prostate, now enrolled on a trial of "
     "carboplatin/etoposide.", "small cell carcinoma of the prostate"),
    ("68M with small cell carcinoma of the prostate and enrolled on NCT01234567.",
     "68M with small cell carcinoma of the prostate"),
    ("Liver biopsy 3/2021 showed small cell carcinoma of the prostate; he consented to "
     "protocol 21-123.", "Liver biopsy 3/2021 showed small cell carcinoma of the prostate"),
    ("Liver, core biopsy: Small cell carcinoma, consistent with prostatic origin per CAP protocol.",
     "Small cell carcinoma, consistent with prostatic origin"),
    ("Compared with the prior study of 3/2021, there are innumerable new osseous metastases.",
     "there are innumerable new osseous metastases"),
    ("PSA 2.1 ng/mL, reported to the tumor registry.", "PSA 2.1 ng/mL"),
    ("He consented to liver biopsy, which showed small cell carcinoma of the prostate.",
     "liver biopsy, which showed small cell carcinoma of the prostate"),
])
def test_patient_findings_beside_ordinary_or_separate_trial_wording_are_retained(text, quote):
    assert TrialContext(text).reason(quote) is None


@pytest.mark.parametrize("heading", ["Diagnosis:", "Plan:", "A/P:", "DIAGNOSIS", "Addendum:"])
def test_more_clinical_headings_end_eligibility_section(heading):
    quote = "small cell carcinoma of the prostate"
    text = f"Eligibility:\n1. Age at least 18.\n{heading}\n68M with {quote}."
    assert TrialContext(text).reason(quote) is None


@pytest.mark.parametrize("text", [
    "Screened for NCT01234567, a phase II trial of small cell carcinoma of the prostate.",
    "Small cell carcinoma of the prostate trial: screening visit next week.",
    "Patients with small cell carcinoma of the prostate are eligible for NCT01234567.",
    "He is enrolled on a trial for small cell carcinoma of the prostate.",
])
def test_trial_descriptions_are_still_rejected(text):
    assert TrialContext(text).reason("small cell carcinoma of the prostate") == (
        "boilerplate:source_context"
    )
