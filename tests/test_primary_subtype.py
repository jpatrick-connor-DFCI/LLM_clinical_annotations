import polars as pl
import pytest

from preprocessing.parquet_io import read_metadata, write_metadata
from tasks import primary_subtype as ps


def _write(tmp_path, name, rows, versions=None):
    spec, _ = ps.PIPELINES[name]
    path = tmp_path / f"{spec.prefix}_labels.parquet"
    pl.DataFrame(rows, strict=False).write_parquet(path)
    meta = {"run_config": "{}", **ps.expected_label_versions(spec), **(versions or {})}
    write_metadata(ps.labels_meta_path(path), meta)
    return path


# mrn: (nepc_status, nepc_t1, is_avpc, c1_met, is_avpc_adjacent, adj_c1)
PATIENTS = {
    1: ("nepc", True, True, True, True, True),        # definitive NEPC
    2: ("nepc", False, True, False, True, False),     # broad-only NE evidence, AVPC
    3: ("non_nepc", False, True, True, True, True),   # small-cell C1, NEPC missed it
    4: ("non_nepc", False, False, False, True, False),
    5: ("non_nepc", False, False, False, False, False),
    6: (None, None, False, False, False, False),      # NEPC review incomplete
    7: ("non_nepc", False, None, None, None, None),   # AVPC review incomplete
    8: ("nepc", None, None, None, None, None),        # partial review, positive stands
}


@pytest.fixture
def paths(tmp_path):
    nepc = [{"DFCI_MRN": m, "review_status": "ok", "gate_rejections": [], "nepc_status": s,
             "nepc_t1": t1} for m, (s, t1, *_) in PATIENTS.items()]
    avpc = [{"DFCI_MRN": m, "review_status": "ok", "gate_rejections": [], "is_avpc": a,
             "c1_met": c1, "c1_quote": "small cell carcinoma" if c1 else None}
            for m, (_, _, a, c1, _, _) in PATIENTS.items()]
    adjacent = [{"DFCI_MRN": m, "avpc_adj_review_status": "ok", "is_avpc_adjacent": j,
                 "avpc_adj_c1_met": c1, "avpc_adj_c1_quote": None}
                for m, (*_, j, c1) in PATIENTS.items()]
    arlpc = [{"DFCI_MRN": 1, "arlpc_strict": True}, {"DFCI_MRN": 99, "arlpc_strict": False}]
    return {"nepc": _write(tmp_path, "nepc", nepc),
            "avpc": _write(tmp_path, "avpc", avpc),
            "avpc_adjacent": _write(tmp_path, "avpc_adjacent", adjacent),
            "arlpc": _write(tmp_path, "arlpc", arlpc)}


def test_labels_and_discordance(paths):
    merged, inputs = ps.build(paths)
    by = {r["DFCI_MRN"]: r for r in merged.iter_rows(named=True)}
    assert list(merged.columns[:4]) == list(ps.LEADING)
    assert {m: by[m]["primary_label"] for m in PATIENTS} == {
        1: "nepc", 2: "nepc", 3: "avpc", 4: "avpc_adjacent", 5: "prad",
        6: None, 7: None, 8: "nepc"}
    assert {m: by[m]["primary_label_definitive"] for m in PATIENTS} == {
        1: "nepc", 2: "avpc", 3: "avpc", 4: "avpc_adjacent", 5: "prad",
        6: None, 7: None, 8: None}
    assert [m for m in PATIENTS if by[m]["small_cell_discordant"]] == [3]
    assert ps.small_cell_discordance(merged)["DFCI_MRN"].to_list() == [3]
    # ARLPC is a left join: patient 99 is not added.
    assert 99 not in by and by[1]["arlpc_strict"] is True and by[2]["arlpc_strict"] is None
    assert set(inputs) == {"nepc", "avpc", "avpc_adjacent", "arlpc"}
    assert {"nepc_review_status", "avpc_review_status", "nepc_gate_rejections"} <= set(merged.columns)


def test_stale_label_versions_raise(tmp_path, paths):
    spec, _ = ps.PIPELINES["nepc"]
    meta = read_metadata(ps.labels_meta_path(paths["nepc"]))
    write_metadata(ps.labels_meta_path(paths["nepc"]), {**meta, "ne_gate_version": "ne-gate-v5"})
    with pytest.raises(ValueError, match="rebuild-labels-only"):
        ps.build(paths)
    merged, _ = ps.build(paths, check_versions=False)
    assert merged.height == len(PATIENTS)


def test_missing_label_meta_raises(paths):
    ps.labels_meta_path(paths["avpc"]).unlink()
    with pytest.raises(FileNotFoundError, match="rebuild-labels-only"):
        ps.build(paths)


def test_missing_required_input_raises(paths, tmp_path):
    with pytest.raises(FileNotFoundError, match="avpc_adjacent"):
        ps.build({**paths, "avpc_adjacent": tmp_path / "nope.parquet"})


def test_arlpc_is_optional(paths):
    merged, inputs = ps.build({k: v for k, v in paths.items() if k != "arlpc"})
    assert "arlpc" not in inputs and "arlpc_strict" not in merged.columns


def test_write_outputs_records_provenance(paths, tmp_path):
    merged, inputs = ps.build(paths)
    out = tmp_path / "out" / "primary_subtype_labels.parquet"
    meta_path = ps.write_outputs(merged, inputs, out)
    assert pl.read_parquet(out).equals(merged)
    meta = read_metadata(meta_path)
    assert meta["primary_subtype_version"] == ps.PRIMARY_SUBTYPE_VERSION
    assert meta["nepc_labels_sha256"] == inputs["nepc"]["sha256"]
    assert "ne_gate_version" in meta["nepc_labels_meta"]
