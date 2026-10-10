"""Historical preimages are data, never a parallel old-precision lane."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.optical.schema import (
    canonical_row_semantics_payload,
    canonical_row_semantics_sha256,
    load_frozen_gate,
    validate_frozen_gate,
)
from optical_worker.render_parameters import (
    BoundRenderParameters,
    bind_render_parameters,
    load_frozen_render_sequence,
)

DATA = Path(__file__).parent / "data"


def test_render_cleanup_ledger_covers_durable_and_comparison_only_paths():
    from optical_worker.render_sequence_contract import RESOLVED_RENDER_SEQUENCE_SHA256

    root = Path(__file__).resolve().parents[3]
    ledger = json.loads((root / "benchmarks/results/q54_p12_4/cleanup_ledger.json").read_text())
    entries = {entry["entry_id"]: entry for entry in ledger["entries"]}
    durable = entries["canonical_resolved_render_contract"]
    temporary = entries["temporary_render_row_migration_snapshots"]
    assert set(durable["paths"]) == {
        "optical_worker/render_parameters.py",
        "optical_worker/render_sequence_contract.py",
        "tests/unit/optics/test_render_parameters.py",
        *(
            f"optical_worker/data/render_sequences/v2/{case}.jsonl"
            for case in RESOLVED_RENDER_SEQUENCE_SHA256
        ),
    }
    assert set(temporary["paths"]) == {
        "tests/unit/benchmarks/data/render_migration_v12.json",
        "tests/unit/benchmarks/data/render_migration_v13.json",
        "tests/unit/benchmarks/data/render_migration_v15.json",
        "benchmarks/results/q54_p12_4/history/gate.v13.json",
        "optical_worker/data/resolved_cameras/v1/asymmetric_synthetic_7x3_v1.jsonl",
        *(f"optical_worker/data/render_sequences/v1/{case}.jsonl"
          for case in RESOLVED_RENDER_SEQUENCE_SHA256),
        *(f"tests/unit/optics/data/native_manifests/{case}.json"
          for case in RESOLVED_RENDER_SEQUENCE_SHA256),
        "tests/unit/benchmarks/test_optical_render_migration.py",
        "tests/unit/optics/test_native_contract_extraction.py",
    }
    for entry, action in ((durable, "retain"), (temporary, "collapse")):
        assert entry["must_resolve_by"] == "P12.4e"
        for branch in ("cuda_selected", "warp_selected"):
            assert entry[branch]["action"] == action
        assert all((root / name).is_file() for name in entry["paths"])


def _sha(payload):
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def test_every_mandatory_row_retains_full_preimage_and_changes_semantic_identity():
    root = Path(__file__).resolve().parents[3]
    old = json.loads((DATA / "render_migration_v13.json").read_text())
    old['gate'] = json.loads((root / "benchmarks/results/q54_p12_4/history/gate.v13.json").read_bytes())
    new = json.loads((DATA / "render_migration_v15.json").read_text())
    gate = load_frozen_gate()
    assert old["gate"]["mandatory_rows"] == gate["mandatory_rows"]
    assert len(old["rows"]) == len(new["rows"]) == len(gate["mandatory_rows"]) == 20
    changed_keys = {"schema_version", "case_contract", "native_execution_binding_contract",
                    "native_input_contract", "resolved_render_contract"}
    for row in gate["mandatory_rows"]:
        before = old["rows"][row["row_id"]]
        after = new["rows"][row["row_id"]]
        actual = canonical_row_semantics_payload(row, gate)
        assert actual == after["preimage"]
        assert set(actual) == set(before["preimage"])
        assert _sha(before["preimage"]) == before["sha256"]
        assert _sha(actual) == after["sha256"] == canonical_row_semantics_sha256(row, gate)
        assert before["sha256"] != after["sha256"]
        for key, value in before["preimage"].items():
            if key not in changed_keys:
                assert actual[key] == value, key
        old_case = old["gate"]["cases"][row["case_id"]]
        for key in ('sha256', 'transform_replay_sha256', 'asset_source_revision', 'asset_aggregate_sha256',
                    'resolved_render_scope'):
            assert gate["cases"][row["case_id"]][key] == old_case[key]
    with pytest.raises(ValueError):
        validate_frozen_gate(old["gate"])
    assert gate["measurement"]["same_process_native_pair"] is False
    # All other frozen gates, including activation, must survive byte-equivalently.
    allowed = {
        "schema_version",
        "result_schema_version",
        "cases",
        "native_input_contract",
        "native_execution_binding_contract",
        "resolved_render_contract",
    }
    for key, value in old["gate"].items():
        if key not in allowed:
            assert gate[key] == value, key


def _bound(*, sample_index=0, diagnostic=False):
    return bind_render_parameters(
        load_frozen_render_sequence("asymmetric_synthetic_7x3_v1"),
        sample_index=sample_index,
        profile="direct_light_full",
        shadows=True,
        pass_mode="diagnostic_profiled" if diagnostic else "primary_unprofiled",
        role_to_mask={"rgb": 2},
    )


def _parse(payload):
    return BoundRenderParameters.from_payload(
        payload,
        case_id="asymmetric_synthetic_7x3_v1",
        profile="direct_light_full",
        shadows=True,
    )


@pytest.mark.parametrize("index", range(24))
def test_descriptive_parser_rejects_changed_native_scalar_even_after_rehash(index):
    payload = copy.deepcopy(_bound().to_payload())
    field, offset, one, zero = (
        ("native_camera_scalars_f64", index, "000000000000f03f", "0000000000000000")
        if index < 17 else ("native_shading_scalars_f32", index - 17, "0000803f", "00000000")
    )
    payload[field][offset] = one if payload[field][offset] != one else zero
    with pytest.raises(ValueError, match="parameters differ"):
        _parse(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 21),
        ("height", 1),
        ("camera_index", 1),
        ("env_idx", 1),
        ("sensor_id", "another"),
        ("sim_time_f64", "0x1.0000000000000p+0"),
        ("frame_id", 0),
        ("frame_id", True),
        ("profile", "rgb_preview"),
        ("shadows", False),
        ("diagnostics", 1),
        ("sensor_role_mask", 0),
    ],
)
def test_descriptive_record_rejects_mismatched_metadata(field, value):
    payload = _bound().to_payload()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        _parse(payload)


@pytest.mark.parametrize("sample_index", [0, 119, 120, 241])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_valid_record_roundtrip_does_not_authorize_execution(sample_index, diagnostic):
    record = _bound(sample_index=sample_index, diagnostic=diagnostic)
    assert _parse(record.to_payload()) == record
    assert not hasattr(record, "enqueue")
    assert not hasattr(record, "result_capable")
