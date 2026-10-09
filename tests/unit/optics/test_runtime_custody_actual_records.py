"""CPU adversarial replay of actual new-schema GPU executions.

ROBOT_SIM_RUNTIME_CUSTODY_EVIDENCE is a JSON list of retained run directories.
No execution is inferred from these tests: the original supervised runs supply
the execution facts. This suite changes only independent in-memory copies.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from optical_worker.native_contract import assert_prepared_storage_registered
from optical_worker.native_manifest import NativeInputManifest
from optical_worker.protocol import canonical_sha256
from optical_worker.runtime_custody import (
    RuntimeCustodyRecord, validate_runtime_custody_attachment, validate_runtime_custody_selection,
)
from optical_worker.runtime_environment import RuntimeEnvironmentContract


def pytest_generate_tests(metafunc):
    if "run_directory" in metafunc.fixturenames:
        paths = json.loads(os.environ.get("ROBOT_SIM_RUNTIME_CUSTODY_EVIDENCE", "[]"))
        metafunc.parametrize("run_directory", paths, scope="module")


def event(payload, kind):
    return next(e for e in payload["journal"] if e["kind"] == kind)


def parse(payload, environment, inputs):
    return RuntimeCustodyRecord.from_payload(payload, environment=environment, trusted_launch_inputs=inputs)


@pytest.fixture(scope="module")
def execution(run_directory):
    directory = Path(run_directory)
    request = json.loads((directory / "request.json").read_bytes())
    environment = RuntimeEnvironmentContract.from_payload(json.loads(Path(request["environment_path"]).read_bytes()))
    inputs = json.loads((directory / "runtime_custody_inputs.json").read_bytes())
    raw = (directory / "runtime_custody.json").read_bytes()
    selection = json.loads((directory / "runtime_custody_selection.json").read_bytes())
    artifact = json.loads((directory / "custody_artifact.json").read_bytes())
    validate_runtime_custody_selection(selection, raw)
    payload = json.loads(raw)
    record = parse(payload, environment, inputs)
    assert record.encoded == raw
    assert validate_runtime_custody_attachment(record, environment, inputs, artifact) == "worker_prefix_matched"
    assert payload["summary"]["input_validation_observed"] is True
    assert payload["summary"]["final_execution_readiness_issued"] is False
    return payload, environment, inputs, artifact


def test_actual_execution_roundtrip_does_not_mint_live_authority(execution):
    payload, environment, inputs, _ = execution
    record = parse(payload, environment, inputs)
    with pytest.raises(TypeError, match="exact production"):
        assert_prepared_storage_registered(record)
    with pytest.raises(TypeError, match="exact production"):
        assert_prepared_storage_registered(record.to_payload())


@pytest.mark.parametrize("replacement", ["session_token", "bootstrap_token", "controller_token"])
@pytest.mark.parametrize("location", ["runtime_binding", "prepared_storage", "both"])
def test_source_session_is_never_canonical_input(execution, replacement, location):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    token = event(payload, "source_bound")["details"][replacement]
    for kind in ("runtime_binding", "prepared_storage"):
        if location in (kind, "both"):
            event(payload, kind)["details"]["source_token"] = token
    with pytest.raises(ValueError, match="canonical source"):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field,replacement", [
    ("controller_token", "source_token"), ("module_token", "runtime_token"),
    ("stream_token", "target_token"), ("source_token", "binding_token"),
    ("target_token", "stream_token"), ("binding_token", "source_token"),
])
def test_runtime_binding_owner_substitutions(execution, field, replacement):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    binding = event(payload, "runtime_binding")["details"]
    binding[field] = binding[replacement]
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field,value", [
    ("device_ordinal", 0), ("device_ordinal", True), ("base_pointer", 0),
    ("storage_identity", 0), ("storage_nbytes", 0), ("active_start", 1),
    ("active_count", 1), ("dtype", "torch.float64"), ("shape", [1]),
    ("strides", "noncontiguous"), ("strides", [-1, 4]), ("active_count", True),
])
def test_registered_storage_descriptor_substitutions_with_recomputed_digest(execution, field, value):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    storage = event(payload, "prepared_storage")["details"]
    row = next(row for row in storage["arrays"] if row["signature"]["active_count"])
    if field == "strides" and value == "noncontiguous":
        # A singleton axis can have another stride without changing layout.
        # Mutate an axis that actually addresses multiple elements instead.
        axis = next(i for i, dimension in enumerate(row["signature"]["shape"]) if dimension > 1)
        value = list(row["signature"]["strides"])
        value[axis] += 1
    assert (type(row["signature"][field]) is not type(value)
            or row["signature"][field] != value)
    row["signature"][field] = value
    storage["storage_sha256"] = canonical_sha256(storage["arrays"])
    with pytest.raises(ValueError, match="storage|integer"):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("kind", ["runtime_binding", "prepared_storage", "input_validated", "binding_consumed"])
def test_observations_cannot_move_to_another_guard(execution, kind):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    row = event(payload, kind)
    row["guard"]["sequence"] += 1
    with pytest.raises(ValueError, match="guard differs"):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("mutation", ["materialization_only", "manifest_only", "complete_alternate"])
def test_go2_cross_case_rejected_even_when_all_arrays_and_downstream_hashes_match(execution, mutation):
    original, environment, inputs, _ = execution
    if not inputs["case_id"].startswith("go2_static_"):
        pytest.skip("Go2 case comparison requires an actual Go2 execution")
    payload = copy.deepcopy(original)
    other = "go2_static_1920x1080_v1" if inputs["case_id"] == "go2_static_960x640_v1" else "go2_static_960x640_v1"
    # Use the retained real CPU manifest, not an invented replacement hash.
    root = Path(os.environ["ROBOT_SIM_REPO_ROOT"])
    alternate = json.loads((root / "tests/unit/optics/data/native_manifests" / (other + ".json")).read_bytes())
    storage = event(payload, "prepared_storage")["details"]
    actual = storage["manifest"]
    assert actual["case_id"] != alternate["case_id"]
    assert actual["arrays"] == alternate["arrays"]
    assert actual["content_sha256"] == alternate["content_sha256"]
    assert actual["topology_sha256"] == alternate["topology_sha256"]
    materialization = next(e for e in payload["journal"] if e["kind"] == "phase_completed"
                           and e["details"]["phase"] == "canonical_materialization")
    if mutation in ("materialization_only", "complete_alternate"):
        materialization["details"]["materialization_case_id"] = other
    if mutation == "manifest_only":
        storage["manifest"]["case_id"] = other
    if mutation == "complete_alternate":
        storage["manifest"] = alternate
        event(payload, "input_validated")["details"]["manifest_sha256"] = NativeInputManifest.from_payload(alternate).sha256
    with pytest.raises(ValueError, match="case"):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("kind", ["runtime_binding", "prepared_storage", "input_validated"])
def test_success_observation_cannot_be_omitted(execution, kind):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    deleted = event(payload, kind)
    payload["journal"].remove(deleted)
    for index, row in enumerate(payload["journal"]):
        row["index"] = index
    payload["observation_cut"]["journal_length"] -= 1
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field", ["source_sha256", "source_path", "interpreter_identity"])
def test_alias_descriptor_transplant_rejected_with_launch_inputs_fixed(execution, field):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    alias = payload["components"]["interpreter_alias"]
    if field == "source_sha256":
        alias[field] = "0" * 64
    elif field == "source_path":
        alias[field] += ".other"
    else:
        alias[field][0] += ".other"
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("component", ["native_globals", "module_replacements", "interpreter_alias"])
def test_component_event_requires_original_global_token_binding(execution, component):
    original, environment, inputs, _ = execution
    kind = {"native_globals": "native_global", "module_replacements": "module_replacement",
            "interpreter_alias": "interpreter_alias"}[component]
    rows = [row for row in original["journal"] if row["kind"] == kind]
    if not rows:
        pytest.skip("this actual execution has no occurrence of " + component)
    payload = copy.deepcopy(original)
    bindings = event(payload, kind)["details"]["token_bindings"]
    bindings[0]["object_token"] = event(payload, "source_bound")["details"]["session_token"]
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field,value", [
    ("origin", "/foreign/numpy/__init__.py"), ("delegate_kind", "ExtensionFileLoader"),
    ("spec_token", 1), ("loader_token", 1), ("module_token", 1),
])
def test_file_resolution_is_bound_to_original_owners_and_environment(execution, field, value):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    details = event(payload, "external")["details"]
    assert details["event"]["operation"] == "file_resolved"
    assert details["provenance"][field] != value
    details["provenance"][field] = value
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field,value", [
    ("observer", "torch"), ("operation", "exec"), ("sequence", 1),
    ("parent_sequence", 0), ("phase", "runtime"), ("state", "completed"),
])
def test_native_call_cannot_be_relabelled(execution, field, value):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    details = event(payload, "file_call")["details"]
    assert details[field] != value
    details[field] = value
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field,value", [("requester", "numpy"), ("requester_token", 1)])
def test_absence_probe_is_bound_to_actual_executing_requester(execution, field, value):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    details = next(e["details"] for e in payload["journal"] if e["kind"] == "external"
                   and e["details"]["event"]["kind"] == "absent_probe")
    assert details["provenance"][field] != value
    details["provenance"][field] = value
    with pytest.raises(ValueError):
        parse(payload, environment, inputs)


def test_canonical_source_cannot_borrow_another_owners_adoption(execution):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    binding = event(payload, "runtime_binding")["details"]
    owners = {row["token"]: row for row in payload["objects"]}
    source, foreign = (owners[binding[key]] for key in ("source_token", "binding_token"))
    source["guardian_sequence"], foreign["guardian_sequence"] = foreign["guardian_sequence"], source["guardian_sequence"]
    with pytest.raises(ValueError, match="adoption"):
        parse(payload, environment, inputs)


@pytest.mark.parametrize("field", ["shape", "strides", "active_count"])
def test_empty_arrays_preserve_actual_registry_signature_and_reject_invalid_types(execution, field):
    original, environment, inputs, _ = execution
    payload = copy.deepcopy(original)
    storage = event(payload, "prepared_storage")["details"]
    rows = [row for row in storage["arrays"] if row["signature"]["active_count"] == 0]
    if not rows:
        pytest.skip("this actual source has no zero-size arrays")
    signature = rows[0]["signature"]
    signature[field] = True if field == "active_count" else [True] * len(signature[field])
    storage["storage_sha256"] = canonical_sha256(storage["arrays"])
    with pytest.raises(ValueError, match="integer"):
        parse(payload, environment, inputs)
