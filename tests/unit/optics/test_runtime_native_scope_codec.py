"""CPU codec attacks using real historical lifecycle/import provenance.

Inserted native rows below are EXPLICITLY SYNTHETIC. These fixtures test the
new codec; they never migrate archived evidence or claim new native execution.
"""
import copy
import hashlib
import json
import os
import tarfile
from pathlib import Path

import pytest

from optical_worker.runtime_custody import RuntimeCustodyRecord
from optical_worker.runtime_environment import RuntimeEnvironmentContract

ROOT = Path(os.environ.get("ROBOT_SIM_REPO_ROOT", Path(__file__).resolve().parents[3]))
pytestmark = pytest.mark.skipif(os.name != "posix", reason="actual POSIX deployment record replay required")


def _load_original(base):
    path = os.environ.get("ROBOT_SIM_RUNTIME_CUSTODY_ARCHIVE")
    if path is None:
        pytest.skip("retained real lifecycle archive required; missing evidence is not a native pass")
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510"
    with tarfile.open(path) as archive:
        def read(name):
            return json.load(archive.extractfile(base + name))
        record, inputs, request = read("runtime_custody.json"), read("runtime_custody_inputs.json"), read("request.json")
    environment_path = Path(request["environment_path"])
    if not environment_path.exists():
        environment_path = ROOT / "tests/data/runtime_environments" / environment_path.name
    environment = RuntimeEnvironmentContract.from_payload(json.loads(environment_path.read_bytes()))
    assert environment.sha256 == request["environment_sha256"]
    assert RuntimeCustodyRecord.from_payload(record, environment=environment, trusted_launch_inputs=inputs).to_payload() == record
    return record, environment, inputs


@pytest.fixture(scope="module")
def original():
    return _load_original("c2d3-9b5702b-runtime-custody-warp-matrix-v1/test_actual_cold_canonical_war0/")


@pytest.fixture(scope="module")
def cuda_original():
    return _load_original("c2d3-9b5702b-runtime-custody-cuda-upload-author-v1/")


def parse(value, original):
    return RuntimeCustodyRecord.from_payload(value, environment=original[1], trusted_launch_inputs=original[2])


def _reindex(record):
    first = {}
    old_to_new = {row["index"]: index for index, row in enumerate(record["journal"])}
    def references(value, index):
        if type(value) is dict:
            for key, item in value.items():
                if key in {"event", "manifest", "signature", "local_token"}:
                    continue
                if key.endswith("_token") and item is not None:
                    first.setdefault(item, index)
                else:
                    references(item, index)
        elif type(value) is list:
            for item in value:
                references(item, index)
    for index, row in enumerate(record["journal"]):
        row["index"] = index
        if row["kind"] == "native_scope" and row["details"]["before_index"] is not None:
            row["details"]["before_index"] = old_to_new[row["details"]["before_index"]]
        references(row["details"], index)
    for row in record["objects"]:
        row["first_journal_index"] = first[row["token"]]
    record["observation_cut"]["journal_length"] = len(record["journal"])


def fixture(original, sites=("helper",)):
    record = copy.deepcopy(original[0])
    # This fixture exercises the historical v2 input grammar. It does not
    # contain a v3 CUDA extension set and must never relabel that evidence.
    record["schema_version"] = "q54.p12.4.runtime-custody.v2"
    binding_row = next(e for e in record["journal"] if e["kind"] == "runtime_binding")
    binding = binding_row["details"]
    cuda = record["lane_id"] == "cuda"
    offset = binding_row["lifecycle_event_index"] - 433
    trace = record["lifecycle"]["transition_trace"]
    library = next(f for f in original[1].to_payload()["files"] if f["path"] == (
        "nvidia/cuda_runtime/lib/libcudart.so.12" if cuda else "libcuda.so.595.71.05"))
    rows = []
    def add(site, edge, operation, cut, *, before=None, interior=False):
        details = {"site": site, "edge": edge, "before_index": before,
                   "operation_event_index": operation, "worker_pid": 12345, "thread_id": 12345,
                   **{key: binding[key] for key in ("module_token", "runtime_token", "target_token", "stream_token", "binding_token")},
                   "libraries": [{key: library[key] for key in ("root", "path", "sha256")}],
                   "state": {"default_device": 1 if interior else 0,
                             "current_context": 9000 if interior else None,
                             "current_context_device": 1 if interior else None,
                             "primary_contexts": [{"ordinal": i, "active": i == 1, "flags": 0} for i in range(8)],
                             "target_context": 9000, "target_current_stream": binding["stream_identity"][1],
                             "held_stream": dict(zip(("ordinal", "handle"), binding["stream_identity"]))}}
        index = 100000 + len(rows)
        if cuda:
            details["state"] = {"initialized": True, "logical_device": 0, "logical_basis": "native_return",
                                "raw_device": 1, "primary_contexts": [{"ordinal": i, "active": i == 1} for i in range(8)],
                                "target_current_stream": 0, "held_stream": details["state"]["held_stream"]}
        rows.append({"index": index, "kind": "native_scope", "guard": dict(trace[cut]["guard"]),
                     "lifecycle_event_index": cut, "details": details})
        return index
    if "stream" in sites:
        returned = next(e for e in record["journal"] if e["kind"] == "returned_owner"
                        and e["details"]["producer"] == "runtime_stream")
        operation = returned["details"]["leaf_event_index"]
        opening = add("runtime_stream", "before", operation, operation)
        before = rows[-1]["details"]
        before["stream_token"] = before["binding_token"] = None
        before["state"]["held_stream"] = before["state"]["target_current_stream"] = None
        for primary in before["state"]["primary_contexts"]:
            primary["active"] = False
        if cuda:
            before["state"].update(initialized=False, logical_basis="cold_source_derived", raw_device=0)
        else:
            before["state"]["target_context"] = None
        add("runtime_stream", "after", operation, returned["lifecycle_event_index"], before=opening)
        rows[-1]["details"]["binding_token"] = None
    if "binding" in sites:
        add("runtime_binding", "point", binding_row["lifecycle_event_index"], binding_row["lifecycle_event_index"])
    if "scope" in sites:
        opening = add("upload_scope", "before", 441 + offset, 442 + offset)
    if "helper" in sites:
        opening_helper = add("empty", "before", 447 + offset, 447 + offset, interior=True)
        add("empty", "after", 447 + offset, 448 + offset, before=opening_helper, interior=True)
    if "scope" in sites:
        storage = next(e for e in record["journal"] if e["kind"] == "prepared_storage")
        add("upload_scope", "after", 441 + offset, storage["lifecycle_event_index"], before=opening)
    if "terminal" in sites:
        operation = next(e["index"] for e in trace if e["operation"] == "emergency_sync_started")
        opening = add("terminal_sync", "before", operation, operation)
        add("terminal_sync", "after", operation, operation, before=opening)
    record["journal"].extend(rows)
    record["journal"].sort(key=lambda e: (e["lifecycle_event_index"],
        -1 if e["kind"] == "native_scope" and e["details"]["site"] == "runtime_stream"
        and e["details"]["edge"] == "before" else 0))
    _reindex(record)
    return record


@pytest.mark.parametrize("sites", [("helper",), ("stream", "binding", "scope", "helper", "terminal")])
def test_lawful_adoptions_nested_helpers_and_emergency_are_distinct(original, sites):
    value = fixture(original, sites)
    assert parse(value, original).to_payload() == value
    assert original[0]["schema_version"] == "q54.p12.4.runtime-custody.v1"


@pytest.mark.parametrize("attack", ["future", "out_of_cut", "stale", "same_kind", "duplicate", "crossed",
                                  "missing_after", "wrong_module", "wrong_runtime", "wrong_binding", "wrong_stream",
                                  "wrong_guard", "v1", "thread", "bool", "derived_warp"])
def test_native_occurrence_and_identity_attacks_reject(original, attack):
    value = fixture(original, ("scope", "helper"))
    rows = [e for e in value["journal"] if e["kind"] == "native_scope"]
    before, after = rows[1:3]
    if attack == "future":
        # The actual review counterexample: lifecycle447 <= journal1343,
        # but it did not exist at lifecycle cut438. Keep the same real guard.
        for row in (before, after):
            row["lifecycle_event_index"] = 438
        value["journal"].sort(key=lambda e: e["lifecycle_event_index"])
        _reindex(value)
    elif attack == "out_of_cut":
        before["details"]["operation_event_index"] = len(value["lifecycle"]["transition_trace"])
    elif attack in {"stale", "same_kind"}:
        cut = 451 if attack == "stale" else 456
        assert value["lifecycle"]["transition_trace"][cut]["operation"] == "low_level_leaf_entered"
        for row in (before, after):
            row["lifecycle_event_index"] = cut
        value["journal"].sort(key=lambda e: e["lifecycle_event_index"])
        _reindex(value)
    elif attack == "duplicate":
        duplicate = copy.deepcopy(after)
        duplicate["index"] = 200000
        value["journal"].insert(value["journal"].index(after) + 1, duplicate)
        _reindex(value)
    elif attack == "crossed":
        after["details"]["before_index"] = rows[0]["index"]
    elif attack == "missing_after":
        value["journal"].remove(after)
        _reindex(value)
    elif attack in {"wrong_module", "wrong_runtime", "wrong_binding", "wrong_stream"}:
        field = {"wrong_module": "module_token", "wrong_runtime": "runtime_token",
                 "wrong_binding": "binding_token", "wrong_stream": "stream_token"}[attack]
        before["details"][field] = value["journal"][0]["details"]["session_token"]
    elif attack == "wrong_guard":
        before["guard"]["sequence"] = 5
    elif attack == "v1":
        value["schema_version"] = "q54.p12.4.runtime-custody.v1"
    elif attack == "thread":
        before["details"]["thread_id"] = 12346
    elif attack == "bool":
        before["details"]["state"]["primary_contexts"][0]["flags"] = False
    else:
        before["details"]["state"]["logical_basis"] = "cold_source_derived"
    with pytest.raises((ValueError, RuntimeError)):
        parse(value, original)


def test_contradictory_native_values_remain_facts_not_codec_success_verdict(original):
    value = fixture(original)
    for row in value["journal"]:
        if row["kind"] == "native_scope":
            row["details"]["state"]["primary_contexts"][0]["active"] = True
    assert parse(value, original).to_payload() == value


def test_cuda_actual_binding_and_helpers_with_distinct_current_stream(cuda_original):
    value = fixture(cuda_original, ("stream", "binding", "helper", "terminal"))
    assert parse(value, cuda_original).to_payload() == value


@pytest.mark.parametrize("field,value", [("logical_basis", "cold_source_derived"), ("initialized", 1),
                                        ("raw_device", True), ("target_current_stream", None),
                                        ("held_stream", {"ordinal": 1, "handle": 7})])
def test_cuda_derived_observed_and_stream_substitutions(cuda_original, field, value):
    record = fixture(cuda_original)
    row = next(e for e in record["journal"] if e["kind"] == "native_scope")
    row["details"]["state"][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        parse(record, cuda_original)


@pytest.mark.parametrize("attack", ["before_after_return", "derived_as_observed", "invalid_derivation", "missing_return"])
def test_pristine_and_retained_return_order(cuda_original, attack):
    value = fixture(cuda_original, ("stream", "binding"))
    before, after = [e for e in value["journal"] if e["kind"] == "native_scope"][:2]
    if attack == "before_after_return":
        value["journal"].remove(before)
        before["lifecycle_event_index"] = after["lifecycle_event_index"]
        value["journal"].insert(value["journal"].index(after), before)
        _reindex(value)
    elif attack == "derived_as_observed":
        before["details"]["state"]["logical_basis"] = "native_return"
    elif attack == "invalid_derivation":
        before["details"]["state"]["raw_device"] = 1
    else:
        returned = next(e for e in value["journal"] if e["kind"] == "returned_owner"
                        and e["details"]["producer"] == "runtime_stream")
        value["journal"].remove(returned)
        _reindex(value)
    with pytest.raises((ValueError, RuntimeError)):
        parse(value, cuda_original)


@pytest.mark.parametrize("attack", ["setup_reference", "after_success", "after_failure_site"])
def test_emergency_observations_cannot_borrow_setup_or_completed_sync(original, attack):
    value = fixture(original, ("terminal",))
    row = next(e for e in value["journal"] if e["kind"] == "native_scope")
    if attack == "setup_reference":
        row["details"]["operation_event_index"] = 447
    elif attack == "after_success":
        row["lifecycle_event_index"] += 1
    else:
        row["details"]["site"] = "event_wait"
    with pytest.raises((ValueError, RuntimeError)):
        parse(value, original)
