"""Descriptive aggregate replay with actual CPU guard/controller evidence."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle
from optical_worker.protocol import canonical_json_bytes
from optical_worker.runtime_custody import (
    PHASES, RUNTIME_CUSTODY_INPUTS_VERSION, RUNTIME_CUSTODY_SELECTION_VERSION, RUNTIME_CUSTODY_VERSION,
    RuntimeCustodyRecord, validate_runtime_custody_selection,
)
from optical_worker.runtime_environment import INTERPRETER_ALIAS_OBSERVATIONS_VERSION, RuntimeEnvironmentContract
from tests.support.guarded_peer import Peer
from tests.unit.optics.test_runtime_environment import deployment


@pytest.fixture
def baseline(deployment):
    _, environment_payload = deployment
    environment = RuntimeEnvironmentContract.from_payload(environment_payload)
    peer = Peer()
    controller = NativeGuardedSetupLifecycle(lane_id="cuda", channel=peer.channel, configuration=peer.config)
    session, bootstrap = object(), object()
    controller.adopt_host_owner(session, role="captured_source_session", operation_id="source_custody_bind")
    controller.adopt_host_owner(bootstrap, role="bootstrap_module", operation_id="source_custody_bind")
    identity = [sys.executable, list(sys.version_info), getattr(sys, "abiflags", ""), sys.platform,
                getattr(sys.implementation, "_multiarch", ""), sys._stdlib_dir, os.name]
    alias = {"schema_version": INTERPRETER_ALIAS_OBSERVATIONS_VERSION, "interpreter_identity": identity,
             "source_path": str(Path(sys._stdlib_dir) / "multiprocessing/__init__.py"), "source_sha256": "a" * 64,
             "resolution": None, "events": []}
    source = {"revision": "1" * 40, "catalog_sha256": "2" * 64, "snapshot_sha256": "3" * 64}
    inputs = {"schema_version": RUNTIME_CUSTODY_INPUTS_VERSION, "source": source, "lane_id": "cuda",
              "configuration": peer.config.to_payload(), "case_id": "asymmetric_synthetic_7x3_v1",
              "interpreter_alias": {key: alias[key] for key in ("interpreter_identity", "source_path", "source_sha256")}}
    inputs["interpreter_alias"].update(resolved_executable=environment_payload["interpreter"]["path"],
                                        executable_sha256=environment_payload["interpreter"]["sha256"])
    lifecycle = controller.evidence().to_payload()
    p = {"schema_version": RUNTIME_CUSTODY_VERSION, "source": copy.deepcopy(source), "environment_sha256": None,
         "configuration": peer.config.to_payload(), "lane_id": "cuda",
         "observation_cut": {"journal_length": 1, "lifecycle_event_count": len(lifecycle["transition_trace"])},
         "journal": [{"index": 0, "kind": "source_bound", "guard": None,
                      "lifecycle_event_index": len(lifecycle["transition_trace"]) - 1,
                      "details": {"session_token": 1, "bootstrap_token": 2, "controller_token": 3}}],
         "objects": [{"token": i + 1, "first_journal_index": 0, "guardian_sequence": i if i < 2 else None}
                     for i in range(3)],
         "components": {"native_globals": None, "module_replacements": None, "interpreter_alias": alias},
         "lifecycle": lifecycle,
         "summary": {"last_completed_phase": None, "external_event_count": 0, "completed_file_count": 0,
                     "successful_guard_sequences": [], "binding_observed": False, "prepared_storage_observed": False,
                     "input_validation_observed": False, "final_execution_readiness_issued": False}}
    try:
        yield p, environment, inputs, controller
    finally:
        if controller.termination_evidence is None:
            controller.terminate(fault="setup_failure", detection_phase="unit_end", emergency_target_synchronize=None)
        peer.close()


def parse(p, environment, inputs):
    return RuntimeCustodyRecord.from_payload(p, environment=environment, trusted_launch_inputs=inputs)


def test_real_guard_baseline_has_no_fabricated_canonical_source(baseline):
    p, environment, inputs, _ = baseline
    record = parse(p, environment, inputs)
    assert record.to_payload() == p
    assert record.sha256 == hashlib.sha256(canonical_json_bytes(p)).hexdigest()
    p["source"]["revision"] = "f" * 40
    assert record.to_payload()["source"] == inputs["source"]
    assert len(record.to_payload()["objects"]) == 3


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=0),
    lambda p: p["source"].update(revision="0" * 40),
    lambda p: p["source"].update(snapshot_sha256="0" * 64),
    lambda p: p.update(environment_sha256="0" * 64),
    lambda p: p.update(lane_id="warp"),
    lambda p: p["objects"][0].update(token=True),
    lambda p: p["objects"][0].update(first_journal_index=1),
    lambda p: p["objects"][0].update(guardian_sequence=None),
    lambda p: p["objects"][0].update(guardian_sequence=1),
    lambda p: p["objects"].append({"token": 4, "first_journal_index": 0, "guardian_sequence": None}),
    lambda p: p["journal"][0].update(index=True),
    lambda p: p["journal"][0].update(lifecycle_event_index=0),
    lambda p: p["journal"][0]["details"].update(controller_token=1),
    lambda p: p["components"]["interpreter_alias"].update(source_sha256="0" * 64),
    lambda p: p["components"].update(interpreter_alias=None),
    lambda p: p["summary"].update(final_execution_readiness_issued=True),
    lambda p: p["summary"].update(external_event_count=True),
    lambda p: p["observation_cut"].update(journal_length=2),
])
def test_baseline_substitution_rejects_with_trusted_inputs_fixed(baseline, mutate):
    p, environment, inputs, _ = baseline
    mutate(p)
    with pytest.raises((ValueError, RuntimeError)):
        parse(p, environment, inputs)


def test_actual_disarm_is_required_for_phase_completion(baseline):
    p, environment, inputs, controller = baseline
    with controller.segment("setup"):
        pass
    lifecycle = controller.evidence().to_payload()
    end = lifecycle["transition_trace"][-1]
    assert end["operation"] == "normal_disarmed"
    p["lifecycle"] = lifecycle
    p["observation_cut"] = {"journal_length": 2, "lifecycle_event_count": len(lifecycle["transition_trace"])}
    p["journal"].append({"index": 1, "kind": "phase_completed", "guard": None,
                         "lifecycle_event_index": end["index"], "details": {"phase": PHASES[0],
                         "disarmed_event_index": end["index"], "materialization_case_id": None}})
    p["summary"].update(last_completed_phase=PHASES[0], successful_guard_sequences=[end["guard"]["sequence"]])
    parse(p, environment, inputs)
    for delta in (1, 2, 3):
        bad = copy.deepcopy(p)
        bad["journal"][-1]["lifecycle_event_index"] -= delta
        bad["journal"][-1]["details"]["disarmed_event_index"] -= delta
        with pytest.raises((ValueError, RuntimeError)):
            parse(bad, environment, inputs)


def test_selection_checks_actual_durable_bytes_not_in_memory_claim(baseline):
    p, environment, inputs, _ = baseline
    record = parse(p, environment, inputs)
    count = p["observation_cut"]["lifecycle_event_count"]
    selection = {"schema_version": RUNTIME_CUSTODY_SELECTION_VERSION, "selection": "current",
                 "runtime_record_sha256": record.sha256, "selected_lifecycle_event_count": count,
                 "observed_lifecycle_event_count": count, "failure_diagnostic_present": False}
    validate_runtime_custody_selection(selection, record.encoded)
    for durable in (None, record.encoded + b"\n", record.encoded[:-1]):
        with pytest.raises(ValueError):
            validate_runtime_custody_selection(selection, durable)
    selection["observed_lifecycle_event_count"] += 1
    with pytest.raises(ValueError, match="older lifecycle"):
        validate_runtime_custody_selection(selection, record.encoded)
    selection["selection"] = "prior_validated_cut"
    validate_runtime_custody_selection(selection, record.encoded)


@pytest.mark.parametrize("failure_window", ["before_record", "before_sidecar"])
def test_failure_between_atomic_writes_never_invents_durable_attachment(baseline, tmp_path, monkeypatch, failure_window):
    from tests.support import native_runtime_process as driver

    p, environment, inputs, controller = baseline
    old = parse(p, environment, inputs)

    def selection(record):
        count = record.to_payload()["observation_cut"]["lifecycle_event_count"]
        return {"schema_version": RUNTIME_CUSTODY_SELECTION_VERSION, "selection": "current",
                "runtime_record_sha256": record.sha256, "selected_lifecycle_event_count": count,
                "observed_lifecycle_event_count": count, "failure_diagnostic_present": False}

    old_selection = selection(old)
    driver._persist_runtime_custody(tmp_path, old, old_selection)
    terminal = controller.terminate(fault="setup_failure", detection_phase="unit_end", emergency_target_synchronize=None)
    p["lifecycle"] = terminal.lifecycle.to_payload()
    p["observation_cut"]["lifecycle_event_count"] = len(p["lifecycle"]["transition_trace"])
    new = parse(p, environment, inputs)
    original_write = driver._write_bytes

    def failing_write(directory, name, data):
        target = "runtime_custody.json" if failure_window == "before_record" else "runtime_custody_selection.json"
        if name == target:
            raise OSError("fixed persistence failure")
        return original_write(directory, name, data)

    monkeypatch.setattr(driver, "_write_bytes", failing_write)
    with pytest.raises(OSError, match="fixed persistence"):
        driver._persist_runtime_custody(tmp_path, new, selection(new))
    durable = (tmp_path / "runtime_custody.json").read_bytes()
    sidecar = json.loads((tmp_path / "runtime_custody_selection.json").read_bytes())
    assert sidecar == old_selection
    if failure_window == "before_record":
        assert durable == old.encoded
        validate_runtime_custody_selection(sidecar, durable)
        with pytest.raises(ValueError, match="observed count mismatch"):
            validate_runtime_custody_selection(sidecar, durable,
                observed_lifecycle_event_count=len(p["lifecycle"]["transition_trace"]))
    else:
        assert durable == new.encoded
        with pytest.raises(ValueError, match="digest mismatch"):
            validate_runtime_custody_selection(sidecar, durable)
