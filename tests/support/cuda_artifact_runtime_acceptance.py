"""Fixed Go2 artifact acceptance through the existing captured worker supervisor.

Uses the existing supervised worker, including its fixed Warp compilation cases.
Private setup render diagnostics use the same worker; no performance/product entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


CASES = (
    ("cuda_fresh", "cuda", "go2_static_960x640_v1", "runtime_artifacts_success"),
    ("cuda_reused_deployment", "cuda", "go2_static_960x640_v1", "runtime_artifacts_success"),
    ("cuda_full_resolution", "cuda", "go2_static_1920x1080_v1", "runtime_artifacts_success"),
    ("warp_go2", "warp", "go2_static_960x640_v1", "runtime_scope_success"),
    ("warp_full_resolution", "warp", "go2_static_1920x1080_v1", "runtime_scope_success"),
    *((name, "cuda", "go2_static_960x640_v1", "runtime_artifacts_" + name) for name in (
        "create_after_raise", "exec_after_raise", "entry_adoption_raise", "factory_after_raise", "factory_hang")),
)
ASSETS = "/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2"


def write(directory, name, value):
    with (directory / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))


def check_bad_artifact_copies(repository, environment, output):
    """Alter private copies of real outputs; never touch the selected deployment."""
    from tools.optical_cuda_artifacts import MANIFEST_NAME, verify_deployment
    from tools.optical_cuda_build_recipe import _create_directory

    payload = environment.to_payload()
    original = Path(next(row["path"] for row in payload["roots"] if row["id"] == "cuda_artifacts"))
    digest = next(row["sha256"] for row in payload["files"]
                  if row["root"] == "cuda_artifacts" and row["path"] == MANIFEST_NAME)
    platform = json.loads((repository / "tests/data/cuda_artifacts/183_platform_v1.json").read_bytes())
    manifest = verify_deployment(original, digest, repository=repository, expected_platform=platform)
    paths = {MANIFEST_NAME, *(row["path"] for row in manifest["artifacts"])}
    for build in manifest["builds"]:
        paths.update(row["path"] for row in (build["stdout"], build["stderr"], build["termination"]["record"], *build["inputs"]))
    for case in ("missing", "changed", "partial"):
        copied = _create_directory(output / ("artifact_" + case))
        omitted = {row["path"] for row in manifest["artifacts"][1:]} if case == "partial" else set()
        if case == "missing":
            omitted.add(manifest["artifacts"][0]["path"])
        for relative in sorted(paths - omitted):
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / relative, target)
        if case == "changed":
            with (copied / manifest["artifacts"][0]["path"]).open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))
        try:
            verify_deployment(copied, digest, repository=repository, expected_platform=platform)
        except (ValueError, FileNotFoundError) as error:
            write(copied, "rejected.json", {"case": case, "error": str(error), "worker_launched": False})
        else:
            raise RuntimeError("invalid real artifact copy passed cold admission: " + case)
    verify_deployment(original, digest, repository=repository, expected_platform=platform)


def _validate_fixed_storage(directory, lane, case, fault, *, successful):
    """Check raw production-boundary observations, in addition to custody replay."""
    from optical_worker.cases import get_case

    hanging = fault.endswith("hang")
    name = "fixed_storage_binding.json" if successful else (
        "fault_entered.json" if hanging else "fixed_storage_failure.json")
    data = json.loads((directory / name).read_bytes())
    guard = {"sequence": 11 if lane == "warp" else 7, "role": "setup"}
    diagnostics = fault == "runtime_fixed_storage_diagnostics"
    dynamic = case == "synthetic_dynamic_rigid_960x640_v1"
    count = (23 if diagnostics else 14) if successful else (2 if "adoption" in fault else 1)
    if dynamic and successful:
        count += 10 if lane == "warp" else 26
    call_count = count + (9 if dynamic and lane == "warp" else 0)
    if (data["guard"] != guard or len(data["arrays"]) != count
            or len(data["calls"]) != (call_count if successful else 2)):
        raise RuntimeError("fixed storage did not reach its exact allocation boundary")
    ids = [row["id"] for row in data["arrays"]]
    if len(set(ids)) != count or sorted(ids) != data["retained_ids"]:
        raise RuntimeError("fixed storage partial arrays lost their production retention")
    for row in data["arrays"]:
        if row["device_ordinal"] != 1:
            raise RuntimeError("fixed storage allocated a foreign array")
    for row in data["calls"]:
        state = row["state"]
        if (row["guard"] != guard or state["held_stream"]["ordinal"] != 1
                or state["target_current_stream"] != state["held_stream"]["handle"]
                or [r["ordinal"] for r in state["primary_contexts"] if r["active"]] != [1]):
            raise RuntimeError("fixed storage allocated outside the held target stream")
        if lane == "warp":
            if state["default_device"] != 1 or state["current_context_device"] != 1:
                raise RuntimeError("Warp allocation was outside its target context")
        elif state["logical_device"] != 1 or state["raw_device"] != 1:
            raise RuntimeError("CUDA allocation was outside its target device")
    if successful:
        before, after = data["before"], data["after"]
        if before != after or (before["default_device"] != 0 or before["current_context"] is not None
                              if lane == "warp" else before["logical_device"] != 0 or before["raw_device"] != 1):
            raise RuntimeError("fixed storage did not restore its original ambient scope")
    if successful:
        workload = get_case(case)
        if (data["traversal_diagnostics"] is not diagnostics or data["sample_submission_count"] != 0
                or data["final_execution_readiness_issued"] is not False
                or sorted(row["id"] for row in data["outputs"]) != sorted(ids)
                or len({row["name"] for row in data["outputs"]}) != count
                or (not dynamic and data["arrays"][0]["shape"] != [workload.width * workload.height, 3])
                or not data["input_ids"]
                or any(row["view_id"] != row["prepared_id"] for row in data["input_ids"])):
            raise RuntimeError("fixed storage binding differs from its prepared inputs or outputs")
    elif not hanging and (data["error_type"], data["error"]) != (
            "RuntimeError", "fixed render storage failure: " + fault):
        raise RuntimeError("fixed storage replaced the original failure")


def _validate_render_graph(directory, lane, fault, *, trace, final_row=None):
    data = json.loads((directory / "render_graph_binding.json").read_bytes())
    diagnostic = fault.endswith("diagnostics")
    guard = {"sequence": 12 if lane == "warp" else 8, "role": "setup"}
    kinds = [row["details"]["descriptor"]["leaf_kind"] for row in trace
             if row["guard"] == guard and row["operation"] == "descriptor_observed"]
    dynamic = final_row is not None and final_row["boundary"] == "R1"
    phases = (["scene_update_kernel", "acceleration_kernel"] if dynamic else []) + [
        "raygen_kernel", "first_hit_kernel", "shade_kernel"]
    count = 2 * len(phases) + 1 if diagnostic else 1
    waits = [row["details"]["descriptor"]["operation_id"] for row in trace
             if row["guard"] == guard and row["operation"] == "descriptor_observed"
             and row["details"]["descriptor"]["leaf_kind"] == "event_wait"]
    expected_waits = (["refit_topology_ready_wait"] if dynamic and lane == "warp" else []) + [
        "slot_event_initialization_wait"]
    if (kinds.count("event_create") != count or kinds.count("event_record") != count
            or waits != expected_waits or kinds[-1:] != ["event_wait"]
            or "warmup_launch" in kinds or "allocation" in kinds):
        raise RuntimeError("render graph did not complete its exact initialized-event path")
    if (data["guard"] != guard or data["before"] != data["after"]
            or data["warmup_frames_completed"] != 0 or data["sample_submission_count"] != 0
            or data["final_execution_readiness_issued"] is not False
            or data["profile"] != ("direct_light_full" if final_row is None else final_row["output_profile"])
            or data["shadows"] is not (True if final_row is None else final_row["shadows"])
            or data["pass_mode"] != ("diagnostic_profiled" if diagnostic else "primary_unprofiled")
            or data["diagnostic_phases"] != (phases if diagnostic else [])
            or [row["frame_id"] for row in data["frames"]] != list(range(1, 11))
            or len(set(data["event_ids"])) != count):
        raise RuntimeError("provisional render graph has incorrect modes, frames, events or scope")
    before = data["before"]
    if (before["default_device"] != 0 or before["current_context"] is not None
            if lane == "warp" else before["logical_device"] != 0 or before["raw_device"] != 1):
        raise RuntimeError("render graph did not restore the required ambient scope")
    adoptions = data["adoptions"]
    if ({row["id"] for row in adoptions if row["role"] == "slot_event"} != set(data["event_ids"])
            or len({row["bundle_id"] for row in data["frames"]}) != 10):
        raise RuntimeError("provisional render graph lost distinct frames or event custody")
    command_ids = []
    for frame in data["frames"]:
        counts = frame["phase_launch_counts"]
        if (len(counts) != len(phases) or counts[-3:] != ([1, 2, 2] if lane == "warp" else [1, 1, 1])
                or (dynamic and (counts[0] != 1 or counts[1] < 1))
                or (dynamic and frame["transform_frame_index"] != (frame["frame_id"] - 1) % 120)
                or len(frame["entries"]) != (sum(counts) if lane == "warp" else 3)):
            raise RuntimeError("provisional render graph has incomplete native commands")
        for entry in frame["entries"]:
            if lane == "warp":
                if entry["executable"] != entry["expected_executable"]:
                    raise RuntimeError("recorded Warp command replaced its compiled executable")
                command_ids.append(entry["id"])
            elif entry["id"] != entry["expected_entry"]:
                raise RuntimeError("CUDA stage captured another executable entry")
    if (len(set(command_ids)) != len(command_ids)
            or set(command_ids) != {row["id"] for row in adoptions if row["role"] == "slot_recorded_command"}):
        raise RuntimeError("recorded Warp commands lack exact distinct custody")
    for row in adoptions:
        state = row["state"]
        if (row["guard"] != guard or row["ordinal"] != 1
                or state["target_current_stream"] != state["held_stream"]["handle"]
                or [item["ordinal"] for item in state["primary_contexts"] if item["active"]] != [1]
                or (state["default_device"] != 1 or state["current_context_device"] != 1
                    if lane == "warp" else state["logical_device"] != 1 or state["raw_device"] != 1)):
            raise RuntimeError("render command/event was prepared outside its held target scope")


def _validate_render_warmups(directory, lane, case, fault, *, trace, request):
    import math
    import struct
    from optical_worker.cases import get_case

    data = json.loads((directory / 'warmup_completion.json').read_bytes())
    evidence = data['evidence']
    graph = json.loads((directory / 'render_graph_binding.json').read_bytes())
    validated = json.loads((directory / 'input_validation_report.json').read_bytes())
    first = 14 if lane == 'warp' else 10
    guards = list(range(first, first + 10))
    if (evidence['schema_version'] != 'q54.p12.4.setup-warmup-completion.v1'
            or evidence['session'] != request['guard_configuration']['session']
            or evidence['target_device_ordinal'] != 1 or evidence['completion_count'] != 10
            or evidence['source_catalog_sha256'] != request['catalog_sha256']
            or evidence['prepared_manifest_sha256'] != validated['manifest_sha256']
            or evidence['provisional_readiness_sha256'] != graph['readiness_sha256']
            or evidence['compiled_modules_identity'] != graph['compiled_modules_identity']
            or evidence['profile'] != graph['profile'] or evidence['shadows'] is not graph['shadows']
            or evidence['pass_mode'] != graph['pass_mode'] or evidence['frame_indices'] != list(range(10))
            or evidence['completed_guard_sequences'] != guards
            or evidence['terminal_event_identities'] != [graph['event_ids'][0]] * 10
            or any(len(evidence[name]) != 10 or len(set(evidence[name])) != 10
                   for name in ('transaction_identities', 'terminal_identities'))
            or data['bundle_ids'] != [row['bundle_id'] for row in graph['frames']]
            or data['provisional_warmup_count'] != 0 or data['sample_submission_count'] != 0
            or data['final_execution_readiness_issued'] is not False or len(data['frames']) != 10):
        raise RuntimeError('warmup observations differ from their original graph, input or completed guards')
    launch, record, wait = 'warmup_launch', 'event_record', 'event_wait'
    for sequence, row, frame in zip(guards, data['frames'], graph['frames'], strict=True):
        kinds = []
        for count in frame['phase_launch_counts']:
            kinds += ([record] if fault.endswith('diagnostics') else [])
            kinds += [launch] * count
            kinds += ([record] if fault.endswith('diagnostics') else [])
        kinds += [record, wait]
        guard = {'sequence': sequence, 'role': 'setup'}
        if (row['guard'] != guard or row['before'] != row['after']
                or [leaf['kind'] for leaf in row['leaves']] != kinds
                or any(leaf['returned'] is not True for leaf in row['leaves'])):
            raise RuntimeError('warmup frame lacks exact leaf order, returned wait or ambient restoration')
        ambient = row['before']
        if (ambient['default_device'] != 0 or ambient['current_context'] is not None
                if lane == 'warp' else ambient['logical_device'] != 0 or ambient['raw_device'] != 1):
            raise RuntimeError('warmup ambient scope was not the expected foreign default')
        for leaf in row['leaves']:
            state = leaf['state']
            if (state['held_stream']['ordinal'] != 1
                    or state['target_current_stream'] != state['held_stream']['handle']
                    or [r['ordinal'] for r in state['primary_contexts'] if r['active']] != [1]
                    or (state['default_device'] != 1 or state['current_context_device'] != 1
                        if lane == 'warp' else state['logical_device'] != 1 or state['raw_device'] != 1)):
                raise RuntimeError('warmup leaf was outside its actual target stream')
        observed = [r for r in trace if r['guard'] == guard]
        declared = [r['details']['descriptor']['leaf_kind'] for r in observed
                    if r['operation'] == 'descriptor_observed'
                    and r['details']['descriptor']['operation_id'].startswith('warmup_')]
        if declared != kinds or not any(r['operation'] == 'normal_disarmed' for r in observed):
            raise RuntimeError('warmup raw operations do not match the successful original guard')
    rgb = json.loads((directory / 'warmup_rgb.json').read_bytes())
    encoded = (directory / 'warmup_rgb.f32').read_bytes()
    workload = get_case(case)
    if ((rgb['width'], rgb['height'], rgb['channels'], rgb['dtype'], rgb['frame_id']) !=
            (workload.width, workload.height, 3, 'float32', 10)
            or rgb['guard'] != {'sequence': first + 10, 'role': 'setup'}
            or rgb['bytes'] != len(encoded) or len(encoded) != workload.width * workload.height * 12
            or rgb['sha256'] != hashlib.sha256(encoded).hexdigest()
            or not all(math.isfinite(value) for (value,) in struct.iter_unpack('<f', encoded))):
        raise RuntimeError('diagnostic RGB is not the finite, exact final-frame readback')
    readback_trace = [row for row in trace if row['guard'] == rgb['guard']]
    copies = [row['details']['descriptor'] for row in readback_trace
              if row['operation'] == 'descriptor_observed'
              and row['details']['descriptor']['leaf_kind'] == 'device_to_host_copy']
    expected_copies = ['warmup_rgb_readback']
    if fault == 'runtime_render_warmup_geometry':
        from tests.support.native_runtime_process import WARMUP_GEOMETRY_CHANNELS
        geometry = json.loads((directory / 'warmup_geometry.json').read_bytes())
        if (geometry['frame_id'] != rgb['frame_id'] or geometry['guard'] != rgb['guard']
                or set(geometry['channels']) != set(WARMUP_GEOMETRY_CHANNELS)):
            raise RuntimeError('diagnostic geometry differs from its final-frame RGB')
        for name in WARMUP_GEOMETRY_CHANNELS:
            row = geometry['channels'][name]
            shape = [workload.width * workload.height]
            if name in {'origins', 'directions', 'position_world', 'normal_world'}:
                shape.append(3)
            dtype = ('float64' if name in {'origins', 'directions'} else
                     'int32' if name in {'hit_mask', 'numeric_instance_id', 'material_index'} else 'float32')
            raw = (directory / ('warmup_' + name + '.bin')).read_bytes()
            if (row['shape'] != shape or row['dtype'] != dtype or row['bytes'] != len(raw)
                    or len(raw) != math.prod(shape) * (8 if dtype == 'float64' else 4)
                    or row['sha256'] != hashlib.sha256(raw).hexdigest()):
                raise RuntimeError('diagnostic geometry channel differs from its native ABI/readback')
            expected_copies.append('warmup_' + name + '_readback')
    if ([row['operation_id'] for row in copies] != expected_copies
            or not any(row['operation'] == 'normal_disarmed' for row in readback_trace)):
        raise RuntimeError('diagnostic RGB lacks its actual completed readback guard')


def run_case(repository, revision, environment_path, environment, directory, lane, case, fault):
    from optical_worker.guarded_evidence import validate_custody_artifact
    from optical_worker.native_sources import CATALOG_PATH
    from optical_worker.runtime_custody import (
        RuntimeCustodyRecord, validate_runtime_custody_selection, validate_runtime_custody_attachment,
    )
    from tests.support.runtime_custody_launch import capture_runtime_custody_inputs
    from tests.support.native_runtime_process import (RUNTIME_FIXED_STORAGE_FAULTS, RUNTIME_RENDER_GRAPH_CASES,
                                                      RUNTIME_RENDER_WARMUP_CASES, RUNTIME_FINAL_GRAPH_CASES,
                                                      RUNTIME_FINAL_GRAPH_FAULTS, RUNTIME_FINAL_IDLE_FAULTS,
                                                      RUNTIME_PHASES)

    request = {"root": str(repository), "revision": revision, "lane": lane,
               "catalog_sha256": hashlib.sha256((repository / CATALOG_PATH).read_bytes()).hexdigest(),
               "environment_path": str(environment_path), "environment_sha256": environment.sha256,
               "case_id": case, "go2_model_dir": ASSETS if case.startswith("go2_static_") else None,
               "guard_configuration": {"protocol_version": "q54.closed-worker.guarded-wait.v1",
                   "session": uuid.uuid4().hex, "device_ordinal": 1,
                   "ack_timeout_ms": 1000, "operation_timeout_ms": 120000}}
    inputs = capture_runtime_custody_inputs(request)
    write(directory, "request.json", request)
    write(directory, "runtime_custody_inputs.json", inputs)
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen([sys.executable, "-I", "-S", "-B",
            str(repository / "tests/support/guarded_custody_process.py"), "supervisor", fault,
            str(read_fd), str(directory)], pass_fds=(read_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except BaseException:
        os.close(write_fd)
        raise
    finally:
        os.close(read_fd)
    try:
        write(directory, "parent.json", {"pid": os.getpid(), "supervisor_pid": process.pid})
        output, error = process.communicate(timeout=1235)
        (directory / "supervisor_stdout.json").write_text(output, encoding="utf-8")
        (directory / "supervisor_stderr.txt").write_text(error, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError("supervisor failed; preserve its exact stderr/evidence")
        result = json.loads(output)
        validate_custody_artifact(result["artifact"])
        write(directory, "custody_artifact.json", result["artifact"])
        control = result["artifact"]["supervisor"]["summary"]
        worker = result["artifact"]["worker"]
        if not control["complete"] or control["state"] != "EXITED":
            raise RuntimeError("supervisor did not observe terminal worker exit")
        if Path(f"/proc/{result['worker_pid']}").exists():
            raise RuntimeError("owned worker PID still exists; inspect without broad signaling")
        if (directory / "ordinary_atexit.json").exists():
            raise RuntimeError("worker performed ordinary Python cleanup")
        graph_case = fault in RUNTIME_RENDER_GRAPH_CASES
        warmup_case = fault in RUNTIME_RENDER_WARMUP_CASES
        final_case = fault in RUNTIME_FINAL_GRAPH_CASES
        storage_case = fault in RUNTIME_FIXED_STORAGE_FAULTS or graph_case
        successful = fault in {"runtime_artifacts_success", "runtime_scope_success", "runtime_warp_modules_success",
                               "runtime_fixed_storage_success", "runtime_fixed_storage_diagnostics"} or (
                                   graph_case and fault not in RUNTIME_FINAL_GRAPH_FAULTS)
        if successful:
            expected_phases = (12 if lane == "warp" else 8) if storage_case else (
                11 if fault == "runtime_warp_modules_success" else (7 if lane == "cuda" else 6))
            expected_phases += int(graph_case)
            expected_phases += 12 * int(warmup_case)  # prepare, ten completions, diagnostic readback
            expected_phases += 2 * int(final_case)  # final composition, final target barrier
            if (result["worker_returncode"] != 0 or error or result["stderr"]
                    or control["normal_complete_count"] != expected_phases
                    or control["arm_count"] != expected_phases + 1
                    or control["deadline_expired"] or control["kill_requested"]):
                raise RuntimeError("positive native case did not complete its exact guarded path")
            summary = worker["summary"]
            expected_disposition = "cancelled_before_submission_exit" if final_case else "aborted_synced_exit"
            if (summary["disposition"] != expected_disposition or summary["release_status"] != "completed"
                    or summary["emergency_sync_attempt_count"] != (0 if final_case else 1)
                    or (directory / "terminal_sync_entered.json").exists() == final_case):
                raise RuntimeError("positive case did not finish its state-appropriate terminal disposal")
            if fault in RUNTIME_FINAL_IDLE_FAULTS:
                started = [event for event in worker["transition_trace"]
                           if event["operation"] == "termination_started"]
                if len(started) != 1 or started[0]["details"]["fault"] != RUNTIME_FINAL_IDLE_FAULTS[fault]:
                    raise RuntimeError("final idle fault was not consumed by the native lifecycle")
            if lane == "cuda" and not (directory / "cuda_factory_binding.json").is_file():
                raise RuntimeError("captured CUDA factory bindings were not observed")
            if fault == "runtime_warp_modules_success" or (storage_case and lane == "warp"):
                compiled = json.loads((directory / "warp_module_binding.json").read_bytes())
                if len(compiled["modules"]) != 3 or len(compiled["calls"]) != 6:
                    raise RuntimeError("Warp compile and same-session reuse observations are incomplete")
                scopes = compiled["outer_scopes"]
                if len(scopes) != 3 or any(row["before"] != row["after"]
                        or row["after"]["default_device"] != 0 or row["after"]["current_context"] is not None
                        for row in scopes):
                    raise RuntimeError("Warp compilation did not preserve its outer ambient scope")
                for record, first, reused in zip(compiled["modules"], compiled["calls"][:3], compiled["calls"][3:]):
                    if (record["executable_id"] != first["executable_id"]
                            or first["executable_id"] != reused["executable_id"]
                            or record["name"] != first["module"] or first["module"] != reused["module"]):
                        raise RuntimeError("Warp module reuse replaced a compiled executable")
            if (directory / "failure.json").exists() or (directory / "fault_entered.json").exists():
                raise RuntimeError("positive native case has a failure marker")
        elif fault == "runtime_optical_idle_entry":
            marker = json.loads((directory / "fault_entered.json").read_bytes())
            failure = json.loads((directory / "failure.json").read_bytes())
            scope = json.loads((directory / "idle_sync_scope.json").read_bytes())
            summary = worker["summary"]
            # This fault precedes either lane's compilation/factory stage.
            completed = len(RUNTIME_PHASES)
            if (result["worker_returncode"] != 74 or control["normal_complete_count"] != completed
                    or control["arm_count"] != completed + 1 or control["deadline_expired"] or control["kill_requested"]
                    or marker != {"fault": fault, "boundary": "post_input_idle", "guard": None}
                    or failure["type"] != "SourceBootstrapError"
                    or failure["message"] != "fixed render storage requires completed compilation, once"
                    or summary["disposition"] != "aborted_synced_exit" or summary["release_status"] != "completed"
                    or summary["emergency_sync_attempt_count"] != 1
                    or any(event["operation"] == "guard_failed" for event in worker["transition_trace"])
                    or scope["before"] != scope["after"]):
                raise RuntimeError("optical idle failure did not preserve its original completed guards and scope")
            inside = scope["inside"]
            if ([item["ordinal"] for item in inside["primary_contexts"] if item["active"]] != [1]
                    or (inside["logical_device"] != 1 or inside["raw_device"] != 1
                        if lane == "cuda" else inside["default_device"] != 1 or inside["current_context_device"] != 1)):
                raise RuntimeError("optical idle disposal synchronized outside its exact target")
        elif fault in RUNTIME_FINAL_GRAPH_FAULTS:
            marker = json.loads((directory / "fault_entered.json").read_bytes())
            partial = json.loads((directory / "final_graph_partial.json").read_bytes())
            guard = {"sequence": 26 if lane == "warp" else 22, "role": "setup"}
            summary = worker["summary"]
            if (result["worker_returncode"] != 74 or control["deadline_expired"] or control["kill_requested"]
                    or marker != {"fault": fault, "boundary": "final_target_barrier", "guard": guard}
                    or partial["barrier_guard"] != guard or not partial["barrier_returned"]
                    or partial["admitted"] or partial["quiescent"] or not partial["sealed_coverage_retained"]
                    or partial["guardian_state"] != "TRANSFERRED_TO_QUARANTINE"
                    or summary["disposition"] != "quarantined_process_exit" or summary["custody"] != "full_quarantine"
                    or summary["emergency_sync_attempt_count"] != 0
                    or summary["final_setup"]["quiescence_sha256"] is not None
                    or summary["final_setup"]["barrier_returned"]
                    or control["normal_complete_count"] != guard["sequence"]
                    or control["arm_count"] != guard["sequence"] + 1):
                raise RuntimeError("final barrier failure did not retain its exact sealed graph and exit")
        else:
            if result["worker_returncode"] == 0 or not (directory / "fault_entered.json").is_file():
                raise RuntimeError("fixed fault did not reach its intended real boundary")
            marker = json.loads((directory / "fault_entered.json").read_bytes())
            from tests.support.native_runtime_process import (
                RUNTIME_WARP_IDLE_FAULTS, RUNTIME_WARP_COMPILE_FAULTS,
                RUNTIME_WARP_ADMISSION_FAULTS, RUNTIME_WARP_ADMISSION_ERRORS,
            )
            idle_fault = fault in RUNTIME_WARP_IDLE_FAULTS
            compile_fault = fault in RUNTIME_WARP_COMPILE_FAULTS
            idle_synced = fault in {"runtime_warp_modules_entry_raise", "runtime_warp_modules_issuance_raise"}
            compile_guard = 6 if fault == "runtime_warp_modules_import_raise" else 7
            expected_guard = None if idle_fault else {
                "sequence": compile_guard if compile_fault else (
                    (11 if lane == "warp" else 7) if storage_case else (
                        10 if fault.startswith("runtime_warp_modules_") else (6 if "factory" in fault else 3))),
                "role": "setup"}
            if marker["fault"] != fault or marker["guard"] != expected_guard:
                raise RuntimeError("fixed fault occurred in another guard or case")
            if fault.endswith("hang"):
                if not control["deadline_expired"] or not control["kill_requested"]:
                    raise RuntimeError("factory hang was not terminated by the original guard deadline")
            elif worker is None or worker["summary"]["emergency_sync_attempt_count"] != int(idle_synced):
                raise RuntimeError("fixed failure has an unexpected synchronization count")
            if storage_case:
                completed = expected_guard["sequence"]
                if control["normal_complete_count"] != completed or control["arm_count"] != completed + 1:
                    raise RuntimeError("fixed storage failure completed or replaced its original guard")
                if (directory / "fixed_storage_binding.json").exists():
                    raise RuntimeError("failed storage issued a successful binding")
                if fault.endswith("hang") and (result["worker_returncode"] != -9
                        or (directory / "worker.json").exists() or (directory / "failure.json").exists()):
                    raise RuntimeError("fixed storage hang lacks original-deadline kill with honest partial evidence")
                if not fault.endswith("hang"):
                    failure = json.loads((directory / "failure.json").read_bytes())
                    events = worker["transition_trace"]
                    failed = [row for row in events if row["operation"] == "guard_failed"]
                    if (result["worker_returncode"] != 74
                            or (failure["type"], failure["message"]) != (
                                "RuntimeError", "fixed render storage failure: " + fault)
                            or worker["summary"]["custody"] != "full_quarantine"
                            or len(failed) != 1 or failed[0]["guard"] != expected_guard
                            or any(row["operation"] == "low_level_leaf_entered"
                                   for row in events[marker["lifecycle_event_count"]:])):
                        raise RuntimeError("fixed storage failure lost custody or continued native work")
            elif compile_fault:
                if control["normal_complete_count"] != compile_guard or control["arm_count"] != compile_guard + 1:
                    raise RuntimeError("failed Warp segment completed or started another guard")
                imports = json.loads((directory / "warp_import_armed.json").read_bytes())["imports"]
                expected_imports = ["optics.device_scene", "optics.device_bvh", "optics.warp_native_render"]
                if compile_guard == 6:
                    expected_imports = expected_imports[:1]
                if ([row["module"] for row in imports] != expected_imports
                        or any(row["guard"] != {"sequence": 6, "role": "setup"}
                               or row["state"] != "ARMED" or row["started"] != "started" for row in imports)):
                    raise RuntimeError("Warp optical import did not begin under its original ARMED guard")
                if fault.endswith("hang"):
                    if (result["worker_returncode"] != -9 or (directory / "worker.json").exists()
                            or (directory / "failure.json").exists()
                            or not marker["mapping_guardian_owned"]):
                        raise RuntimeError("Warp load hang lacks original-deadline kill or preheld mapping")
                else:
                    failure = json.loads((directory / "failure.json").read_bytes())
                    if (result["worker_returncode"] != 74 or failure["type"] != "RuntimeError"
                            or failure["message"] != "fixed Warp guarded compilation failure: " + fault
                            or worker["summary"]["custody"] != "full_quarantine"):
                        raise RuntimeError("Warp failed segment lost its original exception or custody")
                    events = worker["transition_trace"]
                    failed = [row for row in events if row["operation"] == "guard_failed"]
                    if (len(failed) != 1 or failed[0]["guard"] != expected_guard
                            or any(row["operation"] == "low_level_leaf_entered"
                                   for row in events[marker["lifecycle_event_count"]:])):
                        raise RuntimeError("Warp failure changed guards or entered another native leaf")
                    roots = json.loads((directory / "warp_failure_roots.json").read_bytes())
                    if (not roots["python_module_guardian_owned"] or roots["completed_records"] != 0
                            or roots["module_set_issued"] or roots["guardian_state"] != "TRANSFERRED_TO_QUARANTINE"):
                        raise RuntimeError("Warp failure lost its partial module graph")
                    if compile_guard == 6:
                        if roots["load_attempts"] or roots["python_module_id"] != marker["module_id"]:
                            raise RuntimeError("failed import proceeded to native loading")
                    else:
                        observation = json.loads((directory / "warp_compile_failure_scope.json").read_bytes())
                        if (observation["before"] != observation["after"]
                                or observation["after"]["default_device"] != 0
                                or observation["after"]["current_context"] is not None
                                or observation["guard"] != expected_guard):
                            raise RuntimeError("failed Warp compile did not restore its ambient context")
                        if (not roots["module_guardian_owned"] or not roots["mapping_guardian_owned"]
                                or len(roots["load_attempts"]) != 1):
                            raise RuntimeError("failed Warp load lost its preheld module or mapping")
                        published = fault != "runtime_warp_modules_load_before_raise"
                        if (published and (roots["executable_id"] != marker["executable_id"]
                                           or len(roots["calls"]) != 1)):
                            raise RuntimeError("published Warp executable was not retained after failure")
                        if not published and (roots["executable_id"] is not None or roots["calls"]):
                            raise RuntimeError("before-load failure nevertheless published an executable")
                        if fault == "runtime_warp_modules_adoption_raise" and not roots["executable_session_owned"]:
                            raise RuntimeError("returned executable was not retained before failed adoption")
                if (directory / "warp_module_binding.json").exists():
                    raise RuntimeError("failed Warp preparation issued a successful binding")
            elif fault.startswith("runtime_warp_modules_"):
                failure = json.loads((directory / "failure.json").read_bytes())
                issuance = "issuance" in fault
                completed = 10 if not idle_fault or issuance else 6
                arms = completed + (int(idle_synced) if idle_fault else 1)
                expected_error = ("ValueError", "Warp executable or kernel ABI changed")
                if fault in RUNTIME_WARP_ADMISSION_FAULTS:
                    expected_error = RUNTIME_WARP_ADMISSION_ERRORS[fault.removeprefix("runtime_warp_modules_")]
                if idle_fault:
                    if "unknown" in fault:
                        expected_error = (("ValueError", "Warp module issuance requires healthy IDLE target scope")
                                          if issuance else ("SourceBootstrapError",
                                          "Warp module preparation requires healthy IDLE target scope"))
                    elif issuance:
                        expected_error = ("RuntimeError", "fixed Warp CPU issuance failure")
                    else:
                        expected_error = ("SourceBootstrapError", "Warp modules require the original uploaded and validated input")
                if (result["worker_returncode"] != 74 or control["normal_complete_count"] != completed
                        or control["arm_count"] != arms
                        or (failure["type"], failure["message"]) != expected_error):
                    raise RuntimeError("Warp fault did not reject at the intended production boundary")
                if fault in RUNTIME_WARP_ADMISSION_FAULTS:
                    events = worker["transition_trace"]
                    failed = [row for row in events if row["operation"] == "guard_failed"]
                    if (worker["summary"]["custody"] != "full_quarantine"
                            or len(failed) != 1 or failed[0]["guard"] != expected_guard
                            or any(row["operation"] == "low_level_leaf_entered"
                                   for row in events[marker["lifecycle_event_count"]:])
                            or (directory / "warp_module_binding.json").exists()):
                        raise RuntimeError("Warp admission failure proceeded to reuse or lost custody")
                if idle_fault:
                    if any(event["operation"] == "guard_failed" for event in worker["transition_trace"]):
                        raise RuntimeError("IDLE fault fabricated a failed guard")
                    if marker["held_compilation_records"] != (3 if issuance else 0):
                        raise RuntimeError("IDLE fault lost its previously compiled records")
                    summary = worker["summary"]
                    if (idle_synced and (summary["disposition"] != "aborted_synced_exit"
                                         or summary["release_status"] != "completed")):
                        raise RuntimeError("healthy IDLE fault did not complete bounded terminal disposal")
                    if not idle_synced and summary["custody"] != "full_quarantine":
                        raise RuntimeError("unprovable IDLE custody did not retain the graph")
        if storage_case:
            storage_fault = ("runtime_fixed_storage_" + ("diagnostics" if fault.endswith("diagnostics") else "success")
                             if graph_case else fault)
            _validate_fixed_storage(directory, lane, case, storage_fault, successful=successful or final_case)
        final_row = None
        if final_case:
            final = json.loads((directory / ("final_graph_binding.json" if successful else "final_graph_partial.json")).read_bytes())
            gate = json.loads((repository / "benchmarks/results/q54_p12_4/gate.json").read_bytes())
            final_row = next(row for row in gate["mandatory_rows"] if row["row_id"] == final["row"]["row_id"])
            if final_row != final["row"] or final_row["case_id"] != case or final_row["boundary"] not in {"R0", "R1"}:
                raise RuntimeError("final graph row differs from its frozen case")
        if final_case and successful:
            from optical_worker.protocol import canonical_sha256
            final_summary = worker["summary"]["final_setup"]
            if (final_row != final["row"] or final_row["case_id"] != case or final_row["boundary"] not in {"R0", "R1"}
                    or final["before"] != final["after"] or final["before"] != final["after_barrier"]
                    or not final["barrier_returned"] or final["sample_submission_count"] != 0
                    or final["acquire_cancel_observed"] is not True or final["frame_ids"] != [1, 2, 3]
                    or [plan["expected_generation"] for plan in final["plans"]] != [1, 2, 3]
                    or final["readiness"]["warmup_frames_completed"] != 10
                    or final_summary["quiescence_sha256"] != canonical_sha256(final["quiescence"])
                    or final_summary["barrier_guard"] != final["barrier_guard"]
                    or final_summary["graph"]["graph_sha256"] != final["graph_sha256"]):
                raise RuntimeError("final graph does not match its frozen row and acknowledged target barrier")
            state = final["barrier_scope"]
            if (state["target_current_stream"] != state["held_stream"]["handle"]
                    or [item["ordinal"] for item in state["primary_contexts"] if item["active"]] != [1]
                    or (state["default_device"] != 1 or state["current_context_device"] != 1
                        if lane == "warp" else state["logical_device"] != 1 or state["raw_device"] != 1)):
                raise RuntimeError("final barrier ran outside its exact target scope")
        if graph_case:
            _validate_render_graph(directory, lane, fault, trace=worker["transition_trace"], final_row=final_row)
        if warmup_case:
            _validate_render_warmups(directory, lane, case, fault, trace=worker["transition_trace"], request=request)
        raw_path = directory / "runtime_custody.json"
        if raw_path.exists():
            raw = raw_path.read_bytes()
            validate_runtime_custody_selection(json.loads((directory / "runtime_custody_selection.json").read_bytes()), raw)
            record = RuntimeCustodyRecord.from_payload(json.loads(raw), environment=environment, trusted_launch_inputs=inputs)
            validate_runtime_custody_attachment(record, environment, inputs, result["artifact"])
            if successful and not record.to_payload()["summary"]["input_validation_observed"]:
                raise RuntimeError("positive native case lacks input validation")
        else:
            raise RuntimeError("native case lacks its retained runtime custody prefix")
        return {"lane": lane, "case": case, "fault": fault, "worker_pid": result["worker_pid"],
                "worker_returncode": result["worker_returncode"], "supervisor_pid": process.pid}
    finally:
        os.close(write_fd)
        if process.poll() is None:
            # Existing parent-liveness handling owns termination and reaping.
            output, error = process.communicate(timeout=130)
            (directory / "parent_finally_stdout.txt").write_text(output, encoding="utf-8")
            (directory / "parent_finally_stderr.txt").write_text(error, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cuda-environment", type=Path, required=True)
    parser.add_argument("--cuda-environment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise RuntimeError("acceptance requires isolated Linux python -I -S -B")
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"]).decode().strip() != args.revision:
        raise RuntimeError("acceptance revision differs from reviewed command")
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]).strip():
        raise RuntimeError("acceptance checkout is dirty")
    sys.path.insert(0, str(repository))
    from optical_worker.runtime_environment import RuntimeEnvironmentContract
    from tools.optical_cuda_build_recipe import _PREFIX, _create_directory
    if Path(sys.prefix) != _PREFIX or tuple(sys.version_info[:3]) != (3, 12, 13):
        raise RuntimeError("wrong acceptance interpreter")
    forbidden = {"CUDA_VISIBLE_DEVICES", "PYTORCH_NVML_BASED_CUDA_CHECK", "LD_PRELOAD", "LD_AUDIT",
                 "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                 "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_MODULE_LOADING", "TORCH_USE_RTLD_GLOBAL",
                 "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH"}
    if forbidden.intersection(os.environ):
        raise RuntimeError("unreviewed ambient native configuration")
    cuda = RuntimeEnvironmentContract.from_payload(json.loads(args.cuda_environment.read_bytes()))
    cuda.verify_files(expected_sha256=args.cuda_environment_sha256)
    if not any(row["id"] == "cuda_artifacts" for row in cuda.to_payload()["roots"]):
        raise ValueError("CUDA profile lacks the selected fixed deployment")
    warp_path = repository / "tests/data/runtime_environments/183_warp_go2_author_v3.json"
    warp = RuntimeEnvironmentContract.from_payload(json.loads(warp_path.read_bytes()))
    warp.verify_files(expected_sha256=warp.sha256)
    output = _create_directory(args.output)
    check_bad_artifact_copies(repository, cuda, output)
    results = []
    for name, lane, case, fault in CASES:
        directory = _create_directory(output / name)
        print(name + ": starting", flush=True)
        results.append(run_case(repository, args.revision, args.cuda_environment if lane == "cuda" else warp_path,
                                cuda if lane == "cuda" else warp, directory, lane, case, fault))
        write(directory, "accepted.json", results[-1])
        print(name + ": accepted", flush=True)
    write(output, "results.json", {"revision": args.revision, "cases": results})


if __name__ == "__main__":
    main()
