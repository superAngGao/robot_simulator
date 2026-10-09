"""Fixed Go2 artifact acceptance through the existing captured worker supervisor.

Requires exact whole-packet preflight. No compilation, timings, warmups, renderer
launch, product activation or new process-management implementation lives here.
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


def run_case(repository, revision, environment_path, environment, directory, lane, case, fault):
    from optical_worker.guarded_evidence import validate_custody_artifact
    from optical_worker.native_sources import CATALOG_PATH
    from optical_worker.runtime_custody import (
        RuntimeCustodyRecord, validate_runtime_custody_selection, validate_runtime_custody_attachment,
    )
    from tests.support.runtime_custody_launch import capture_runtime_custody_inputs

    request = {"root": str(repository), "revision": revision, "lane": lane,
               "catalog_sha256": hashlib.sha256((repository / CATALOG_PATH).read_bytes()).hexdigest(),
               "environment_path": str(environment_path), "environment_sha256": environment.sha256,
               "case_id": case, "go2_model_dir": ASSETS,
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
        successful = fault in {"runtime_artifacts_success", "runtime_scope_success", "runtime_warp_modules_success"}
        if successful:
            expected_phases = 11 if fault == "runtime_warp_modules_success" else (7 if lane == "cuda" else 6)
            if (result["worker_returncode"] != 0 or error or result["stderr"]
                    or control["normal_complete_count"] != expected_phases
                    or control["arm_count"] != expected_phases + 1
                    or control["deadline_expired"] or control["kill_requested"]):
                raise RuntimeError("positive native case did not complete its exact guarded path")
            summary = worker["summary"]
            if (summary["disposition"] != "aborted_synced_exit" or summary["release_status"] != "completed"
                    or summary["emergency_sync_attempt_count"] != 1):
                raise RuntimeError("positive case did not finish its original terminal disposal")
            if lane == "cuda" and not (directory / "cuda_factory_binding.json").is_file():
                raise RuntimeError("captured CUDA factory bindings were not observed")
            if fault == "runtime_warp_modules_success":
                compiled = json.loads((directory / "warp_module_binding.json").read_bytes())
                if len(compiled["modules"]) != 3 or len(compiled["calls"]) != 6:
                    raise RuntimeError("Warp compile and same-session reuse observations are incomplete")
                for record, first, reused in zip(compiled["modules"], compiled["calls"][:3], compiled["calls"][3:]):
                    if (record["executable_id"] != first["executable_id"]
                            or first["executable_id"] != reused["executable_id"]
                            or record["name"] != first["module"] or first["module"] != reused["module"]):
                        raise RuntimeError("Warp module reuse replaced a compiled executable")
            if (directory / "failure.json").exists() or (directory / "fault_entered.json").exists():
                raise RuntimeError("positive native case has a failure marker")
        else:
            if result["worker_returncode"] == 0 or not (directory / "fault_entered.json").is_file():
                raise RuntimeError("fixed fault did not reach its intended real boundary")
            marker = json.loads((directory / "fault_entered.json").read_bytes())
            if marker["fault"] != fault or marker["guard"] != {
                    "sequence": 6 if "factory" in fault else 3, "role": "setup"}:
                raise RuntimeError("fixed fault occurred in another guard or case")
            if fault.endswith("hang"):
                if not control["deadline_expired"] or not control["kill_requested"]:
                    raise RuntimeError("factory hang was not terminated by the original guard deadline")
            elif worker is None or worker["summary"]["emergency_sync_attempt_count"] != 0:
                raise RuntimeError("failed armed setup attempted another synchronization")
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
