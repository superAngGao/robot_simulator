"""Fixed captured CUDA compilation child and dedicated compiler-tree supervisor.

This is a sacrificial validation driver, never a product or arbitrary command
runner. Native execution requires the exact candidate's independent preflight.
"""

from __future__ import annotations

import importlib.util
import json
import os
import select
import sys
from pathlib import Path


def _runtime_driver():
    path = Path(__file__).with_name("native_runtime_process.py")
    spec = importlib.util.spec_from_file_location("fixed_native_runtime_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request_path(request_path, directory):
    request_path = Path(request_path).resolve(strict=True)
    directory = Path(directory).resolve(strict=True)
    if request_path != directory / "request.json":
        raise ValueError("fixed compilation request must belong to its output directory")
    return request_path, directory


def worker(fd, request_path, directory):
    _, directory = _request_path(request_path, directory)
    _runtime_driver().child("runtime_scope_success", fd, directory, compilation=True)
    raise AssertionError("sacrificial worker returned without hard exit")


def supervise(liveness_fd, request_path, directory, policy_path):
    request_path, directory = _request_path(request_path, directory)
    root = Path(__file__).resolve(strict=True).parents[2]
    request = json.loads(request_path.read_bytes())
    if Path(request["root"]).resolve(strict=True) != root or request["lane"] != "cuda":
        raise ValueError("fixed CUDA supervisor source/lane mismatch")
    policy = json.loads(Path(policy_path).read_bytes())
    if (set(policy) != {"schema_version", "worker_environment", "libc_file"}
            or policy["schema_version"] != "q54.p12.4.compiler-supervisor-launch.v1"):
        raise ValueError("fixed compiler supervisor launch policy required")

    # Only the dedicated supervisor imports OS process ownership. The worker
    # continues through the cold captured bootstrap without this sys.path edit.
    sys.path.insert(0, str(root))
    from optical_worker.compiler_supervision import _CompilerJob
    from optical_worker.wait_protocol import GuardConfiguration, GuardProtocolError
    from optical_worker.wait_supervision import GuardSupervisorCore, GuardSupervisorTransport
    from optical_worker.guarded_evidence import VALIDATION_VERSION, validate_custody_artifact

    driver = _runtime_driver()
    core = GuardSupervisorCore(GuardConfiguration.from_payload(request["guard_configuration"]))
    job = _CompilerJob(core, libc_file=policy["libc_file"])
    parent_alive = True
    diagnostics = []
    transport = None

    def retain_failure(error):
        if len(diagnostics) < 16:
            diagnostics.append(error)
        if core.state != "EXITED":
            core.fault("process_observation_failed")
            if job.obligation.replay.state != "NEW":
                job.after_core_transition()
        job.obligation.observation_failed("proc_unavailable" if isinstance(error, GuardProtocolError) else "other_os")

    try:
        channel = job.launch(repository_root=root, request_path=request_path,
                             output_directory=directory, environment=policy["worker_environment"])
        transport = GuardSupervisorTransport(channel, core, compiler_job=job)
        driver._write(directory, "supervisor_identity.json", {
            "pid": os.getpid(), "ppid": os.getppid(), "worker_pid": job.process.pid,
        })
    except BaseException as error:
        if job.process is None or not getattr(job.process, "_child_created", False):
            raise  # No worker was forked; no CLOSED evidence is issued.
        channel = job.transport_channel
        retain_failure(error)
    last_prefix = -1
    while job.obligation.replay.state != "CLOSED":
        try:
            job.service_processes()
            if job.obligation.replay.state == "CLOSED":
                break
            readers = [channel] if core.state not in {"TERMINATING", "EXITED"} else []
            writers = [channel] if transport is not None and transport.wants_write else []
            if parent_alive:
                readers.append(liveness_fd)
            readable, writable, _ = select.select(readers, writers, [], 0.01)
            if parent_alive and liveness_fd in readable and os.read(liveness_fd, 1) == b"":
                parent_alive = False
                if core.state != "EXITED":
                    core.fault("parent_lost")
                    if job.obligation.replay.state != "NEW":
                        job.after_core_transition()
            if transport is not None and core.state != "EXITED":
                transport.service(readable=channel in readable, writable=channel in writable)
        except BaseException as error:
            # Retain ownership after process/proc/transport failure. The same
            # frozen deadline and unique pidfd attempts continue; no broad kill,
            # Popen waiter, renewed timeout or ordinary partial cleanup.
            retain_failure(error)
        if len(job.obligation.events) != last_prefix:
            try:
                driver._write(directory, "compiler_job_prefix.json", job.obligation.record().to_payload())
                last_prefix = len(job.obligation.events)
            except BaseException as error:
                if len(diagnostics) < 16:
                    diagnostics.append(error)

    channel.close()
    job_record = job.obligation.record(require_closed=True)
    worker_path = directory / "worker.json"
    if not worker_path.exists():
        worker_path = directory / "prefix.json"
    worker_record = json.loads(worker_path.read_bytes()) if worker_path.exists() else None
    artifact = {"schema_version": VALIDATION_VERSION, "worker": worker_record,
                "supervisor": core.evidence(require_final=True).to_payload()}
    validate_custody_artifact(artifact)
    result = {
        "schema_version": "q54.c2d.cuda-compilation-supervision.v0", "artifact": artifact,
        "compiler_job": job_record.to_payload(), "parent_alive": parent_alive,
        "pid": os.getpid(), "worker_pid": job.process.pid, "worker_returncode": job.process.returncode,
        "stdout": bytes(job.output["stdout"]).decode("utf-8", errors="replace"),
        "stderr": bytes(job.output["stderr"]).decode("utf-8", errors="replace"),
        "output_truncated": job.output_truncated,
        "sampled_processes": list(job.observed_members.values()),
        "diagnostics": [{"type": type(error).__name__, "message": str(error)[:512]} for error in diagnostics],
    }
    driver._write(directory, "supervisor.json", result)
    return result


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif len(sys.argv) == 6 and sys.argv[1] == "supervisor":
        supervise(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise ValueError("fixed compiler worker/supervisor invocation required")


if __name__ == "__main__":
    main()
