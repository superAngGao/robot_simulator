"""Fixed, CPU-only installed-manager acceptance; execute only after preflight.

This replaces the old compiler-process campaign. It exercises the deployment
service argv/footer/closure checks without importing a compiler or a backend.
It is not a worker dependency or a publisher. All outputs remain test evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


CASES = ("normal", "descendant", "timeout", "launcher_loss", "signal", "nonzero")


def write_record(directory, name, value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    temporary = directory / (name + ".tmp")
    with temporary.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(directory / name)


def identity():
    return {"pid": os.getpid(), "pgid": os.getpgrp(),
            "cgroup": Path("/proc/self/cgroup").read_text().strip(),
            "monotonic_ns": time.monotonic_ns()}


def fixture(case, directory):
    """Known stdlib processes only; descendants intentionally escape the PGID."""
    write_record(directory, "main.json", identity())
    if case == "sentinel":
        time.sleep(90)
        return
    if case in {"descendant", "timeout", "launcher_loss"}:
        read_fd, write_fd = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            os.setsid()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            write_record(directory, "child.json", identity())
            os.write(write_fd, b"R")
            os.close(write_fd)
            if case == "descendant":
                deadline = time.monotonic() + 30
                while not (directory / "release-child").exists():
                    if time.monotonic() >= deadline:
                        os._exit(9)
                    time.sleep(0.02)
                write_record(directory, "child-complete.json", identity())
                os._exit(0)
            time.sleep(30)  # Manager must terminate this, including ignored TERM.
            os._exit(9)
        os.close(write_fd)
        if os.read(read_fd, 1) != b"R":
            raise RuntimeError("descendant did not become ready")
        os.close(read_fd)
    write_record(directory, "expected-output.json", identity())
    if case == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
        raise RuntimeError("SIGTERM unexpectedly returned")
    if case == "nonzero":
        raise SystemExit(7)


def fixture_argv(repository, job, unit):
    """Reuse production settings; replace only the recipe and fixture budgets."""
    from tools.optical_cuda_build_recipe import _BUILD_ENVIRONMENT, _PREFIX, build_service_argv

    environment = {**_BUILD_ENVIRONMENT, "PATH": str(_PREFIX / "bin") + ":/usr/bin:/bin",
                   "LD_LIBRARY_PATH": "", "TMPDIR": str(job), "LC_ALL": "C"}
    original = build_service_argv(repository, job, unit, environment)
    shortened = {"--property=RuntimeMaxSec=120s": "--property=RuntimeMaxSec=6s",
                 "--property=TimeoutStartSec=120s": "--property=TimeoutStartSec=6s",
                 "--property=TimeoutStopSec=10s": "--property=TimeoutStopSec=1s"}
    if not set(shortened).issubset(original):
        raise RuntimeError("production service budgets changed; review the fixture")
    argv = [shortened.get(item, item) for item in original]
    command = argv.index(str(repository / "tools/optical_cuda_build_recipe.py"))
    return argv[:command] + [str(repository / "tests/support/cuda_artifact_service_acceptance.py"),
                            "--fixture", job.name, "--output", str(job)]


def until(predicate, deadline, description):
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(description)
        time.sleep(0.05)


def accept_case(case, directory, repository, manager, sentinel):
    from tools.optical_cuda_build_status import require_normal_exit, validate_status
    from tools.optical_cuda_deploy import observe_group_closure

    unit = "robot-sim-cuda-" + uuid.uuid4().hex + "-0.service"
    argv = fixture_argv(repository, directory, unit)
    write_record(directory, "invocation.json", {"unit": unit, "argv": argv})
    deadline = time.monotonic() + 20
    with (directory / "waiter.log").open("xb") as log:
        waiter = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        try:
            write_record(directory, "waiter.json", {"pid": waiter.pid})
            if case in {"descendant", "timeout", "launcher_loss"}:
                until(lambda: (directory / "expected-output.json").is_file(), deadline, "main output absent")
                main = json.loads((directory / "main.json").read_bytes())
                child = json.loads((directory / "child.json").read_bytes())
                if child["pgid"] == main["pgid"] or child["cgroup"] != main["cgroup"]:
                    raise RuntimeError("fixture did not escape the PGID within the service")
                until(lambda: not Path(f"/proc/{main['pid']}").exists(), deadline, "main has not exited")
                if waiter.poll() is not None or not Path(f"/proc/{child['pid']}").exists():
                    raise RuntimeError("waiter failed to remain attached to the live descendant")
                write_record(directory, "after-main-exit.json", {
                    "waiter_returncode": None, "child_present": True, "monotonic_ns": time.monotonic_ns()})
                if case == "descendant":
                    (directory / "release-child").touch(exist_ok=False)
                elif case == "launcher_loss":
                    # The owned systemd-run process is the deployment waiter's
                    # connection. Kill only this Popen handle; the service stays.
                    waiter.kill()
            returncode = waiter.wait(timeout=max(0.1, deadline - time.monotonic()))
        finally:
            if waiter.poll() is None:
                try:
                    waiter.wait(timeout=10)  # Manager's independent 6s + 1s bound.
                except subprocess.TimeoutExpired:
                    waiter.kill()
                    waiter.wait(timeout=5)
                    raise RuntimeError(f"manager did not terminate {unit}; keep its evidence and inspect")

    main = json.loads((directory / "main.json").read_bytes())
    if not main["cgroup"].startswith("0::") or "\n" in main["cgroup"]:
        raise RuntimeError("fixture is not in one unified service cgroup")
    group = main["cgroup"][3:]
    # A lost waiter cannot observe its own terminal cgroup. The campaign observes
    # it after the bounded service ends, without treating waiter death as closure.
    closure = None

    def closed():
        nonlocal closure
        try:
            closure = observe_group_closure(group, unit=unit, manager=manager)
            return True
        except RuntimeError as error:
            if str(error) != "build cgroup is still populated":
                raise
            return False

    until(closed, deadline, "owned service cgroup did not close")
    # ExecStopPost belongs to this group too. Read its non-atomic raw record
    # only after closure, including when the original waiter has been killed.
    raw = validate_status(json.loads((directory / "termination.json").read_bytes()), unit=unit)
    if raw["cgroup_path"] != group:
        raise RuntimeError("manager footer and fixture differ on their owned group")
    try:
        require_normal_exit(raw, unit=unit, waiter_returncode=returncode)
        eligible = True
    except ValueError:
        eligible = False
    if eligible != (case in {"normal", "descendant"}):
        raise RuntimeError("production exit gate accepted/rejected the wrong service outcome")
    if case in {"timeout", "launcher_loss"} and raw["service_result"] != "timeout":
        raise RuntimeError("ignored-TERM service did not end by the manager deadline")
    if case == "launcher_loss" and returncode != -signal.SIGKILL:
        raise RuntimeError("fixture did not lose its deployment waiter")
    if case == "signal" and (raw["exit_code"], raw["exit_status"]) != ("killed", "TERM"):
        raise RuntimeError("manager did not preserve main SIGTERM")
    if case == "nonzero" and (raw["exit_code"], raw["exit_status"]) != ("exited", "7"):
        raise RuntimeError("manager did not preserve nonzero main exit")
    if not (directory / "expected-output.json").is_file():
        raise RuntimeError("fixture never produced its output")
    if case == "descendant" and not (directory / "child-complete.json").is_file():
        raise RuntimeError("descendant did not finish before successful group closure")
    if sentinel.poll() is not None:
        raise RuntimeError("unrelated sentinel was affected")
    sentinel_identity = json.loads((directory.parent / "sentinel/main.json").read_bytes())
    if sentinel_identity["pid"] != sentinel.pid or sentinel_identity["cgroup"] == main["cgroup"]:
        raise RuntimeError("sentinel is not the independent owned process")
    result = {"case": case, "unit": unit, "waiter_returncode": returncode,
              "termination": raw, "cgroup_outcome": closure, "normal_exit_eligible": eligible,
              "sentinel_alive": True}
    write_record(directory, "result.json", result)
    return result


def campaign(output):
    if sys.platform != "linux" or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise RuntimeError("acceptance requires isolated Linux python -I -S -B")
    repository = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repository))
    from tools.optical_cuda_build_recipe import _PREFIX, _create_directory
    if Path(sys.prefix) != _PREFIX or tuple(sys.version_info[:3]) != (3, 12, 13):
        raise RuntimeError("acceptance requires the frozen deployment interpreter")
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"]).strip():
        raise RuntimeError("acceptance requires a clean reviewed checkout")
    manager = subprocess.check_output(["/usr/bin/systemctl", "--user", "show", "--property=ControlGroup", "--value"], timeout=10).decode().strip()
    if manager != f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service":
        raise RuntimeError("unexpected user manager")
    output = _create_directory(output)
    sentinel_directory = _create_directory(output / "sentinel")
    sentinel = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                                 "--fixture", "sentinel", "--output", str(sentinel_directory)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until(lambda: (sentinel_directory / "main.json").is_file(), time.monotonic() + 5, "sentinel did not start")
        results = []
        for case in CASES:
            results.append(accept_case(case, _create_directory(output / case), repository, manager, sentinel))
            print(case + ": accepted", flush=True)
        write_record(output, "results.json", {"revision": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"]).decode().strip(), "cases": results})
    finally:
        if sentinel.poll() is None:
            sentinel.terminate()
        sentinel.wait(timeout=5)
        write_record(output, "sentinel-terminal.json", {"pid": sentinel.pid, "returncode": sentinel.returncode})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", choices=(*CASES, "sentinel"))
    args = parser.parse_args()
    if args.fixture:
        fixture(args.fixture, args.output)
    else:
        campaign(args.output)
