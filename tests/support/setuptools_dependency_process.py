"""Fixed CPU dependency proof; execution requires its own exact reviewed preflight.

Actual source session, custodian finder, external loaders, initialization owner,
guardian and guard transport run here. Input upload/device context/toolchain are
explicit outer stand-ins. No compilation receipt, COMPLETE, readiness or native
execution can be claimed from this driver. Each process runs once and exits.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
from types import SimpleNamespace


PROFILE = "tests/data/runtime_environments/183_cuda_compilation_author_v1.json"
CATALOG = "optical_worker/data/native_sources/v13.json"
PROFILE_SHA256 = "0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753"
CATALOG_SHA256 = "17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399"


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _create_run_directory(directory):
    assert directory.is_absolute() and directory.resolve() == directory
    directory.mkdir(mode=0o700)  # Never reuse a previous run directory.
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        assert stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
        # mkdir inherits a shared parent's setgid bit even with mode0700.
        # Set the exact mode on the newly owned directory's open descriptor.
        os.fchmod(fd, 0o700)
        current = directory.lstat()
        assert (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino)
        assert stat.S_IMODE(current.st_mode) == 0o700
    finally:
        os.close(fd)


def _cpu_only(root, revision):
    commands = (("git", "rev-parse", "HEAD"), ("git", "show", f"{revision}:{CATALOG}"))

    def audit(event, args):
        if event == "import" and args[0].split(".")[0] in {"torch", "warp", "numpy", "scipy", "trimesh"}:
            raise RuntimeError("dependency-only driver forbids backend import")
        if event == "subprocess.Popen":
            executable, argv, cwd, env = args
            if (executable == "git" and type(argv) is list and tuple(argv) in commands
                    and type(cwd) is type(root) and cwd == root and env is None):
                return  # Exact read-only source revalidation; still owned descendants.
            raise RuntimeError("dependency-only driver forbids this process")
        if event in {"os.system", "os.fork", "os.posix_spawn", "ctypes.dlopen"}:
            raise RuntimeError("dependency-only driver forbids process/library execution")

    return audit


def _reap_cpu_group(process):
    # Caller created this child's private session/group with start_new_session.
    # Include its synchronous read-only Git children on timeout or early death.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def child(fd, directory):
    request = json.loads((directory / "request.json").read_bytes())
    assert set(request) == {"root", "revision", "guard_configuration"}
    root = Path(request["root"])
    assert root.is_absolute() and root.resolve(strict=True) == root
    assert sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode
    assert sys.version_info[:3] == (3, 12, 13) and sys.platform == "linux"
    info = directory.lstat()
    assert directory.resolve(strict=True) == directory and stat.S_ISDIR(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.getuid()
    assert not os.path.lexists(directory / "pybuilddir.txt")
    os.chdir(directory)
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "local"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    profile_bytes = (root / PROFILE).read_bytes()
    assert hashlib.sha256(profile_bytes).hexdigest() == PROFILE_SHA256
    assert hashlib.sha256((root / CATALOG).read_bytes()).hexdigest() == CATALOG_SHA256
    payload = json.loads(profile_bytes)
    _write(directory / "identity.json", {"pid": os.getpid(), "ppid": os.getppid()})
    spec = importlib.util.spec_from_file_location("source_test", root / "optical_worker/native_bootstrap.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.__name__ = "__main__"
    sys.modules["__main__"] = entry
    session = controller = owner = None
    phase = "source_verification"
    try:
        session = entry._install(root, request["revision"], "cuda", CATALOG_SHA256)
        entry._verify(session)
        # Keep the two fixed Git source-audit commands available at every audit;
        # no compiler or arbitrary subprocess/library operation is admitted.
        sys.addaudithook(_cpu_only(root, request["revision"]))
        session.bind_guarded_custody(socket.socket(fileno=fd), request["guard_configuration"])
        controller = session.bound_controller()
        observation = session.runtime_observation
        phase = "cpu_outer_fixture"
        with controller.segment("setup"):
            session.phase = "importing"
            from optical_worker import native_setup

            assert native_setup is session.modules["optical_worker.native_setup"][0]
            session.audit()
        session.phase = "imported"
        # Deliberately no configure_runtime_custody/checkpoint: the CPU harness
        # cannot supply a real uploaded input cut or a device observation.
        with controller.segment("setup"):
            session.phase = "configuring_runtime"
            from optical_worker.runtime_environment import RuntimeEnvironmentContract

            environment = RuntimeEnvironmentContract.from_payload(payload)
            admission = object.__new__(entry._ExternalAdmission)
            session.external = admission
            entry._EXTERNAL_CUSTODY[session] = (
                admission, environment, environment.sha256, controller, entry._thread.get_ident())
            admission.__init__(session, environment, environment.sha256)
            observation.environment = environment
            session.audit()
            admission.audit()
        session.phase = "runtime_configured"
        for _ in range(4):
            with controller.segment("setup"):
                pass  # No materialization, runtime init, upload or readback.
        phase = "dependency_initialization"
        with controller.segment("setup"):
            assert controller._guard_reference == {"sequence": 6, "role": "setup"}
            session.phase = "compiling_cuda_scene"
            admission.phase = "runtime"
            input_payload = {"objects": [{"token": token} for token, _ in observation.owners.values()]}
            encoded = json.dumps(input_payload, sort_keys=True).encode()
            input_standin = SimpleNamespace(encoded=encoded, to_payload=lambda: json.loads(encoded))
            compile_root = entry._CompileCustodyRoot(session, input_standin, tuple(session.trace), {})
            session.runtime_owners.append(compile_root)
            controller.adopt_host_owner(compile_root, role="cuda_compile_custody", operation_id="cuda_compile_custody")
            entry._CUDA_COMPILATIONS[id(session)] = (session, compile_root)
            compile_root.root_token = observation.token(compile_root)
            observation.append("compile_root_registered", {"root_token": compile_root.root_token})
            # Explicit fixture for the outer toolchain prerequisite, never a
            # _Compilation instance and never serialized as inventory evidence.
            compile_root.collector = SimpleNamespace(toolchain=SimpleNamespace(inventory_checked=True))
            owner = session.begin_setuptools_initialization(directory)
            import setuptools
            import setuptools.command.build_ext

            assert owner.state == "RESTORED"
            assert setuptools.__version__ == "83.0.0"
            assert sys.modules["setuptools.version"].__version__ == "83.0.0"
            assert sys.modules["distutils"] is sys.modules["setuptools._distutils"] is owner.source
            assert owner.metadata_reads > 0 and owner.handles
            assert all(h.closed for h in owner.handles)
            assert hashlib.sha256(owner.resource_bytes).hexdigest() == "37fedcffbf73c4eb9f058f47677cb33203a436ff9390e4d38a8e01c9dad28e0b"
            assert tuple(sys.path) == owner.original_paths + (owner.vendor_path,)
            assert sys.meta_path == [session.finder]
            owner.check_hooks(restored=True)
            owner.check_alias_methods()
            owner.check_source_metadata()
            assert owner.restore_attempted == {"insert_shim", "_remove_shim"}
            admission.audit()
            session.audit()
            compile_root.audit_retention()
            assert all(record["state"] == "completed" for record in admission.records.values())
            assert not any(n.split(".")[0] in {"torch", "warp", "numpy", "scipy", "trimesh"} for n in sys.modules)
            initialization = [e for e in observation.journal if e["kind"] == "setuptools_initialization"]
            assert not any(e["details"]["action"] == "completed" for e in initialization)
            result = {
                "schema_version": "q54.setuptools.cpu-dependency-observation.v0",
                "scope": "real CPU dependency loading; descriptive outer input/device/toolchain stand-ins",
                "revision": request["revision"], "profile_sha256": PROFILE_SHA256,
                "catalog_sha256": CATALOG_SHA256, "state": owner.state,
                "same_alias_object": True, "setuptools_version": setuptools.__version__,
                "resource_sha256": hashlib.sha256(owner.resource_bytes).hexdigest(),
                "metadata_read_count": owner.metadata_reads, "resource_read_count": len(owner.handles),
                "original_paths": list(owner.original_paths), "active_paths": list(sys.path),
                "repository_trace": list(session.trace), "external_events": list(admission.events),
                "journal": json.loads(json.dumps(observation.journal)),
                "backend_imported": False, "compilation_complete": False,
                "native_scope_observed": False, "readiness_issued": False,
            }
        # Only HOST_OWNER resources were adopted; no sync leaf is supplied.
        phase = "cpu_terminal"
        terminal = controller.terminate(fault="setup_failure", detection_phase="cpu_dependency_end",
                                        emergency_target_synchronize=None)
        result["terminal"] = terminal.lifecycle.to_payload()
        _write(directory / "dependency.json", result)
        os._exit(0)
    except BaseException as error:
        if session is not None:
            try:
                session.fail(error)
            except BaseException:
                pass
        _write(directory / "failure.json", {
            "phase": phase, "error_type": type(error).__name__, "error": str(error),
            "state": None if owner is None else owner.state,
            "source_failures": [] if session is None else [str(e) for e in session.failures],
            "external_events": [] if session is None or session.external is None else session.external.events,
            "journal": [] if session is None else session.runtime_observation.journal,
        })
        os._exit(74)


def parent(root, directory):
    assert os.name == "posix" and sys.version_info[:3] == (3, 12, 13)
    root, directory = root.resolve(strict=True), directory.absolute()
    _create_run_directory(directory)
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Parent-only CPU guard transport; it never loads backend packages.
    sys.path.insert(0, str(root))
    from optical_worker.wait_protocol import GuardConfiguration
    from tests.support.guarded_peer import Peer
    import uuid

    config = GuardConfiguration(uuid.uuid4().hex, 1, 1000, 120000)
    peer = Peer(configuration=config)
    _write(directory / "request.json", {"root": str(root), "revision": revision,
                                        "guard_configuration": config.to_payload()})
    process = None
    group_reaped = False
    try:
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.Popen([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()),
                                        "child", str(peer.channel.fileno()), str(directory)],
                                       pass_fds=(peer.channel.fileno(),), stdout=out, stderr=err,
                                       start_new_session=True)
            timed_out = False
            try:
                code = process.wait(timeout=150)
            except subprocess.TimeoutExpired:
                timed_out = True
                _reap_cpu_group(process)
                group_reaped = True
                code = process.returncode
        _write(directory / "parent.json", {"child_pid": process.pid, "returncode": code,
                                           "timed_out": timed_out, "waited": True})
        assert not timed_out and code == 0, (code, str(directory))
        assert (directory / "dependency.json").is_file() and not (directory / "failure.json").exists()
        print(json.dumps({"directory": str(directory), "child_pid": process.pid, "returncode": code}))
    finally:
        if process is not None and not group_reaped:
            _reap_cpu_group(process)
        peer.close()


if __name__ == "__main__":
    assert len(sys.argv) == 4
    if sys.argv[1] == "child":
        child(int(sys.argv[2]), Path(sys.argv[3]))
    else:
        assert sys.argv[1] == "parent"
        parent(Path(sys.argv[2]), Path(sys.argv[3]))
