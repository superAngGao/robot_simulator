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
import threading
from types import ModuleType, SimpleNamespace


PROFILE = "tests/data/runtime_environments/183_cuda_compilation_author_v1.json"
CATALOG = "optical_worker/data/native_sources/v13.json"
PROFILE_SHA256 = "0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753"
CATALOG_SHA256 = "17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399"

# Fixed test-only selectors. Each runs in its own cold child through the same
# source session and installed dependency as the positive proof below.
FAULTS = {
    "preloaded_setuptools": "setuptools dependency preloaded",
    "preloaded__distutils_hack": "setuptools dependency preloaded",
    "preloaded_distutils": "setuptools dependency preloaded",
    "preloaded_backports": "setuptools dependency preloaded",
    "extra_path": "closed import topology changed",
    "copied_path": "setuptools import path object changed",
    "vendor_first": "closed import topology changed",
    "repeated_begin": "setuptools initialization repeated",
    "early_shim": "outside its exact source execution",
    "repeated_shim": "shim entry order",
    "foreign_thread": "failed or wrong thread",
    "foreign_shim": "shim binding changed",
    "function_code": "function code/globals/closure changed",
    "wrapper_closure": "function code/globals/closure changed",
    "alias_spec": "original/alias spec changed",
    "alias_closure": "function code/globals/closure changed",
    "alias_module": "alias exec order/publication",
    "alias_loader": "alias method changed",
    "missing_alias": "alias publication order",
    "unlisted_child": "unlisted external import",
    "resource_traversal": "unlisted setuptools resource",
    "resource_caught": "unlisted setuptools resource",
    "resource_changed": "resource handle identity",
    "metadata_path": "unlisted setuptools metadata file",
    "metadata_enumeration": "metadata query context",
    "partial_install_before": "injected:partial_install_before",
    "partial_install_after": "injected:partial_install_after",
    "partial_create": "injected:partial_create",
    "partial_exec": "injected:partial_exec",
    "restoration_before": "injected:restoration_before",
    "restoration_after": "injected:restoration_after",
    "late_publication": "injected:late_publication",
}


class _DependencyFault:
    """Faults at real callbacks; never supplies source, specs or an importer."""

    def __init__(self, name):
        assert name in FAULTS
        self.name, self.reached, self.assignments = name, [], []
        self.injected = RuntimeError("injected:" + name)
        self.owner = self.helper = None
        self.thread = None

    def mark(self, point):
        self.reached.append(point)

    def before_begin(self):
        if self.name.startswith("preloaded_"):
            # Load the real helper before placing a foreign preload in the
            # interpreter, so begin() is the boundary under test.
            from optical_worker import setuptools_admission

            assert setuptools_admission.__name__ == "optical_worker.setuptools_admission"
            name = self.name.removeprefix("preloaded_")
            assert name not in sys.modules
            sys.modules[name] = ModuleType(name)
            self.mark("foreign_preload_before_begin")

    def install(self, owner):
        self.owner = owner
        name = self.name
        if name == "unlisted_child":
            return  # Attempted only after the real positive imports return.
        if name in {"extra_path", "copied_path", "vendor_first"}:
            self.mark("path_after_begin")
            if name == "extra_path":
                sys.path.append(str(owner.run_root))
            elif name == "copied_path":
                sys.path = list(sys.path)
            else:
                sys.path.insert(0, sys.path.pop())
        elif name == "repeated_begin":
            self.mark("second_begin")
            owner.begin()
        elif name == "early_shim":
            self.mark("before_dependency")
            owner.enter_shim()
        elif name in {"repeated_shim", "foreign_thread", "wrapper_closure"}:
            original = owner.enter_shim

            def enter():
                self.mark("actual_override_entry")
                if name == "wrapper_closure":
                    # Both real wrappers share this cell. Mutate after the
                    # actual wrapper dispatched, then run its real owner check.
                    owner.wrappers["_remove_shim"].__closure__[0].cell_contents = object()
                    return original()
                if name == "repeated_shim":
                    original()
                    return original()
                errors = []

                def foreign():
                    try:
                        original()
                    except BaseException as error:
                        errors.append(error)

                self.thread = threading.Thread(target=foreign, daemon=True)
                self.thread.start()
                self.thread.join(10)
                assert not self.thread.is_alive() and len(errors) == 1
                raise errors[0]

            owner.enter_shim = enter
        elif name in {"foreign_shim", "function_code"}:
            original = owner.after_file_complete

            def after_file(module_name, module, loader):
                original(module_name, module, loader)
                if module_name == "_distutils_hack":
                    self.mark("actual_hack_completed")
                    if name == "foreign_shim":
                        module._remove_shim = object()
                    else:
                        owner.originals["insert_shim"].__code__ = (lambda: None).__code__

            owner.after_file_complete = after_file
        elif name.startswith("partial_install_") or name.startswith("restoration_"):
            # Intercept only this helper's actual assignment, as in the existing
            # unit counterexample. The real ordinary loader owns its finally.
            self.helper = sys.modules[type(owner).__module__]
            assert "setattr" not in vars(self.helper)

            def assign(target, attribute, value):
                self.assignments.append((attribute, value))
                installing = value is owner.wrappers.get(attribute)
                restoring = value is owner.originals.get(attribute)
                trigger = attribute == "_remove_shim" and (
                    name.startswith("partial_install_") and installing
                    or name.startswith("restoration_") and restoring)
                if trigger:
                    self.mark("install_assignment" if installing else "restore_assignment")
                    if name.endswith("_after"):
                        setattr(target, attribute, value)
                    raise self.injected
                setattr(target, attribute, value)

            self.helper.setattr = assign
        elif name in {"partial_create", "alias_spec", "alias_closure", "late_publication"}:
            original = owner.alias_create

            def create(wrapper, spec):
                self.mark("actual_alias_create")
                if name == "alias_closure":
                    owner.alias_methods[0].__closure__[0].cell_contents = object()
                    return original(wrapper, spec)
                if name == "alias_spec":
                    spec.origin = "invalid-origin"
                    return original(wrapper, spec)
                module = original(wrapper, spec)
                assert module is owner.source and owner.state == "ALIAS_CREATED"
                if name == "late_publication":
                    owner.fail(self.injected)
                    sys.modules["distutils"] = module
                    owner.leave_shim()  # The real failed-path intake retains it.
                    self.mark("publication_after_poison")
                raise self.injected

            owner.alias_create = create
        elif name in {"partial_exec", "alias_module", "alias_loader"}:
            original = owner.alias_exec

            def execute(wrapper, module):
                self.mark("actual_alias_exec")
                if name == "alias_module":
                    return original(wrapper, ModuleType("distutils"))
                if name == "alias_loader":
                    wrapper.delegate = object()
                    return original(wrapper, module)
                original(wrapper, module)
                assert owner.state == "ALIAS_EXECUTED"
                raise self.injected

            owner.alias_exec = execute
        elif name == "missing_alias":
            original = owner.leave_shim

            def leave():
                self.mark("actual_shim_exit")
                assert sys.modules.pop("distutils") is owner.source
                return original()

            owner.leave_shim = leave
        elif name.startswith("resource_"):
            original = owner.open_resource

            def resource(reader, resource_name):
                self.mark("actual_resource_expression")
                if name == "resource_changed":
                    handle = original(reader, resource_name)
                    owner.resource_bytes = b"changed after actual capture"
                    return handle
                if name == "resource_caught":
                    try:
                        original(reader, "../PKG-INFO")
                    except ValueError as error:
                        assert str(error) == FAULTS[name]
                        self.mark("denial_caught")
                    return original(reader, resource_name)
                return original(reader, "../PKG-INFO")

            owner.open_resource = resource
        elif name == "metadata_path":
            original = owner.read_metadata

            def metadata(distribution, filename):
                self.mark("actual_metadata_read")
                return original(distribution, "../PKG-INFO")

            owner.read_metadata = metadata
        else:
            assert name == "metadata_enumeration"
            original = owner.find_distributions

            def distributions(context):
                self.mark("actual_metadata_query")
                return original(type(context)())

            owner.find_distributions = distributions

    def after_loaded(self):
        assert self.name == "unlisted_child", "injected dependency fault did not reject"
        assert self.owner.state == "RESTORED"
        self.mark("unlisted_child_after_actual_loading")
        __import__("distutils._q54_unlisted_child")

    def observe_failure(self, caught):
        owner = self.owner
        if self.helper is not None:
            del self.helper.setattr
        assert self.reached, "fault boundary was not reached"
        assert owner.state == "FAILED" and owner.session.failed and owner.admission.failed
        first = owner.errors[0]
        assert FAULTS[self.name] in str(first), (self.name, str(first), str(caught))
        if FAULTS[self.name].startswith("injected:"):
            assert first is self.injected
        assert any(error is first for error in owner.session.failures)
        assert owner.controller.termination_evidence is not None
        terminal = owner.controller.termination_evidence.lifecycle.to_payload()["summary"]
        assert terminal["disposition"] == "cancelled_before_submission_exit"
        assert terminal["guardian_state"] == "RELEASED" and terminal["release_status"] == "completed"
        assert terminal["setup_operation_count"] == terminal["entered_leaf_count"] == 0
        assert terminal["emergency_sync_attempt_count"] == 0
        assert terminal["possible_dirty_device_ordinals"] == [] and not terminal["unknown_device_scope"]
        assert self.thread is None or not self.thread.is_alive()
        assert owner.restore_attempted == set(owner.installed)
        if self.name != "restoration_before":
            assert all(vars(owner.hack)[name] is owner.originals[name] for name in owner.installed)
        else:
            assert vars(owner.hack)["_remove_shim"] is owner.wrappers["_remove_shim"]
        if self.assignments:
            for name in owner.installed:
                assert sum(n == name and value is owner.originals[name] for n, value in self.assignments) == 1
        if self.name in {"partial_create", "partial_exec", "late_publication"}:
            assert owner.observation.owners[id(owner.source)][1] is owner.source
        if self.name == "late_publication":
            assert sys.modules["distutils"] is owner.source
        # Denial remains live after the dependency unwinds, including when a
        # dependency callback caught its first error. These are real checks.
        denials = []
        for check in (owner.check, owner.session.check_live, owner.admission.check_live):
            try:
                check()
            except BaseException as error:
                denials.append(str(error))
            else:
                raise AssertionError("poisoned dependency accepted a subsequent check")
        actions = [event["details"]["action"] for event in owner.observation.journal
                   if event["kind"] == "setuptools_initialization"]
        assert actions.count("failed") == 1 and actions[actions.index("failed"):] == ["failed"]
        assert "completed" not in actions
        if self.name == "resource_caught":
            assert "denial_caught" in self.reached
        return {"case": self.name, "reached": self.reached, "first_error": str(first),
                "caught_error": str(caught), "state": owner.state, "actions": actions,
                "restore_attempted": sorted(owner.restore_attempted), "subsequent_denials": denials,
                "foreign_thread_joined": self.thread is not None and not self.thread.is_alive()}


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
    assert set(request) == {"root", "revision", "guard_configuration", "case"}
    case = request["case"]
    assert case == "positive" or case in FAULTS
    fault = None if case == "positive" else _DependencyFault(case)
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
            if fault is not None:
                fault.before_begin()
            owner = session.begin_setuptools_initialization(directory)
            if fault is not None:
                fault.install(owner)
            import setuptools
            import setuptools.command.build_ext

            if fault is not None:
                fault.after_loaded()
            assert fault is None, "injected dependency fault did not reject"
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
        if fault is not None and fault.name.startswith("preloaded_") and session is not None:
            fault.owner = owner = session.setuptools_owner()
        if fault is not None and fault.owner is not None:
            try:
                facts = fault.observe_failure(error)
                assert not any(n.split(".")[0] in {"torch", "warp", "numpy", "scipy", "trimesh"} for n in sys.modules)
                _write(directory / "negative.json", {
                    "schema_version": "q54.setuptools.cpu-dependency-failure-observation.v0",
                    "scope": "actual CPU dependency denial; outer input/device/toolchain stand-ins",
                    "revision": request["revision"], "profile_sha256": PROFILE_SHA256,
                    "catalog_sha256": CATALOG_SHA256, **facts,
                    "repository_trace": list(session.trace), "external_events": list(session.external.events),
                    "journal": session.runtime_observation.journal,
                    "terminal": controller.termination_evidence.lifecycle.to_payload(),
                    "backend_imported": False, "compilation_complete": False,
                    "native_scope_observed": False, "readiness_issued": False,
                })
                os._exit(0)  # Harness succeeded in proving this expected denial.
            except BaseException as validation_error:
                error = validation_error
        if session is not None:
            try:
                session.fail(error)
            except BaseException:
                pass
        _write(directory / "failure.json", {
            "case": case, "fault_reached": [] if fault is None else fault.reached,
            "phase": phase, "error_type": type(error).__name__, "error": str(error),
            "state": None if owner is None else owner.state,
            "source_failures": [] if session is None else [str(e) for e in session.failures],
            "external_events": [] if session is None or session.external is None else session.external.events,
            "journal": [] if session is None else session.runtime_observation.journal,
        })
        os._exit(74)


def parent(root, directory, case="positive"):
    assert os.name == "posix" and sys.version_info[:3] == (3, 12, 13)
    assert case == "positive" or case in FAULTS
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
    _write(directory / "request.json", {"root": str(root), "revision": revision, "case": case,
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
        artifact = "dependency.json" if case == "positive" else "negative.json"
        assert (directory / artifact).is_file() and not (directory / "failure.json").exists()
        print(json.dumps({"directory": str(directory), "case": case, "child_pid": process.pid, "returncode": code}))
    finally:
        if process is not None and not group_reaped:
            _reap_cpu_group(process)
        peer.close()


if __name__ == "__main__":
    assert len(sys.argv) in (4, 5)
    if sys.argv[1] == "child":
        assert len(sys.argv) == 4
        child(int(sys.argv[2]), Path(sys.argv[3]))
    else:
        assert sys.argv[1] == "parent"
        parent(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] if len(sys.argv) == 5 else "positive")
