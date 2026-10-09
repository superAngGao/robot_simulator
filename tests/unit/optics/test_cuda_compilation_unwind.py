"""Production build orchestration with CPU stand-ins, not native-scope proof."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from optical_worker import cuda_compilation as compilation


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    source = compilation.CUDA_EXTENSION_RECIPES[0].source_module
    observation = SimpleNamespace(token=lambda value: id(value), append=lambda *args: None)
    session = SimpleNamespace(
        bound_controller=lambda: SimpleNamespace(_guard_reference={"sequence": 6, "role": "setup"}),
        runtime_observation=observation,
        modules={source.__name__: (source, SimpleNamespace(source=b"fixture", origin="fixture.py"), "completed")},
    )
    owner = compilation._Compilation(session, object(), (), object(), 1, object())
    owner.toolchain = SimpleNamespace(root=tmp_path, audit=lambda: None)
    monkeypatch.setattr(owner, "audit", lambda: None)
    monkeypatch.setattr(owner, "audit_module", lambda index: None)
    monkeypatch.setattr(owner, "_enter", lambda site: None)
    monkeypatch.setattr(compilation, "_private_directory", lambda path: path)
    monkeypatch.setattr(compilation, "_ACTIVE_HOOK", None)
    dependency = SimpleNamespace(_import_module_from_library=object())
    owner.dependency = dependency
    owner.original_hook = dependency._import_module_from_library
    return owner


@pytest.mark.parametrize("build_fails", [False, True])
def test_hook_drift_and_exit_audit_retain_all_errors_and_original_priority(compiler, monkeypatch, build_fails):
    build_error = RuntimeError("compiler failed first")
    audit_error = RuntimeError("environment drift after build")

    def fail_audit():
        raise audit_error

    def build(*args):
        compiler.hook = object()
        compilation._ACTIVE_HOOK = compiler
        compiler.dependency._import_module_from_library = object()  # Foreign replacement during build.
        compiler.toolchain.audit = fail_audit
        if build_fails:
            raise build_error
        module = object()
        compiler.live[0]["module"] = module
        compiler.owners.append(object())
        return module

    monkeypatch.setattr(compilation, "_torch_compile_exact", build)
    with pytest.raises(RuntimeError) as raised:
        compiler.compile_recipe(0)
    assert compiler.failed_flag
    assert compiler.dependency._import_module_from_library is compiler.original_hook
    assert compilation._ACTIVE_HOOK is None
    assert audit_error in compiler.failure_objects
    drift = next(error for error in compiler.failure_objects if "direct helper changed" in str(error))
    assert raised.value is (build_error if build_fails else drift)
    assert "hook_restored" not in compiler.steps[0]


def test_dependency_import_failure_never_installs_or_restores_a_hook(compiler, monkeypatch):
    compiler.dependency = None
    original = RuntimeError("dependency import failed")

    def build(*args):
        raise original

    monkeypatch.setattr(compilation, "_torch_compile_exact", build)
    with pytest.raises(RuntimeError) as raised:
        compiler.compile_recipe(0)
    assert raised.value is original and compiler.failure_objects == [original]
    assert compiler.hook is None and compilation._ACTIVE_HOOK is None


def test_failed_restoration_is_attempted_once_and_cannot_skip_exit_audit(compiler, monkeypatch):
    original = RuntimeError("build error")
    restore_error = RuntimeError("restore error")
    attempts, audits = [], []

    class Dependency:
        _import_module_from_library = compiler.original_hook

        def __setattr__(self, name, value):
            if value is compiler.original_hook:
                attempts.append(value)
                raise restore_error
            object.__setattr__(self, name, value)

    compiler.dependency = Dependency()

    def build(*args):
        compiler.hook = object()
        compilation._ACTIVE_HOOK = compiler
        compiler.dependency._import_module_from_library = compiler.hook
        compiler.toolchain.audit = lambda: audits.append("exit")
        raise original

    monkeypatch.setattr(compilation, "_torch_compile_exact", build)
    with pytest.raises(RuntimeError) as raised:
        compiler.compile_recipe(0)
    assert raised.value is original
    assert attempts == [compiler.original_hook] and audits == ["exit"]
    assert compiler.failure_objects == [original, restore_error] and compiler.failed_flag
    assert compilation._ACTIVE_HOOK is None


def test_failed_journal_retains_errors_without_replacing_original(compiler, monkeypatch):
    original = RuntimeError("build error")
    diagnostic = RuntimeError("journal error")

    def append(kind, details):
        if details.get("operation") == "failed":
            raise diagnostic

    def build(*args):
        raise original

    compiler.observation.append = append
    monkeypatch.setattr(compilation, "_torch_compile_exact", build)
    with pytest.raises(RuntimeError) as raised:
        compiler.compile_recipe(0)
    assert raised.value is original and compiler.failed_flag
    assert compiler.failure_objects == [original, diagnostic]


@pytest.mark.parametrize("fail_at", [None, "properties", "hook", "build"])
def test_compile_leaf_enters_indexed_scope_before_dependency_and_restores_on_error(fail_at):
    calls = []
    state = {"device": 0, "stream": "ambient"}
    original = RuntimeError("leaf failure")

    @contextmanager
    def scope(key, value):
        old = state[key]
        state[key] = value
        try:
            yield
        finally:
            state[key] = old

    def step(name):
        assert state == {"device": 1, "stream": "held"}
        calls.append(name)
        if name == fail_at:
            raise original

    def properties(target):
        assert target == 1
        step("properties")
        return SimpleNamespace(major=9, minor=0)

    def build(**kwargs):
        step("build")
        return "module"

    torch = SimpleNamespace(cuda=SimpleNamespace(
        device=lambda target: scope("device", target),
        stream=lambda stream: scope("stream", stream), get_device_properties=properties))
    owner = SimpleNamespace(_retain=lambda value: None, install_hook=lambda: step("hook"),
                            dependency=SimpleNamespace(load_inline=build))
    if fail_at is None:
        assert compilation._torch_compile_exact(torch, 1, "held", owner, {}) == "module"
        assert calls == ["properties", "hook", "build"]
    else:
        with pytest.raises(RuntimeError) as raised:
            compilation._torch_compile_exact(torch, 1, "held", owner, {})
        assert raised.value is original and calls[-1] == fail_at
    assert state == {"device": 0, "stream": "ambient"}
