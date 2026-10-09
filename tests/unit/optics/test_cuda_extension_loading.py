"""CPU-only resolver/factory checks. Dummy artifacts are never natively loaded."""

import builtins
import importlib.machinery
import importlib.util
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from optics import cuda_extension_compatibility as loading
from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from tests.unit.optics.test_cuda_artifact_manifest import artifact, _publish_fixture


@pytest.fixture
def selected(artifact, monkeypatch):
    root, payload = artifact
    monkeypatch.setattr(loading, "_DEPLOYMENT", None)
    monkeypatch.setattr(loading, "_LOADED", {})
    monkeypatch.setattr(sys, "path", list(sys.path))
    for recipe in CUDA_EXTENSION_RECIPES:
        monkeypatch.delitem(sys.modules, recipe.name, raising=False)
    digest = _publish_fixture(root, payload)
    loading.select_cuda_artifact_deployment(root, digest, expected_platform=payload["platform"])
    return root, payload


def test_selection_is_explicit_single_use_and_does_not_import_dummy_outputs(selected):
    root, payload = selected
    assert all(recipe.name not in sys.modules for recipe in CUDA_EXTENSION_RECIPES)
    with pytest.raises(RuntimeError, match="single-use"):
        loading.select_cuda_artifact_deployment(root, "0" * 64, expected_platform=payload["platform"])


def test_missing_selection_cannot_fall_back_to_jit(monkeypatch):
    monkeypatch.setattr(loading, "_DEPLOYMENT", None)
    with pytest.raises(RuntimeError, match="select a verified"):
        loading.load_cuda_extension(CUDA_EXTENSION_RECIPES[0].name)
    assert "torch.utils.cpp_extension" not in sys.modules


@pytest.mark.parametrize("attack", (None, "entry", "module", "spec", "artifact", "manifest"))
def test_ordinary_resolution_rechecks_files_and_actual_entry_identity(selected, monkeypatch, attack):
    root, payload = selected
    recipe = CUDA_EXTENSION_RECIPES[0]
    name = recipe.name
    module = ModuleType(name)
    origin = str(root / payload["artifacts"][0]["path"])
    delegate = importlib.machinery.ExtensionFileLoader(name, origin)
    spec = importlib.util.spec_from_file_location(name, origin, loader=delegate)
    module.__spec__, module.__loader__, module.__file__ = spec, delegate, origin
    for entry_name in recipe.entries:
        def entry():
            raise AssertionError("optical entry must not execute during resolution")
        entry.__name__, entry.__module__ = entry_name, name
        setattr(module, entry_name, entry)
    real_import, real_find = builtins.__import__, importlib.machinery.PathFinder.find_spec
    imports = []

    def import_dummy(request, *args, **kwargs):
        if request == name:
            imports.append(request)
            monkeypatch.setitem(sys.modules, name, module)
            return module
        return real_import(request, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_dummy)
    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec",
                        lambda request, *args, **kwargs: spec if request == name else real_find(request, *args, **kwargs))
    assert loading.load_cuda_extension(name) is module
    if attack == "entry":
        old = getattr(module, recipe.entries[0])
        def replacement():
            pass
        replacement.__name__, replacement.__module__ = old.__name__, old.__module__
        setattr(module, recipe.entries[0], replacement)
    elif attack == "module":
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    elif attack == "spec":
        module.__spec__ = importlib.util.spec_from_file_location(name, origin, loader=delegate)
    elif attack == "artifact":
        (root / payload["artifacts"][0]["path"]).write_bytes(b"changed")
    elif attack == "manifest":
        (root / "cuda-artifacts.json").write_bytes(b"changed")
    if attack:
        with pytest.raises(ValueError):
            loading.load_cuda_extension(name)
    else:
        assert loading.load_cuda_extension(name) is module
    assert imports == [name]


def test_factory_modules_do_not_import_warp_or_compiler_tooling():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import optics.cuda_scene, optics.cuda_lbvh, optics.cuda_direct_light; "
        "assert 'warp' not in sys.modules; assert 'torch.utils.cpp_extension' not in sys.modules"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_lbvh_query_and_factory_use_one_resolved_module(monkeypatch):
    from optics import cuda_lbvh as lbvh, cuda_timeline as timeline

    calls = []
    device = SimpleNamespace(type="cuda", index=1)
    stream = SimpleNamespace(device=device, cuda_stream=17)
    context = object.__new__(timeline.CudaOpticalExecutionContext)
    context.device, context.stream, context.stream_handle = device, stream, 17
    torch = SimpleNamespace(device=lambda value: value, cuda=SimpleNamespace(
        device_count=lambda: 2, device=lambda ordinal: nullcontext(),
        get_device_properties=lambda ordinal: SimpleNamespace(major=9, minor=0)))
    monkeypatch.setattr(timeline, "torch", torch)
    monkeypatch.setattr(lbvh, "torch", torch)
    monkeypatch.setattr(lbvh, "_require_torch_cuda_lbvh", lambda: None)

    def query(count, ordinal):
        calls.append(("query", count, ordinal))
        return 128, 256, 256

    module = SimpleNamespace(cuda_lbvh_workspace_requirements=query, cuda_lbvh_rebuild_out=lambda: None)
    extensions = object()
    def resolve(name, **kwargs):
        assert kwargs == {"extensions": extensions, "context": context}
        calls.append(("resolve", name))
        return module
    monkeypatch.setattr(lbvh, "load_cuda_extension", resolve)
    factory = lbvh.CudaLbvhRebuildStageFactory.prepare(
        context=context, primitive_count=4, native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64, input_contract="canonical_native_replay_v1", extensions=extensions)
    assert factory.module is module and factory.requirements.required_bytes == 256
    assert calls == [("resolve", "robot_sim_cuda_lbvh_native_v3"), ("query", 4, 1)]


def test_direct_light_rejects_bare_module_construction():
    from optics.cuda_direct_light import CudaDirectLightNativeRenderer

    with pytest.raises(TypeError, match="prepare"):
        CudaDirectLightNativeRenderer(context=object(), module=object())


def test_copied_or_unregistered_captured_set_never_uses_compatibility_fallback(monkeypatch):
    from optical_worker.native_setup import CudaExtensionSet
    from optics.cuda_extension_loading import load_cuda_extension

    monkeypatch.setattr(loading, "_DEPLOYMENT", None)
    owner = object.__new__(CudaExtensionSet)
    object.__setattr__(owner, "session", object())
    with pytest.raises(ValueError, match="unregistered"):
        load_cuda_extension(CUDA_EXTENSION_RECIPES[0].name, extensions=owner, context=object())
