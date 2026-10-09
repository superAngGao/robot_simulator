"""Non-captured ordinary imports of a selected fixed deployment; never JIT."""

from __future__ import annotations

import hashlib
import importlib.machinery
import sys
from pathlib import Path

from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES, cuda_extension_entries, cuda_extension_recipe

_DEPLOYMENT = None
_LOADED = {}


def select_cuda_artifact_deployment(deployment, manifest_sha256, *, expected_platform):
    """Select a verified deployment before using non-captured compatibility APIs.

    The trusted launcher supplies the explicit platform/path/digest. Captured
    workers instead use their bound environment and session-issued extension set.
    """
    global _DEPLOYMENT
    from tools.optical_cuda_artifacts import verify_deployment

    if _DEPLOYMENT is not None or any(r.name in sys.modules for r in CUDA_EXTENSION_RECIPES):
        raise RuntimeError("CUDA compatibility deployment is single-use and must precede imports")
    root = Path(deployment)
    manifest = verify_deployment(root, manifest_sha256, repository=Path(__file__).resolve().parents[1],
                                 expected_platform=expected_platform)
    rows = tuple((r["name"], r["path"], r["size"], r["sha256"]) for r in manifest["artifacts"])
    selected = (root, manifest_sha256, rows)
    # Ordinary imports resolve through the standard path machinery. Every load
    # checks its selected exact origin before invoking Python's extension loader.
    sys.path.insert(0, str(root))
    _DEPLOYMENT = selected


def _verify_bytes(root, relative, size, digest):
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("CUDA artifact path changed")
    data = path.read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("CUDA artifact bytes changed")
    return str(path)


def load_cuda_extension(recipe_name):
    """Resolve one fixed recipe, rechecking authority/entries on every prepare."""
    recipe = cuda_extension_recipe(recipe_name)
    if _DEPLOYMENT is None:
        raise RuntimeError("select a verified CUDA artifact deployment before compatibility preparation")
    root, manifest_digest, rows = _DEPLOYMENT
    manifest = root / "cuda-artifacts.json"
    if manifest.is_symlink() or manifest.resolve(strict=True) != manifest or hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_digest:
        raise ValueError("CUDA artifact manifest changed")
    name, path, size, digest = next(row for row in rows if row[0] == recipe.name)
    origin = _verify_bytes(root, path, size, digest)
    previous = _LOADED.get(name)
    if previous is None:
        if name in sys.modules:
            raise RuntimeError("CUDA extension was imported outside the selected compatibility route")
        spec = importlib.machinery.PathFinder.find_spec(name)
        if (spec is None or spec.origin != origin or type(spec.loader) is not importlib.machinery.ExtensionFileLoader
                or spec.name != name or spec.loader.name != name or spec.loader.path != origin):
            raise ValueError("CUDA extension resolution differs from selected deployment")
        _LOADED[name] = (None, None, None, None)
        if name == "robot_sim_cuda_scene_v3":
            import robot_sim_cuda_scene_v3 as module
        elif name == "robot_sim_cuda_lbvh_native_v3":
            import robot_sim_cuda_lbvh_native_v3 as module
        else:
            import robot_sim_cuda_direct_light_v8 as module
        # Hold before metadata/ABI validation can fail. A partial load cannot be
        # retried as an unobserved new module in this process.
        _LOADED[name] = (module, None, None, None)
        entries = cuda_extension_entries(module, name)
        _LOADED[name] = (module, module.__spec__, module.__loader__, entries)
    module, spec, loader, entries = _LOADED[name]
    if (entries is None or sys.modules.get(name) is not module or module.__spec__ is not spec
            or module.__loader__ is not loader or spec.loader is not loader or spec.name != name
            or spec.origin != origin or module.__file__ != origin
            or type(loader) is not importlib.machinery.ExtensionFileLoader or loader.name != name or loader.path != origin):
        raise ValueError("CUDA extension module identity changed or admission is incomplete")
    actual_entries = cuda_extension_entries(module, name)
    if any(actual is not original for actual, original in zip(actual_entries, entries, strict=True)):
        raise ValueError("CUDA extension entry was replaced")
    return module
