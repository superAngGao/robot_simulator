"""Closed source set for the fixed disposable Warp validation program.

The test manifest freezes the complete repository import closure, including
package __init__ files. This does not extend the production worker allowlist.
Installed runtime/stdlib imports are trusted; arbitrary test callbacks and
caller-selected modules are not inputs to this program.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import sys
from pathlib import Path

REPOSITORY_SOURCES = frozenset(
    """
benchmarks/optical/__init__.py
benchmarks/optical/arena.py
benchmarks/optical/assets.py
benchmarks/optical/cases.py
benchmarks/optical/closed_worker.py
benchmarks/optical/cuda_arena.py
benchmarks/optical/eligibility.py
benchmarks/optical/instrumentation.py
benchmarks/optical/native.py
benchmarks/optical/receipts.py
benchmarks/optical/runner.py
benchmarks/optical/schema.py
benchmarks/optical/timing.py
benchmarks/optical/workload_contract.py
benchmarks/optical/workloads.py
optical_worker/__init__.py
optical_worker/lifecycle.py
optical_worker/manifest.py
optical_worker/native_contract.py
optical_worker/protocol.py
optics/__init__.py
optics/_cuda_lbvh_source.py
optics/builder.py
optics/cuda_arena.py
optics/cuda_direct_light.py
optics/cuda_lbvh.py
optics/cuda_scene.py
optics/cuda_timeline.py
optics/device.py
optics/device_bvh.py
optics/device_channel.py
optics/device_scene.py
optics/execution.py
optics/gpu_runtime.py
optics/native_arena.py
optics/output_contract.py
optics/registry.py
optics/scene.py
optics/scene_contract.py
optics/warp_execution.py
optics/warp_native_render.py
physics/__init__.py
physics/_robot_tree_base.py
physics/collision.py
physics/collision_filter.py
physics/constraint_solver.py
physics/constraint_solvers.py
physics/contact.py
physics/contact_tolerances.py
physics/cuda_publication.py
physics/dynamics_cache.py
physics/dynamics_utils.py
physics/force_source.py
physics/geometry.py
physics/integrator.py
physics/joint.py
physics/publish.py
physics/robot_tree.py
physics/spatial.py
physics/step_pipeline.py
physics/telemetry.py
physics/terrain.py
robot/__init__.py
robot/model.py
robot/urdf_loader.py
sensing/__init__.py
sensing/builders.py
sensing/optical.py
sensing/readings.py
sensing/state_sample.py
sensing/surface_query.py
tests/__init__.py
tests/support/__init__.py
tests/support/sacrificial_lifecycle.py
tests/support/sacrificial_watchdog.py
tests/support/sacrificial_sources.py
tests/support/sacrificial_warp_case.py
tests/support/sacrificial_warp_child.py
tests/support/sacrificial_supervisor.py
tools/optical_sacrificial_warp_supervisor.py
""".split()
)


def canonical_bytes(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("ascii")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_source_manifest(root: Path) -> dict:
    root = root.resolve()
    entries = []
    for relative in sorted(REPOSITORY_SOURCES):
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise RuntimeError(f"missing or redirected trusted source: {relative}")
        entries.append(dict(path=relative, sha256=_digest(path)))
    return dict(schema="c4c1c.sacrificial-source.v1", repository_root=str(root), sources=entries)


class RepositoryImportLedger:
    """Before-exec repository source admission plus post-import ledger checks."""

    def __init__(self, manifest: dict):
        if type(manifest) is not dict or set(manifest) != {"schema", "repository_root", "sources"}:
            raise RuntimeError("invalid test source manifest")
        self.root = Path(manifest["repository_root"]).resolve()
        if manifest != frozen_source_manifest(self.root):
            raise RuntimeError("test source manifest does not match its frozen repository closure")
        self._digests = {item["path"]: item["sha256"] for item in manifest["sources"]}
        self._observed = set()

    def _check(self, origin):
        if not origin or origin in ("built-in", "frozen"):
            return
        path = Path(origin).resolve()
        if not path.is_relative_to(self.root):
            return  # Fixed installed Python/runtime paths are the declared trusted runtime.
        relative = path.relative_to(self.root).as_posix()
        if relative not in self._digests or _digest(path) != self._digests[relative]:
            raise RuntimeError(f"repository import outside frozen closure: {relative}")
        self._observed.add(relative)

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None:
            self._check(spec.origin)
            if spec.origin and Path(spec.origin).resolve().is_relative_to(self.root):
                origin = Path(spec.origin).resolve()
                spec.loader = VerifiedRepositorySourceLoader(
                    fullname, str(origin), self._digests[origin.relative_to(self.root).as_posix()]
                )
        return spec

    def verify(self):
        for module in tuple(sys.modules.values()):
            # Torch's two installed runtime namespace objects have synthetic,
            # relative __file__ values; they are not repository source files.
            # Admit only the runtime's exact objects and exact installed types.
            torch = sys.modules.get("torch")
            runtime_namespaces = (("ops", "torch._ops", "_Ops"), ("classes", "torch._classes", "_Classes"))
            if torch is not None and any(
                module is vars(torch).get(attribute)
                and type(module) is vars(sys.modules[implementation]).get(type_name)
                for attribute, implementation, type_name in runtime_namespaces
                if implementation in sys.modules
            ):
                self._check(vars(sys.modules[type(module).__module__]).get("__file__"))
                continue
            self._check(getattr(module, "__file__", None))
        return tuple(sorted(self._observed))

    def install(self):
        self.verify()
        sys.meta_path.insert(0, self)


class VerifiedRepositorySourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the verified bytes, never timestamp/hash bytecode caches."""

    def __init__(self, fullname, path, expected_sha256):
        super().__init__(fullname, path)
        self._expected_sha256 = expected_sha256

    def get_code(self, fullname):
        if fullname != self.name:
            raise RuntimeError("repository source loader module identity changed")
        source = self.get_data(self.path)
        if hashlib.sha256(source).hexdigest() != self._expected_sha256:
            raise RuntimeError("repository source changed before compilation")
        # SourceFileLoader.get_code() would consult .pyc first. Deliberately use
        # only the checked byte buffer for both ordinary modules and packages.
        return self.source_to_code(source, self.path)
