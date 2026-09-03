"""Runtime-enforced operation instrumentation for Q54 optical lanes."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
from unittest.mock import patch

from .timing import (
    TimedBoundaryViolation,
    activate_runtime_hooks,
    active_timed_boundary,
    assert_operation_allowed,
    record_runtime_hook_operation,
)

RUNTIME_HOOK_REGISTRY_VERSION = "q54.p12.4.runtime-hook-registry.v2"


@dataclass(frozen=True)
class RuntimeHookSpec:
    """One concrete callable replaced while a native timed boundary is active."""

    hook_id: str
    target: str
    contamination: str
    entry_point_prefixes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, str]:
        return {
            "hook_id": self.hook_id,
            "target": self.target,
            "contamination": self.contamination,
            "entry_point_prefixes": list(self.entry_point_prefixes),
        }


# These hooks cover the allocation, upload/materialization, synchronization,
# interop-view, and extension-load APIs reachable from the current optical
# product calls. They are installed around every submitted lane transaction;
# source manifests do not activate or satisfy this registry.
FORBIDDEN_RUNTIME_HOOKS = (
    RuntimeHookSpec("numpy.empty", "numpy.empty", "allocation"),
    RuntimeHookSpec("numpy.empty_like", "numpy.empty_like", "allocation"),
    RuntimeHookSpec("numpy.zeros", "numpy.zeros", "allocation"),
    RuntimeHookSpec("numpy.zeros_like", "numpy.zeros_like", "allocation"),
    RuntimeHookSpec("numpy.ones", "numpy.ones", "allocation"),
    RuntimeHookSpec("numpy.ones_like", "numpy.ones_like", "allocation"),
    RuntimeHookSpec("numpy.full", "numpy.full", "allocation"),
    RuntimeHookSpec("numpy.array", "numpy.array", "host_materialization"),
    RuntimeHookSpec("numpy.asarray", "numpy.asarray", "host_materialization"),
    RuntimeHookSpec("numpy.copy", "numpy.copy", "host_materialization"),
    RuntimeHookSpec("torch.empty", "torch.empty", "allocation"),
    RuntimeHookSpec("torch.empty_like", "torch.empty_like", "allocation"),
    RuntimeHookSpec("torch.empty_strided", "torch.empty_strided", "allocation"),
    RuntimeHookSpec("torch.zeros", "torch.zeros", "allocation"),
    RuntimeHookSpec("torch.zeros_like", "torch.zeros_like", "allocation"),
    RuntimeHookSpec("torch.ones", "torch.ones", "allocation"),
    RuntimeHookSpec("torch.ones_like", "torch.ones_like", "allocation"),
    RuntimeHookSpec("torch.full", "torch.full", "allocation"),
    RuntimeHookSpec("torch.full_like", "torch.full_like", "allocation"),
    RuntimeHookSpec("torch.arange", "torch.arange", "allocation"),
    RuntimeHookSpec("torch.linspace", "torch.linspace", "allocation"),
    RuntimeHookSpec("torch.logspace", "torch.logspace", "allocation"),
    RuntimeHookSpec("torch.eye", "torch.eye", "allocation"),
    RuntimeHookSpec("torch.rand", "torch.rand", "allocation"),
    RuntimeHookSpec("torch.randn", "torch.randn", "allocation"),
    RuntimeHookSpec("torch.randint", "torch.randint", "allocation"),
    RuntimeHookSpec("torch.rand_like", "torch.rand_like", "allocation"),
    RuntimeHookSpec("torch.randn_like", "torch.randn_like", "allocation"),
    RuntimeHookSpec("torch.clone", "torch.clone", "allocation"),
    RuntimeHookSpec("torch.cat", "torch.cat", "allocation"),
    RuntimeHookSpec("torch.stack", "torch.stack", "allocation"),
    RuntimeHookSpec("torch.sort", "torch.sort", "allocation"),
    RuntimeHookSpec("torch.min", "torch.min", "allocation"),
    RuntimeHookSpec("torch.max", "torch.max", "allocation"),
    RuntimeHookSpec("torch.Tensor.contiguous", "torch.Tensor.contiguous", "allocation"),
    RuntimeHookSpec("torch.Tensor.to", "torch.Tensor.to", "allocation"),
    RuntimeHookSpec("torch.Tensor.clone", "torch.Tensor.clone", "allocation"),
    RuntimeHookSpec("torch.Tensor.new_empty", "torch.Tensor.new_empty", "allocation"),
    RuntimeHookSpec("torch.Tensor.new_zeros", "torch.Tensor.new_zeros", "allocation"),
    RuntimeHookSpec("torch.Tensor.new_ones", "torch.Tensor.new_ones", "allocation"),
    RuntimeHookSpec("torch.Tensor.new_full", "torch.Tensor.new_full", "allocation"),
    RuntimeHookSpec("torch.Tensor.resize_", "torch.Tensor.resize_", "allocation"),
    RuntimeHookSpec("warp.empty", "warp.empty", "allocation"),
    RuntimeHookSpec("warp.empty_like", "warp.empty_like", "allocation"),
    RuntimeHookSpec("warp.zeros", "warp.zeros", "allocation"),
    RuntimeHookSpec("warp.zeros_like", "warp.zeros_like", "allocation"),
    RuntimeHookSpec("warp.ones", "warp.ones", "allocation"),
    RuntimeHookSpec("warp.ones_like", "warp.ones_like", "allocation"),
    RuntimeHookSpec("warp.full", "warp.full", "allocation"),
    RuntimeHookSpec("warp.full_like", "warp.full_like", "allocation"),
    RuntimeHookSpec("warp.array", "warp.types.array.__init__", "allocation"),
    RuntimeHookSpec("torch.tensor", "torch.tensor", "host_upload"),
    RuntimeHookSpec("torch.as_tensor", "torch.as_tensor", "host_upload"),
    RuntimeHookSpec("torch.from_numpy", "torch.from_numpy", "host_upload"),
    RuntimeHookSpec("torch.asarray", "torch.asarray", "host_upload"),
    RuntimeHookSpec("torch.Tensor.cuda", "torch.Tensor.cuda", "host_upload"),
    RuntimeHookSpec("warp.from_numpy", "warp.from_numpy", "host_upload"),
    RuntimeHookSpec("torch.Tensor.cpu", "torch.Tensor.cpu", "host_materialization"),
    RuntimeHookSpec("torch.Tensor.numpy", "torch.Tensor.numpy", "host_materialization"),
    RuntimeHookSpec("torch.Tensor.item", "torch.Tensor.item", "host_materialization"),
    RuntimeHookSpec("torch.Tensor.tolist", "torch.Tensor.tolist", "host_materialization"),
    RuntimeHookSpec("torch.cuda.synchronize", "torch.cuda.synchronize", "global_device_synchronize"),
    RuntimeHookSpec("warp.synchronize", "warp.synchronize", "global_device_synchronize"),
    RuntimeHookSpec(
        "warp.synchronize_device",
        "warp.synchronize_device",
        "global_device_synchronize",
    ),
    RuntimeHookSpec("warp.synchronize_stream", "warp.synchronize_stream", "host_synchronize"),
    RuntimeHookSpec("warp.synchronize_event", "warp.synchronize_event", "host_synchronize"),
    RuntimeHookSpec("torch.cuda.Event.synchronize", "torch.cuda.Event.synchronize", "host_synchronize"),
    RuntimeHookSpec("torch.cuda.Stream.synchronize", "torch.cuda.Stream.synchronize", "host_synchronize"),
    RuntimeHookSpec("warp.to_torch", "warp.to_torch", "warp_to_torch_bridge"),
    RuntimeHookSpec("warp.from_torch", "warp.from_torch", "torch_to_warp_bridge"),
    RuntimeHookSpec("torch.from_dlpack", "torch.from_dlpack", "bridge_creation"),
    RuntimeHookSpec(
        "torch.utils.dlpack.from_dlpack",
        "torch.utils.dlpack.from_dlpack",
        "bridge_creation",
    ),
    RuntimeHookSpec(
        "torch.utils.dlpack.to_dlpack",
        "torch.utils.dlpack.to_dlpack",
        "bridge_creation",
    ),
    RuntimeHookSpec(
        "optics.cuda_direct_light.channel_to_torch",
        "optics.cuda_direct_light.channel_to_torch",
        "warp_to_torch_bridge",
        ("optics.cuda_direct_light.",),
    ),
    RuntimeHookSpec(
        "tools.optical_pipeline_lab.delivery.stage_optical_channels",
        "tools.optical_pipeline_lab.delivery.stage_optical_channels",
        "host_materialization",
        ("tools.optical_pipeline_lab.delivery.",),
    ),
    RuntimeHookSpec(
        "torch.utils.cpp_extension.load",
        "torch.utils.cpp_extension.load",
        "extension_load",
    ),
    RuntimeHookSpec(
        "torch.utils.cpp_extension.load_inline",
        "torch.utils.cpp_extension.load_inline",
        "extension_load",
    ),
    RuntimeHookSpec(
        "optics.cuda_lbvh.load_inline",
        "optics.cuda_lbvh.load_inline",
        "extension_load",
        (
            "optics.cuda_lbvh.",
            "benchmarks.optical.lanes.warp.WarpCudaLbvhComparisonCommandBundle.",
        ),
    ),
    RuntimeHookSpec(
        "optics.cuda_direct_light.load_inline",
        "optics.cuda_direct_light.load_inline",
        "extension_load",
        ("optics.cuda_direct_light.",),
    ),
)

_FORBIDDEN_HOOK_BY_ID = {spec.hook_id: spec for spec in FORBIDDEN_RUNTIME_HOOKS}
if len(_FORBIDDEN_HOOK_BY_ID) != len(FORBIDDEN_RUNTIME_HOOKS):  # pragma: no cover
    raise AssertionError("runtime hook IDs must be unique")


@dataclass(frozen=True)
class RuntimeHookPlan:
    """Resolved hook plan bound into one entry-point audit."""

    operation: str
    entry_point: str
    registered_hook_ids: tuple[str, ...]
    manifest_sha256: str


@dataclass
class RuntimeHookObservation:
    """Invocation evidence captured by wrappers, including a rejected call."""

    invoked_hook_ids: list[str] = field(default_factory=list)
    violation_hook_id: str | None = None
    violation_contamination: str | None = None

    def observe(self, hook_id: str, contamination: str | None = None) -> None:
        self.invoked_hook_ids.append(hook_id)
        if contamination is not None and self.violation_hook_id is None:
            self.violation_hook_id = hook_id
            self.violation_contamination = contamination


_ACTIVE_OBSERVATION: ContextVar[RuntimeHookObservation | None] = ContextVar(
    "q54_p12_4_runtime_hook_observation",
    default=None,
)


def entry_point_hook_id(entry_point: str) -> str:
    return f"entry_point:{entry_point}"


def runtime_hook_registry_sha256() -> str:
    payload = {
        "schema_version": RUNTIME_HOOK_REGISTRY_VERSION,
        "forbidden_hooks": [spec.to_payload() for spec in FORBIDDEN_RUNTIME_HOOKS],
    }
    return _canonical_sha256(payload)


def declared_runtime_hook_plan(*, operation: str, entry_point: str) -> RuntimeHookPlan:
    """Build the immutable registry claim without treating it as runtime proof."""

    if not operation or not entry_point:
        raise ValueError("runtime hook operation and entry point must be non-empty")
    registered = (
        entry_point_hook_id(entry_point),
        *tuple(spec.hook_id for spec in _hook_specs_for_entry_point(entry_point)),
    )
    payload = {
        "registry_sha256": runtime_hook_registry_sha256(),
        "operation": operation,
        "entry_point": entry_point,
        "registered_hook_ids": list(registered),
    }
    return RuntimeHookPlan(
        operation=operation,
        entry_point=entry_point,
        registered_hook_ids=registered,
        manifest_sha256=_canonical_sha256(payload),
    )


def resolve_runtime_hook_plan(*, operation: str, entry_point: str) -> RuntimeHookPlan:
    """Resolve every registered target; absence makes instrumentation unavailable."""

    plan = declared_runtime_hook_plan(operation=operation, entry_point=entry_point)
    owner, attribute = _resolve_owner(entry_point)
    if not callable(getattr(owner, attribute)):
        raise RuntimeError(f"runtime entry point is not callable: {entry_point}")
    for spec in _hook_specs_for_entry_point(entry_point):
        owner, attribute = _resolve_owner(spec.target)
        if not callable(getattr(owner, attribute)):
            raise RuntimeError(f"runtime hook target is not callable: {spec.target}")
    return plan


def validate_runtime_hook_plan(
    *,
    operation: str,
    entry_point: str,
    registered_hook_ids: Sequence[str],
    verified_hook_ids: Sequence[str],
    manifest_sha256: str | None,
) -> bool:
    """Re-resolve and compare a claimed plan instead of trusting receipt booleans."""

    try:
        expected = resolve_runtime_hook_plan(operation=operation, entry_point=entry_point)
    except (ImportError, AttributeError, RuntimeError, ValueError):
        return False
    registered = tuple(registered_hook_ids)
    verified = tuple(verified_hook_ids)
    return (
        registered == expected.registered_hook_ids
        and verified == expected.registered_hook_ids
        and manifest_sha256 == expected.manifest_sha256
    )


@contextmanager
def instrument_runtime_entry_points(
    entry_point_audits: Sequence[Any],
) -> Iterator[RuntimeHookObservation]:
    """Install mandatory high- and low-level wrappers for one lane submission."""

    audits = tuple(entry_point_audits)
    if not audits:
        raise TimedBoundaryViolation("runtime instrumentation requires entry-point audits")
    plans: list[RuntimeHookPlan] = []
    for audit in audits:
        if not validate_runtime_hook_plan(
            operation=audit.operation,
            entry_point=audit.entry_point,
            registered_hook_ids=audit.registered_runtime_hooks,
            verified_hook_ids=audit.verified_runtime_hooks,
            manifest_sha256=audit.runtime_hook_manifest_sha256,
        ):
            raise TimedBoundaryViolation(f"runtime hook plan is unresolved or stale for {audit.entry_point}")
        plans.append(resolve_runtime_hook_plan(operation=audit.operation, entry_point=audit.entry_point))

    observation = RuntimeHookObservation()
    observation_token = _ACTIVE_OBSERVATION.set(observation)
    all_hook_ids = frozenset(hook_id for plan in plans for hook_id in plan.registered_hook_ids)
    try:
        with ExitStack() as stack, activate_runtime_hooks(all_hook_ids) as hook_capabilities:
            for spec in FORBIDDEN_RUNTIME_HOOKS:
                if spec.hook_id not in all_hook_ids:
                    continue
                owner, attribute = _resolve_owner(spec.target)
                original = getattr(owner, attribute)
                stack.enter_context(patch.object(owner, attribute, _forbidden_wrapper(spec, original)))
            for plan in plans:
                owner, attribute = _resolve_owner(plan.entry_point)
                original = getattr(owner, attribute)
                stack.enter_context(
                    patch.object(
                        owner,
                        attribute,
                        _entry_point_wrapper(
                            plan,
                            original,
                            hook_capabilities[entry_point_hook_id(plan.entry_point)],
                        ),
                    )
                )
            yield observation
    finally:
        _ACTIVE_OBSERVATION.reset(observation_token)


def invoke_instrumented_entry_point(entry_point: str, /, *args: Any, **kwargs: Any) -> Any:
    """Resolve the currently wrapped product callable and invoke it."""

    owner, attribute = _resolve_owner(entry_point)
    return getattr(owner, attribute)(*args, **kwargs)


def _entry_point_wrapper(plan: RuntimeHookPlan, original: Any, capability: object):
    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        if active_timed_boundary() is not None:
            hook_id = entry_point_hook_id(plan.entry_point)
            observation = _ACTIVE_OBSERVATION.get()
            if observation is not None:
                observation.observe(hook_id)
            record_runtime_hook_operation(
                plan.operation,
                entry_point=plan.entry_point,
                runtime_hook_id=hook_id,
                _capability=capability,
            )
        return original(*args, **kwargs)

    return wrapped


def _forbidden_wrapper(spec: RuntimeHookSpec, original: Any):
    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        if active_timed_boundary() is not None:
            observation = _ACTIVE_OBSERVATION.get()
            if observation is not None:
                observation.observe(spec.hook_id, spec.contamination)
            assert_operation_allowed(spec.contamination)
        return original(*args, **kwargs)

    return wrapped


def _resolve_owner(path: str) -> tuple[Any, str]:
    parts = path.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError(f"invalid runtime hook path {path!r}")
    module = None
    remainder: list[str] = []
    for index in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError as exc:
            if exc.name != ".".join(parts[:index]) and not ".".join(parts[:index]).startswith(f"{exc.name}."):
                raise
            continue
        remainder = parts[index:]
        break
    if module is None or not remainder:
        raise ImportError(f"cannot import runtime hook path {path!r}")
    owner = module
    for name in remainder[:-1]:
        owner = getattr(owner, name)
    attribute = remainder[-1]
    if not hasattr(owner, attribute):
        raise AttributeError(f"runtime hook target does not exist: {path}")
    return owner, attribute


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hook_specs_for_entry_point(entry_point: str) -> tuple[RuntimeHookSpec, ...]:
    return tuple(
        spec
        for spec in FORBIDDEN_RUNTIME_HOOKS
        if not spec.entry_point_prefixes
        or any(entry_point.startswith(prefix) for prefix in spec.entry_point_prefixes)
    )
