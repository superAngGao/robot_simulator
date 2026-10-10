from __future__ import annotations

import pytest
from contextlib import contextmanager

import optics.cuda_direct_light as cuda_direct_light
import optics.cuda_timeline as cuda_timeline
from optics.cuda_timeline import (
    CudaDeferredProfile,
    CudaOpticalCompletion,
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransactionPoisonedError,
    CudaOpticalTransactionTemplate,
)


def test_cuda_timeline_contract_imports_without_runtime_initialization():
    assert issubclass(CudaOpticalTransactionPoisonedError, CudaOpticalTimelineError)
    assert CudaOpticalCompletion.__dataclass_fields__["recorded"].default is True
    assert CudaOpticalTransactionTemplate.__doc__


def test_cuda_timeline_requires_torch_only_when_context_is_constructed(monkeypatch):
    monkeypatch.setattr(cuda_timeline, "torch", None)

    with pytest.raises(ImportError, match="PyTorch"):
        CudaOpticalExecutionContext(device="cuda:0", stream=object())


def test_deferred_profile_resolution_is_explicit():
    class _Event:
        def __init__(self, value: float) -> None:
            self.value = value

        def elapsed_time(self, end: "_Event") -> float:
            return end.value - self.value

    profile = CudaDeferredProfile((("raygen_kernel", _Event(1.25), _Event(3.75)),))

    assert profile.resolve_ms() == (("raygen_kernel", 2.5),)


def test_cuda_shade_extension_abi_uses_fixed_diagnostic_arguments():
    source = cuda_direct_light._CPP_SOURCE
    cuda_source = cuda_direct_light._CUDA_SOURCE

    assert "std::vector<torch::Tensor>" not in source
    assert "std::vector<torch::Tensor>" not in cuda_source
    assert "#include <vector>" not in source
    assert "#include <vector>" not in cuda_source
    assert ": shadow_stack_overflow_count.data_ptr<int32_t>()" not in cuda_source

    for base_name in (
        "shade_direct_light_no_shadow_out",
        "shade_direct_light_with_shadows_out",
    ):
        start = source.index(f"void {base_name}(")
        signature = source[start : source.index(");", start)]
        assert "traversal_counter_" not in signature

    for diagnostics_name in (
        "shade_direct_light_no_shadow_diagnostics_out",
        "shade_direct_light_with_shadows_diagnostics_out",
    ):
        start = source.index(f"void {diagnostics_name}(")
        signature = source[start : source.index(");", start)]
        counter_positions = [signature.index(f"traversal_counter_{index}") for index in range(9)]
        assert counter_positions == sorted(counter_positions)


@pytest.mark.parametrize("triangles,planes", [(False, False), (True, False), (False, True), (True, True)])
def test_scene_launch_latches_follow_actual_entries_not_diagnostic_phases(triangles, planes):
    from optics.cuda_scene import CudaSceneUpdateStage

    calls = []
    stage = object.__new__(CudaSceneUpdateStage)
    for name, present in (("triangle", triangles), ("plane", planes)):
        object.__setattr__(stage, "_" + name + "_entry",
                           (lambda *args, name=name: calls.append(name)) if present else None)
        object.__setattr__(stage, "_" + name + "_args", ())

    class Transaction:
        @contextmanager
        def enqueue_stage(self, actual):
            assert actual is stage
            yield

        def record_diagnostic_start(self, name):
            calls.append("start")

        def record_diagnostic_end(self, name):
            calls.append("end")

        def before_native_launch(self):
            calls.append("launch")

    stage.enqueue(Transaction())
    assert calls == ["start", *(["launch", "triangle"] if triangles else []),
                     *(["launch", "plane"] if planes else []), "end"]


def test_native_launch_outside_active_transaction_stage_rejects():
    transaction = object.__new__(cuda_timeline.CudaOpticalTransaction)
    transaction._state = "validated_open"
    transaction._active_stage = None
    transaction._poison_reason = None
    with pytest.raises(CudaOpticalTransactionPoisonedError, match="active CUDA stage"):
        transaction.before_native_launch()
    assert transaction.state == "poisoned"


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_lbvh_launch_latch_requires_an_actual_native_entry(populated, diagnostic):
    from optics.cuda_lbvh import CudaLbvhRebuildStage

    calls = []
    stage = object.__new__(CudaLbvhRebuildStage)
    object.__setattr__(stage, "_extension_entry", (lambda: calls.append("native")) if populated else None)
    object.__setattr__(stage, "_extension_args", ())

    class Transaction:
        @contextmanager
        def enqueue_stage(self, actual):
            assert actual is stage
            yield

        def record_diagnostic_start(self, name):
            if diagnostic:
                calls.append("start")

        def record_diagnostic_end(self, name):
            if diagnostic:
                calls.append("end")

        def before_native_launch(self):
            calls.append("launch")

    stage.enqueue(Transaction())
    assert calls == (["start"] if diagnostic else []) + (["launch", "native"] if populated else []) + (
        ["end"] if diagnostic else [])
