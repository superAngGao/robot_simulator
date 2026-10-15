from __future__ import annotations

import pytest

import optics.cuda_direct_light as cuda_direct_light
import optics.cuda_timeline as cuda_timeline
from optics.cuda_timeline import (
    CudaDeferredProfile,
    CudaOpticalCompletion,
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransactionPoisonedError,
)


def test_cuda_timeline_contract_imports_without_runtime_initialization():
    assert issubclass(CudaOpticalTransactionPoisonedError, CudaOpticalTimelineError)
    assert CudaOpticalCompletion.__dataclass_fields__["recorded"].default is True


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
