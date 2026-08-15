from __future__ import annotations

import pytest

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
