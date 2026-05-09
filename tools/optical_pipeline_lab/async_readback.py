"""Async device-to-host readback helpers for Optical Pipeline Lab experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    import torch
except Exception as exc:  # pragma: no cover - optional GPU dependency.
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional GPU dependency.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


@dataclass
class TorchAsyncReadbackSlot:
    """One reusable pinned host slot in an ordered D2H readback ring."""

    index: int
    host_tensors: dict[str, object]
    copy_start_event: object
    copy_done_event: object


@dataclass
class TorchAsyncReadbackJob:
    """Submitted async readback copy and its timing metadata."""

    frame_index: int
    slot: TorchAsyncReadbackSlot
    submit_ms: float
    result: object

    def synchronize(self) -> float:
        """Block until the D2H copy is complete and return wall wait time in ms."""
        wait_start = time.perf_counter()
        self.slot.copy_done_event.synchronize()
        return (time.perf_counter() - wait_start) * 1000.0

    def copy_elapsed_ms(self) -> float:
        return float(self.slot.copy_start_event.elapsed_time(self.slot.copy_done_event))

    def host_channels(self) -> dict[str, object]:
        return {name: tensor.numpy() for name, tensor in self.slot.host_tensors.items()}


class TorchAsyncReadbackRing:
    """Pinned-host ordered D2H ring backed by Torch non-blocking CUDA copies."""

    def __init__(
        self,
        *,
        channels: Sequence[str],
        ring_depth: int,
        copy_stream: object,
        slots: list[TorchAsyncReadbackSlot],
    ) -> None:
        if ring_depth <= 0:
            raise ValueError("ring_depth must be > 0")
        self.channels = tuple(channels)
        self.ring_depth = int(ring_depth)
        self.copy_stream = copy_stream
        self.slots = slots

    @classmethod
    def from_warmup_result(
        cls,
        warmup_result,
        *,
        channels: Sequence[str],
        ring_depth: int,
    ) -> "TorchAsyncReadbackRing":
        """Allocate pinned slots from one representative device result."""
        _require_torch_and_warp()
        if ring_depth <= 0:
            raise ValueError("ring_depth must be > 0")
        device_tensors = _torch_device_tensors_for_channels(warmup_result, channels)
        first_tensor = next(iter(device_tensors.values()))
        copy_stream = torch.cuda.Stream(device=first_tensor.device)
        slots = [
            TorchAsyncReadbackSlot(
                index=slot_index,
                host_tensors={
                    name: torch.empty(
                        tuple(device_tensor.shape),
                        dtype=device_tensor.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    for name, device_tensor in device_tensors.items()
                },
                copy_start_event=torch.cuda.Event(enable_timing=True),
                copy_done_event=torch.cuda.Event(enable_timing=True),
            )
            for slot_index in range(int(ring_depth))
        ]

        ring = cls(
            channels=channels,
            ring_depth=ring_depth,
            copy_stream=copy_stream,
            slots=slots,
        )
        warmup_job = ring.submit(warmup_result, frame_index=-1)
        warmup_job.synchronize()
        return ring

    def submit(self, result, *, frame_index: int) -> TorchAsyncReadbackJob:
        submit_start = time.perf_counter()
        slot = self.slots[int(frame_index) % len(self.slots)]
        # The ring only owns the Torch copy stream. Callers must ensure
        # result.ready_event has completed or been synchronized before submit.
        _copy_torch_async_channels(
            _torch_device_tensors_for_channels(result, self.channels),
            slot,
            copy_stream=self.copy_stream,
        )
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        return TorchAsyncReadbackJob(
            frame_index=int(frame_index),
            slot=slot,
            submit_ms=submit_ms,
            result=result,
        )


def torch_async_readback_available() -> bool:
    return torch is not None and wp is not None


def torch_async_readback_import_error() -> Exception | None:
    return _TORCH_IMPORT_ERROR if torch is None else _WARP_IMPORT_ERROR


def _torch_device_tensors_for_channels(result, channels: Sequence[str]) -> dict[str, object]:
    _require_torch_and_warp()
    device_tensors = {}
    for name in channels:
        device_tensor = wp.to_torch(result.channel(name))
        if not device_tensor.is_contiguous():
            device_tensor = device_tensor.contiguous()
        device_tensors[name] = device_tensor
    return device_tensors


def _copy_torch_async_channels(
    device_tensors: dict[str, object],
    slot: TorchAsyncReadbackSlot,
    *,
    copy_stream,
) -> None:
    with torch.cuda.stream(copy_stream):
        slot.copy_start_event.record(copy_stream)
        for name, device_tensor in device_tensors.items():
            slot.host_tensors[name].copy_(device_tensor, non_blocking=True)
        slot.copy_done_event.record(copy_stream)


def _require_torch_and_warp() -> None:
    if torch is None:
        raise ImportError("Torch async readback requires torch with CUDA support") from _TORCH_IMPORT_ERROR
    if wp is None:
        raise ImportError("Torch async readback requires warp") from _WARP_IMPORT_ERROR
