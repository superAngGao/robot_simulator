"""Device-channel materialization helpers for optical compute results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def channel_is_device(value: object) -> bool:
    """Return whether a channel value is backed by device memory."""

    torch = _optional_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return bool(value.is_cuda)
    device = getattr(value, "device", None)
    if device is None:
        return False
    return "cpu" not in str(device).lower()


def channel_to_torch(value: object):
    """Return a contiguous Torch tensor view/copy for a device channel."""

    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value if value.is_contiguous() else value.contiguous()

    wp = _optional_warp()
    if wp is not None:
        try:
            tensor = wp.to_torch(value)
        except Exception:
            pass
        else:
            # Warp owns the underlying allocation. Callers must keep the
            # original Warp array/result resources alive for at least as long
            # as they use this Torch view.
            return tensor if tensor.is_contiguous() else tensor.contiguous()

    if hasattr(value, "__array__") or isinstance(value, (list, tuple)):
        tensor = torch.as_tensor(value)
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    raise TypeError(f"unsupported optical channel type for Torch conversion: {type(value).__name__}")


def channel_to_numpy(value: object) -> np.ndarray:
    """Return a host NumPy array for a channel value."""

    torch = _optional_torch()
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()

    wp = _optional_warp()
    if wp is not None:
        try:
            tensor = wp.to_torch(value)
        except Exception:
            pass
        else:
            return channel_to_numpy(tensor)

    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def stage_channels_to_host(result, channels: "Sequence[str]") -> dict[str, np.ndarray]:
    """Stage selected result channels to host NumPy arrays without dtype canonicalization."""

    return {name: np.asarray(channel_to_numpy(result.channel(name))).copy() for name in channels}


def _optional_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _require_torch():
    torch = _optional_torch()
    if torch is None:
        raise ImportError("Optical device channel Torch conversion requires torch")
    return torch


def _optional_warp():
    try:
        import warp as wp
    except Exception:
        return None
    return wp
