"""GPU RGB delivery packing helpers for Optical Pipeline Lab experiments."""

from __future__ import annotations

from optics.execution import OpticalComputeResult

try:
    import warp as wp
except Exception as exc:  # pragma: no cover - optional GPU dependency.
    wp = None
    _WARP_IMPORT_ERROR = exc
else:
    _WARP_IMPORT_ERROR = None


def pack_linear_rgb_to_preview_uint8(result, *, source: str = "rgb", target: str = "rgb8"):
    """Pack a device linear RGB channel into display-oriented uint8 RGB."""
    _require_warp()
    rgb = result.channel(source)
    rgb8 = wp.empty(rgb.shape, dtype=wp.uint8, device=rgb.device)
    stream = _event_stream(result.ready_event, rgb.device)
    with _scoped_stream(stream):
        _wait_on_event(result.ready_event, stream=stream, device=rgb.device)
        wp.launch(
            _linear_rgb_to_preview_uint8_kernel,
            dim=int(rgb.shape[0]),
            inputs=[rgb, rgb8],
            device=rgb.device,
            stream=stream,
        )
        ready_event = (stream or wp.get_stream(rgb.device)).record_event()
    channels = dict(result.channels)
    channels[target] = rgb8
    return OpticalComputeResult(
        frame_id=result.frame_id,
        sim_time=result.sim_time,
        env_idx=result.env_idx,
        sensor_id=result.sensor_id,
        location=result.location,
        channels=channels,
        output_profile=result.output_profile,
        ready_event=ready_event,
        resources=result.resources + (rgb8,),
    )


def rgb_pack_available() -> bool:
    return wp is not None


def rgb_pack_import_error() -> Exception | None:
    return _WARP_IMPORT_ERROR


def _event_stream(ready_event, device):
    if ready_event is None:
        return None
    stream = getattr(ready_event, "stream", None)
    if stream is not None:
        return stream
    return wp.get_stream(device)


def _wait_on_event(ready_event, *, stream, device) -> None:
    if ready_event is None:
        return
    target_stream = stream or wp.get_stream(device)
    try:
        target_stream.wait_event(ready_event)
    except AttributeError:
        wp.synchronize_event(ready_event)


class _scoped_stream:
    def __init__(self, stream) -> None:
        self.stream = stream

    def __enter__(self):
        if self.stream is None:
            return None
        self.context = wp.ScopedStream(self.stream)
        return self.context.__enter__()

    def __exit__(self, exc_type, exc, tb):
        if self.stream is None:
            return None
        return self.context.__exit__(exc_type, exc, tb)


def _require_warp() -> None:
    if wp is None:
        raise ImportError("RGB8 packing requires warp") from _WARP_IMPORT_ERROR


if wp is not None:

    @wp.kernel
    def _linear_rgb_to_preview_uint8_kernel(
        rgb: wp.array2d(dtype=wp.float32),
        rgb8: wp.array2d(dtype=wp.uint8),
    ):
        ray = wp.tid()
        for channel in range(3):
            value = rgb[ray, channel]
            if value != value:
                value = 0.0
            if value < 0.0:
                value = 0.0
            if value > 1.0:
                value = 1.0
            display = wp.pow(value, 1.0 / 2.2)
            packed = int(display * 255.0 + 0.5)
            if packed < 0:
                packed = 0
            if packed > 255:
                packed = 255
            rgb8[ray, channel] = wp.uint8(packed)
