"""Deprecated compatibility shim for the Menagerie static runner.

Go2/Menagerie is a legacy static preview scenario, not the Optical Pipeline
Lab backend. New code should import ``menagerie_static_runner`` or the P11
preset workflow APIs instead.
"""

from __future__ import annotations

from . import menagerie_static_runner as _impl

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)

_build_video_camera = _impl._build_video_camera
_pack_video_rgb8 = _impl._pack_video_rgb8
wp = _impl.wp


def _render_video_frame(pipeline, args, frame_index, ray_cache):
    return _impl.video_loop.render_video_frame(
        pipeline,
        args,
        frame_index,
        ray_cache,
        build_video_camera=_build_video_camera,
    )


def _run_video_benchmark(pipeline, args, out_dir):
    return _impl.video_loop.run_video_benchmark(
        pipeline,
        args,
        out_dir,
        build_video_camera=_build_video_camera,
        pack_rgb8=_pack_video_rgb8,
        synchronize_event=getattr(wp, "synchronize_event", lambda event: None),
        render_frame=_render_video_frame,
    )


def _build_torch_async_warmup_result(pipeline, args, delivery_request):
    return _impl.video_loop.build_torch_async_warmup_result(
        pipeline,
        args,
        delivery_request,
        build_video_camera=_build_video_camera,
    )


__all__ = tuple(
    name for name in globals() if not (name.startswith("__") and name.endswith("__")) and name != "_impl"
)

del _name
