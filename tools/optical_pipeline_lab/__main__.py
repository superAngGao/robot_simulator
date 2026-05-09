"""Command-line entry point for the Optical Pipeline Lab foundation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .matrix import AVAILABLE_MATRIX_SUITES, MatrixRunOptions, get_suite, run_matrix_suite
from .presets import get_preset
from .runner import (
    DEFAULT_LAB_WARMUP_RENDERS,
    LabRunOptions,
    apply_run_overrides,
    run_scenario,
    validate_scenario,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical Pipeline Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Describe a lab preset and validate reserved modes.")
    describe.add_argument("--preset", default="go2_video_ordered_static")

    run = subparsers.add_parser("run", help="Run an implemented lab preset.")
    run.add_argument("--preset", default="go2_video_ordered_static")
    run.add_argument("--out", help="Output directory. Defaults to out/optical_pipeline_lab/<preset>.")
    run.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    run.add_argument("--model-xml", default="go2.xml")
    run.add_argument("--device")
    run.add_argument("--width", type=int)
    run.add_argument("--height", type=int)
    run.add_argument("--frames", type=int, default=10)
    run.add_argument("--fps", type=float, default=30.0)
    run.add_argument("--warmup-renders", type=int, default=DEFAULT_LAB_WARMUP_RENDERS)
    run.add_argument("--progress-every", type=int, default=5)
    run.add_argument("--video-raygen", choices=("host", "gpu"), default="gpu")
    run.add_argument("--video-ray-cache", choices=("off", "precompute"), default="off")
    run.add_argument("--video-readback-delivery", choices=("sync", "torch_async"), default="sync")
    run.add_argument("--video-readback-ring-depth", type=int, default=2)
    run.add_argument("--readback", choices=("none", "rgb", "rgb8", "full"))
    run.add_argument("--render-profile", action="store_true")
    run.add_argument("--fail-on-overflow", dest="fail_on_overflow", action="store_true", default=True)
    run.add_argument("--no-fail-on-overflow", dest="fail_on_overflow", action="store_false")
    run.add_argument("--shadows", dest="shadows", action="store_true", default=None)
    run.add_argument("--no-shadows", dest="shadows", action="store_false")
    run.add_argument("--write-frames", dest="write_frames", action="store_true", default=None)
    run.add_argument("--no-write-frames", dest="write_frames", action="store_false")
    run.add_argument("--verbose-warp", action="store_true")

    matrix = subparsers.add_parser("matrix", help="Run a named baseline matrix suite serially.")
    matrix.add_argument("--suite", choices=AVAILABLE_MATRIX_SUITES, default="go2_video_ordered_baseline")
    matrix.add_argument("--out", help="Output directory. Defaults to out/optical_pipeline_lab/<suite>.")
    matrix.add_argument("--model-dir", default="out/external/mujoco_menagerie/unitree_go2")
    matrix.add_argument("--model-xml", default="go2.xml")
    matrix.add_argument("--device")
    matrix.add_argument("--frames", type=int, default=10)
    matrix.add_argument("--fps", type=float, default=30.0)
    matrix.add_argument("--warmup-renders", type=int, default=DEFAULT_LAB_WARMUP_RENDERS)
    matrix.add_argument("--progress-every", type=int, default=5)
    matrix.add_argument("--video-raygen", choices=("host", "gpu"), default="gpu")
    matrix.add_argument("--video-ray-cache", choices=("off", "precompute"), default="off")
    matrix.add_argument("--render-profile", action="store_true")
    matrix.add_argument("--include-full-debug", action="store_true")
    matrix.add_argument("--stop-on-failure", action="store_true")
    matrix.add_argument("--verbose-warp", action="store_true")

    args = parser.parse_args()
    if args.command == "describe":
        config = get_preset(args.preset)
        validate_scenario(config)
        for field, value in config.__dict__.items():
            print(f"{field}: {value.value if hasattr(value, 'value') else value}")
    elif args.command == "run":
        if args.frames < 0:
            parser.error("--frames must be >= 0")
        if args.fps <= 0.0:
            parser.error("--fps must be > 0")
        if args.progress_every < 0:
            parser.error("--progress-every must be >= 0")
        config = apply_run_overrides(
            get_preset(args.preset),
            device=args.device,
            width=args.width,
            height=args.height,
            readback=args.readback,
            shadows=args.shadows,
            write_frames=args.write_frames,
        )
        out = Path(args.out) if args.out else Path("out") / "optical_pipeline_lab" / args.preset
        options = LabRunOptions(
            out=out,
            model_dir=args.model_dir,
            model_xml=args.model_xml,
            frames=args.frames,
            fps=args.fps,
            warmup_renders=args.warmup_renders,
            progress_every=args.progress_every,
            video_raygen=args.video_raygen,
            video_ray_cache=args.video_ray_cache,
            video_readback_delivery=args.video_readback_delivery,
            video_readback_ring_depth=args.video_readback_ring_depth,
            render_profile=args.render_profile,
            fail_on_overflow=args.fail_on_overflow,
            verbose_warp=args.verbose_warp,
        )
        try:
            run_scenario(config, options)
        except (NotImplementedError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "matrix":
        if args.frames < 0:
            parser.error("--frames must be >= 0")
        if args.fps <= 0.0:
            parser.error("--fps must be > 0")
        if args.progress_every < 0:
            parser.error("--progress-every must be >= 0")
        out = Path(args.out) if args.out else Path("out") / "optical_pipeline_lab" / args.suite
        try:
            suite = get_suite(args.suite, include_full_debug=args.include_full_debug)
            run_matrix_suite(
                suite,
                MatrixRunOptions(
                    out=out,
                    device=args.device,
                    model_dir=args.model_dir,
                    model_xml=args.model_xml,
                    frames=args.frames,
                    fps=args.fps,
                    warmup_renders=args.warmup_renders,
                    progress_every=args.progress_every,
                    video_raygen=args.video_raygen,
                    video_ray_cache=args.video_ray_cache,
                    render_profile=args.render_profile,
                    verbose_warp=args.verbose_warp,
                    keep_going=not args.stop_on_failure,
                ),
            )
        except (NotImplementedError, ValueError) as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    main()
