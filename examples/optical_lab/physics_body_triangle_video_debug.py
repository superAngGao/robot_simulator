"""Run the physics body-triangle preset with video and debug products."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("runs/examples/physics_body_triangle_video_debug"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    preset = "physics_body_triangle_video_smoke"
    products = ("video", "debug")

    if args.dry_run:
        print(f"Would run: preset={preset} frames={args.frames} products={products} out={args.out}")
        return

    result = run_optical_lab_preset(
        preset,
        frames=args.frames,
        products=products,
        out=args.out,
        device=args.device,
    )
    print(f"artifacts={dict(result.artifacts)}")
    print(f"products={sorted(result.product_results)}")


if __name__ == "__main__":
    main()
