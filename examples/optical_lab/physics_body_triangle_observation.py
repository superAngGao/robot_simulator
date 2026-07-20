"""Run the physics body-triangle preset with an explicit observation product."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_env.obs import locomotion_obs_schema
from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset
from tools.optical_pipeline_lab.presets import get_preset
from tools.optical_pipeline_lab.product_specs import ObservationProductSpec


def build_observation_spec() -> ObservationProductSpec:
    """Build the reviewed explicit observation spec for the example preset."""

    config = get_preset("physics_body_triangle_video_smoke")
    schema = locomotion_obs_schema(
        num_actuated_joints=2,
        num_contact_bodies=2,
        include_contact_mask=True,
    )
    return ObservationProductSpec.from_scenario(
        config,
        schema=schema,
        actuated_q_indices=np.array([7, 8], dtype=np.intp),
        actuated_v_indices=np.array([6, 7], dtype=np.intp),
        contact_body_names=("left_foot", "right_foot"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("runs/examples/physics_body_triangle_observation"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    preset = "physics_body_triangle_video_smoke"
    observation = build_observation_spec()
    products = ("debug", observation)

    if args.dry_run:
        print(
            f"Would run: preset={preset} frames={args.frames} products=('debug', observation) out={args.out}"
        )
        return

    result = run_optical_lab_preset(
        preset,
        frames=args.frames,
        products=products,
        out=args.out,
        device=args.device,
    )
    print(result.product_results["observation"][-1].payload["observation"])


if __name__ == "__main__":
    main()
