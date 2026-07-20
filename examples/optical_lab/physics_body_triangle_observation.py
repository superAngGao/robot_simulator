"""Run the physics body-triangle preset with an explicit observation product."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.optical_pipeline_lab.preset_workflows import run_optical_lab_preset


def build_observation_spec():
    """Build the reviewed explicit observation spec for the example preset."""

    import numpy as np

    from rl_env.obs import locomotion_obs_schema
    from tools.optical_pipeline_lab.presets import get_preset
    from tools.optical_pipeline_lab.product_specs import ObservationProductSpec

    config = get_preset("physics_body_triangle_video_smoke")
    schema = locomotion_obs_schema(
        num_actuated_joints=0,
        include_contact_mask=False,
    )
    # physics_body_triangle is a single free-joint ball:
    # q[0:7] is the root pose, v[0:6] is the root velocity, and there are no
    # actuated joints in the reviewed smoke model.
    return ObservationProductSpec.from_scenario(
        config,
        schema=schema,
        actuated_q_indices=np.array([], dtype=np.intp),
        actuated_v_indices=np.array([], dtype=np.intp),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("runs/examples/physics_body_triangle_observation"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    preset = "physics_body_triangle_video_smoke"

    if args.dry_run:
        print(
            f"Would run: preset={preset} frames={args.frames} products=('debug', observation) out={args.out}"
        )
        return

    observation = build_observation_spec()
    result = run_optical_lab_preset(
        preset,
        frames=args.frames,
        products=("debug", observation),
        out=args.out,
        device=args.device,
    )
    print(result.product_results["observation"][-1].payload["observation"])


if __name__ == "__main__":
    main()
