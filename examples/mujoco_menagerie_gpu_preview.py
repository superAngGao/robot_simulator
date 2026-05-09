"""Render a MuJoCo Menagerie robot model with the GPU optical pipeline.

This example is now a thin CLI wrapper around the Optical Pipeline Lab Go2
backend. The implementation lives in ``tools.optical_pipeline_lab.go2_backend``
so lab runners and examples do not share a private example API.

Example:

    conda run -n env_tilelang_20260119 python examples/mujoco_menagerie_gpu_preview.py \
      --model-dir out/external/mujoco_menagerie/unitree_go2 \
      --model-xml go2.xml \
      --out out/menagerie_go2_gpu_preview
"""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.optical_pipeline_lab.go2_backend import main  # noqa: E402


if __name__ == "__main__":
    main()
