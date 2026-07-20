"""Render a MuJoCo Menagerie robot model with the GPU optical pipeline.

This example is a legacy static-preview CLI wrapper. The implementation lives
in ``tools.optical_pipeline_lab.menagerie_static_runner`` while P11 preset
workflows provide the primary user-facing Optical Pipeline Lab API.

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

from tools.optical_pipeline_lab.menagerie_static_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
