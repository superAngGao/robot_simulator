"""Thin smoke caller for the production caller-owned CUDA LBVH rebuild.

Run from the repository root, for example::

    python examples/cuda_lbvh_extension_spike.py --primitive-count 8

The extension source and lifecycle contracts live under :mod:`optics`; this
example deliberately owns no CUDA source or alternate implementation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optics.cuda_lbvh import (
    CudaLbvhRebuildStageFactory,
    allocate_cuda_lbvh_buffers,
    allocate_cuda_lbvh_workspace,
)
from optics.cuda_scene import CudaWorldSceneBuffers
from optics.cuda_timeline import CudaOpticalExecutionContext


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Torch CUDA is unavailable")
    target = torch.device(args.device)
    if target.type != "cuda" or target.index is None:
        raise SystemExit("--device must be an indexed CUDA device")
    if args.primitive_count < 0:
        raise SystemExit("--primitive-count must be non-negative")

    with torch.cuda.device(target):
        stream = torch.cuda.Stream(device=target)
        generator = torch.Generator(device=target).manual_seed(args.seed)
        centers = torch.rand(
            (args.primitive_count, 3),
            dtype=torch.float32,
            device=target,
            generator=generator,
        )
        half_extent = 0.01 + 0.09 * torch.rand(
            (args.primitive_count, 3),
            dtype=torch.float32,
            device=target,
            generator=generator,
        )
        world = _world_buffers(centers - half_extent, centers + half_extent, target)
        context = CudaOpticalExecutionContext(device=target, stream=stream)
        provenance = hashlib.sha256(
            f"thin-cuda-lbvh-smoke:{args.primitive_count}:{args.seed}".encode()
        ).hexdigest()
        factory = CudaLbvhRebuildStageFactory.prepare(
            context=context,
            primitive_count=args.primitive_count,
            native_input_manifest_sha256=provenance,
            scene_contract_sha256=provenance,
            input_contract="thin_example_only_v1",
        )
        outputs = allocate_cuda_lbvh_buffers(
            primitive_count=args.primitive_count,
            device=target,
        )
        workspace = allocate_cuda_lbvh_workspace(factory.requirements)
        stage = factory.bind(world=world, outputs=outputs, workspace=workspace)
        completion_event = torch.cuda.Event(enable_timing=False)
        completion_event.record(stream)
        completion_event.synchronize()
        transaction = context.begin_transaction(
            stages=(stage,),
            completion_event=completion_event,
        )
        stage.enqueue(transaction)
        completion = transaction.finish()
        completion.event.synchronize()

    print(
        json.dumps(
            {
                "device": str(target),
                "primitive_count": args.primitive_count,
                "node_count": outputs.node_count,
                "sorted_primitive_ids": outputs.sorted_primitive_ids.cpu().tolist(),
                "workspace": factory.requirements.to_payload(),
                "stage_evidence_sha256": stage.evidence.sha256,
                "stage_names": list(completion.stage_names),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _world_buffers(minimum, maximum, device: torch.device) -> CudaWorldSceneBuffers:
    primitive_count = int(minimum.shape[0])
    triangle = torch.empty((primitive_count, 3), dtype=torch.float32, device=device)
    plane = torch.empty((0, 3), dtype=torch.float32, device=device)
    return CudaWorldSceneBuffers(
        device_ordinal=int(device.index),
        triangle_v0_world=triangle,
        triangle_e1_world=torch.empty_like(triangle),
        triangle_e2_world=torch.empty_like(triangle),
        triangle_normal_world=torch.empty_like(triangle),
        plane_normal_world=plane,
        plane_point_world=torch.empty_like(plane),
        triangle_aabb_min=minimum,
        triangle_aabb_max=maximum,
        resources=(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--primitive-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    main()
