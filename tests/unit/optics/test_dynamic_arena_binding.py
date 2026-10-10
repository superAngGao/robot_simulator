"""Dynamic C3 allocation arrows with native metadata doubles, not GPU evidence."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from optical_worker.arena import native_optical_arena_spec_from_row
from optical_worker.native_contract import WarpPreparedNativeInput, register_prepared_native_input
from optical_worker.native_input import CanonicalNativeInput
from optical_worker.provenance import DYNAMIC_SCENE_OUTPUT_NAMES, NativeStorageSignature
from optical_worker.render_parameters import bind_render_parameters, resolve_render_source
from optical_worker.workloads import build_canonical_transform_replay
from optics import warp_arena
from tests.unit.benchmarks.test_optical_benchmark_c4a import _readiness
from tests.unit.optics.test_guarded_native_readback import _WarpArray


@pytest.mark.parametrize("row_id", ["r1_dynamic_full_shadow", "r1_dynamic_render_only_no_shadow"])
@pytest.mark.parametrize("sample_index", [0, 119, 120, 121])
def test_dynamic_c3_preserves_replay_frame_and_slot_arrows(monkeypatch, row_id, sample_index):
    from benchmarks.optical.schema import load_frozen_gate

    gate = load_frozen_gate()
    row = next(row for row in gate["mandatory_rows"] if row["row_id"] == row_id)
    render = resolve_render_source(row["case_id"])
    source = CanonicalNativeInput.from_scene_contract(
        render.packed_scene,
        input_contract="canonical_native_replay_v1",
        replay=build_canonical_transform_replay(row["case_id"]),
    )
    device = NS(ordinal=1, is_cuda=True)
    prepared = WarpPreparedNativeInput(
        source.manifest,
        1,
        {name: _WarpArray(value, device) for name, value in source.arrays.items()},
        object(),
        source,
    )
    register_prepared_native_input(prepared, source)
    spec = native_optical_arena_spec_from_row(
        row=row,
        case_contract=gate["cases"][row["case_id"]],
        prepared=prepared,
        readiness=_readiness(prepared),
        execution_stream_token="warp-stream:1:17",
        primitive_count=1,
        node_capacity=1,
    )
    parameters = bind_render_parameters(
        render.sequence,
        sample_index=sample_index,
        profile=row["output_profile"],
        shadows=row["shadows"],
        pass_mode="primary_unprofiled",
        role_to_mask=render.packed_scene.role_to_mask,
    )
    world = NS(**{name: np.empty((1, 3)) for name in DYNAMIC_SCENE_OUTPUT_NAMES})
    bvh = NS(
        **{
            name: np.empty((1,))
            for name in ("bounds_min", "bounds_max", "left", "right", "start", "count", "prim_ids")
        }
    )
    stages = (
        NS(outputs=world, evidence=NS(operation_id=None, dispatch_plan_sha256="a" * 64)),
        NS(bvh=bvh, evidence=NS(operation_id=None, dispatch_plan_sha256="b" * 64, topology_sha256="c" * 64)),
        NS(
            binding=NS(
                factory=NS(prepared_input=prepared, device=device),
                transform_frame_index=sample_index % source.manifest.transform_frame_count,
            ),
            evidence=NS(sha256="d" * 64),
            validate_resolved_parameters=lambda p: None,
        ),
    )
    # Read operation identities from the actual factory constants, so this test
    # checks composition rather than introducing another operation registry.
    from optics.device_bvh import WARP_BVH_REFIT_OPERATION_ID
    from optics.device_scene import WARP_SCENE_UPDATE_OPERATION_ID

    stages[0].evidence.operation_id = WARP_SCENE_UPDATE_OPERATION_ID
    stages[1].evidence.operation_id = WARP_BVH_REFIT_OPERATION_ID

    def signature(name, value, device, owner):
        return NativeStorageSignature(
            name=name,
            backend="warp",
            device_ordinal=device.ordinal,
            storage_token=f"allocation:{id(value)}",
            dtype="float32",
            shape=value.shape,
            nbytes=value.nbytes,
            active_start=0,
            active_count=value.nbytes,
            owner_token=owner,
        )

    monkeypatch.setattr(warp_arena, "_storage_signature", signature)
    outputs = (signature("rgb", np.empty((614400, 3)), device, "warp-slot-owner:0"),)
    binding = warp_arena._bind_c3(spec, parameters, stages, outputs)
    scene, acceleration, final = (stage.evidence for stage in binding.ordered_stage_evidence)
    assert [stage.stage_name for stage in (scene, acceleration, final)] == [
        "scene_update",
        "acceleration",
        "render",
    ]
    assert binding.transform_frame_index == sample_index % 120
    assert final.bound_render_parameters.to_payload()["frame_id"] == sample_index + 1
    assert final.bound_render_parameters == parameters
    by_name = {value.name: value for value in scene.output_storage}
    assert acceleration.input_storage == tuple(
        by_name[name] for name in ("triangle_aabb_min", "triangle_aabb_max")
    )
    render_inputs = {value.name: value for value in final.input_storage}
    assert all(render_inputs[value.name] == value for value in acceleration.output_storage)
    assert [value.owner_token for value in acceleration.output_storage] == [
        "warp-slot-owner:0",
        "warp-slot-owner:0",
        *("warp-session-owner",) * 5,
    ]
    transforms = [value for value in scene.input_storage if value.name.startswith("replay_")]
    assert all(
        value.active_start == binding.transform_frame_index * value.active_count for value in transforms
    )
