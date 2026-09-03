from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.optical.eligibility import (
    CANONICAL_NATIVE_REPLAY_INPUT,
    verified_runtime_entry_point_audit,
)
from benchmarks.optical.instrumentation import (
    instrument_runtime_entry_points,
    invoke_instrumented_entry_point,
)
from benchmarks.optical.native import (
    CanonicalNativeInput,
    build_prepared_cuda_lbvh_rebuild_factory,
    build_prepared_cuda_scene_update_factory,
    build_prepared_cuda_transform_inputs,
    prepare_cuda_native_input,
)
from benchmarks.optical.timing import RuntimeOperationTrace, native_timed_submission
from benchmarks.optical.workloads import (
    build_canonical_packed_scene,
    build_canonical_transform_replay,
)
from optics.cuda_direct_light import (
    CudaDirectLightNativeRenderer,
    CudaFirstHitInputs,
    CudaShadeInputs,
    allocate_cuda_direct_light_buffers,
)
from optics.cuda_lbvh import (
    CUDA_LBVH_ACCELERATION_ID,
    CUDA_LBVH_MAX_PRIMITIVES,
    CUDA_LBVH_OPERATION_ID,
    CUDA_LBVH_TOPOLOGY_ORIGIN_ID,
    CudaLbvhRebuildStageFactory,
    _load_cuda_lbvh_extension,
    allocate_cuda_lbvh_buffers,
    allocate_cuda_lbvh_workspace,
    cuda_lbvh_workspace_requirements,
)
from optics.cuda_scene import CudaWorldSceneBuffers, allocate_cuda_world_scene_buffers
from optics.cuda_timeline import CudaOpticalExecutionContext
from optics.execution import OpticalOutputProfile

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.cuda_ext,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA is unavailable"),
]


@pytest.mark.parametrize(
    "aabb_min,aabb_max",
    (
        (
            [
                [0.0, 0.0, 0.0],
                [2.0, -1.0, 0.5],
                [-2.0, 3.0, -1.0],
                [0.25, 0.25, 0.25],
                [0.0, 0.0, 0.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [3.5, 0.0, 2.0],
                [-1.0, 4.0, 0.0],
                [0.75, 0.75, 0.75],
                [1.0, 1.0, 1.0],
            ],
        ),
        (
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        ),
    ),
    ids=("asymmetric_and_tied_centroids", "degenerate_scene_extent"),
)
def test_cuda_lbvh_rebuild_matches_morton_topology_and_descendant_bounds(
    aabb_min,
    aabb_max,
):
    setup = _raw_setup(aabb_min, aabb_max)
    completion = _run(setup)
    completion.event.synchronize()

    minimum = np.asarray(aabb_min, dtype=np.float32)
    maximum = np.asarray(aabb_max, dtype=np.float32)
    expected_keys, expected_ids = _morton_reference(minimum, maximum)
    assert setup.outputs.sorted_morton_keys.cpu().tolist() == expected_keys
    assert setup.outputs.sorted_primitive_ids.cpu().tolist() == expected_ids
    np.testing.assert_array_equal(
        setup.outputs.primitive_ids.cpu().numpy(),
        np.arange(len(aabb_min), dtype=np.int32),
    )
    np.testing.assert_allclose(
        setup.outputs.scene_bounds_min.cpu().numpy(),
        minimum.min(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        setup.outputs.scene_bounds_max.cpu().numpy(),
        maximum.max(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    _assert_topology_and_bounds(setup.outputs, minimum, maximum, expected_ids)
    assert setup.stage.evidence.acceleration_id == CUDA_LBVH_ACCELERATION_ID
    assert setup.stage.evidence.operation_id == CUDA_LBVH_OPERATION_ID
    assert setup.stage.evidence.topology_origin_id == CUDA_LBVH_TOPOLOGY_ORIGIN_ID
    assert setup.stage.evidence.extension_dispatch_count == 1
    assert completion.stage_names == ("cuda_lbvh_rebuild",)


def test_cuda_lbvh_native_enqueue_has_no_observed_timed_contamination():
    setup = _raw_setup(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
    )
    transaction = setup.context.begin_transaction(
        stages=(setup.stage,),
        completion_event=_initialized_event(setup.context.stream),
    )
    audits = (
        verified_runtime_entry_point_audit(
            operation="acceleration",
            entry_point="optics.cuda_lbvh.CudaLbvhRebuildStage.enqueue",
        ),
        verified_runtime_entry_point_audit(
            operation="render",
            entry_point="optics.cuda_timeline.CudaOpticalTransaction.finish",
        ),
    )
    trace = RuntimeOperationTrace(row_id="unit_cuda_c3d", lane="cuda", boundary="R1")
    with instrument_runtime_entry_points(audits) as observation:
        with native_timed_submission(
            row_id=trace.row_id,
            lane=trace.lane,
            boundary=trace.boundary,
            operation_trace=trace,
        ):
            invoke_instrumented_entry_point(
                "optics.cuda_lbvh.CudaLbvhRebuildStage.enqueue",
                setup.stage,
                transaction,
            )
            completion = invoke_instrumented_entry_point(
                "optics.cuda_timeline.CudaOpticalTransaction.finish",
                transaction,
            )
    assert observation.violation_hook_id is None
    completion.event.synchronize()
    assert [value.operation for value in trace.evidence().records] == [
        "acceleration",
        "render",
    ]


@pytest.mark.parametrize("primitive_count", (2, 4, 17, 257))
def test_cuda_lbvh_topology_is_complete_for_even_odd_and_multiblock_counts(
    primitive_count,
):
    generator = np.random.default_rng(primitive_count)
    minimum = generator.uniform(-10.0, 10.0, size=(primitive_count, 3)).astype(np.float32)
    maximum = minimum + generator.uniform(0.001, 1.0, size=(primitive_count, 3)).astype(np.float32)
    setup = _raw_setup(minimum, maximum)
    _run(setup).event.synchronize()
    _, expected_ids = _morton_reference(minimum, maximum)
    _assert_topology_and_bounds(setup.outputs, minimum, maximum, expected_ids)


@pytest.mark.parametrize("primitive_count", (0, 1))
def test_cuda_lbvh_zero_and_single_primitive_have_explicit_dispatch_contract(
    primitive_count,
):
    minimum = np.asarray([[1.0, 2.0, 3.0]][:primitive_count], dtype=np.float32).reshape(primitive_count, 3)
    maximum = np.asarray([[4.0, 5.0, 6.0]][:primitive_count], dtype=np.float32).reshape(primitive_count, 3)
    setup = _raw_setup(minimum, maximum)
    completion = _run(setup)
    completion.event.synchronize()
    evidence = setup.stage.evidence
    if primitive_count == 0:
        assert setup.factory.requirements.required_bytes == 0
        assert evidence.extension_entry is None
        assert evidence.extension_dispatch_count == 0
        assert evidence.kernel_dispatch_count == 0
        assert evidence.cub_dispatch_count == 0
        assert evidence.async_reset_count == 0
        assert setup.outputs.node_count == 0
    else:
        assert evidence.extension_entry == "cuda_lbvh_rebuild_out"
        assert evidence.kernel_dispatch_count == 4
        assert evidence.cub_dispatch_count == 7
        assert evidence.async_reset_count == 5
        assert setup.outputs.left.cpu().tolist() == [-1]
        assert setup.outputs.right.cpu().tolist() == [-1]
        assert setup.outputs.parent.cpu().tolist() == [-1]
        assert setup.outputs.start.cpu().tolist() == [0]
        assert setup.outputs.count.cpu().tolist() == [1]
        np.testing.assert_array_equal(setup.outputs.node_bounds_min.cpu().numpy(), minimum)
        np.testing.assert_array_equal(setup.outputs.node_bounds_max.cpu().numpy(), maximum)


def test_cuda_lbvh_cpp_workspace_query_matches_flattened_index_boundary():
    requirements = cuda_lbvh_workspace_requirements(
        CUDA_LBVH_MAX_PRIMITIVES,
        0,
    )
    assert requirements.primitive_count == CUDA_LBVH_MAX_PRIMITIVES
    assert requirements.node_capacity == CUDA_LBVH_MAX_PRIMITIVES * 2 - 1
    with pytest.raises(RuntimeError, match="int32 flattened-node range"):
        _load_cuda_lbvh_extension().cuda_lbvh_workspace_requirements(
            CUDA_LBVH_MAX_PRIMITIVES + 1,
            0,
        )


def test_cuda_lbvh_overwrites_stale_output_and_is_deterministic():
    setup = _raw_setup(
        [[-1.0, -2.0, 0.0], [3.0, 1.0, 4.0], [0.0, 8.0, -3.0]],
        [[0.0, -1.0, 1.0], [4.0, 2.0, 5.0], [2.0, 9.0, -2.0]],
    )
    _run(setup).event.synchronize()
    first = {name: value.cpu().clone() for name, value in setup.outputs.tensor_items()}
    for _, value in setup.outputs.tensor_items():
        if value.dtype.is_floating_point:
            value.fill_(12345.0)
        else:
            value.fill_(12345)
    _run(setup).event.synchronize()
    for name, value in setup.outputs.tensor_items():
        assert torch.equal(value.cpu(), first[name]), name


def test_cuda_lbvh_rejects_workspace_storage_and_binding_drift_before_enqueue():
    setup = _raw_setup(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
    )
    with pytest.raises(AttributeError, match="immutable"):
        setup.factory._sealed = False
    with pytest.raises(AttributeError, match="immutable"):
        setup.factory.native_input_manifest_sha256 = "0" * 64

    drifted_factory = CudaLbvhRebuildStageFactory.prepare(
        context=setup.context,
        primitive_count=2,
        native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64,
        input_contract="unit_native_replay_v1",
    )
    object.__setattr__(drifted_factory, "scene_contract_sha256", "0" * 64)
    with pytest.raises(AttributeError, match="immutable"):
        drifted_factory._frozen_signature = drifted_factory._factory_signature()
    with pytest.raises(ValueError, match="changed after construction"):
        drifted_factory.bind(
            world=setup.world,
            outputs=setup.outputs,
            workspace=setup.workspace,
        )

    owner = setup.workspace.resources[0]
    misaligned = replace(setup.workspace, storage=owner[1 : 1 + setup.workspace.storage.numel()])
    with pytest.raises(ValueError, match="aligned"):
        setup.factory.bind(world=setup.world, outputs=setup.outputs, workspace=misaligned)

    wrong_dtype = replace(
        setup.outputs,
        sorted_primitive_ids=torch.empty((2,), dtype=torch.int64, device=setup.context.device),
    )
    with pytest.raises(ValueError, match="wrong dtype"):
        setup.factory.bind(world=setup.world, outputs=wrong_dtype, workspace=setup.workspace)

    wrong_shape = replace(
        setup.outputs,
        sorted_primitive_ids=torch.empty((1,), dtype=torch.int32, device=setup.context.device),
    )
    with pytest.raises(ValueError, match="wrong shape"):
        setup.factory.bind(world=setup.world, outputs=wrong_shape, workspace=setup.workspace)

    noncontiguous = replace(
        setup.outputs,
        sorted_primitive_ids=torch.empty((2, 2), dtype=torch.int32, device=setup.context.device)[:, 0],
    )
    with pytest.raises(ValueError, match="contiguous"):
        setup.factory.bind(
            world=setup.world,
            outputs=noncontiguous,
            workspace=setup.workspace,
        )

    aliased = replace(setup.outputs, sorted_primitive_ids=setup.outputs.primitive_ids)
    with pytest.raises(ValueError, match="overlap"):
        setup.factory.bind(world=setup.world, outputs=aliased, workspace=setup.workspace)

    stage = setup.stage
    with pytest.raises(AttributeError, match="immutable"):
        stage._sealed = False
    with pytest.raises(RuntimeError, match="one-way"):
        stage._freeze()
    arguments = list(stage._extension_args)
    arguments[6], arguments[8] = arguments[8], arguments[6]
    object.__setattr__(stage, "_extension_args", tuple(arguments))
    with pytest.raises(ValueError, match="binding changed"):
        setup.context.begin_transaction(
            stages=(stage,),
            completion_event=_initialized_event(setup.context.stream),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cuda_lbvh_restores_wrong_ambient_device_and_target_stream():
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    setup = _raw_setup(
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
        device="cuda:1",
    )
    assert torch.cuda.current_device() == 0
    target_previous_stream = torch.cuda.current_stream(torch.device("cuda:1"))
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0
    completion = _run(setup)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(torch.device("cuda:1")) == target_previous_stream
    completion.event.synchronize()
    assert torch.cuda.current_device() == 0

    failing_args = list(setup.stage._extension_args)
    failing_args[0] = torch.empty(
        (setup.factory.primitive_count, 3),
        dtype=torch.float32,
        device="cuda:0",
    )
    torch.cuda.set_device(0)
    with pytest.raises(RuntimeError, match="anchor device"):
        setup.factory.module.cuda_lbvh_rebuild_out(*failing_args)
    assert torch.cuda.current_device() == 0
    assert torch.cuda.current_stream(torch.device("cuda:1")) == target_previous_stream


def test_cuda_scene_lbvh_and_render_compose_on_one_transaction():
    packed = build_canonical_packed_scene("synthetic_dynamic_rigid_960x640_v1")
    replay = build_canonical_transform_replay("synthetic_dynamic_rigid_960x640_v1")
    source = CanonicalNativeInput.from_scene_contract(
        packed,
        input_contract=CANONICAL_NATIVE_REPLAY_INPUT,
        replay=replay,
    )
    prepared = prepare_cuda_native_input(source, device="cuda:0")
    context = _context("cuda:0")
    scene_factory = build_prepared_cuda_scene_update_factory(prepared, context=context)
    transforms = build_prepared_cuda_transform_inputs(prepared, 1)
    world = allocate_cuda_world_scene_buffers(scene_factory.local_scene)
    scene_stage = scene_factory.bind(transforms=transforms, outputs=world)
    lbvh_factory = build_prepared_cuda_lbvh_rebuild_factory(prepared, context=context)
    lbvh_outputs = allocate_cuda_lbvh_buffers(
        primitive_count=lbvh_factory.primitive_count,
        device=context.device,
    )
    lbvh_workspace = allocate_cuda_lbvh_workspace(lbvh_factory.requirements)
    lbvh_stage = lbvh_factory.bind(
        world=world,
        outputs=lbvh_outputs,
        workspace=lbvh_workspace,
    )
    scene = scene_factory.local_scene
    first_hit = CudaFirstHitInputs(
        plane_normal_world=world.plane_normal_world,
        plane_point_world=world.plane_point_world,
        plane_numeric_instance_id=scene.arrays["plane_numeric_instance_id"],
        plane_source_order_key=scene.arrays["plane_source_order_key"],
        plane_role_mask=scene.arrays["plane_role_mask"],
        plane_material_index=scene.arrays["plane_material_index"],
        num_planes=scene.num_planes,
        triangle_v0_world=world.triangle_v0_world,
        triangle_e1_world=world.triangle_e1_world,
        triangle_e2_world=world.triangle_e2_world,
        triangle_normal_world=world.triangle_normal_world,
        triangle_numeric_instance_id=scene.arrays["triangle_numeric_instance_id"],
        triangle_source_order_key=scene.arrays["triangle_source_order_key"],
        triangle_role_mask=scene.arrays["triangle_role_mask"],
        triangle_material_index=scene.arrays["triangle_material_index"],
        bvh_bounds_min=lbvh_outputs.node_bounds_min,
        bvh_bounds_max=lbvh_outputs.node_bounds_max,
        bvh_left=lbvh_outputs.left,
        bvh_right=lbvh_outputs.right,
        bvh_start=lbvh_outputs.start,
        bvh_count=lbvh_outputs.count,
        bvh_prim_ids=lbvh_outputs.sorted_primitive_ids,
        num_bvh_nodes=lbvh_outputs.node_count,
        sensor_role_mask=_all_role_mask(scene),
        max_distance=10.0,
    )
    shade = CudaShadeInputs(
        material_albedo_rgb=scene.arrays["material_albedo_rgb"],
        light_kind=scene.arrays["light_kind"],
        light_position_or_direction_world=scene.arrays["light_position_or_direction_world"],
        light_intensity=scene.arrays["light_intensity"],
        light_color_rgb=scene.arrays["light_color_rgb"],
        num_lights=scene.num_lights,
    )
    render_buffers = allocate_cuda_direct_light_buffers(
        num_rays=2,
        device=context.device,
        origins=torch.tensor(
            [[0.05, 0.05, 2.0], [2.0, 2.0, 2.0]],
            dtype=torch.float32,
            device=context.device,
        ),
        directions=torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=context.device,
        ),
        direct_light=True,
        traversal_diagnostics=False,
    )
    render_stage = CudaDirectLightNativeRenderer.prepare(context).bind_direct_light(
        first_hit_inputs=first_hit,
        shade_inputs=shade,
        buffers=render_buffers,
        output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL,
        shadows=False,
        ambient_rgb=(0.1, 0.1, 0.1),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
    )
    transaction = context.begin_transaction(
        stages=(scene_stage, lbvh_stage, render_stage),
        completion_event=_initialized_event(context.stream),
    )
    scene_stage.enqueue(transaction)
    lbvh_stage.enqueue(transaction)
    render_stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()
    assert completion.stage_names == (
        "cuda_scene_update",
        "cuda_lbvh_rebuild",
        "cuda_direct_light_render",
    )
    assert render_buffers.hit_mask.cpu().tolist() == [1, 0]


def _raw_setup(aabb_min, aabb_max, *, device: str = "cuda:0"):
    context = _context(device)
    target = context.device
    minimum = torch.as_tensor(aabb_min, dtype=torch.float32, device=target).reshape(-1, 3)
    maximum = torch.as_tensor(aabb_max, dtype=torch.float32, device=target).reshape(-1, 3)
    primitive_count = int(minimum.shape[0])
    triangle = torch.empty((primitive_count, 3), dtype=torch.float32, device=target)
    plane = torch.empty((0, 3), dtype=torch.float32, device=target)
    world = CudaWorldSceneBuffers(
        device_ordinal=int(target.index),
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
    factory = CudaLbvhRebuildStageFactory.prepare(
        context=context,
        primitive_count=primitive_count,
        native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64,
        input_contract="unit_native_replay_v1",
    )
    outputs = allocate_cuda_lbvh_buffers(primitive_count=primitive_count, device=target)
    workspace = allocate_cuda_lbvh_workspace(factory.requirements)
    stage = factory.bind(world=world, outputs=outputs, workspace=workspace)
    return SimpleNamespace(
        context=context,
        world=world,
        factory=factory,
        outputs=outputs,
        workspace=workspace,
        stage=stage,
    )


def _run(setup):
    transaction = setup.context.begin_transaction(
        stages=(setup.stage,),
        completion_event=_initialized_event(setup.context.stream),
    )
    setup.stage.enqueue(transaction)
    return transaction.finish()


def _context(device: str):
    target = torch.device(device)
    with torch.cuda.device(target):
        stream = torch.cuda.Stream(device=target)
    return CudaOpticalExecutionContext(device=target, stream=stream)


def _initialized_event(stream):
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
    event.synchronize()
    return event


def _expand_bits_10(value: int) -> int:
    value = (value * 0x00010001) & 0xFF0000FF
    value = (value * 0x00000101) & 0x0F00F00F
    value = (value * 0x00000011) & 0xC30C30C3
    return (value * 0x00000005) & 0x49249249


def _morton_reference(minimum: np.ndarray, maximum: np.ndarray):
    scene_min = minimum.min(axis=0)
    scene_max = maximum.max(axis=0)
    extent = np.maximum(scene_max - scene_min, np.float32(1.0e-12))
    keys = []
    for primitive, center in enumerate(np.float32(0.5) * (minimum + maximum)):
        normalized = np.clip((center - scene_min) / extent, 0.0, 0.999999)
        xyz = [int(float(value) * 1024.0) for value in normalized]
        morton = (_expand_bits_10(xyz[0]) << 2) | (_expand_bits_10(xyz[1]) << 1) | _expand_bits_10(xyz[2])
        keys.append((morton << 32) | primitive)
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return [keys[index] for index in order], order


def _assert_topology_and_bounds(outputs, minimum, maximum, sorted_ids):
    primitive_count = len(sorted_ids)
    node_count = primitive_count * 2 - 1
    leaf_base = primitive_count - 1
    left = outputs.left.cpu().numpy()
    right = outputs.right.cpu().numpy()
    parent = outputs.parent.cpu().numpy()
    start = outputs.start.cpu().numpy()
    count = outputs.count.cpu().numpy()
    bounds_min = outputs.node_bounds_min.cpu().numpy()
    bounds_max = outputs.node_bounds_max.cpu().numpy()
    assert parent[0] == -1
    assert sorted(np.flatnonzero(parent == -1).tolist()) == [0]
    visited = set()

    def visit(node):
        assert 0 <= node < node_count
        assert node not in visited
        visited.add(node)
        if node >= leaf_base:
            rank = node - leaf_base
            assert left[node] == right[node] == -1
            assert start[node] == rank
            assert count[node] == 1
            primitive = sorted_ids[rank]
            np.testing.assert_array_equal(bounds_min[node], minimum[primitive])
            np.testing.assert_array_equal(bounds_max[node], maximum[primitive])
            return [primitive]
        assert 0 <= left[node] < node_count
        assert 0 <= right[node] < node_count
        assert parent[left[node]] == node
        assert parent[right[node]] == node
        descendants = visit(int(left[node])) + visit(int(right[node]))
        np.testing.assert_array_equal(bounds_min[node], minimum[descendants].min(axis=0))
        np.testing.assert_array_equal(bounds_max[node], maximum[descendants].max(axis=0))
        assert start[node] == -1
        assert count[node] == 0
        return descendants

    assert sorted(visit(0)) == list(range(primitive_count))
    assert visited == set(range(node_count))


def _all_role_mask(scene) -> int:
    mask = 0
    for value in scene.role_to_mask.values():
        mask |= int(value)
    return mask
