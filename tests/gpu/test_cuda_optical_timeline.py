from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from optics.cuda_direct_light import (
    CudaCameraParameters,
    CudaDirectLightNativeRenderer,
    CudaFirstHitInputs,
    CudaShadeInputs,
    allocate_cuda_direct_light_buffers,
)
from optics.cuda_timeline import (
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransactionPoisonedError,
)
from optics.execution import OpticalOutputProfile

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.cuda_ext,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Torch CUDA is unavailable"),
]


def _initialized_event(stream, *, enable_timing: bool = False):
    with torch.cuda.device(stream.device), torch.cuda.stream(stream):
        event = torch.cuda.Event(enable_timing=enable_timing)
        event.record(stream)
    event.synchronize()
    return event


class _TensorStage:
    def __init__(self, *, name: str, context, output, delta: float, fail: bool = False) -> None:
        self.cuda_transaction_stage_name = name
        self.context = context
        self.output = output
        self.delta = delta
        self.fail = fail
        self.signature = (id(output), output.data_ptr(), tuple(output.shape), output.device)

    def _validate_cuda_transaction_stage(self, context) -> None:
        if context is not self.context or self.output.device != context.device:
            raise ValueError("stage context mismatch")

    def _assert_cuda_transaction_stage_stable(self, context) -> None:
        self._validate_cuda_transaction_stage(context)
        assert self.signature == (
            id(self.output),
            self.output.data_ptr(),
            tuple(self.output.shape),
            self.output.device,
        )

    def enqueue(self, transaction) -> None:
        with transaction.enqueue_stage(self):
            self.output.add_(self.delta)
            if self.fail:
                raise RuntimeError("injected stage failure")


def _context(device: str = "cuda:0"):
    resolved = torch.device(device)
    with torch.cuda.device(resolved):
        stream = torch.cuda.Stream(device=resolved)
    return CudaOpticalExecutionContext(device=resolved, stream=stream)


def _empty_first_hit_inputs(device) -> CudaFirstHitInputs:
    def f3():
        return torch.empty((0, 3), dtype=torch.float32, device=device)

    def i32():
        return torch.empty((0,), dtype=torch.int32, device=device)

    def i64():
        return torch.empty((0,), dtype=torch.int64, device=device)

    return CudaFirstHitInputs(
        plane_normal_world=f3(),
        plane_point_world=f3(),
        plane_numeric_instance_id=i32(),
        plane_source_order_key=i64(),
        plane_role_mask=i64(),
        plane_material_index=i32(),
        num_planes=0,
        triangle_v0_world=f3(),
        triangle_e1_world=f3(),
        triangle_e2_world=f3(),
        triangle_normal_world=f3(),
        triangle_numeric_instance_id=i32(),
        triangle_source_order_key=i64(),
        triangle_role_mask=i64(),
        triangle_material_index=i32(),
        bvh_bounds_min=f3(),
        bvh_bounds_max=f3(),
        bvh_left=i32(),
        bvh_right=i32(),
        bvh_start=i32(),
        bvh_count=i32(),
        bvh_prim_ids=i32(),
        num_bvh_nodes=0,
        sensor_role_mask=1,
        max_distance=10.0,
    )


def _empty_shade_inputs(device) -> CudaShadeInputs:
    return CudaShadeInputs(
        material_albedo_rgb=torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32, device=device),
        light_kind=torch.empty((0,), dtype=torch.int32, device=device),
        light_position_or_direction_world=torch.empty((0, 3), dtype=torch.float32, device=device),
        light_intensity=torch.empty((0,), dtype=torch.float32, device=device),
        light_color_rgb=torch.empty((0, 3), dtype=torch.float32, device=device),
        num_lights=0,
    )


def test_transaction_orders_stages_and_finishes_once_on_explicit_stream():
    context = _context()
    output = torch.zeros((1,), dtype=torch.float32, device=context.device)
    first = _TensorStage(name="scene", context=context, output=output, delta=2.0)
    second = _TensorStage(name="render", context=context, output=output, delta=3.0)
    terminal = _initialized_event(context.stream)
    transaction = context.begin_transaction(
        stages=(first, second),
        completion_event=terminal,
    )

    with pytest.raises(CudaOpticalTimelineError, match="before successful finish"):
        _ = transaction.completion
    first.enqueue(transaction)
    second.enqueue(transaction)
    completion = transaction.finish()

    consumer = torch.cuda.Stream(device=context.device)
    observed = torch.empty_like(output)
    consumer.wait_event(completion.event)
    with torch.cuda.stream(consumer):
        observed.copy_(output)
    consumer.synchronize()
    assert observed.item() == pytest.approx(5.0)
    assert completion.stage_names == ("scene", "render")
    assert transaction.state == "finished"
    with pytest.raises(CudaOpticalTimelineError, match="already finished"):
        transaction.finish()
    with pytest.raises(CudaOpticalTimelineError, match="cannot enqueue"):
        first.enqueue(transaction)


def test_transaction_template_reuses_validation_but_not_generation_state():
    context = _context()
    output = torch.zeros((1,), dtype=torch.float32, device=context.device)
    stage = _TensorStage(name="render", context=context, output=output, delta=1.0)
    terminal = _initialized_event(context.stream)
    template = context.prepare_transaction_template(
        stages=(stage,),
        completion_event=terminal,
    )

    first = template.activate()
    second = template.activate()

    assert first is not second
    assert first._completion_event is terminal
    assert second._completion_event is terminal
    assert first.state == second.state == "validated_open"
    with pytest.raises(AttributeError, match="immutable"):
        template._completion_event = _initialized_event(context.stream)
    object.__setattr__(template, "_stages", ())
    with pytest.raises(CudaOpticalTimelineError, match="binding changed"):
        template.activate()


def test_transaction_protocol_errors_and_stage_failure_poison_slots():
    context = _context()
    output = torch.zeros((1,), dtype=torch.float32, device=context.device)
    first = _TensorStage(name="first", context=context, output=output, delta=1.0)
    second = _TensorStage(name="second", context=context, output=output, delta=1.0)

    with pytest.raises(ValueError, match="initialized"):
        context.begin_transaction(stages=(first,), completion_event=torch.cuda.Event())
    if torch.cuda.device_count() >= 2:
        other_context = _context("cuda:1")
        with pytest.raises(ValueError, match="device must match"):
            context.begin_transaction(
                stages=(first,),
                completion_event=_initialized_event(other_context.stream),
            )
    duplicate_name = _TensorStage(
        name="first",
        context=context,
        output=output,
        delta=1.0,
    )
    with pytest.raises(ValueError, match="stage names must be unique"):
        context.begin_transaction(
            stages=(first, duplicate_name),
            completion_event=_initialized_event(context.stream),
        )

    out_of_order = context.begin_transaction(
        stages=(first, second),
        completion_event=_initialized_event(context.stream),
    )
    with pytest.raises(CudaOpticalTransactionPoisonedError, match="out of order"):
        second.enqueue(out_of_order)
    assert out_of_order.state == "poisoned"
    with pytest.raises(CudaOpticalTransactionPoisonedError):
        out_of_order.finish()

    early_finish = context.begin_transaction(
        stages=(first,),
        completion_event=_initialized_event(context.stream),
    )
    with pytest.raises(CudaOpticalTransactionPoisonedError, match="before all"):
        early_finish.finish()
    assert early_finish.state == "poisoned"

    failing = _TensorStage(
        name="failing",
        context=context,
        output=output,
        delta=4.0,
        fail=True,
    )
    failed = context.begin_transaction(
        stages=(failing,),
        completion_event=_initialized_event(context.stream),
    )
    with pytest.raises(RuntimeError, match="injected"):
        failing.enqueue(failed)
    assert failed.state == "poisoned"
    with pytest.raises(CudaOpticalTransactionPoisonedError):
        failed.finish()
    torch.cuda.synchronize(context.device)


def test_transaction_restores_ambient_device_and_current_stream():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    context = _context("cuda:1")
    output = torch.zeros((1,), dtype=torch.float32, device=context.device)
    stage = _TensorStage(name="render", context=context, output=output, delta=1.0)
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )

    torch.cuda.set_device(0)
    ambient_stream = torch.cuda.Stream(device="cuda:0")
    with torch.cuda.stream(ambient_stream):
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream("cuda:0") == ambient_stream
        stage.enqueue(transaction)
        completion = transaction.finish()
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream("cuda:0") == ambient_stream
    completion.event.synchronize()
    assert torch.cuda.current_device() == 0


def test_native_binding_rejects_dtype_overlay_overlap_and_post_bind_resize():
    context = _context()
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    first_hit = _empty_first_hit_inputs(context.device)
    shade = _empty_shade_inputs(context.device)
    buffers = allocate_cuda_direct_light_buffers(
        num_rays=4,
        device=context.device,
        direct_light=True,
        traversal_diagnostics=True,
    )
    bind_kwargs = {
        "first_hit_inputs": first_hit,
        "shade_inputs": shade,
        "camera": None,
        "output_profile": OpticalOutputProfile.DIRECT_LIGHT_FULL,
        "shadows": False,
        "ambient_rgb": (0.0, 0.0, 0.0),
        "background_rgb": (0.0, 0.0, 0.0),
        "shadow_bias": 1.0e-6,
    }

    with pytest.raises(ValueError, match="direct-light output profile"):
        renderer.bind_direct_light(
            buffers=buffers,
            **{**bind_kwargs, "output_profile": OpticalOutputProfile.GEOMETRY_FULL},
        )
    with pytest.raises(ValueError, match="missing first_hit_rays_out"):
        CudaDirectLightNativeRenderer(context=context, module=object()).bind_direct_light(
            buffers=buffers,
            **bind_kwargs,
        )

    wrong_dtype = replace(
        buffers,
        hit_mask=torch.empty((4,), dtype=torch.float32, device=context.device),
    )
    with pytest.raises(ValueError, match="hit_mask has the wrong dtype"):
        renderer.bind_direct_light(buffers=wrong_dtype, **bind_kwargs)

    partial_overlay = replace(
        buffers,
        shadow_traversal_counters=buffers.shadow_traversal_counters[:1],
    )
    with pytest.raises(ValueError, match="completely absent or present"):
        renderer.bind_direct_light(buffers=partial_overlay, **bind_kwargs)

    overlapping = replace(buffers, range_m=buffers.intensity)
    with pytest.raises(ValueError, match="storage overlap"):
        renderer.bind_direct_light(buffers=overlapping, **bind_kwargs)

    stage = renderer.bind_direct_light(buffers=buffers, **bind_kwargs)
    with pytest.raises(AttributeError, match="immutable"):
        stage.shadows = True
    object.__setattr__(stage, "shadows", True)
    with pytest.raises(ValueError, match="extension entry identity changed"):
        context.begin_transaction(
            stages=(stage,),
            completion_event=_initialized_event(context.stream),
        )
    object.__setattr__(stage, "shadows", False)
    mutation_transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )
    object.__setattr__(stage, "output_profile", OpticalOutputProfile.RGB_PREVIEW)
    with pytest.raises(ValueError, match="binding changed"):
        stage.enqueue(mutation_transaction)
    assert mutation_transaction.state == "poisoned"
    object.__setattr__(stage, "output_profile", OpticalOutputProfile.DIRECT_LIGHT_FULL)
    original_pointer = buffers.range_m.data_ptr()
    buffers.range_m.resize_(1_000_000)
    buffers.range_m.resize_(4)
    assert buffers.range_m.data_ptr() != original_pointer
    with pytest.raises(ValueError, match="binding changed"):
        context.begin_transaction(
            stages=(stage,),
            completion_event=_initialized_event(context.stream),
        )


def test_empty_first_hit_still_resets_primary_counters():
    context = _context()
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    buffers = allocate_cuda_direct_light_buffers(
        num_rays=0,
        device=context.device,
        direct_light=False,
        traversal_diagnostics=False,
    )
    with context.activate():
        buffers.bvh_stack_overflow_count.fill_(91)
        buffers.bvh_max_stack_depth.fill_(92)
    stage = renderer.bind_first_hit(
        first_hit_inputs=_empty_first_hit_inputs(context.device),
        buffers=buffers,
    )
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )
    stage.enqueue(transaction)
    completion = transaction.finish()
    completion.event.synchronize()

    np.testing.assert_array_equal(buffers.bvh_stack_overflow_count.cpu().numpy(), [0])
    np.testing.assert_array_equal(buffers.bvh_max_stack_depth.cpu().numpy(), [0])


@pytest.mark.parametrize("shadows", [False, True])
def test_native_render_enqueue_is_out_only_and_returns_no_early_completion(shadows):
    context = _context()
    first_hit = _empty_first_hit_inputs(context.device)
    shade = _empty_shade_inputs(context.device)
    camera = CudaCameraParameters(
        width=5,
        height=3,
        fx=4.25,
        fy=2.75,
        cx=1.3,
        cy=0.7,
        origin=(0.25, -0.5, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    buffers = allocate_cuda_direct_light_buffers(
        num_rays=camera.width * camera.height,
        device=context.device,
        direct_light=True,
        traversal_diagnostics=True,
    )
    for tensor in buffers.resource_tuple():
        tensor.fill_(71)
    torch.cuda.synchronize(context.device)
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    stage = renderer.bind_direct_light(
        first_hit_inputs=first_hit,
        shade_inputs=shade,
        buffers=buffers,
        camera=camera,
        output_profile=OpticalOutputProfile.DIRECT_LIGHT_FULL,
        shadows=shadows,
        ambient_rgb=(0.0, 0.0, 0.0),
        background_rgb=(0.1, 0.2, 0.3),
        shadow_bias=1.0e-6,
    )
    assert len(stage._shade_args) == (44 if shadows else 29)
    assert all(
        stage._shade_args[-9 + index] is buffers.shadow_traversal_counters[index] for index in range(9)
    )
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
    )
    stage.enqueue(transaction)
    with pytest.raises(CudaOpticalTimelineError, match="before successful finish"):
        _ = transaction.completion
    completion = transaction.finish()
    completion.event.synchronize()

    np.testing.assert_array_equal(buffers.hit_mask.cpu().numpy(), 0)
    assert np.isinf(buffers.range_m.cpu().numpy()).all()
    assert np.isnan(buffers.position_world.cpu().numpy()).all()
    assert np.isnan(buffers.normal_world.cpu().numpy()).all()
    np.testing.assert_array_equal(buffers.numeric_instance_id.cpu().numpy(), 0)
    np.testing.assert_array_equal(buffers.material_index.cpu().numpy(), 0)
    np.testing.assert_allclose(
        buffers.rgb.cpu().numpy(),
        np.broadcast_to(np.array([0.1, 0.2, 0.3], dtype=np.float32), (15, 3)),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(buffers.intensity.cpu().numpy(), 0.0)
    for counter in (
        buffers.bvh_stack_overflow_count,
        buffers.bvh_max_stack_depth,
        buffers.shadow_stack_overflow_count,
        buffers.shadow_max_stack_depth,
        *buffers.shadow_traversal_counters,
    ):
        np.testing.assert_array_equal(counter.cpu().numpy(), [0])
    assert len({tensor.data_ptr() for tensor in buffers.shadow_traversal_counters}) == 9
    evidence = completion.stage_evidence[0]
    assert evidence.output_profile == OpticalOutputProfile.DIRECT_LIGHT_FULL.value
    assert evidence.shadows is shadows
    assert evidence.ray_source == "camera"
    assert evidence.extension_entry == (
        "shade_direct_light_with_shadows_diagnostics_out"
        if shadows
        else "shade_direct_light_no_shadow_diagnostics_out"
    )


def test_native_render_diagnostic_events_are_deferred_until_terminal_wait():
    context = _context()
    renderer = CudaDirectLightNativeRenderer.prepare(context)
    camera = CudaCameraParameters(
        width=2,
        height=1,
        fx=2.0,
        fy=2.5,
        cx=0.3,
        cy=0.4,
        origin=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    buffers = allocate_cuda_direct_light_buffers(
        num_rays=2,
        device=context.device,
        direct_light=True,
        traversal_diagnostics=False,
    )
    stage = renderer.bind_direct_light(
        first_hit_inputs=_empty_first_hit_inputs(context.device),
        shade_inputs=_empty_shade_inputs(context.device),
        buffers=buffers,
        camera=camera,
        output_profile=OpticalOutputProfile.RGB_PREVIEW,
        shadows=False,
        ambient_rgb=(0.0, 0.0, 0.0),
        background_rgb=(0.0, 0.0, 0.0),
        shadow_bias=1.0e-6,
    )
    assert len(stage._shade_args) == 20
    diagnostics = {
        name: (
            _initialized_event(context.stream, enable_timing=True),
            _initialized_event(context.stream, enable_timing=True),
        )
        for name in ("raygen_kernel", "first_hit_kernel", "shade_kernel")
    }
    transaction = context.begin_transaction(
        stages=(stage,),
        completion_event=_initialized_event(context.stream),
        diagnostic_events=diagnostics,
    )

    stage.enqueue(transaction)
    completion = transaction.finish()
    assert completion.deferred_profile is not None
    assert completion.stage_evidence[0].extension_entry == "shade_direct_light_no_shadow_out"
    completion.event.synchronize()
    resolved = dict(completion.deferred_profile.resolve_ms())

    assert tuple(resolved) == ("raygen_kernel", "first_hit_kernel", "shade_kernel")
    assert all(value >= 0.0 for value in resolved.values())
