from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

import optics.cuda_direct_light as cuda_direct_light
import optics.cuda_timeline as cuda_timeline
from optics.cuda_timeline import (
    CudaDeferredProfile,
    CudaOpticalCompletion,
    CudaOpticalExecutionContext,
    CudaOpticalTimelineError,
    CudaOpticalTransactionPoisonedError,
    CudaOpticalTransactionTemplate,
)


def test_cuda_timeline_contract_imports_without_runtime_initialization():
    assert issubclass(CudaOpticalTransactionPoisonedError, CudaOpticalTimelineError)
    assert CudaOpticalCompletion.__dataclass_fields__["recorded"].default is True
    assert CudaOpticalTransactionTemplate.__doc__


def test_cuda_timeline_requires_torch_only_when_context_is_constructed(monkeypatch):
    monkeypatch.setattr(cuda_timeline, "torch", None)

    with pytest.raises(ImportError, match="PyTorch"):
        CudaOpticalExecutionContext(device="cuda:0", stream=object())


def test_deferred_profile_resolution_is_explicit():
    class _Event:
        def __init__(self, value: float) -> None:
            self.value = value

        def elapsed_time(self, end: "_Event") -> float:
            return end.value - self.value

    profile = CudaDeferredProfile((("raygen_kernel", _Event(1.25), _Event(3.75)),))

    assert profile.resolve_ms() == (("raygen_kernel", 2.5),)


@pytest.fixture
def scope_transaction(monkeypatch):
    """CPU native-scope model: save the ambient state at entry, as Torch does."""
    @dataclass(frozen=True)
    class Device:
        index: int
        type: str = "cuda"

    target = Device(1)
    stream = NS(device=target, cuda_stream=17)
    state = NS(device=Device(0), stream=object(), calls=[], fault=None)
    failure = RuntimeError("original scope failure")

    @contextmanager
    def scope(owner, kind):
        previous = getattr(state, kind)
        state.calls.append(kind + "_enter")
        setattr(state, kind, owner)
        try:
            if state.fault == kind + "_enter":
                raise failure
            yield
        finally:
            setattr(state, kind, previous)
            state.calls.append(kind + "_exit")
            if state.fault == kind + "_exit":
                raise failure

    class Event:
        device = target

        def record(self, actual):
            assert actual is stream and state.stream is stream and state.device == target
            state.calls.append("record")
            if state.fault == "record":
                raise failure

    monkeypatch.setattr(cuda_timeline, "torch", NS(device=lambda value: value,
        cuda=NS(is_available=lambda: True, Event=Event,
                device=lambda value: scope(value, "device"), stream=lambda value: scope(value, "stream"))))
    context = CudaOpticalExecutionContext(device=target, stream=stream)

    class Stage:
        cuda_transaction_stage_name = "scope_stage"

        def _validate_cuda_transaction_stage(self, actual):
            assert actual is context

        def _assert_cuda_transaction_stage_stable(self, actual):
            assert actual is context

    stage = Stage()
    transaction = context.begin_transaction(stages=(stage,), completion_event=Event(),
        diagnostic_events={"phase": (Event(), Event())})
    assert not state.calls
    return NS(context=context, stage=stage, transaction=transaction, state=state,
              target=target, stream=stream, failure=failure, Device=Device)


@pytest.mark.parametrize(
    "fault", [None, "device_enter", "stream_enter", "stream_exit", "device_exit", "leaf"],
)
def test_prepared_stage_scope_restores_entry_time_ambient_and_preserves_failure(
    scope_transaction, fault,
):
    x = scope_transaction
    # Preparation used device 0; execution must restore the new device/stream.
    ambient_device, ambient_stream = x.Device(2), object()
    x.state.device, x.state.stream, x.state.fault = ambient_device, ambient_stream, fault

    def run():
        with x.transaction.enqueue_stage(x.stage):
            assert x.state.device == x.target and x.state.stream is x.stream
            if fault == "leaf":
                raise x.failure

    if fault is None:
        run()
        assert x.transaction._next_stage == 1
    else:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is x.failure
        assert x.transaction.state == "poisoned" and x.transaction._next_stage == 0
    assert x.transaction._active_stage is None
    assert x.state.device == ambient_device and x.state.stream is ambient_stream
    assert x.state.calls.count("device_enter") == x.state.calls.count("device_exit") == 1
    assert x.state.calls.count("stream_enter") == x.state.calls.count("stream_exit")
    assert x.state.calls.count("stream_enter") == (fault != "device_enter")


@pytest.mark.parametrize("failure", [False, True])
def test_prepared_completion_publishes_only_after_original_event_record(
    scope_transaction, monkeypatch, failure,
):
    x = scope_transaction
    transaction = x.transaction
    completion = transaction._prepared_completion
    profile = completion.deferred_profile
    with pytest.raises(CudaOpticalTimelineError, match="unavailable"):
        transaction.completion
    assert not completion.recorded

    def unexpected_creation(*args, **kwargs):
        pytest.fail("timed path created a scope/completion/profile")

    monkeypatch.setattr(x.context, "activate", unexpected_creation)
    monkeypatch.setattr(CudaOpticalCompletion, "__init__", unexpected_creation)
    monkeypatch.setattr(CudaDeferredProfile, "__init__", unexpected_creation)
    with transaction.enqueue_stage(x.stage):
        transaction.record_diagnostic_start("phase")
        transaction.before_native_launch()
        transaction.record_diagnostic_end("phase")
    if failure:
        x.state.fault = "record"
        with pytest.raises(RuntimeError) as caught:
            transaction.finish()
        assert caught.value is x.failure
        assert not completion.recorded and transaction.state == "poisoned"
        with pytest.raises(CudaOpticalTimelineError, match="unavailable"):
            transaction.completion
    else:
        assert transaction.finish() is completion and completion.recorded
        assert transaction.completion.deferred_profile is profile
    prior = tuple(x.state.calls)
    with pytest.raises(CudaOpticalTimelineError):
        transaction.finish()
    assert tuple(x.state.calls) == prior


@pytest.mark.parametrize(
    "replacement", ["equal_device", "foreign_device", "equal_handle_stream", "new_handle"],
)
def test_prepared_scope_binds_original_stream_and_indexed_device_value(scope_transaction, replacement):
    x = scope_transaction
    if replacement == "equal_device":
        x.context.device = x.Device(1)
        assert x.context.device is not x.target
    elif replacement == "foreign_device":
        x.context.device = x.Device(2)
        x.context.stream = NS(device=x.context.device, cuda_stream=x.stream.cuda_stream)
    elif replacement == "equal_handle_stream":
        x.context.stream = NS(device=x.target, cuda_stream=x.stream.cuda_stream)
    else:
        x.context.stream_handle = 23
        x.stream.cuda_stream = 23
    if replacement == "equal_device":
        with x.transaction.enqueue_stage(x.stage):
            assert x.state.stream is x.stream and x.state.device == x.target
    else:
        with pytest.raises(CudaOpticalTimelineError, match="scope binding changed"):
            with x.transaction.enqueue_stage(x.stage):
                pytest.fail("rebound prepared scope entered")
        assert not x.state.calls and x.transaction.state == "poisoned"


def test_execution_validates_device_value_without_reconstructing_it(scope_transaction, monkeypatch):
    x = scope_transaction

    def late_device(*args):
        pytest.fail("execution reconstructed a device value")

    monkeypatch.setattr(cuda_timeline.torch, "device", late_device)
    with x.transaction.enqueue_stage(x.stage):
        x.transaction.record_diagnostic_start("phase")
        x.transaction.record_diagnostic_end("phase")
    assert x.transaction.finish().recorded


@pytest.mark.parametrize("handle", [17.0, "17", True, None])
def test_stream_handle_metadata_cannot_change_type(scope_transaction, handle):
    x = scope_transaction
    x.stream.cuda_stream = handle
    with pytest.raises(CudaOpticalTimelineError, match="handle identity changed"):
        with x.transaction.enqueue_stage(x.stage):
            pytest.fail("invalid stream handle reached execution")
    assert not x.state.calls


def test_cuda_shade_extension_abi_uses_fixed_diagnostic_arguments():
    source = cuda_direct_light._CPP_SOURCE
    cuda_source = cuda_direct_light._CUDA_SOURCE

    assert "std::vector<torch::Tensor>" not in source
    assert "std::vector<torch::Tensor>" not in cuda_source
    assert "#include <vector>" not in source
    assert "#include <vector>" not in cuda_source
    assert ": shadow_stack_overflow_count.data_ptr<int32_t>()" not in cuda_source

    for base_name in (
        "shade_direct_light_no_shadow_out",
        "shade_direct_light_with_shadows_out",
    ):
        start = source.index(f"void {base_name}(")
        signature = source[start : source.index(");", start)]
        assert "traversal_counter_" not in signature

    for diagnostics_name in (
        "shade_direct_light_no_shadow_diagnostics_out",
        "shade_direct_light_with_shadows_diagnostics_out",
    ):
        start = source.index(f"void {diagnostics_name}(")
        signature = source[start : source.index(");", start)]
        counter_positions = [signature.index(f"traversal_counter_{index}") for index in range(9)]
        assert counter_positions == sorted(counter_positions)


def test_prepared_first_hit_arguments_follow_the_declared_cpp_abi():
    import re

    source = cuda_direct_light._CPP_SOURCE
    start = source.index("void first_hit_rays_out(")
    names = re.findall(r"(?:torch::Tensor|double|int64_t)\s+(\w+)", source[start:source.index(");", start)])
    assert len(names) == 35
    arguments = {name: object() for name in names}
    output_names = {
        "origins", "directions", "hit_mask", "range_m", "position_world", "normal_world",
        "numeric_instance_id", "material_index", "bvh_stack_overflow_count", "bvh_max_stack_depth",
    }
    aliases = {
        "plane_normals": "plane_normal_world", "plane_points": "plane_point_world",
        "plane_numeric_ids": "plane_numeric_instance_id", "plane_source_keys": "plane_source_order_key",
        "plane_role_masks": "plane_role_mask", "plane_material_indices": "plane_material_index",
        "triangle_v0": "triangle_v0_world", "triangle_e1": "triangle_e1_world",
        "triangle_e2": "triangle_e2_world",
        "triangle_normal": "triangle_normal_world", "triangle_numeric_ids": "triangle_numeric_instance_id",
        "triangle_source_keys": "triangle_source_order_key", "triangle_role_masks": "triangle_role_mask",
        "triangle_material_indices": "triangle_material_index",
    }
    stage = object.__new__(cuda_direct_light.CudaDirectLightRenderStage)
    stage.buffers = NS(**{name: arguments[name] for name in output_names})
    stage.first_hit_inputs = NS(**{
        aliases.get(name, name): value for name, value in arguments.items() if name not in output_names
    })
    actual = stage._bind_first_hit_arguments()
    assert len(actual) == len(names)
    assert all(value is arguments[name] for name, value in zip(names, actual, strict=True))


@pytest.mark.parametrize("triangles,planes", [(False, False), (True, False), (False, True), (True, True)])
def test_scene_launch_latches_follow_actual_entries_not_diagnostic_phases(triangles, planes):
    from optics.cuda_scene import CudaSceneUpdateStage

    calls = []
    stage = object.__new__(CudaSceneUpdateStage)
    for name, present in (("triangle", triangles), ("plane", planes)):
        object.__setattr__(stage, "_" + name + "_entry",
                           (lambda *args, name=name: calls.append(name)) if present else None)
        object.__setattr__(stage, "_" + name + "_args", ())

    class Transaction:
        @contextmanager
        def enqueue_stage(self, actual):
            assert actual is stage
            yield

        def record_diagnostic_start(self, name):
            calls.append("start")

        def record_diagnostic_end(self, name):
            calls.append("end")

        def before_native_launch(self):
            calls.append("launch")

    stage.enqueue(Transaction())
    assert calls == ["start", *(["launch", "triangle"] if triangles else []),
                     *(["launch", "plane"] if planes else []), "end"]


def test_native_launch_outside_active_transaction_stage_rejects():
    transaction = object.__new__(cuda_timeline.CudaOpticalTransaction)
    transaction._state = "validated_open"
    transaction._active_stage = None
    transaction._poison_reason = None
    with pytest.raises(CudaOpticalTransactionPoisonedError, match="active CUDA stage"):
        transaction.before_native_launch()
    assert transaction.state == "poisoned"


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_lbvh_launch_latch_requires_an_actual_native_entry(populated, diagnostic):
    from optics.cuda_lbvh import CudaLbvhRebuildStage

    calls = []
    stage = object.__new__(CudaLbvhRebuildStage)
    object.__setattr__(stage, "_extension_entry", (lambda: calls.append("native")) if populated else None)
    object.__setattr__(stage, "_extension_args", ())

    class Transaction:
        @contextmanager
        def enqueue_stage(self, actual):
            assert actual is stage
            yield

        def record_diagnostic_start(self, name):
            if diagnostic:
                calls.append("start")

        def record_diagnostic_end(self, name):
            if diagnostic:
                calls.append("end")

        def before_native_launch(self):
            calls.append("launch")

    stage.enqueue(Transaction())
    assert calls == (["start"] if diagnostic else []) + (["launch", "native"] if populated else []) + (
        ["end"] if diagnostic else [])
