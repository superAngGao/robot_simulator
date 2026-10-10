"""Register-before-copy a2 preparation for the closed worker.

The module itself is stdlib-only at import time.  Backend imports happen only
inside the two exact-lane entry points, after their setup lifecycle and
guardian already exist.
"""

from __future__ import annotations


import sys
import weakref
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .lifecycle import (
    DeviceScopeDescriptor,
    NativeWorkerSetupLifecycle,
    SetupQuiescenceEvidence,
    TransportTerminationEvidence,
    _array_operation_id,
)
from .native_contract import (
    CudaPreparedNativeInput,
    WarpPreparedNativeInput,
    register_prepared_native_input,
    warp_dtype_from_host_dtype,
)


_CUDA_EXTENSION_SETS = {}
SETUP_WARMUP_FRAME_COUNT = 10
_RENDER_PHASE_NAMES = ("raygen_kernel", "first_hit_kernel", "shade_kernel")


def _render_phase_names(*, boundary, diagnostic):
    if boundary not in {"R0", "R1", "R2"} or type(diagnostic) is not bool:
        raise ValueError("invalid native render event mode")
    if not diagnostic:
        return ()
    return (() if boundary == "R0" else ("scene_update_kernel", "acceleration_kernel")) + _RENDER_PHASE_NAMES


class CudaExtensionSet:
    """Only the captured source session registers a usable extension set."""

    __slots__ = ("session", "controller", "binding", "admission", "recipes")

    def __init__(self):
        raise TypeError("CUDA extension sets are issued by their source session")

    def __setattr__(self, name, value):
        raise TypeError("CUDA extension set is frozen")

    def __delattr__(self, name):
        raise TypeError("CUDA extension set is frozen")


def _require_cuda_extension(owner, recipe_name, context):
    if type(owner) is not CudaExtensionSet:
        raise TypeError("captured preparation needs its issued CUDA extension set")
    record = _CUDA_EXTENSION_SETS.get(owner.session)
    if record is None or record[0] is not owner:
        raise ValueError("CUDA extension set is copied, foreign or unregistered")
    session = record[1][0].session
    try:
        module = session.require_cuda_extension(owner, recipe_name)
        # The set is checked before inspecting caller context metadata. The
        # registered stream must be the actual stream used by preparation.
        from optics.cuda_timeline import CudaOpticalExecutionContext

        binding = record[1][0]
        if (type(context) is not CudaOpticalExecutionContext or context.device != binding.target
                or context.stream is not binding.stream or context.stream_handle != binding.stream_identity[1]):
            raise ValueError("CUDA factory context differs from its issued runtime binding")
        context._validate_identity()
        return module
    except BaseException as error:
        session.fail(error)
        raise


def _cuda_artifact_create_exact(torch, target, stream, loader, spec, frame):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            module = loader.delegate.create_module(spec)
            loader._retain_created_module(module, frame)
            return module


def _cuda_artifact_exec_exact(torch, target, stream, delegate, module):
    with torch.cuda.device(target):
        with torch.cuda.stream(stream):
            delegate.exec_module(module)

@contextmanager
def _optical_setup_operation(session):
    """One failure boundary for the explicit optical preparation stages."""
    source_checked = False
    try:
        session.check_live()
        source_checked = True
        yield
    except BaseException as error:
        try:
            _fail_optical_preparation(session, error, source_integrity_failed=not source_checked)
        except BaseException as disposal_error:
            # A secondary retention/diagnostic failure must not replace the
            # original stage exception. The source still owns partial intake.
            session.failures.append(disposal_error)
            try:
                session.fail(error)
            except BaseException as retention_error:
                session.failures.append(retention_error)
        raise


def _fail_optical_preparation(session, error, *, source_integrity_failed):
    controller = session.bound_controller()
    controller._hold_failure(error)
    healthy_idle = (
        not source_integrity_failed
        and controller.termination_evidence is None
        and not controller._terminating
        and not controller._replay.failed
        and controller._guard_client.state == "IDLE"
    )
    synchronize = None
    if (
        healthy_idle
        and not controller._replay.compromised
        and not controller._replay.may_cancel_without_sync
    ):
        try:
            uploaded = session._retained_setup_input()
            session._require_setup_condition(uploaded is not None, "no retained runtime for idle disposal")
            record = uploaded[0]
            module, target, _stream = session._validate_runtime_binding_resources(record[0], record)
            setup = session.modules["optical_worker.native_setup"][0]
            session._require_setup_condition(
                sys.modules.get("optical_worker.native_setup") is setup, "setup module identity drift"
            )

            def synchronize():
                if session.lane == "warp":
                    setup._warp_synchronize_target_exact(module, target)
                else:
                    with module.cuda.device(target):
                        module.cuda.synchronize(target)
        except BaseException as custody_error:
            controller._hold_failure(custody_error)
            session.runtime_owners.append(custody_error)
            controller.mark_unknown_device_scope(
                source="optical_preparation_failure", reason="runtime_custody_not_validated"
            )
    try:
        session._retain_failure(error)
    except BaseException as retention_error:
        controller._fail_guard("operation", retention_error)
    if not healthy_idle:
        controller._fail_guard("operation", error)
    if controller.termination_evidence is None and not controller._terminating:
        controller.terminate(
            fault="setup_failure",
            detection_phase="optical_preparation_failure",
            emergency_target_synchronize=synchronize,
        )


def _validated_optical_input(session):
    uploaded = session._retained_setup_input()
    session._require_setup_condition(
        uploaded is not None
        and uploaded[1] is session.prepared_input
        and session.validation_report is not None,
        "optical preparation requires the original uploaded and validated input",
    )
    record, prepared = uploaded
    runtime, target, stream = session._validate_runtime_binding_record(record[0], record)
    session.bound_controller().require_adopted(prepared)
    return prepared, runtime, target, stream


def _audit_optical_setup(session):
    session.audit()
    session.external.audit()


def _publish_optical_stage(session, *, preparing, ready):
    """Checkpoint the completed guard before making its result usable."""
    session._require_setup_condition(session.phase == preparing, "optical preparation phase changed")
    session.runtime_observation.checkpoint()
    session.check_live()
    session._require_setup_condition(session.phase == preparing, "optical preparation phase changed")
    session.phase = ready


def _session_prepare_cuda_factories(session):
    """Parent setup foundation only: no slots, warmups or execution readiness."""
    with _optical_setup_operation(session):
        session._require_setup_condition(
            session.lane == "cuda" and session.phase == "input_validated",
            "CUDA factory preparation requires this session's validated input, once",
        )
        controller = session.bound_controller()
        with controller.segment("setup"):
            session.phase = "preparing_cuda_factories"
            prepared, module, target, stream = _validated_optical_input(session)
            extensions = session.cuda_extensions()

            try:
                factories = _prepare_cuda_factory_owners(
                    session, prepared, module, target, stream, extensions
                )
            except BaseException:
                if not controller._terminating and not controller._replay.unknown:
                    try:
                        controller.mark_unknown_device_scope(
                            source="partial_cuda_factory_preparation", reason="cuda_factories_not_completed"
                        )
                    except BaseException as scope_error:
                        session.runtime_owners.append(scope_error)
                raise
            session.cuda_factories = factories
            _audit_optical_setup(session)
        # The six input phases remain unchanged. This is a later parent
        # setup guard, not another input phase or a final readiness record.
        _publish_optical_stage(session, preparing="preparing_cuda_factories", ready="cuda_factories_ready")
        return factories


def _session_prepare_warp_modules(session):
    """Compile existing optical modules; does not issue execution readiness."""
    uploaded = session._retained_setup_input()
    runtime_record = None if uploaded is None else uploaded[0]
    with _optical_setup_operation(session):
        session._require_setup_condition(
            session.lane == "warp"
            and session.phase == "input_validated"
            and not session.warp_preparation_started,
            "Warp module preparation requires this session's validated input, once",
        )
        session.warp_preparation_started = True
        controller = session.bound_controller()
        # This must precede adopted-resource APIs: their compromised-scope
        # rejection poisons normal work, even when no segment is active.
        session._require_setup_condition(
            controller._guard_client.state == "IDLE"
            and not controller._context_active
            and controller._guard_reference is None
            and not controller._replay.compromised,
            "Warp module preparation requires healthy IDLE target scope",
        )
        session._require_setup_condition(
            uploaded is not None
            and uploaded[1] is session.prepared_input
            and session.validation_report is not None,
            "Warp modules require the original uploaded and validated input",
        )
        binding = runtime_record[0]
        session._validate_runtime_binding_resources(binding, runtime_record)
        controller.require_adopted(session.prepared_input)
        records = []
        with controller.segment("setup"):
            session.phase = "preparing_warp_modules"
            _audit_optical_setup(session)
            from optical_worker import warp_setup

            warp_setup.validate_compilation_inputs(session)
            warp_setup.validate_warp_cache(session, binding.module)
            session._runtime_leaf("warp_optical_imports", "runtime_import_or_init")
            from optics import device_scene, device_bvh, warp_native_render

            for index, imported in enumerate((device_scene, device_bvh, warp_native_render)):
                python_module, module, definitions = warp_setup.definition_owners(session, index)
                session._require_setup_condition(
                    python_module is imported, "Warp captured module identity drift"
                )
                session._hold_runtime_owner(module, "warp_imported_module")
                session._hold_runtime_owner(definitions, "warp_imported_definitions")
            _audit_optical_setup(session)
        for index in range(3):
            with controller.segment("setup"):
                session._validate_runtime_binding_record(binding, runtime_record)
                _audit_optical_setup(session)
                records.append(warp_setup.compile_module(session, binding, index))
                _audit_optical_setup(session)
        session.warp_modules = warp_setup.issue_module_set(session, binding, tuple(records))
        _publish_optical_stage(session, preparing="preparing_warp_modules", ready="warp_modules_ready")
        return session.warp_modules


def _session_prepare_fixed_render_storage(session, *, traversal_diagnostics=False):
    """One supervised R0 storage construction, without execution readiness."""
    with _optical_setup_operation(session):
        expected_phase = "cuda_factories_ready" if session.lane == "cuda" else "warp_modules_ready"
        session._require_setup_condition(
            session.phase == expected_phase and type(traversal_diagnostics) is bool,
            "fixed render storage requires completed compilation, once",
        )
        session._require_setup_condition(
            session.prepared_input.manifest.input_contract in {"fixed_native_scene_v1", "canonical_native_replay_v1"},
            "fixed render storage requires canonical R0 or R1 input",
        )
        controller = session.bound_controller()
        with controller.segment("setup"):
            session.phase = "preparing_fixed_render_storage"
            prepared, module, target, stream = _validated_optical_input(session)
            _audit_optical_setup(session)
            if session.lane == "warp":
                from optical_worker import warp_setup

                warp_setup.validate_module_set(session, session.warp_modules)
            else:
                session.cuda_extensions()

            storage = _prepare_fixed_render_storage(
                session, module, target, stream, traversal_diagnostics=traversal_diagnostics
            )
            session._hold_runtime_owner(storage, "fixed_render_storage")
            session.fixed_render_storage = storage
            if session.lane == "warp":
                warp_setup.validate_module_set(session, session.warp_modules)
            _audit_optical_setup(session)
        _publish_optical_stage(
            session, preparing="preparing_fixed_render_storage", ready="fixed_render_storage_ready"
        )
        return storage


def _session_prepare_fixed_render_graph(session, *, profile, shadows, pass_mode):
    """Prepare provisional frame commands/events, without executing optical work."""
    with _optical_setup_operation(session):
        session._require_setup_condition(
            session.phase == "fixed_render_storage_ready", "fixed render graph requires its storage, once"
        )
        session._require_setup_condition(
            type(profile) is str
            and profile in {"direct_light_full", "rgb_preview", "render_only"}
            and type(shadows) is bool
            and type(pass_mode) is str
            and pass_mode in {"primary_unprofiled", "diagnostic_profiled"},
            "invalid fixed render graph modes",
        )
        controller = session.bound_controller()
        with controller.segment("setup"):
            session.phase = "preparing_fixed_render_graph"
            prepared, module, target, stream = _validated_optical_input(session)
            controller.require_adopted(session.fixed_render_storage)
            _audit_optical_setup(session)
            if session.lane == "warp":
                from optical_worker import warp_setup

                compiled = warp_setup.validate_module_set(session, session.warp_modules)
            else:
                compiled = session.cuda_extensions()
            from optical_worker.native_input import certify_native_readiness

            readiness = certify_native_readiness(
                prepared, session.validation_report, compile_complete=True, warmup_frames_completed=0
            )
            session._hold_runtime_owner(readiness, "provisional_input_readiness")
            graph = _prepare_fixed_render_graph(
                session,
                module,
                target,
                stream,
                readiness=readiness,
                profile=profile,
                shadows=shadows,
                pass_mode=pass_mode,
            )
            session._hold_runtime_owner(graph, "provisional_fixed_render_graph")
            if session.lane == "warp":
                session._require_setup_condition(
                    warp_setup.validate_module_set(session, session.warp_modules) is compiled,
                    "render recording changed its compiled modules",
                )
            else:
                session._require_setup_condition(
                    session.cuda_extensions() is compiled, "render binding changed its CUDA extensions"
                )
            _audit_optical_setup(session)
        session.fixed_render_graph = graph
        _publish_optical_stage(
            session, preparing="preparing_fixed_render_graph", ready="fixed_render_graph_ready"
        )
        return graph


def _session_run_fixed_render_warmups(session):
    """Execute and observe ten setup-only frames; no final/sample authority."""
    with _optical_setup_operation(session):
        session._require_setup_condition(
            session.phase == "fixed_render_graph_ready", "fixed render warmups require their graph, once"
        )
        with session.bound_controller().segment("setup"):
            session.phase = "warming_fixed_render_graph"
            _prepared, runtime, target, stream = _validated_optical_input(session)
            from optical_worker.warmup import _prepare_fixed_render_warmups, _run_fixed_render_warmups

            authority = _prepare_fixed_render_warmups(session, runtime, target, stream)
        evidence = _run_fixed_render_warmups(authority)
        session.fixed_render_warmups = evidence
        _publish_optical_stage(
            session, preparing="warming_fixed_render_graph", ready="fixed_render_warmups_ready"
        )
        return evidence


def _session_prepare_fixed_final_graph(session, *, sample_count):
    """Fixed-driver R0 composition and barrier; product IPC remains closed."""
    with _optical_setup_operation(session):
        session._require_setup_condition(
            session.phase == "fixed_render_warmups_ready", "final graph requires observed warmups, once"
        )
        controller = session.bound_controller()
        with controller.segment("setup"):
            session.phase = "preparing_fixed_final_graph"
            _prepared, runtime, target, stream = _validated_optical_input(session)
            _audit_optical_setup(session)

            from optical_worker.arena import _spec_for_final_graph

            spec = _spec_for_final_graph(session, sample_count=sample_count)
            graph = _prepare_fixed_final_graph(
                session, runtime, target, stream, spec=spec, sample_count=sample_count
            )
            _audit_optical_setup(session)
        session.final_render_graph = graph
        session.phase = "fixed_final_graph_ready"
        session.final_quiescence = controller.commit_target_barrier(_source_session=session, _graph=graph)
        session.phase = "fixed_final_graph_admitted"
        return graph

def _prepare_cuda_factory_owners(session, prepared, torch, target, stream, extensions):
    """Reuse ordinary factories and retain each return inside its device scope."""
    session._runtime_leaf("cuda_factory_definitions", "runtime_import_or_init")
    from optics.cuda_timeline import CudaOpticalExecutionContext
    from optics.cuda_direct_light import CudaDirectLightNativeRenderer
    from optical_worker.native_input import (
        build_prepared_cuda_scene_update_factory, build_prepared_cuda_lbvh_rebuild_factory,
    )
    from optical_worker.native_contract import assert_prepared_storage_registered

    session._runtime_leaf("cuda_factory_context", "runtime_import_or_init")
    with torch.cuda.device(target), torch.cuda.stream(stream):
        context = CudaOpticalExecutionContext(device=target, stream=stream)
        session._hold_runtime_owner(context, "cuda_factory_context")
        session._runtime_leaf("cuda_scene_factory", "runtime_import_or_init")
        assert_prepared_storage_registered(prepared)
        scene = build_prepared_cuda_scene_update_factory(prepared, context=context, extensions=extensions)
        session._hold_runtime_owner(scene, "cuda_scene_factory")
        session._runtime_leaf("cuda_lbvh_factory", "runtime_import_or_init")
        lbvh = build_prepared_cuda_lbvh_rebuild_factory(prepared, context=context, extensions=extensions)
        session._hold_runtime_owner(lbvh, "cuda_lbvh_factory")
        session._runtime_leaf("cuda_render_factory", "runtime_import_or_init")
        renderer = CudaDirectLightNativeRenderer.prepare(context, extensions=extensions)
        session._hold_runtime_owner(renderer, "cuda_render_factory")
        factories = (context, scene, lbvh, renderer)
        session._hold_runtime_owner(factories, "cuda_factory_owners")
        return factories


def _require_cuda_render_factory(session, extensions):
    """Resolve the retained renderer through the existing extension/binding checks."""
    from optics.cuda_direct_light import CudaDirectLightNativeRenderer

    factories = session.cuda_factories
    session.bound_controller().require_adopted(factories)
    context, _scene, _lbvh, renderer = factories
    session.bound_controller().require_adopted(context, renderer)
    if type(renderer) is not CudaDirectLightNativeRenderer or renderer.context is not context:
        raise ValueError("CUDA render factory differs from its retained context")
    module = _require_cuda_extension(extensions, "robot_sim_cuda_direct_light_v10", context)
    if renderer.module is not module:
        raise ValueError("CUDA render factory differs from its issued module")
    return renderer, context


def _prepare_fixed_render_storage(session, runtime, target, stream, *, traversal_diagnostics):
    """Connect retained R0 inputs to one set of existing render outputs."""
    from optical_worker.cases import get_case
    from optical_worker.native_input import (
        build_prepared_cuda_fixed_render_inputs, build_prepared_warp_fixed_render_inputs,
    )

    prepared = session.prepared_input
    if prepared.manifest.input_contract == "canonical_native_replay_v1":
        return _prepare_dynamic_render_storage(
            session, runtime, target, stream, traversal_diagnostics=traversal_diagnostics)
    if session.render_source.packed_scene is not prepared.source.packed_scene_owner:
        raise ValueError("fixed render storage requires its original render source")
    case = get_case(prepared.manifest.case_id)
    controller = session.bound_controller()
    controller._require_setup()
    controller.require_adopted(prepared, runtime, target, stream)
    # Covers constructor metadata and CUDA allocator availability preambles,
    # as well as device/stream context entry, before any runtime call.
    session._runtime_leaf("fixed_render_storage_scope", "runtime_import_or_init")
    device_scope = runtime.cuda.device(target) if session.lane == "cuda" else runtime.ScopedDevice(target)
    with device_scope:
        stream_scope = runtime.cuda.stream(stream) if session.lane == "cuda" else runtime.ScopedStream(stream)
        with stream_scope:
            if session.lane == "warp":
                from optics.warp_native_render import allocate_warp_direct_light_buffers

                inputs = build_prepared_warp_fixed_render_inputs(prepared)
            else:
                from optics.cuda_direct_light import allocate_cuda_direct_light_buffers

                frame = session.render_source.sequence.to_payload()["frames"][0]
                inputs = build_prepared_cuda_fixed_render_inputs(
                    prepared,
                    sensor_role_mask=session.render_source.packed_scene.role_to_mask[frame["sensor_role"]],
                    max_distance=float.fromhex(frame["max_distance_f64"]),
                )
            session._hold_runtime_owner(inputs, "fixed_render_input_views")
            context = _create_upload_context(controller, runtime, target, stream)
            if session.lane == "warp":
                outputs = allocate_warp_direct_light_buffers(
                    num_rays=case.width * case.height, device=target, stream=stream,
                    traversal_diagnostics=traversal_diagnostics, _context=context,
                )
            else:
                outputs = allocate_cuda_direct_light_buffers(
                    num_rays=case.width * case.height, device=target, direct_light=True,
                    traversal_diagnostics=traversal_diagnostics, _context=context,
                )
            session._hold_runtime_owner(outputs, "fixed_render_output_buffers")
            return (*inputs, outputs)


@dataclass(frozen=True, eq=False)
class _DynamicRenderStorage:
    scene: object
    world: object
    acceleration: object
    outputs: object
    topology: object = None
    workspace: object = None


def _prepare_dynamic_render_storage(session, runtime, target, stream, *, traversal_diagnostics):
    """One R1 slot, using existing update, topology and allocation implementations."""
    from dataclasses import replace
    from optical_worker.cases import get_case

    prepared = session.prepared_input
    if (prepared.manifest.input_contract != "canonical_native_replay_v1"
            or session.render_source.packed_scene is not prepared.source.packed_scene_owner):
        raise ValueError("dynamic storage requires its original canonical replay source")
    case = get_case(prepared.manifest.case_id)
    controller = session.bound_controller()
    controller._require_setup()
    controller.require_adopted(prepared, runtime, target, stream)
    session._runtime_leaf("dynamic_storage_scope", "runtime_import_or_init")
    device_scope = runtime.cuda.device(target) if session.lane == "cuda" else runtime.ScopedDevice(target)
    stream_scope = runtime.cuda.stream(stream) if session.lane == "cuda" else runtime.ScopedStream(stream)
    with device_scope, stream_scope:
        topology = workspace = None
        if session.lane == "warp":
            from optics.device_scene import allocate_warp_world_scene_buffers
            from optics.device_bvh import build_device_bvh_from_snapshot
            from optics.warp_native_render import allocate_warp_direct_light_buffers
            from optical_worker.native_input import (
                build_prepared_warp_scene_update_factory, build_prepared_warp_transform_inputs,
            )
            from optical_worker import warp_setup

            compiled = warp_setup.validate_module_set(session, session.warp_modules)
            initial_factory = build_prepared_warp_scene_update_factory(prepared, stream=stream)
            session._hold_runtime_owner(initial_factory, "initial_scene_factory")
            scene = initial_factory.local_scene
            session._hold_runtime_owner(scene, "dynamic_local_scene")
            world = allocate_warp_world_scene_buffers(scene, stream=stream,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(world, "dynamic_world_buffers")
            transforms = build_prepared_warp_transform_inputs(prepared, 0)
            session._hold_runtime_owner(transforms, "initial_scene_transforms")
            commands = initial_factory.seal(transforms=transforms, outputs=world,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(commands, "initial_scene_commands")
            if (warp_setup.validate_module_set(session, session.warp_modules) is not compiled
                    or any(command.module_exec is not compiled[0].executable
                           or command.kernel.module is not compiled[0].module for command in commands._commands)):
                raise ValueError("initial scene commands differ from the compiled executable")
            # This untimed initial snapshot establishes the immutable tree.
            # It is not one of the ten observed complete warmup transactions.
            context = _create_upload_context(controller, runtime, target, stream)
            with _native_resource_scope(runtime, target, lane="warp", stream=stream, context=context,
                    owner_role="initial_scene_custody", scope_id="initial_scene_scope") as (_, _, operation):
                for command in commands._commands:
                    with operation("initial_scene_launch", "warmup_launch"):
                        command.launch(stream=stream)
            completion, _ = _prepare_native_events(runtime, target, stream, lane="warp", phase_names=(),
                context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(completion, "initial_scene_completion")
            snapshot = world.as_snapshot(scene, frame_id=0, sim_time=0.0, env_idx=0)
            session._hold_runtime_owner(snapshot, "initial_scene_snapshot")
            topology = build_device_bvh_from_snapshot(snapshot, device=target, stream=stream,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(topology, "dynamic_immutable_topology")
            with _output_array_allocation(runtime, target, lane="warp", stream=stream,
                    context=_create_upload_context(controller, runtime, target, stream)) as empty:
                bounds_min = empty("dynamic_bvh_bounds_min", (topology.num_nodes, 3), runtime.float32)
                bounds_max = empty("dynamic_bvh_bounds_max", (topology.num_nodes, 3), runtime.float32)
            acceleration = replace(topology, bounds_min=bounds_min, bounds_max=bounds_max, ready_event=None,
                                   resources=(topology, bounds_min, bounds_max))
            session._hold_runtime_owner(acceleration, "dynamic_slot_bvh")
            outputs = allocate_warp_direct_light_buffers(num_rays=case.width * case.height, device=target,
                stream=stream, traversal_diagnostics=traversal_diagnostics,
                _context=_create_upload_context(controller, runtime, target, stream))
        else:
            from optics.cuda_scene import allocate_cuda_world_scene_buffers
            from optics.cuda_lbvh import allocate_cuda_lbvh_buffers, allocate_cuda_lbvh_workspace
            from optics.cuda_direct_light import allocate_cuda_direct_light_buffers

            _require_cuda_render_factory(session, session.cuda_extensions())
            _, scene_factory, lbvh_factory, _ = session.cuda_factories
            scene = scene_factory.local_scene
            session._hold_runtime_owner(scene, "dynamic_local_scene")
            world = allocate_cuda_world_scene_buffers(scene,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(world, "dynamic_world_buffers")
            workspace = allocate_cuda_lbvh_workspace(lbvh_factory.requirements,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(workspace, "dynamic_lbvh_workspace")
            acceleration = allocate_cuda_lbvh_buffers(primitive_count=scene.num_triangles, device=target,
                _context=_create_upload_context(controller, runtime, target, stream))
            session._hold_runtime_owner(acceleration, "dynamic_slot_bvh")
            outputs = allocate_cuda_direct_light_buffers(num_rays=case.width * case.height, device=target,
                direct_light=True, traversal_diagnostics=traversal_diagnostics,
                _context=_create_upload_context(controller, runtime, target, stream))
        session._hold_runtime_owner(outputs, "dynamic_render_outputs")
        storage = _DynamicRenderStorage(scene, world, acceleration, outputs, topology, workspace)
        session._hold_runtime_owner(storage, "dynamic_render_storage")
    return storage


@dataclass(frozen=True, eq=False)
class _FixedRenderPreparation:
    """Retained provisional graph; not warmup, final readiness or sample authority."""

    readiness: object
    compiled_modules: object
    factory: object
    frames: tuple
    completion_event: object
    diagnostic_events: tuple
    profile: str
    shadows: bool
    pass_mode: str
    dynamic_factories: tuple = ()


@dataclass(frozen=True, eq=False)
class _DynamicFrameStages:
    """The existing three native stages and their fixed replay selection."""

    stages: tuple
    transform_frame_index: int

    @property
    def render(self):
        return self.stages[2]


def _frame_stages(bundle):
    return bundle.stages if type(bundle) is _DynamicFrameStages else (bundle,)


def _prepare_dynamic_factories(session, runtime, target, stream, compiled):
    storage = session.fixed_render_storage
    if type(storage) is not _DynamicRenderStorage:
        return ()
    prepared = session.prepared_input
    controller = session.bound_controller()
    session._runtime_leaf("dynamic_scene_factory", "runtime_import_or_init")
    if session.lane == "warp":
        from optics.device_scene import WarpSceneUpdateCommandFactory
        from optical_worker.native_input import build_prepared_warp_bvh_refit_factory

        scene = WarpSceneUpdateCommandFactory(storage.scene, stream=stream,
            native_input_manifest_sha256=prepared.manifest.sha256, input_contract=prepared.manifest.input_contract)
        session._hold_runtime_owner(scene, "dynamic_scene_factory")
        acceleration = build_prepared_warp_bvh_refit_factory(prepared, storage.topology, stream=stream,
            _context=_create_upload_context(controller, runtime, target, stream))
    else:
        from optical_worker.native_input import (
            build_prepared_cuda_scene_update_factory, build_prepared_cuda_lbvh_rebuild_factory,
        )

        _, context = _require_cuda_render_factory(session, compiled)
        scene = build_prepared_cuda_scene_update_factory(prepared, context=context, extensions=compiled)
        session._hold_runtime_owner(scene, "dynamic_scene_factory")
        session._runtime_leaf("dynamic_acceleration_factory", "runtime_import_or_init")
        acceleration = build_prepared_cuda_lbvh_rebuild_factory(prepared, context=context, extensions=compiled)
    session._hold_runtime_owner(acceleration, "dynamic_acceleration_factory")
    factories = (scene, acceleration)
    session._hold_runtime_owner(factories, "dynamic_stage_factories")
    return factories


def _prepare_render_events(runtime, target, stream, *, lane, diagnostic, context, boundary="R0"):
    """Initialize and complete one slot's events under the caller's setup guard."""
    if context is None or type(diagnostic) is not bool:
        raise TypeError("render events require a guarded context and exact diagnostic mode")
    return _prepare_native_events(runtime, target, stream, lane=lane,
        phase_names=_render_phase_names(boundary=boundary, diagnostic=diagnostic), context=context)


def _prepare_native_events(runtime, target, stream, *, lane, phase_names, context=None):
    """Shared native event initialization; terminal wait covers all prior records.

    Captured setup supplies its existing custody context. Standalone arena and
    compatibility callers use the same loop under an explicit device/stream.
    """
    if (type(phase_names) is not tuple or len(set(phase_names)) != len(phase_names)
            or any(name not in ("scene_update_kernel", "acceleration_kernel", *_RENDER_PHASE_NAMES)
                   for name in phase_names)):
        raise ValueError("invalid native event phase names")
    with _native_resource_scope(runtime, target, lane=lane, stream=stream, context=context,
            owner_role="slot_event_custody", scope_id="slot_event_scope") as (lifecycle, owners, operation):
        def create(timing):
            with operation("slot_event_create", "event_create"):
                event = (runtime.cuda.Event(enable_timing=timing) if lane == "cuda"
                         else runtime.Event(device=target, enable_timing=timing))
                owners.append(event)
                if lifecycle is not None:
                    ordinal = _event_ordinal(event, cuda=lane == "cuda", lifecycle=lifecycle, allow_unbound=True)
                    lifecycle.adopt_returned_device_owner(event, actual_device_ordinal=ordinal,
                        role="slot_event", operation_id="slot_event_create")
                elif event.device is not None and event.device != target:
                    raise ValueError("native event differs from its target device")
                return event

        def record(event):
            with operation("slot_event_initialize", "event_record"):
                if lane == "cuda":
                    event.record(stream)
                else:
                    stream.record_event(event)
                if lifecycle is not None:
                    _require_ready_event_target(event, cuda=lane == "cuda", lifecycle=lifecycle)
                elif event.device != target:
                    raise ValueError("native event record differs from its target device")

        completion = create(False)
        pairs = []
        for phase in phase_names:
            start, end = create(True), create(True)
            record(start)
            record(end)
            pairs.append((phase, start, end))
        # The terminal initialization is last, so its guarded wait covers all
        # diagnostic event initialization on the same stream as well.
        record(completion)
        with operation("slot_event_initialization_wait", "event_wait"):
            if lane == "cuda":
                completion.synchronize()
            else:
                runtime.synchronize_event(completion)
        return completion, tuple(pairs)


def _prepare_fixed_render_graph(session, runtime, target, stream, *, readiness, profile, shadows, pass_mode):
    """Bind the ten provisional warmup frames using existing renderer builders."""
    from optical_worker.render_parameters import load_frozen_render_sequence

    controller = session.bound_controller()
    storage = session.fixed_render_storage
    dynamic = type(storage) is _DynamicRenderStorage
    if dynamic:
        scene, outputs = storage.scene, storage.outputs
    else:
        scene, _, _, outputs = storage
    diagnostic = pass_mode == "diagnostic_profiled"
    allocated_diagnostic = (outputs.traversal_diagnostics if session.lane == "warp"
                            else bool(outputs.shadow_traversal_counters))
    if diagnostic != allocated_diagnostic:
        raise ValueError("render graph diagnostics differ from its allocated storage")
    frames = []
    session._hold_runtime_owner(frames, "provisional_render_frames")
    sequence = session.render_source.sequence
    prepared = session.prepared_input
    if (session.render_source.packed_scene is not prepared.source.packed_scene_owner
            or sequence.canonical_bytes != load_frozen_render_sequence(prepared.manifest.case_id).canonical_bytes):
        raise ValueError("render graph lost its original canonical source")
    controller.require_adopted(session.fixed_render_storage, readiness)
    session._runtime_leaf("provisional_render_graph_scope", "runtime_import_or_init")
    device_scope = runtime.cuda.device(target) if session.lane == "cuda" else runtime.ScopedDevice(target)
    with device_scope:
        stream_scope = runtime.cuda.stream(stream) if session.lane == "cuda" else runtime.ScopedStream(stream)
        with stream_scope:
            if session.lane == "warp":
                from optical_worker import warp_setup
                from optics.warp_native_render import WarpDirectLightRenderCommandFactory

                compiled_owner = session.warp_modules
                warp_setup.validate_module_set(session, compiled_owner)
                factory = WarpDirectLightRenderCommandFactory.prepare(
                    prepared_input=session.prepared_input, readiness=readiness, local_scene=scene,
                    device=target, stream=stream, _context=_create_upload_context(controller, runtime, target, stream))
            else:
                compiled_owner = session.cuda_extensions()
                factory, cuda_context = _require_cuda_render_factory(session, compiled_owner)
            session._hold_runtime_owner(factory, "provisional_render_factory")
            dynamic_factories = _prepare_dynamic_factories(session, runtime, target, stream, compiled_owner)
            for index in range(SETUP_WARMUP_FRAME_COUNT):
                frames.append(_bind_fixed_render_frame(
                    session, runtime, target, stream, factory=factory, compiled_owner=compiled_owner,
                    index=index, profile=profile, shadows=shadows, pass_mode=pass_mode,
                    owner_prefix="provisional", dynamic_factories=dynamic_factories))
            if session.lane == "cuda":
                actual_factory, actual_context = _require_cuda_render_factory(session, compiled_owner)
                if actual_factory is not factory or actual_context is not cuda_context:
                    raise ValueError("CUDA render factory changed during frame binding")
                for _, bundle in frames:
                    if _frame_stages(bundle)[-1].renderer is not factory:
                        raise ValueError("CUDA render stage differs from its retained factory")
                    for stage in _frame_stages(bundle):
                        stage._validate_cuda_transaction_stage(cuda_context)
            completion, events = _prepare_render_events(runtime, target, stream, lane=session.lane,
                diagnostic=diagnostic, boundary="R1" if dynamic else "R0",
                context=_create_upload_context(controller, runtime, target, stream))
            return _FixedRenderPreparation(readiness, compiled_owner, factory, tuple(frames), completion, events,
                                           profile, shadows, pass_mode, dynamic_factories)


def _bind_fixed_render_frame(session, runtime, target, stream, *, factory, compiled_owner,
                             index, profile, shadows, pass_mode, owner_prefix, dynamic_factories=()):
    """Bind one canonical frame over retained storage for setup or final plans."""
    from dataclasses import replace
    from optical_worker.render_parameters import bind_render_parameters, NATIVE_SCALAR_ORDER, decode_native_scalars

    controller = session.bound_controller()
    controller._require_setup()
    controller.require_adopted(factory, session.fixed_render_storage)
    storage = session.fixed_render_storage
    dynamic = type(storage) is _DynamicRenderStorage
    if dynamic:
        first, second, outputs = storage.world, storage.acceleration, storage.outputs
        if len(dynamic_factories) != 2:
            raise ValueError("dynamic frame requires its scene and acceleration factories")
        controller.require_adopted(dynamic_factories, *dynamic_factories)
        transform_index = index % session.prepared_input.manifest.transform_frame_count
    else:
        _, first, second, outputs = storage
    sequence = session.render_source.sequence
    parameters = bind_render_parameters(sequence, sample_index=index, profile=profile,
        shadows=shadows, pass_mode=pass_mode,
        role_to_mask=session.render_source.packed_scene.role_to_mask)
    payload = parameters.to_payload()
    scalars = decode_native_scalars(payload)
    values = dict(zip(NATIVE_SCALAR_ORDER, scalars, strict=True))
    camera_values = dict(width=payload["width"], height=payload["height"],
        fx=values["fx"], fy=values["fy"], cx=values["cx"], cy=values["cy"],
        origin=scalars[4:7], rotation=scalars[7:16])
    colors = dict(ambient_rgb=scalars[17:20], background_rgb=scalars[20:23],
                  shadow_bias=values["shadow_bias"])
    if session.lane == "warp":
        from optical_worker import warp_setup
        from optics.warp_native_render import WarpCameraParameters, WarpRenderQueryParameters
        from optics.execution import normalize_output_profile

        source = sequence.to_payload()
        frame = source["frames"][index % len(source["frames"])]
        query = WarpRenderQueryParameters(
            num_rays=payload["width"] * payload["height"], frame_id=payload["frame_id"],
            sim_time=float.fromhex(payload["sim_time_f64"]), env_idx=payload["env_idx"],
            sensor_id=payload["sensor_id"], max_distance=values["max_distance"],
            sensor_role=frame["sensor_role"], sensor_role_mask=payload["sensor_role_mask"])
        camera = WarpCameraParameters(query=query, **camera_values)
        # Frame metadata borrows the same immutable R0 world/BVH allocations.
        bvh = replace(second, frame_id=query.frame_id, env_idx=query.env_idx)
        session._hold_runtime_owner(bvh, owner_prefix + "_fixed_bvh_view")
        compiled = warp_setup.validate_module_set(session, compiled_owner)
        bundle = factory.seal_camera(world=first, bvh=bvh, camera=camera, buffers=outputs,
            output_profile=normalize_output_profile(profile), shadows=shadows,
            traversal_diagnostics=pass_mode == "diagnostic_profiled", **colors,
            transform_frame_index=transform_index if dynamic else None,
            _context=_create_upload_context(controller, runtime, target, stream))
    else:
        from optics.cuda_direct_light import CudaCameraParameters

        camera = CudaCameraParameters(**camera_values)
        if dynamic:
            from optics.cuda_direct_light import _dynamic_first_hit_inputs, _shade_inputs

            inputs = _dynamic_first_hit_inputs(scene=storage.scene, world=first, lbvh=second,
                sensor_role_mask=payload["sensor_role_mask"], max_distance=values["max_distance"])
            shade = _shade_inputs(storage.scene)
        else:
            inputs = replace(first, sensor_role_mask=payload["sensor_role_mask"],
                             max_distance=values["max_distance"])
            shade = second
        bundle = factory.bind_direct_light(first_hit_inputs=inputs, shade_inputs=shade,
            buffers=outputs, camera=camera, output_profile=profile, shadows=shadows, **colors)
    session._hold_runtime_owner(bundle, owner_prefix + "_render_bundle")
    if session.lane == "warp":
        if (warp_setup.validate_module_set(session, compiled_owner) is not compiled
                or any(command.module_exec is not compiled[2].executable
                       or command.kernel.module is not compiled[2].module for command in bundle.commands)):
            raise ValueError("recorded render commands differ from the compiled executable")
        bundle.validate_resolved_parameters(parameters)
    else:
        bundle._cuda_transaction_stage_evidence().validate_render_parameters(payload)
    if not dynamic:
        return parameters, bundle
    scene_factory, acceleration_factory = dynamic_factories
    if session.lane == "warp":
        from optical_worker.native_input import build_prepared_warp_transform_inputs

        transforms = build_prepared_warp_transform_inputs(session.prepared_input, transform_index)
        session._hold_runtime_owner(transforms, owner_prefix + "_replay_view")
        update = scene_factory.seal(transforms=transforms, outputs=storage.world,
            _context=_create_upload_context(controller, runtime, target, stream))
        session._hold_runtime_owner(update, owner_prefix + "_scene_bundle")
        acceleration = acceleration_factory.seal(world=storage.world, bvh=storage.acceleration,
            _context=_create_upload_context(controller, runtime, target, stream))
        session._hold_runtime_owner(acceleration, owner_prefix + "_acceleration_bundle")
        if warp_setup.validate_module_set(session, compiled_owner) is not compiled:
            raise ValueError("dynamic recording changed its compiled modules")
        for stage, owner in ((update, compiled[0]), (acceleration, compiled[1])):
            if any(command.module_exec is not owner.executable or command.kernel.module is not owner.module
                   for command in stage._commands):
                raise ValueError("dynamic commands differ from their compiled executable")
    else:
        from optical_worker.native_input import build_prepared_cuda_transform_inputs

        transforms = build_prepared_cuda_transform_inputs(session.prepared_input, transform_index)
        session._hold_runtime_owner(transforms, owner_prefix + "_replay_view")
        update = scene_factory.bind(transforms=transforms, outputs=storage.world)
        session._hold_runtime_owner(update, owner_prefix + "_scene_bundle")
        acceleration = acceleration_factory.bind(world=storage.world, outputs=storage.acceleration,
                                                   workspace=storage.workspace)
        session._hold_runtime_owner(acceleration, owner_prefix + "_acceleration_bundle")
    stages = _DynamicFrameStages((update, acceleration, bundle), transform_index)
    session._hold_runtime_owner(stages, owner_prefix + "_dynamic_stages")
    return parameters, stages


class NativeWorkerPreparationAborted(RuntimeError):
    """A setup exception reached its required terminal resource disposition."""

    def __init__(self, message: str, *, evidence: TransportTerminationEvidence) -> None:
        super().__init__(message)
        self.evidence = evidence


def _descriptor(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    operation_id: str,
    leaf_kind: str,
    runtime_identity: str,
) -> DeviceScopeDescriptor:
    return DeviceScopeDescriptor.exact_target(
        operation_id=operation_id,
        leaf_kind=leaf_kind,
        device_ordinal=lifecycle.target_device_ordinal,
        scope_source="closed_worker_allowlist",
        runtime_identity=runtime_identity,
    )



_FINAL_READINESS_ORIGINS = {}
_FINAL_RENDER_GRAPHS = {}
_CAPTURED_ARENAS = {}


@dataclass(frozen=True, eq=False)
class FinalReadinessOrigin:
    """Observed upstream facts; no reference to the later graph or barrier."""

    session: object
    prepared: object
    warmups: object
    compiled_modules: object
    readiness: object


@dataclass(frozen=True, eq=False)
class _FinalRenderGraph:
    origin: FinalReadinessOrigin
    runtime: object
    target: object
    stream: object
    spec: object
    factory: object
    frames: tuple
    arena: object
    samples: tuple

    @property
    def sha256(self):
        from .protocol import canonical_sha256

        return canonical_sha256({
            "schema_version": "q54.p12.4.final-render-graph.v1",
            "native_readiness_sha256": self.origin.readiness.sha256,
            "warmup_completion": self.origin.warmups.to_payload(),
            "arena_spec_sha256": self.spec.sha256,
            "arena_resources_sha256": self.arena.resource_manifest.sha256,
            "sample_plan_sha256s": [sample._control_plan.evidence.sha256
                if self.spec.lane == "cuda" else sample._plan.evidence.sha256 for sample in self.samples],
        })


def _require_final_graph(session, graph):
    if (type(graph) is not _FinalRenderGraph or _FINAL_RENDER_GRAPHS.get(session) is not graph
            or graph.origin.session is not session or _FINAL_READINESS_ORIGINS.get(session) is not graph.origin
            or graph.origin.prepared is not session.prepared_input
            or graph.origin.warmups is not session.fixed_render_warmups
            or _CAPTURED_ARENAS.get(graph.arena._control, (None,))[0] is not session
            or session.phase != "fixed_final_graph_ready"):
        raise ValueError("final barrier requires the exact registered completed graph")
    session.check_live()
    controller = session.bound_controller()
    controller.require_adopted(graph, graph.origin, graph.runtime, graph.target, graph.stream,
                               graph.arena, graph.origin.readiness)
    if graph.spec.lane == "cuda":
        graph.arena._validate_configuration()
    else:
        graph.arena._control._validate_frozen()
        graph.arena._validate_native_scope()
        if (graph.stream is not graph.arena._stream or graph.target is not graph.arena._device
                or graph.stream.cuda_stream != graph.arena._stream_handle):
            raise ValueError("final Warp graph lost its original native scope")
        for parameters, bundle in graph.frames:
            from optics.warp_timeline import _validate_dynamic_stages

            stages = _frame_stages(bundle)
            stages[-1].validate_resolved_parameters(parameters)
            _validate_dynamic_stages(stages[-1], parameters, graph.stream, stages[:-1])
    return graph


def _require_captured_arena_admission(control, *, for_enqueue=False):
    """Fixed-cost acquisition check; standalone arenas retain their own contract."""
    retained = _CAPTURED_ARENAS.get(control)
    if retained is None:
        return
    session, controller, guardian = retained
    graph = _FINAL_RENDER_GRAPHS.get(session)
    if (graph is None or graph.arena._control is not control
            or session.phase != "fixed_final_graph_admitted" or session.failed
            or controller._final_admission is not graph or controller.guardian is not guardian
            or guardian.state != "SEALED" or controller._terminating
            or controller._replay.controller != "RUNNING" or controller._replay.failed
            or controller._replay.compromised):
        raise ValueError("captured arena has no acknowledged final target barrier")
    if for_enqueue:
        # Final setup admits plans, while measured submission still belongs to
        # the guarded worker entry. Its current closed implementation rejects
        # attempts through either lane's public arena/lease methods as well.
        controller.latch_measured_sample_submission()


def _require_final_readiness_origin(session, prepared, readiness):
    """Reuse observed input validation without another unguarded GPU readback."""
    from .native_contract import assert_prepared_storage_registered

    origin = _FINAL_READINESS_ORIGINS.get(session)
    if (type(origin) is not FinalReadinessOrigin or origin.session is not session
            or origin.prepared is not prepared or origin.readiness is not readiness
            or session.prepared_input is not prepared
            or session.fixed_render_warmups is not origin.warmups
            or session.phase != "preparing_fixed_final_graph"):
        raise ValueError("final C3 requires its exact locally observed readiness origin")
    session.check_live()
    session.bound_controller()._require_setup()
    session.bound_controller().require_adopted(origin, prepared, readiness, origin.warmups)
    assert_prepared_storage_registered(prepared)
    return origin


def _prepare_fixed_final_graph(session, runtime, target, stream, *, spec, sample_count):
    """Compose final R0/R1 factories/C3/arena plans over the warmed physical slot.

    The fixed trusted driver supplies a previously frozen-row-derived spec.
    This private construction is not product row admission or barrier success.
    """
    from dataclasses import replace
    from optics.native_arena import NativeOpticalArenaSpec
    from .native_input import certify_native_readiness
    from .warmup import require_observed_warmups

    observed = require_observed_warmups(session, session.fixed_render_warmups)
    if session in _FINAL_RENDER_GRAPHS:
        raise ValueError("final graph construction is one-shot")
    if (type(spec) is not NativeOpticalArenaSpec or spec.boundary not in {"R0", "R1"}
            or spec.lane != session.lane or spec.payload != "native_only" or spec.ring_depth != 1
            or type(sample_count) is not int or not 1 <= sample_count <= 1_000_000):
        raise ValueError("final graph requires a fixed R0/R1 spec and protocol-bounded sample count")
    provisional = session.fixed_render_graph
    prepared = session.prepared_input
    storage = session.fixed_render_storage
    dynamic = type(storage) is _DynamicRenderStorage
    if dynamic != (spec.boundary == "R1"):
        raise ValueError("final graph boundary differs from its warmed physical storage")
    if (spec.case_id != prepared.manifest.case_id or spec.case_sha256 != prepared.manifest.case_sha256
            or spec.native_input_manifest_sha256 != prepared.manifest.sha256
            or spec.native_readiness_sha256 != provisional.readiness.sha256
            or spec.scene_contract_sha256 != prepared.manifest.scene_contract_sha256
            or spec.fixed_render_fixture_sha256 != prepared.manifest.fixed_render_fixture_sha256
            or spec.input_contract != prepared.manifest.input_contract
            or spec.transform_replay_sha256 != prepared.manifest.transform_replay_sha256
            or spec.transform_frame_count != prepared.manifest.transform_frame_count
            or spec.device_ordinal != prepared.device_ordinal
            or spec.output_profile != provisional.profile or spec.shadows != provisional.shadows
            or spec.diagnostic_mode != provisional.pass_mode):
        raise ValueError("final graph spec differs from its actual warmed input/mode")
    controller = session.bound_controller()
    controller._require_setup()
    controller.require_adopted(prepared, provisional, observed, session.fixed_render_storage)
    _FINAL_RENDER_GRAPHS[session] = None
    session.phase = "preparing_fixed_final_graph"
    if session.lane == "warp":
        from . import warp_setup
        compiled = session.warp_modules
        warp_setup.validate_module_set(session, compiled)
    else:
        compiled = session.cuda_extensions()
    if compiled is not observed.compiled_modules or observed.readiness is not provisional.readiness:
        raise ValueError("final graph differs from the warmed compiled/readiness owners")
    readiness = certify_native_readiness(prepared, session.validation_report, compile_complete=True,
                                         warmup_frames_completed=len(observed.completions))
    session._hold_runtime_owner(readiness, "final_input_readiness")
    before, after = provisional.readiness.to_payload(), readiness.to_payload()
    before["warmup_frames_completed"] = after["warmup_frames_completed"]
    if before != after or readiness is provisional.readiness:
        raise ValueError("final readiness changed more than its observed warmup count")
    origin = FinalReadinessOrigin(session, prepared, observed, compiled, readiness)
    session._hold_runtime_owner(origin, "final_readiness_origin")
    _FINAL_READINESS_ORIGINS[session] = origin
    final_spec = replace(spec, native_readiness_sha256=readiness.sha256)
    if dynamic:
        scene, first, second, outputs = storage.scene, storage.world, storage.acceleration, storage.outputs
    else:
        scene, first, second, outputs = storage
    frames, samples = [], []
    session._hold_runtime_owner(frames, "final_render_frames")
    session._hold_runtime_owner(samples, "final_render_samples")
    session._runtime_leaf("final_render_graph_scope", "runtime_import_or_init")
    device_scope = runtime.cuda.device(target) if session.lane == "cuda" else runtime.ScopedDevice(target)
    stream_scope = runtime.cuda.stream(stream) if session.lane == "cuda" else runtime.ScopedStream(stream)
    with device_scope, stream_scope:
        if session.lane == "warp":
            from optics.warp_native_render import WarpDirectLightRenderCommandFactory
            from optics.warp_arena import WarpNativeOpticalArena

            factory = WarpDirectLightRenderCommandFactory.prepare(
                prepared_input=prepared, readiness=readiness, local_scene=scene, device=target, stream=stream,
                _context=_create_upload_context(controller, runtime, target, stream))
        else:
            from optics.cuda_direct_light import CudaDirectLightNativeRenderer
            from optics.cuda_arena import _seal_cuda_arena, cuda_execution_stream_token
            from .cuda_arena import build_cuda_c3_binding_from_arena_target

            _, context = _require_cuda_render_factory(session, compiled)
            if (spec.execution_stream_token != cuda_execution_stream_token(context)
                    or spec.primitive_count != int(first.triangle_v0_world.shape[0])
                    or spec.node_capacity != (second.node_count if dynamic else first.num_bvh_nodes)):
                raise ValueError("final CUDA spec differs from warmed physical inputs")
            factory = CudaDirectLightNativeRenderer.prepare(context, extensions=compiled)
        session._hold_runtime_owner(factory, "final_render_factory")
        if factory is provisional.factory:
            raise ValueError("final graph cannot reuse its provisional factory identity")
        dynamic_factories = _prepare_dynamic_factories(session, runtime, target, stream, compiled)
        if any(final is setup for final, setup in zip(dynamic_factories, provisional.dynamic_factories, strict=True)):
            raise ValueError("final graph cannot reuse its provisional dynamic factories")
        for index in range(sample_count):
            frames.append(_bind_fixed_render_frame(
                session, runtime, target, stream, factory=factory, compiled_owner=compiled,
                index=index, profile=provisional.profile, shadows=provisional.shadows,
                pass_mode=provisional.pass_mode, owner_prefix="final", dynamic_factories=dynamic_factories))
        frozen_frames = tuple(frames)
        if session.lane == "warp":
            arena = WarpNativeOpticalArena(spec=final_spec, runtime=runtime, frames=frozen_frames,
                completion_event=provisional.completion_event, diagnostic_events=provisional.diagnostic_events,
                dynamic_storage=storage if dynamic else None)
        else:
            templates = {}
            session._hold_runtime_owner(templates, "final_cuda_templates")
            diagnostics = {name: (start, end) for name, start, end in provisional.diagnostic_events}
            # One capability owns this slot's shared events, as in the existing
            # replay arena. Each camera still has its own template/plan identity.
            capability = context._new_controlled_transaction_capability()
            session._hold_runtime_owner(capability, "final_cuda_template_capability")
            for index, (_, bundle) in enumerate(frames):
                template = context.prepare_transaction_template(stages=_frame_stages(bundle),
                    completion_event=provisional.completion_event, diagnostic_events=diagnostics,
                    _activation_capability=capability)
                session._hold_runtime_owner(template, "final_cuda_template")
                transform_index = bundle.transform_frame_index if dynamic else None
                templates[transform_index, index] = (template, capability)
            runtime_arrays = {name: prepared.arrays[name] for name in prepared.manifest.scene_array_names}
            fixture_arrays = {name: prepared.arrays[name] for name in prepared.manifest.fixed_render_fixture_arrays}
            session_tensors = tuple(prepared.arrays.items())
            if dynamic:
                session_tensors += (("lbvh.workspace", storage.workspace.storage),)
            arena = _seal_cuda_arena(spec=final_spec, slot_builds=({
                "slot_id": 0, "world": first if dynamic else None,
                "lbvh": second if dynamic else None, "render": outputs,
                "completion": provisional.completion_event, "diagnostics": diagnostics,
                "templates": templates, "stage_owners": frozen_frames,
            },), session_objects=(origin, factory, context, stream, first, second, prepared, storage, dynamic_factories),
                session_tensors=session_tensors,
                runtime_scene_arrays=runtime_arrays, derived_fixture_arrays=fixture_arrays)
        session._hold_runtime_owner(arena, "final_render_arena")
        _CAPTURED_ARENAS[arena._control] = (session, controller, controller.guardian)
        for index in range(sample_count):
            if session.lane == "warp":
                sample = arena.prepare_sample(render_sample_index=index, planned_acquisition_offset=index)
            else:
                transform_index = index % prepared.manifest.transform_frame_count if dynamic else None
                plan_target = arena.plan_target(transform_frame_index=transform_index, render_sample_index=index,
                                                planned_acquisition_offset=index)
                binding = build_cuda_c3_binding_from_arena_target(plan_target,
                    prepared_input=prepared, native_readiness=readiness, _captured_session=session)
                provenance = arena.bind_c3_provenance(target=plan_target, binding=binding,
                    prepared_input=prepared, native_readiness=readiness)
                sample = arena.prepare_sample(target=plan_target, provenance=provenance,
                                               planned_acquisition_offset=index)
                session._hold_runtime_owner(sample._active, "final_active_sample")
                session._hold_runtime_owner(sample._active._transaction, "final_transaction")
            session._hold_runtime_owner(sample, "final_prepared_sample")
            if session.lane == "warp":
                session._hold_runtime_owner(sample._transaction, "final_transaction")
            samples.append(sample)
        if session.lane == "warp":
            warp_setup.validate_module_set(session, compiled)
        elif session.cuda_extensions() is not compiled:
            raise ValueError("final graph changed its warmed CUDA modules")
    graph = _FinalRenderGraph(origin, runtime, target, stream, final_spec, factory, frozen_frames, arena, tuple(samples))
    session._hold_runtime_owner(graph, "final_render_graph")
    _FINAL_RENDER_GRAPHS[session] = graph
    return graph


def _enter_device_leaf(
    lifecycle: NativeWorkerSetupLifecycle,
    descriptor: DeviceScopeDescriptor,
) -> None:
    permit = lifecycle.begin_device_leaf(descriptor)
    lifecycle.enter_device_leaf(permit)


def _torch_empty_exact(torch: Any, *, shape: tuple[int, ...], dtype: Any, target: Any) -> Any:
    return torch.empty(shape, dtype=dtype, device=target)


def _torch_copy_exact(torch: Any, destination: Any, source: Any, stream: Any) -> None:
    # StreamContext restores its source stream via set_stream(). Enter the
    # indexed device first so that source and destination streams belong to the
    # target, rather than restoring an ambient foreign-device stream. The outer
    # device guard uses Torch's lazy device restoration on CUDA 12.
    with torch.cuda.device(stream.device):
        with torch.cuda.stream(stream):
            destination.copy_(source, non_blocking=True)


def _torch_event_exact(torch: Any) -> Any:
    return torch.cuda.Event(enable_timing=False)


def _torch_record_event_exact(event: Any, stream: Any) -> None:
    event.record(stream)


def _torch_wait_event_host_exact(event: Any) -> None:
    event.synchronize()


def _warp_empty_exact(wp: Any, *, shape: tuple[int, ...], dtype: Any, target: Any) -> Any:
    return wp.empty(shape=shape, dtype=dtype, device=target)


def _warp_copy_exact(wp: Any, destination: Any, source: Any, stream: Any) -> None:
    wp.copy(destination, source, stream=stream)


def _warp_event_exact(wp: Any, *, target: Any) -> Any:
    return wp.Event(device=target)


def _warp_record_event_exact(stream: Any, event: Any) -> None:
    stream.record_event(event)


def _warp_wait_event_host_exact(wp: Any, event: Any) -> None:
    wp.synchronize_event(event)


def _warp_synchronize_target_exact(wp: Any, target: Any) -> None:
    with wp.ScopedDevice(target):
        wp.synchronize_device(target)


def _event_ordinal(event, *, cuda, lifecycle, allow_unbound=False):
    """Read held event metadata; only an unrecorded CUDA event may be unbound."""
    try:
        device = event.device
        if cuda and device is None and allow_unbound:
            return lifecycle.target_device_ordinal
        ordinal = device.index if cuda else device.ordinal
        if (type(ordinal) is not int or not 0 <= ordinal <= 1023
                or (cuda and device.type != "cuda") or (not cuda and not device.is_cuda)):
            raise ValueError("event has no valid indexed CUDA device")
        return ordinal
    except BaseException:
        lifecycle.mark_unknown_device_scope(source="event_device_metadata", reason="event_device_unavailable")
        raise


def _require_ready_event_target(event, *, cuda, lifecycle):
    actual = _event_ordinal(event, cuda=cuda, lifecycle=lifecycle)
    if actual != lifecycle.target_device_ordinal:
        # Register the observed foreign scope before any native wait. The
        # existing lifecycle rejects this descriptor without admitting work.
        _enter_device_leaf(lifecycle, DeviceScopeDescriptor.exact_target(
            operation_id="foreign_ready_event_wait", leaf_kind="event_wait",
            device_ordinal=actual, scope_source="returned_ready_event",
            runtime_identity="observed_event_device_v1",
        ))
        raise ValueError("foreign ready event scope was unexpectedly accepted")


def _abort(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    exception: BaseException,
    sync: Callable[[], None] | None,
) -> NativeWorkerPreparationAborted:
    if lifecycle.guardian.state == "CONSTRUCTING" and not lifecycle.guardian.contains_exact_owner(exception):
        lifecycle.adopt_host_owner(
            exception,
            role="setup_exception_graph",
            operation_id="setup_exception",
        )
    evidence = lifecycle.terminate(
        fault="setup_failure",
        detection_phase="a2_native_input_preparation",
        emergency_target_synchronize=sync,
    )
    return NativeWorkerPreparationAborted(
        "closed-worker native input preparation aborted",
        evidence=evidence,
    )


@dataclass(frozen=True, eq=False)
class _NativeUploadContext:
    """Internal mechanics shared by C0 and the guarded runtime binding.

    This is NOT readiness or a public execution capability. Only the fixed
    wrappers create it; one consumption burns its registration before upload.
    """

    lifecycle: NativeWorkerSetupLifecycle
    runtime: object
    target: object
    stream: object


_UPLOAD_CONTEXTS = {}


def _create_upload_context(lifecycle, runtime, target, stream):
    context = _NativeUploadContext(lifecycle, runtime, target, stream)
    key = id(context)

    def discard(reference):
        record = _UPLOAD_CONTEXTS.get(key)
        if record is not None and record[0] is reference:
            _UPLOAD_CONTEXTS.pop(key, None)

    _UPLOAD_CONTEXTS[key] = (
        weakref.ref(context, discard),
        lifecycle,
        runtime,
        target,
        stream,
    )
    try:
        lifecycle.adopt_host_owner(
            context,
            role="native_upload_context",
            operation_id="native_upload_context_create",
        )
    except BaseException:
        # The C0 lifecycle is slotted/non-weakrefable. Its exact strong identity
        # is retained only until immediate consumption, or failed construction.
        # Context construction enqueues no device work; existing operand custody
        # and the caller's failure policy remain responsible for runtime owners.
        _UPLOAD_CONTEXTS.pop(key, None)
        raise
    return context


def _consume_upload_context(context):
    if type(context) is not _NativeUploadContext:
        raise TypeError("upload core requires its exact internal context")
    record = _UPLOAD_CONTEXTS.pop(id(context), None)
    if (
        record is None
        or record[0]() is not context
        or record[1] is not context.lifecycle
        or record[2] is not context.runtime
        or record[3] is not context.target
        or record[4] is not context.stream
    ):
        raise ValueError("upload context is unregistered, consumed or has changed")
    context.lifecycle.require_adopted(context)
    return context.lifecycle, context.runtime, context.target, context.stream


@contextmanager
def _native_resource_scope(runtime, target, *, lane, stream=None, context=None,
                           owner_role, scope_id):
    """Reuse one guarded ticket and retain every partial native resource graph."""
    lifecycle = None
    owners = []
    if lane not in {"cuda", "warp"}:
        raise ValueError("unknown native resource lane")
    if context is not None:
        from .guarded_lifecycle import NativeGuardedSetupLifecycle

        lifecycle, held_runtime, held_target, held_stream = _consume_upload_context(context)
        if type(lifecycle) is not NativeGuardedSetupLifecycle:
            raise TypeError("native resource setup requires its guarded lifecycle")
        lifecycle._require_setup()
        ticket = lifecycle._guard_ticket
        if (runtime is not held_runtime or lifecycle._lane_id != lane
                or (target != held_target if lane == "cuda" else target is not held_target)
                or (stream is not None and stream is not held_stream)):
            raise ValueError("native resource setup differs from its runtime binding")
        target, stream = held_target, held_stream
        lifecycle.require_adopted(runtime, target, stream)
        lifecycle.adopt_host_owner(owners, role=owner_role, operation_id=owner_role)
        _enter_device_leaf(lifecycle, _descriptor(lifecycle,
            operation_id=scope_id, leaf_kind="runtime_import_or_init",
            runtime_identity=lane + "_resource_scope_v1"))

    active = False

    @contextmanager
    def operation(name, kind):
        if not active:
            raise ValueError("native resource operation is outside its original scope")
        with lifecycle._observe_public_operation() if lifecycle is not None else nullcontext():
            if lifecycle is not None:
                lifecycle._require_setup()
                if lifecycle._guard_ticket is not ticket:
                    raise ValueError("native resource operation is outside its original guard")
                _enter_device_leaf(lifecycle, _descriptor(lifecycle,
                    operation_id=name, leaf_kind=kind,
                    runtime_identity=lane + "_explicit_resource_v1"))
            yield

    with lifecycle._observe_public_operation() if lifecycle is not None else nullcontext():
        device_scope = runtime.cuda.device(target) if lane == "cuda" else runtime.ScopedDevice(target)
        with device_scope:
            stream_scope = (runtime.cuda.stream(stream) if lane == "cuda" else runtime.ScopedStream(stream)) \
                if stream is not None else nullcontext()
            with stream_scope:
                active = True
                try:
                    yield lifecycle, owners, operation
                finally:
                    active = False


@contextmanager
def _output_array_allocation(runtime, target, *, lane, stream=None, context=None):
    """Existing allocator, sharing the original ticket/custody scope with recording."""
    with _native_resource_scope(runtime, target, lane=lane, stream=stream, context=context,
            owner_role="slot_array_custody", scope_id="slot_allocation_scope") as (lifecycle, owners, operation):
        target = context.target if context is not None else target
        def empty(name, shape, dtype):
            with operation(_array_operation_id("slot_allocate", name), "allocation"):
                result = runtime.empty(shape, dtype=dtype, device=target)
                owners.append(result)
                if lifecycle is not None:
                    ordinal = int(result.device.index if lane == "cuda" else result.device.ordinal)
                    lifecycle.adopt_returned_device_owner(result, actual_device_ordinal=ordinal,
                        role="slot_device_array", operation_id=_array_operation_id("slot_allocate", name))
                return result
        yield empty


@contextmanager
def _warp_command_recording(runtime, target, stream, *, context=None, ready_event=None, _descriptors=None):
    """Record existing kernel definitions, retaining each return before the next leaf."""
    with _native_resource_scope(runtime, target, lane="warp", stream=stream, context=context,
            owner_role="slot_command_custody", scope_id="slot_command_scope") as (lifecycle, owners, operation):
        if ready_event is not None:
            with operation("slot_command_input_wait", "event_wait"):
                owners.append(ready_event)
                if lifecycle is not None:
                    lifecycle.require_adopted(ready_event, stream)
                    if ready_event.device is not target:
                        raise ValueError("recorded command dependency differs from its target")
                stream.wait_event(ready_event)
        def record(kernel, *, dim, inputs):
            with operation("slot_record_command", "compile_or_jit"):
                # Hold operands before record_cmd can initialize a native entry.
                owners.append((kernel, inputs))
                result = runtime.launch(kernel, dim=dim, inputs=inputs, device=target,
                                        stream=stream, record_tape=False, record_cmd=True)
                owners.append(result)
                if lifecycle is not None:
                    lifecycle.adopt_returned_device_owner(result,
                        actual_device_ordinal=int(result.device.ordinal),
                        role="slot_recorded_command", operation_id="slot_record_command")
                if _descriptors is not None:
                    from optics.warp_native_render import _WarpRenderDescriptor

                    _descriptors.append(_WarpRenderDescriptor(
                        entry_id=kernel.func.__name__, kernel=kernel, dim=dim,
                        argument_names=tuple(argument.label for argument in kernel.adj.args),
                        arguments=tuple(inputs)))
                return result
        yield record


def _upload_cuda_native_input(source, *, context):
    """Single allocation/upload implementation; the caller owns failure policy."""
    lifecycle, torch, target, stream = _consume_upload_context(context)
    import numpy as np

    prepared: dict[str, Any] = {}
    for manifest in source.manifest.arrays:
        value = source.arrays[manifest.name]
        host = np.array(value, copy=True, order="C")
        lifecycle.adopt_host_owner(
            host,
            role="cuda_h2d_host_staging",
            operation_id=_array_operation_id("cuda_stage", manifest.name),
        )
        host_tensor = torch.from_numpy(host)
        lifecycle.adopt_host_owner(
            host_tensor,
            role="cuda_h2d_torch_host_view",
            operation_id=_array_operation_id("cuda_host_view", manifest.name),
        )
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id=_array_operation_id("cuda_allocate", manifest.name),
                leaf_kind="allocation",
                runtime_identity="torch_empty_explicit_device_v1",
            ),
        )
        destination = _torch_empty_exact(
            torch,
            shape=tuple(host.shape),
            dtype=host_tensor.dtype,
            target=target,
        )
        lifecycle.adopt_returned_device_owner(
            destination,
            actual_device_ordinal=(
                destination.device.index
                if destination.device.index is not None
                else lifecycle.target_device_ordinal
            ),
            role="cuda_a2_device_array",
            operation_id=_array_operation_id("cuda_allocate", manifest.name),
        )
        lifecycle.require_adopted(host, host_tensor, destination, stream)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id=_array_operation_id("cuda_upload", manifest.name),
                leaf_kind="host_to_device_copy",
                runtime_identity="torch_copy_explicit_stream_v1",
            ),
        )
        _torch_copy_exact(torch, destination, host_tensor, stream)
        prepared[manifest.name] = destination

    ready_owners = []
    lifecycle.adopt_host_owner(ready_owners, role="ready_event_custody", operation_id="ready_event_custody")
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_create",
            leaf_kind="event_create",
            runtime_identity="torch_cuda_event_v1",
        ),
    )
    ready_event = _torch_event_exact(torch)
    ready_owners.append(ready_event)
    lifecycle.adopt_returned_device_owner(
        ready_event,
        actual_device_ordinal=_event_ordinal(ready_event, cuda=True, lifecycle=lifecycle, allow_unbound=True),
        role="cuda_a2_ready_event",
        operation_id="cuda_ready_event_create",
    )
    lifecycle.require_adopted(ready_event, stream)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_record",
            leaf_kind="event_record",
            runtime_identity="torch_event_record_indexed_stream_v1",
        ),
    )
    _torch_record_event_exact(ready_event, stream)
    _require_ready_event_target(ready_event, cuda=True, lifecycle=lifecycle)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_ready_event_host_wait",
            leaf_kind="event_wait",
            runtime_identity="torch_event_synchronize_v1",
        ),
    )
    _torch_wait_event_host_exact(ready_event)
    result = CudaPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=lifecycle.target_device_ordinal,
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
        source=source,
    )
    lifecycle.adopt_returned_device_owner(
        result,
        actual_device_ordinal=lifecycle.target_device_ordinal,
        role="cuda_a2_prepared_input",
        operation_id="cuda_a2_result",
    )
    register_prepared_native_input(result, source)
    return result


def _upload_warp_array(lifecycle, wp, target, stream, name, value):
    """One adopted host staging / allocation / explicit-stream copy body."""
    import numpy as np

    host = np.array(value, copy=True, order="C")
    lifecycle.adopt_host_owner(
        host,
        role="warp_h2d_host_staging",
        operation_id=_array_operation_id("warp_stage", name),
    )
    host_array = wp.array(
        host,
        dtype=warp_dtype_from_host_dtype(wp, value.dtype),
        device="cpu",
        copy=False,
    )
    lifecycle.adopt_host_owner(
        host_array,
        role="warp_h2d_cpu_array_view",
        operation_id=_array_operation_id("warp_host_view", name),
    )
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id=_array_operation_id("warp_allocate", name),
            leaf_kind="allocation",
            runtime_identity="warp_empty_explicit_device_v1",
        ),
    )
    destination = _warp_empty_exact(
        wp,
        shape=tuple(host.shape),
        dtype=warp_dtype_from_host_dtype(wp, value.dtype),
        target=target,
    )
    lifecycle.adopt_returned_device_owner(
        destination,
        actual_device_ordinal=int(destination.device.ordinal),
        role="warp_a2_device_array",
        operation_id=_array_operation_id("warp_allocate", name),
    )
    lifecycle.require_adopted(host, host_array, destination, stream)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id=_array_operation_id("warp_upload", name),
            leaf_kind="host_to_device_copy",
            runtime_identity="warp_copy_explicit_stream_v1",
        ),
    )
    _warp_copy_exact(wp, destination, host_array, stream)
    return destination


def _upload_warp_native_input(source, *, context):
    """Single allocation/upload implementation; the caller owns failure policy."""
    lifecycle, wp, target, stream = _consume_upload_context(context)
    prepared: dict[str, Any] = {}
    with wp.ScopedDevice(target):
        for manifest in source.manifest.arrays:
            value = source.arrays[manifest.name]
            destination = _upload_warp_array(lifecycle, wp, target, stream, manifest.name, value)
            prepared[manifest.name] = destination

        ready_owners = []
        lifecycle.adopt_host_owner(ready_owners, role="ready_event_custody", operation_id="ready_event_custody")
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_create",
                leaf_kind="event_create",
                runtime_identity="warp_event_explicit_device_v1",
            ),
        )
        ready_event = _warp_event_exact(wp, target=target)
        ready_owners.append(ready_event)
        lifecycle.adopt_returned_device_owner(
            ready_event,
            actual_device_ordinal=_event_ordinal(ready_event, cuda=False, lifecycle=lifecycle),
            role="warp_a2_ready_event",
            operation_id="warp_ready_event_create",
        )
        lifecycle.require_adopted(ready_event, stream)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_record",
                leaf_kind="event_record",
                runtime_identity="warp_stream_record_event_v1",
            ),
        )
        _warp_record_event_exact(stream, ready_event)
        _require_ready_event_target(ready_event, cuda=False, lifecycle=lifecycle)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_ready_event_host_wait",
                leaf_kind="event_wait",
                runtime_identity="warp_synchronize_event_v1",
            ),
        )
        _warp_wait_event_host_exact(wp, ready_event)
    result = WarpPreparedNativeInput(
        manifest=source.manifest,
        device_ordinal=lifecycle.target_device_ordinal,
        arrays=MappingProxyType(prepared),
        ready_event=ready_event,
        source=source,
    )
    lifecycle.adopt_returned_device_owner(
        result,
        actual_device_ordinal=lifecycle.target_device_ordinal,
        role="warp_a2_prepared_input",
        operation_id="warp_a2_result",
    )
    register_prepared_native_input(result, source)
    return result


def prepare_cuda_native_input_for_closed_worker(
    source: object,
    *,
    lifecycle: NativeWorkerSetupLifecycle,
) -> object:
    """Prepare exact Torch-owned a2 allocations without allocation+copy calls."""

    if lifecycle.state != "NO_SETUP_GPU_WORK":
        raise RuntimeError("closed-worker CUDA a2 preparation must start a fresh setup epoch")
    if not lifecycle.guardian.contains_exact_owner(source):
        lifecycle.adopt_host_owner(
            source,
            role="canonical_native_input",
            operation_id="cuda_a2_source",
        )
    torch: Any | None = None
    target: Any | None = None
    try:
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="cuda_runtime_import",
                leaf_kind="runtime_import_or_init",
                runtime_identity="torch_cuda_indexed_v1",
            ),
        )
        import torch as torch_module

        torch = torch_module
        target = torch.device("cuda", lifecycle.target_device_ordinal)
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="cuda_setup_stream_create",
                leaf_kind="allocation",
                runtime_identity="torch_cuda_stream_explicit_device_v1",
            ),
        )
        stream = torch.cuda.Stream(device=target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=lifecycle.target_device_ordinal,
            role="cuda_setup_stream",
            operation_id="cuda_setup_stream_create",
        )

        context = _create_upload_context(lifecycle, torch, target, stream)
        return _upload_cuda_native_input(source, context=context)
    except BaseException as exc:
        if torch is None or target is None:
            lifecycle.mark_unknown_device_scope(
                source="cuda_runtime_import",
                reason="target_sync_unavailable_after_import_failure",
            )
            sync = None
        else:

            def sync() -> None:
                torch.cuda.synchronize(target)

        raise _abort(lifecycle, exception=exc, sync=sync) from exc


def prepare_warp_native_input_for_closed_worker(
    source: object,
    *,
    lifecycle: NativeWorkerSetupLifecycle,
) -> object:
    """Prepare exact Warp-owned a2 allocations with registered H2D operands."""

    if lifecycle.state != "NO_SETUP_GPU_WORK":
        raise RuntimeError("closed-worker Warp a2 preparation must start a fresh setup epoch")
    if not lifecycle.guardian.contains_exact_owner(source):
        lifecycle.adopt_host_owner(
            source,
            role="canonical_native_input",
            operation_id="warp_a2_source",
        )
    wp: Any | None = None
    target: Any | None = None
    try:
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_runtime_import",
                leaf_kind="runtime_import_or_init",
                runtime_identity="warp_cuda_indexed_v1",
            ),
        )
        import warp as warp_module

        wp = warp_module
        target = wp.get_device(f"cuda:{lifecycle.target_device_ordinal}")
        if not target.is_cuda or int(target.ordinal) != lifecycle.target_device_ordinal:
            raise RuntimeError("Warp resolved a device different from the frozen target")
        lifecycle.adopt_returned_device_owner(
            target,
            actual_device_ordinal=int(target.ordinal),
            role="warp_target_device",
            operation_id="warp_runtime_import",
        )
        _enter_device_leaf(
            lifecycle,
            _descriptor(
                lifecycle,
                operation_id="warp_setup_stream_resolve",
                leaf_kind="runtime_import_or_init",
                runtime_identity="warp_get_stream_explicit_device_v1",
            ),
        )
        stream = wp.get_stream(target)
        lifecycle.adopt_returned_device_owner(
            stream,
            actual_device_ordinal=int(stream.device.ordinal),
            role="warp_setup_stream",
            operation_id="warp_setup_stream_resolve",
        )

        context = _create_upload_context(lifecycle, wp, target, stream)
        return _upload_warp_native_input(source, context=context)
    except BaseException as exc:
        if wp is None or target is None:
            lifecycle.mark_unknown_device_scope(
                source="warp_runtime_import",
                reason="target_sync_unavailable_after_import_failure",
            )
            sync = None
        else:

            def sync() -> None:
                _warp_synchronize_target_exact(wp, target)

        raise _abort(lifecycle, exception=exc, sync=sync) from exc


def seal_and_quiesce_cuda_setup(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    a2_readiness_sha256: str,
    prepared_graph_sha256: str,
) -> SetupQuiescenceEvidence:
    """Seal, then issue the final exact-target Torch setup barrier."""

    import torch

    target = torch.device("cuda", lifecycle.target_device_ordinal)
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="cuda_final_setup_barrier",
            leaf_kind="device_synchronize",
            runtime_identity="torch_synchronize_explicit_device_v1",
        ),
    )
    lifecycle.seal_guardian()
    try:
        torch.cuda.synchronize(target)
    except BaseException as exc:
        raise _abort(
            lifecycle,
            exception=exc,
            sync=lambda: torch.cuda.synchronize(target),
        ) from exc
    return lifecycle.commit_target_barrier(
        a2_readiness_sha256=a2_readiness_sha256,
        prepared_graph_sha256=prepared_graph_sha256,
    )


def seal_and_quiesce_warp_setup(
    lifecycle: NativeWorkerSetupLifecycle,
    *,
    a2_readiness_sha256: str,
    prepared_graph_sha256: str,
) -> SetupQuiescenceEvidence:
    """Seal, then issue the final exact-target Warp setup barrier."""

    import warp as wp

    target = wp.get_device(f"cuda:{lifecycle.target_device_ordinal}")
    _enter_device_leaf(
        lifecycle,
        _descriptor(
            lifecycle,
            operation_id="warp_final_setup_barrier",
            leaf_kind="device_synchronize",
            runtime_identity="warp_synchronize_explicit_device_v1",
        ),
    )
    lifecycle.seal_guardian()
    try:
        _warp_synchronize_target_exact(wp, target)
    except BaseException as exc:
        raise _abort(
            lifecycle,
            exception=exc,
            sync=lambda: _warp_synchronize_target_exact(wp, target),
        ) from exc
    return lifecycle.commit_target_barrier(
        a2_readiness_sha256=a2_readiness_sha256,
        prepared_graph_sha256=prepared_graph_sha256,
    )
