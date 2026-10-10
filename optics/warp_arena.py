"""Warp recorded commands over the existing native FIFO/lease control plane.

R0 construction consumes already allocated buffers and recorded final frames.
It performs no allocation, upload, recording or synchronization on the device.
This adapter does not grant closed-worker admission or product activation.
"""

from dataclasses import dataclass, field

from .native_arena import (
    NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
    NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
    NativeOpticalArena,
    NativeOpticalArenaBackpressure,
    NativeOpticalArenaProtocolError,
    NativeOpticalArenaResourceManifest,
    NativeOpticalArenaSpec,
    NativeOpticalSlotResourceManifest,
)


def warp_execution_stream_token(device, stream):
    return f"warp-stream:{device.ordinal}:{stream.cuda_stream}"


def _digest(payload):
    from optical_worker.provenance import _canonical_sha256

    return _canonical_sha256(payload)


def _storage_signature(name, value, device, owner):
    from optical_worker.provenance import NativeStorageSignature

    from .warp_native_render import _render_array_snapshot

    snapshot = _render_array_snapshot(value, name=name, device=device)
    root, pointer, capacity, _, _ = snapshot[-1][-1]
    return NativeStorageSignature(
        name=name,
        backend="warp",
        device_ordinal=device.ordinal,
        storage_token=(f"warp:0x{pointer:x}" if capacity else f"warp:empty:{id(root):x}"),
        dtype=value.dtype.__name__,
        shape=snapshot[3],
        nbytes=capacity,
        active_start=snapshot[5] - pointer if snapshot[6] else 0,
        active_count=snapshot[6],
        owner_token=owner,
    )


def _bind_r0_c3(spec, parameters, bundle, outputs):
    from optical_worker.native_input import prepared_storage_signatures
    from optical_worker.provenance import (
        NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
        NativeStageEvidence,
        build_c3_one_shot_native_execution_binding,
    )

    bundle.validate_resolved_parameters(parameters)
    prepared = bundle.binding.factory.prepared_input
    runtime = prepared_storage_signatures(prepared, prepared.manifest.scene_array_names)
    fixture = prepared_storage_signatures(prepared, prepared.manifest.fixed_render_fixture_arrays)
    stage = NativeStageEvidence(
        schema_version=NATIVE_STAGE_EVIDENCE_SCHEMA_VERSION,
        stage_name="render",
        operation_id=spec.renderer_id,
        output_profile=spec.output_profile,
        shadows=spec.shadows,
        scene_contract_sha256=spec.scene_contract_sha256,
        input_contract=spec.input_contract,
        transform_replay_sha256=None,
        transform_frame_index=None,
        fixed_render_fixture_sha256=spec.fixed_render_fixture_sha256,
        topology_sha256=None,
        input_storage=runtime + fixture,
        output_storage=outputs,
        dispatch_plan_sha256=bundle.evidence.sha256,
        bound_render_parameters=parameters,
        bound_render_parameters_sha256=parameters.sha256,
    )
    return build_c3_one_shot_native_execution_binding(
        row_id=spec.row_id,
        case_id=spec.case_id,
        case_sha256=spec.case_sha256,
        boundary="R0",
        lane="warp",
        renderer_id=spec.renderer_id,
        acceleration_class=spec.acceleration_class,
        acceleration_id=spec.acceleration_id,
        output_profile=spec.output_profile,
        shadows=spec.shadows,
        input_contract=spec.input_contract,
        native_input_manifest_sha256=spec.native_input_manifest_sha256,
        native_readiness_sha256=spec.native_readiness_sha256,
        scene_contract_sha256=spec.scene_contract_sha256,
        transform_replay_sha256=None,
        transform_frame_index=None,
        fixed_render_fixture_sha256=spec.fixed_render_fixture_sha256,
        runtime_scene_storage=runtime,
        derived_fixture_storage=fixture,
        stage_evidence=(stage,),
        bound_render_parameters=parameters,
    )


@dataclass(frozen=True, eq=False, init=False, slots=True)
class WarpNativeOpticalArena:
    """Immutable physical graph; all slot transitions belong to NativeOpticalArena."""

    _control: object
    _runtime: object
    _device: object
    _stream: object
    _stream_handle: int
    _frames: tuple
    _bindings: tuple
    _completion: object
    _diagnostics: tuple
    _event_bindings: tuple
    _prepared: dict
    _transactions: dict

    def __init__(self, *, spec, runtime, frames, completion_event, diagnostic_events):
        from optical_worker.native_contract import assert_warp_prepared_authority
        from optical_worker.native_setup import SETUP_WARMUP_FRAME_COUNT, _render_phase_names

        from .warp_native_render import WarpDirectLightRenderCommandBundle

        if hasattr(self, "_control"):
            raise RuntimeError("Warp arena cannot be initialized again")
        if (
            type(spec) is not NativeOpticalArenaSpec
            or spec.lane != "warp"
            or spec.boundary != "R0"
            or spec.payload != "native_only"
            or spec.ring_depth != 1
        ):
            raise ValueError("Warp fixed arena requires one native-only R0 slot")
        if type(frames) is not tuple or not frames:
            raise ValueError("Warp arena requires pre-recorded final frames")
        if any(type(bundle) is not WarpDirectLightRenderCommandBundle for _, bundle in frames):
            raise TypeError("Warp arena requires exact recorded bundles")
        factory = frames[0][1].binding.factory
        prepared, readiness = factory.prepared_input, factory.readiness
        assert_warp_prepared_authority(prepared, readiness)
        device, stream = factory.device, factory.stream
        if (
            readiness.warmup_frames_completed != SETUP_WARMUP_FRAME_COUNT
            or spec.native_input_manifest_sha256 != prepared.manifest.sha256
            or spec.native_readiness_sha256 != readiness.sha256
            or spec.execution_stream_token != warp_execution_stream_token(device, stream)
            or spec.device_ordinal != device.ordinal
        ):
            raise ValueError("Warp arena differs from its final input readiness/scope")
        expected = _render_phase_names(
            boundary="R0", diagnostic=spec.diagnostic_mode == "diagnostic_profiled"
        )
        if type(diagnostic_events) is not tuple or tuple(p for p, _, _ in diagnostic_events) != expected:
            raise ValueError("Warp arena diagnostics differ from its frozen pass")
        events = (completion_event, *(e for _, a, b in diagnostic_events for e in (a, b)))
        if len({id(e) for e in events}) != len(events) or any(
            type(e) is not runtime.Event or e.device is not device for e in events
        ):
            raise ValueError("Warp arena events are foreign or aliased")
        event_bindings = tuple((event, event.cuda_event) for event in events)
        if any(type(handle) is not int or handle <= 0 for _, handle in event_bindings) or len(
            {handle for _, handle in event_bindings}
        ) != len(events):
            raise ValueError("Warp arena events require distinct initialized native handles")
        object.__setattr__(self, "_event_bindings", event_bindings)
        manifest_input = prepared.manifest
        if any(
            getattr(spec, name) != getattr(manifest_input, name)
            for name in (
                "case_id",
                "case_sha256",
                "input_contract",
                "scene_contract_sha256",
                "fixed_render_fixture_sha256",
                "transform_replay_sha256",
                "transform_frame_count",
            )
        ):
            raise ValueError("Warp arena spec differs from its actual prepared input")
        buffers = frames[0][1].binding.buffers
        for index, (parameters, bundle) in enumerate(frames):
            bundle.validate_resolved_parameters(parameters)
            bound = bundle.binding
            if (
                bound.factory is not factory
                or bound.buffers is not buffers
                or bound.transform_frame_index is not None
                or bound.query.frame_id != index + 1
                or bound.output_profile.value != spec.output_profile
                or bound.shadows != spec.shadows
                or bound.traversal_diagnostics != bool(expected)
                or bound.query.num_rays != spec.num_rays
                or bound.camera.width != spec.width
                or bound.camera.height != spec.height
                or bound.bvh.num_primitives != spec.primitive_count
                or bound.bvh.num_nodes != spec.node_capacity
            ):
                raise ValueError("Warp final frames differ from their slot/mode/sequence")
        outputs = tuple(
            _storage_signature(name, value, device, "warp-slot-owner:0") for name, value in buffers.arrays
        )
        # The renderer validates output intervals against every actual input.
        # Group allocation roots once so views do not inflate memory accounting.
        output_roots = {s.storage_token: s.nbytes for s in outputs}
        event_tokens = tuple(f"warp-event:{id(e):x}" for e in events)
        slot = NativeOpticalSlotResourceManifest(
            schema_version=NATIVE_OPTICAL_SLOT_RESOURCE_SCHEMA_VERSION,
            slot_id=0,
            slot_storage_manifest_sha256=_digest(
                {"arrays": [s.to_payload() for s in outputs], "events": list(event_tokens)}
            ),
            terminal_event_token=event_tokens[0],
            resource_tokens=(*output_roots, *event_tokens),
            resource_owner_tokens=("warp-slot-owner:0",) * (len(output_roots) + len(events)),
            resource_nbytes=(*output_roots.values(), *((0,) * len(events))),
        )
        input_roots = {}
        for name, value in prepared.arrays.items():
            storage = _storage_signature(name, value, device, "warp-session-owner")
            if storage.storage_token in output_roots:
                raise ValueError("Warp arena output aliases an input allocation")
            input_roots[storage.storage_token] = storage.nbytes
        owner_token = f"warp-session-control:{id(factory):x}"
        manifest = NativeOpticalArenaResourceManifest(
            schema_version=NATIVE_OPTICAL_RESOURCE_MANIFEST_SCHEMA_VERSION,
            arena_spec_sha256=spec.sha256,
            session_resource_tokens=(*input_roots, owner_token),
            session_resource_owner_tokens=("warp-session-owner",) * (len(input_roots) + 1),
            session_resource_nbytes=(*input_roots.values(), 0),
            slots=(slot,),
            total_resource_nbytes=sum(input_roots.values()) + slot.total_resource_nbytes,
        )
        owners = {
            "warp-session-owner": (factory, prepared, readiness, runtime, device, stream),
            "warp-slot-owner:0": (frames, buffers, events),
        }
        control = NativeOpticalArena(spec=spec, resource_manifest=manifest, resource_owners=owners)
        bindings = tuple(_bind_r0_c3(spec, p, b, outputs) for p, b in frames)
        for name, value in (
            ("_control", control),
            ("_runtime", runtime),
            ("_device", device),
            ("_stream", stream),
            ("_stream_handle", stream.cuda_stream),
            ("_frames", frames),
            ("_bindings", bindings),
            ("_completion", completion_event),
            ("_diagnostics", diagnostic_events),
            ("_prepared", {}),
            ("_transactions", {}),
        ):
            object.__setattr__(self, name, value)

    @property
    def spec(self):
        return self._control.spec

    @property
    def resource_manifest(self):
        return self._control.resource_manifest

    def prepare_sample(self, *, render_sample_index, planned_acquisition_offset=0):
        from .warp_timeline import _WarpArenaTransaction

        if type(render_sample_index) is not int or not 0 <= render_sample_index < len(self._frames):
            raise ValueError("Warp sample is outside its final recorded frames")
        parameters, bundle = self._frames[render_sample_index]
        bundle.validate_resolved_parameters(parameters)
        binding = self._bindings[render_sample_index]
        plan = self._control.prepare_plan(
            transform_frame_index=None,
            c3_one_shot_binding_sha256=binding.sha256,
            ordered_stage_evidence_sha256=_digest([v.to_payload() for v in binding.ordered_stage_evidence]),
            dataflow_binding_sha256=binding.dataflow_binding_sha256,
            planned_acquisition_offset=planned_acquisition_offset,
        )
        transaction = _WarpArenaTransaction(
            arena=self,
            bundle=bundle,
            parameters=parameters,
            runtime=self._runtime,
            target=self._device,
            stream=self._stream,
            completion_event=self._completion,
            diagnostic_events=self._diagnostics,
        )
        sample = WarpPreparedArenaSample(self, plan, transaction, binding)
        self._prepared[sample] = (plan, transaction, binding)
        self._transactions[transaction] = sample
        return sample

    def acquire(self, sample):
        if type(sample) is not WarpPreparedArenaSample or sample._arena is not self:
            raise NativeOpticalArenaProtocolError("Warp prepared sample is foreign")
        record = self._prepared.get(sample)
        if (
            record is None
            or sample._plan is not record[0]
            or sample._transaction is not record[1]
            or sample._binding is not record[2]
            or sample._state != "PREPARED"
        ):
            raise NativeOpticalArenaProtocolError("Warp sample is copied, consumed or rebound")
        lease = self._control.acquire(sample._plan)
        if isinstance(lease, NativeOpticalArenaBackpressure):
            return lease
        object.__setattr__(sample, "_lease", lease)
        object.__setattr__(sample, "_state", "ACQUIRED")
        del self._prepared[sample]
        return sample

    def _require_active_transaction(self, transaction):
        self._validate_native_scope()
        sample = self._transactions.get(transaction)
        if (
            sample is None
            or sample._transaction is not transaction
            or sample._arena is not self
            or sample._operation not in ("enqueue", "finish")
            or sample._state != ("ACQUIRED" if sample._operation == "enqueue" else "ENQUEUED")
            or sample._lease is not sample._plan._live_lease
            or sample._lease.state != "ENQUEUING"
        ):
            raise NativeOpticalArenaProtocolError("Warp transaction has no exact active lease scope")
        if (
            transaction.target is not self._device
            or transaction.stream is not self._stream
            or transaction.runtime is not self._runtime
            or transaction._terminal.event is not self._completion
            or self._stream.device is not self._device
            or self._stream.cuda_stream != self._stream_handle
        ):
            raise NativeOpticalArenaProtocolError("Warp transaction native scope changed")

    def _validate_native_scope(self):
        if (
            self._stream.device is not self._device
            or self._stream.cuda_stream != self._stream_handle
            or any(
                type(event) is not self._runtime.Event
                or event.device is not self._device
                or event.cuda_event != handle
                for event, handle in self._event_bindings
            )
        ):
            raise NativeOpticalArenaProtocolError("Warp stream or native event binding changed")

    def lifecycle_evidence(self):
        return self._control.lifecycle_evidence()


@dataclass(frozen=True, eq=False, slots=True)
class WarpPreparedArenaSample:
    """Preallocated one-shot transaction and lease binding, never a serialized token."""

    _arena: WarpNativeOpticalArena
    _plan: object
    _transaction: object
    _binding: object
    _lease: object = None
    _state: str = "PREPARED"
    _operation: object = None
    _device_scope: object = field(init=False, repr=False)
    _stream_scope: object = field(init=False, repr=False)

    def __post_init__(self):
        # Runtime context objects are prepared before acquire, not allocated per
        # native launch. The worker's custody graph retains the sample itself.
        object.__setattr__(self, "_device_scope", self._arena._runtime.ScopedDevice(self._arena._device))
        object.__setattr__(self, "_stream_scope", self._arena._runtime.ScopedStream(self._arena._stream))

    @property
    def state(self):
        return self._state

    @property
    def completion(self):
        if self._state not in ("COMPLETION_RECORDED", "PAYLOAD_READY", "RECLAIMABLE", "RECLAIMED"):
            raise NativeOpticalArenaProtocolError("Warp completion is not recorded")
        return self._transaction.completion

    def _require(self, state):
        if (
            self._state != state
            or self._arena._transactions.get(self._transaction) is not self
            or self._lease is not self._plan._live_lease
        ):
            raise NativeOpticalArenaProtocolError("Warp active sample is stale, foreign or out of order")
        self._arena._control._slot_for_live_lease(self._lease)

    def _fail(self, error):
        try:
            self._lease.mark_submitted_failure(f"Warp execution failed: {error}")
        except BaseException:
            pass
        object.__setattr__(self, "_state", "POISONED")

    def enqueue(self):
        self._require("ACQUIRED")
        try:
            self._lease.begin_enqueue()
            object.__setattr__(self, "_operation", "enqueue")
            with self._device_scope, self._stream_scope:
                self._transaction.enqueue()
            object.__setattr__(self, "_state", "ENQUEUED")
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            object.__setattr__(self, "_operation", None)

    def finish(self):
        self._require("ENQUEUED")
        try:
            object.__setattr__(self, "_operation", "finish")
            with self._device_scope, self._stream_scope:
                completion = self._transaction.finish()
            self._lease.record_completion()
            object.__setattr__(self, "_state", "COMPLETION_RECORDED")
            return completion
        except BaseException as error:
            self._fail(error)
            raise
        finally:
            object.__setattr__(self, "_operation", None)

    def _validate_resolution(self):
        self._require("COMPLETION_RECORDED")
        arena, tx = self._arena, self._transaction
        arena._validate_native_scope()
        tx.bundle.validate_resolved_parameters(tx.parameters)
        if (
            self._lease.state != "COMPLETION_RECORDED"
            or tx.runtime is not arena._runtime
            or tx.target is not arena._device
            or tx.stream is not arena._stream
            or tx.stream.device is not arena._device
            or tx.stream.cuda_stream != arena._stream_handle
            or tx.completion.event is not arena._completion
            or type(arena._completion) is not arena._runtime.Event
            or arena._completion.device is not arena._device
        ):
            raise NativeOpticalArenaProtocolError("Warp completion scope changed")

    def resolve(self, *, _guarded_controller=None):
        self._require("COMPLETION_RECORDED")
        try:
            self._validate_resolution()
            if _guarded_controller is None:
                self._arena._runtime.synchronize_event(self.completion.event)
            else:
                from optical_worker.guarded_lifecycle import NativeGuardedSetupLifecycle

                controller = _guarded_controller
                if (
                    type(controller) is not NativeGuardedSetupLifecycle
                    or controller._lane_id != "warp"
                    or controller.target_device_ordinal != self._arena._device.ordinal
                    or controller.unknown_device_scope
                    or controller.possible_dirty_device_ordinals != (controller.target_device_ordinal,)
                ):
                    raise ValueError("Warp resolution requires its retained guarded target")
                controller.observe_terminal_completion(
                    owner=self,
                    transaction=self._transaction,
                    event=self.completion.event,
                    stream=self._arena._stream,
                    target=self._arena._device,
                    runtime=self._arena._runtime,
                )
            self._validate_resolution()
            self._lease.resolve_terminal_completion()
            object.__setattr__(self, "_state", "PAYLOAD_READY")
        except BaseException as error:
            self._fail(error)
            raise

    def release_device_result(self):
        self._require("PAYLOAD_READY")
        self._lease.release_device_result()
        object.__setattr__(self, "_state", "RECLAIMABLE")

    def reclaim(self):
        self._require("RECLAIMABLE")
        self._lease.reclaim()
        object.__setattr__(self, "_state", "RECLAIMED")

    def cancel_before_enqueue(self):
        self._require("ACQUIRED")
        self._lease.cancel_before_enqueue()
        object.__setattr__(self, "_state", "CANCELLED")
