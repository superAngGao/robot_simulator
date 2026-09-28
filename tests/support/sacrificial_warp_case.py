"""Fixed disposable Warp correctness case; imported only after guardian creation.

All rendering mathematics comes from the production recorded bundle. This
module only builds registered caller-owned resources and independent CPU truth.
It is not an executor, arena, benchmark lane or publicly callable worker API.
"""

from __future__ import annotations

import os
import weakref
from contextlib import nullcontext
from types import MappingProxyType, SimpleNamespace

from optical_worker.lifecycle import DeviceScopeDescriptor
from tests.support.sacrificial_lifecycle import State


class FixedWarpSetup:
    def __init__(self, guardian, channel, journal, *, target_ordinal, fault, fault_position):
        self.guardian = guardian
        self.channel = channel
        self.journal = journal
        self.ordinal = target_ordinal
        self.fault = fault
        self.fault_position = fault_position
        self.wp = None
        self.target = None
        self.stream = None
        self.uploads = 0
        self.records = 0
        self.owner_refs = []
        self.np = None
        self._watched_ids = set()

    def watch_owner(self, owner, label):
        # Observes the actual allocation owner, not a surrogate lifecycle flag.
        # The callback does not own the allocation or perform CUDA cleanup.
        supported = (self.wp is not None and type(owner) is self.wp.array) or (
            self.np is not None and type(owner) is self.np.ndarray
        )
        if supported and id(owner) not in self._watched_ids:
            self._watched_ids.add(id(owner))
            finalizer = weakref.finalize(owner, self.journal, "allocation_owner_finalized", label=label)
            finalizer.atexit = False  # Only actual collection, not interpreter-exit notifications.
            self.owner_refs.append((label, weakref.ref(owner)))

    def assert_owner_retention(self):
        import gc

        gc.collect()
        for label, reference in self.owner_refs:
            owner = reference()
            assert owner is not None, f"registered CUDA allocation lost: {label}"
            assert self.guardian._guardian.contains_exact_owner(owner), label
        self.journal("allocation_custody_checked", count=len(self.owner_refs))

    def transport_hold(self, location):
        if self.fault == f"hold_{location}":
            import time

            self.journal("transport_hold", location=location)
            while True:
                time.sleep(60)

    def exit_failure(self, reason):
        # Never unwind a partially submitted graph into a caller/pytest process.
        try:
            if self.guardian.state is not State.QUARANTINED:
                if self.guardian.state is not State.POISONED:
                    self.guardian.poison(reason)
                if self.wp is None or self.target is None:
                    self.guardian.quarantine("runtime_target_unavailable")
                else:
                    self.guardian.emergency_drain(self.channel, self._emergency_sync)
            self.journal("failure_disposition", reason=reason, evidence=self.guardian.snapshot())
            if self.guardian.state is State.SYNCED:
                self.journal("release_begin", evidence=self.guardian.snapshot())
                self.guardian.release()
                self.journal("failure_release_authorized", evidence=self.guardian.snapshot())
        except BaseException:
            self.guardian.quarantine("failure_disposition_failed")
            self.journal("quarantined", evidence=self.guardian.snapshot())
        finally:
            os._exit(74)

    def _emergency_sync(self):
        self.journal("emergency_sync_entered")
        if self.fault == "emergency_raise":
            raise RuntimeError("fixed emergency failure")
        if self.fault == "emergency_hang":
            import time

            while True:
                time.sleep(60)
        self.wp.synchronize_device(self.target)

    def leaf(self, operation, function, *, produces_owner=False, kind="allocation"):
        permit = self.guardian.begin_setup_leaf(
            DeviceScopeDescriptor.exact_target(
                operation_id="".join(f"_{c.lower()}" if c.isupper() else c for c in operation),
                leaf_kind=kind,
                device_ordinal=self.ordinal,
                scope_source="fixed_sacrificial_warp_case",
                runtime_identity="warp_explicit_device_v1",
                unreturned_resource_possible=produces_owner,
            )
        )
        try:
            scope = (
                self.wp.ScopedStream(self.stream)
                if self.stream is not None
                else self.wp.ScopedDevice(self.target)
                if self.target is not None
                else nullcontext()
            )
            with scope:
                result = function()
            self.guardian.finish_setup_leaf(permit, owners=(result,) if produces_owner else ())
            if produces_owner:
                self.watch_owner(result, operation)
            return result
        except BaseException as exc:
            if self.guardian.state is State.SETUP:
                self.guardian.fail_setup_leaf(permit, partial_owners=(exc,))
            self.exit_failure(operation)

    def host(self, owner, role="host_setup"):
        self.guardian.adopt_host(owner, role=role)
        self.watch_owner(owner, role)
        return owner

    def readback_input(self, value, manifest):
        # No .numpy()/to()/clone(): each physical destination and CPU view is
        # retained before copy, and before the next array's GPU leaf.
        host = self.np.empty(manifest.shape, dtype=self.np.dtype(manifest.dtype))
        self.host(host, "input_readback_storage")
        if self.fault == "readback_storage_failure":
            raise RuntimeError("fixed failure after readback host storage adoption")
        view = self.wp.array(host, dtype=value.dtype, device="cpu", copy=False)
        self.host(view, "input_readback_view")
        if self.fault == "readback_transfer_failure":
            raise RuntimeError("fixed failure after readback view transfer, before GPU copy")
        assert self.guardian._guardian.contains_exact_owner(host)
        assert self.guardian._guardian.contains_exact_owner(view)
        self.journal("input_readback_custody", name=manifest.name)
        if self.fault == "readback_before_copy_failure":
            self.exit_failure("readback_before_copy_failure")

        def copy():
            self.wp.copy(view, value, stream=self.stream)
            if self.fault == "readback_submit_then_raise":
                raise RuntimeError("fixed input readback copy submitted then raised")

        self.leaf(f"readback_{manifest.name}", copy, kind="device_to_host_copy")
        if self.fault == "readback_return_failure":
            raise RuntimeError("fixed failure after registered readback copy, before host return")
        return host

    def setup_wait(self, operation, wait):
        # Intermediate upload/validation waits also require ACK; final seal uses
        # the separate final guardian setup barrier after all setup operations.
        def guarded():
            try:
                ticket = self.channel.arm("S")
                self.journal("setup_wait_entered", operation_id=operation)
                wait()
                self.channel.complete(ticket)
            except BaseException:
                self.guardian.quarantine("setup_wait_failed")
                self.journal("quarantined", evidence=self.guardian.snapshot())
                os._exit(74)

        return self.leaf(operation, guarded, kind="event_wait")

    def initialize(self):
        def runtime_import():
            import warp as wp

            if self.fault == "runtime_import_failure":
                raise RuntimeError("fixed import failure before owner transfer")
            return wp

        self.wp = self.leaf("warp_import", runtime_import, produces_owner=True, kind="runtime_import_or_init")
        self.target = self.leaf(
            "warp_target",
            lambda: self.wp.get_device(f"cuda:{self.ordinal}"),
            produces_owner=True,
            kind="runtime_import_or_init",
        )
        assert self.target.ordinal == self.ordinal and self.target.is_cuda
        self.stream = self.leaf(
            "warp_stream", lambda: self.wp.Stream(device=self.target), produces_owner=True
        )
        assert self.stream.owner is True

    def upload(self, host, *, name, dtype):
        wp = self.wp
        self.host(host, "host_staging")
        host_view = self.host(wp.array(host, dtype=dtype, device="cpu", copy=False), "host_view")
        destination = self.leaf(
            f"allocate_{name}",
            lambda: wp.empty(shape=tuple(host.shape), dtype=dtype, device=self.target),
            produces_owner=True,
        )
        self.leaf(
            f"upload_{name}",
            lambda: wp.copy(destination, host_view, stream=self.stream),
            kind="host_to_device_copy",
        )
        self.uploads += 1
        if self.uploads == 1:
            self.transport_hold("upload")
        if self.fault == "first_upload_failure" and self.uploads == 1:
            self.exit_failure("after_first_upload_before_ready")
        return destination

    def allocate(self, shape, dtype, *, name):
        def allocate_owner():
            owner = self.wp.empty(shape=shape, dtype=dtype, device=self.target)
            if self.fault == "allocation_return_failure":
                self.watch_owner(owner, "unreturned_allocation")
                raise RuntimeError("fixed allocation failure before owner return")
            return owner

        return self.leaf(
            f"allocate_{name}",
            allocate_owner,
            produces_owner=True,
        )

    def record(self, original_launch, kernel, *args, **kwargs):
        if kwargs.get("record_cmd") is not True:
            raise RuntimeError("fixed recording hook cannot submit an ordinary launch")
        position = self.records

        def construct():
            command = original_launch(kernel, *args, **kwargs)
            if self.fault == "record_failure" and position == self.fault_position:
                raise RuntimeError("fixed unreturned recording owner")
            return command

        command = self.leaf(f"record_{position}", construct, produces_owner=True, kind="compile_or_jit")
        self.records += 1
        if position == 0:
            self.transport_hold("record")
        return command


def build_case(driver: FixedWarpSetup, *, profile, shadows, diagnostics, resident_rays=None):
    """Untimed, registered allocation/recording over one exact canonical source."""
    driver.initialize()
    wp, target, stream = driver.wp, driver.target, driver.stream

    def imports():
        import numpy as np

        from benchmarks.optical import native as native_input
        from benchmarks.optical import workloads
        from optics import warp_native_render as native_render

        return SimpleNamespace(
            np=np, native_input=native_input, workloads=workloads, native_render=native_render
        )

    modules = driver.leaf(
        "render_module_imports", imports, produces_owner=True, kind="runtime_import_or_init"
    )
    np, a2, workloads, native = modules.np, modules.native_input, modules.workloads, modules.native_render
    driver.np = np
    from optical_worker.native_contract import (
        WarpPreparedNativeInput,
        register_prepared_native_input,
        warp_dtype_from_host_dtype,
    )
    from optics.device_bvh import DeviceBvhBuildStats, DeviceOpticalBvh
    from optics.device_scene import WarpWorldSceneBuffers
    from optics.execution import OpticalOutputProfile
    from optics.output_contract import SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
    from physics.spatial import SpatialTransform
    from sensing.optical import OpticalPinholeCameraSpec

    packed = driver.host(workloads.build_canonical_packed_scene("asymmetric_synthetic_7x3_v1"))
    fixture = driver.host(workloads.build_canonical_fixed_fixture(packed))
    source = driver.host(
        a2.CanonicalNativeInput.from_scene_contract(
            packed,
            input_contract="fixed_native_scene_v1",
            fixed_fixture=fixture,
        )
    )
    arrays = {}
    for spec in source.manifest.arrays:
        host = np.array(source.arrays[spec.name], copy=True, order="C")
        arrays[spec.name] = driver.upload(
            host, name=spec.name, dtype=warp_dtype_from_host_dtype(wp, host.dtype)
        )
    event = driver.leaf(
        "ready_event", lambda: wp.Event(device=target), produces_owner=True, kind="event_create"
    )
    driver.leaf("record_ready", lambda: stream.record_event(event), kind="event_record")
    driver.setup_wait("input_ready", lambda: wp.synchronize_event(event))
    prepared = driver.host(
        WarpPreparedNativeInput(
            manifest=source.manifest,
            device_ordinal=driver.ordinal,
            arrays=MappingProxyType(arrays),
            ready_event=event,
            source=source,
        )
    )
    register_prepared_native_input(prepared, source)
    observations = tuple(
        (manifest.name, driver.readback_input(arrays[manifest.name], manifest))
        for manifest in source.manifest.arrays
    )
    driver.setup_wait("input_validation_ready", lambda: wp.synchronize_stream(stream))
    validation = driver.host(a2._validate_native_input_host_observations(source, prepared, observations))
    module_owner = driver.host(wp.get_module(native.__name__), "warp_module")
    driver.host(native._render_kernel_owners(), "kernel_owners")

    def compile_module():
        wp.load_module(module=native.__name__, device=target)
        if driver.fault == "compile_failure":
            raise RuntimeError("fixed compilation failure after module load")
        return module_owner

    driver.leaf("compile_module", compile_module, produces_owner=True, kind="compile_or_jit")
    readiness = driver.host(
        a2.certify_native_readiness(prepared, validation, compile_complete=True, warmup_frames_completed=0)
    )
    scene = driver.host(a2.build_prepared_warp_local_scene(prepared))
    world_names = (
        "triangle_v0_world",
        "triangle_e1_world",
        "triangle_e2_world",
        "triangle_normal_world",
        "plane_normal_world",
        "plane_point_world",
        "triangle_aabb_min",
        "triangle_aabb_max",
    )
    world = driver.host(
        WarpWorldSceneBuffers(
            device=target, **{name: arrays[name] for name in world_names}, resources=(prepared,)
        )
    )
    host = fixture.arrays
    levels = tuple(tuple(int(v) for v in row) for row in host["bvh_level_ranges"])
    stats = DeviceBvhBuildStats(
        num_nodes=len(host["bvh_left"]),
        num_leaves=int((host["bvh_count"] > 0).sum()),
        max_depth=len(levels) - 1 if levels else 0,
        leaf_size=4,
        sah_quality_cost=0.0,
        level_ranges=levels,
    )
    bvh = driver.host(
        DeviceOpticalBvh(
            device=target,
            frame_id=0,
            env_idx=0,
            stats=stats,
            **{
                name: arrays[f"bvh_{name}"]
                for name in (
                    "bounds_min",
                    "bounds_max",
                    "left",
                    "right",
                    "start",
                    "count",
                    "node_depth",
                    "level_ranges",
                    "prim_ids",
                )
            },
            primitive_global_id=scene.triangle_primitive_global_id,
            primitive_instance_index=scene.triangle_instance_index,
            primitive_index_within_instance=scene.triangle_primitive_index_within_instance,
            primitive_geometry_index=scene.triangle_geometry_index,
            primitive_geometry_primitive_index=scene.triangle_geometry_primitive_index,
            primitive_source_order_key=scene.triangle_source_order_key,
            scene_contract_sha256=scene.scene_contract_sha256,
            resources=(prepared,),
        )
    )
    angle = 0.03
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    spec = OpticalPinholeCameraSpec(
        frame_id=0,
        sim_time=0.0,
        env_idx=0,
        sensor_id="sacrificial",
        width=31,
        height=13,
        fx=40.0,
        fy=43.0,
        cx=15.2,
        cy=5.7,
        max_distance=50.0,
        sensor_role="rgb",
        X_world_camera=SpatialTransform(rotation, np.array([0.01, -0.01, 0.0])),
    )
    camera = native.WarpCameraParameters.from_spec(spec, role_table=scene.role_table)
    n = camera.query.num_rays if resident_rays is None else resident_rays
    outputs = {}
    for name, dtype, shape in native.warp_direct_light_buffer_layout(n, traversal_diagnostics=diagnostics):
        outputs[name] = driver.allocate(shape, getattr(wp, dtype), name=name)
    overlay = tuple(outputs.pop(name) for name in SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES if diagnostics)
    buffers = driver.host(
        native.WarpDirectLightBuffers(
            num_rays=n,
            traversal_diagnostics=diagnostics,
            traversal_counters=overlay,
            **outputs,
        )
    )
    factory = driver.host(
        native.WarpDirectLightRenderCommandFactory(
            prepared,
            readiness,
            scene,
            target,
            stream,
        )
    )
    original_launch = wp.launch
    wp.launch = lambda kernel, *args, **kwargs: driver.record(original_launch, kernel, *args, **kwargs)
    try:
        options = dict(
            world=world,
            bvh=bvh,
            buffers=buffers,
            output_profile=OpticalOutputProfile(profile),
            shadows=shadows,
            traversal_diagnostics=diagnostics,
            ambient_rgb=(0.03, 0.05, 0.07),
            background_rgb=(0.02, 0.03, 0.07),
        )
        if resident_rays is None:
            bundle = factory.seal_camera(camera=camera, **options)
        else:
            from dataclasses import replace

            bundle = factory.seal_device_rays(query=replace(camera.query, num_rays=n), **options)
    finally:
        wp.launch = original_launch
    driver.host(bundle, "recorded_bundle")
    bundle.validate_bound()
    return SimpleNamespace(
        bundle=bundle,
        buffers=buffers,
        camera_spec=spec,
        modules=modules,
        prepared=prepared,
        outputs=outputs,
        overlay=overlay,
        scene=scene,
        source=source,
        profile=profile,
        shadows=shadows,
        diagnostics=diagnostics,
    )
