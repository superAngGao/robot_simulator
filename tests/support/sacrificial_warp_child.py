"""Fixed one-shot Warp validation child. No public/product dispatch surface."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import sysconfig
import time
from pathlib import Path


def main():
    if os.name != "posix" or not sys.flags.isolated or not sys.flags.no_site or len(sys.argv) != 4:
        return 64
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    source_manifest = payload["sources"]
    if Path(source_manifest["repository_root"]).resolve() != root:
        return 65
    for item in source_manifest["sources"]:
        path = root / item["path"]
        if (
            not path.resolve().is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            return 65
    # Before the first repository import (including the loader itself), use a
    # fresh empty cache namespace. -B alone would still read existing .pyc.
    bootstrap_cache = Path(sys.argv[1]).resolve().parent / "child-source-cache"
    bootstrap_cache.mkdir(exist_ok=False)
    sys.pycache_prefix = str(bootstrap_cache)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from tests.support.sacrificial_lifecycle import SacrificialRenderGuardian, State

    guardian = SacrificialRenderGuardian(session_id=payload["session"], target_ordinal=payload["target"])
    journal_fd = int(sys.argv[3])

    def journal(operation, **facts):
        try:
            encoded = (
                json.dumps(dict(operation=operation, **facts), allow_nan=False, sort_keys=True) + "\n"
            ).encode()
            if os.write(journal_fd, encoded) != len(encoded):
                os._exit(74)
        except BaseException:
            # The journal is the last diagnostic channel, not cleanup authority.
            # Encoding/write failure after a GPU leaf must never unwind owners.
            os._exit(74)

    journal(
        "guardian_created",
        imported_cuda_modules=[
            name for name in sys.modules if name.split(".")[0] in {"warp", "torch", "optics"}
        ],
    )
    if any(name.split(".")[0] in {"warp", "torch", "optics"} for name in sys.modules):
        os._exit(74)
    sys.path.append(sysconfig.get_paths()["purelib"])
    from tests.support.sacrificial_sources import RepositoryImportLedger
    from tests.support.sacrificial_watchdog import GuardedOperationChannel

    ledger = RepositoryImportLedger(source_manifest)
    ledger.install()
    channel = GuardedOperationChannel(
        socket.socket(fileno=int(sys.argv[2])), session=payload["session"], ack_timeout=2.0
    )
    from tests.support.sacrificial_warp_case import FixedWarpSetup, build_case

    driver = FixedWarpSetup(
        guardian,
        channel,
        journal,
        target_ordinal=payload["target"],
        fault=payload["fault"],
        fault_position=payload["position"],
    )
    try:
        case = build_case(
            driver,
            profile=payload["profile"],
            shadows=payload["shadows"],
            diagnostics=payload["diagnostics"],
            resident_rays=payload["resident_rays"],
        )
        wp, np = driver.wp, case.modules.np
        ambient_device = wp.get_device()
        from tests.support.sacrificial_leaf_audit import RecordedLeafAudit

        current_stream_fixture = driver.leaf(
            "current_stream_fixture", lambda: wp.Stream(device=driver.target), produces_owner=True
        )
        driver.leaf(
            "install_current_stream_fixture",
            lambda: driver.target.set_stream(current_stream_fixture, sync=False),
            kind="runtime_import_or_init",
            scope_stream=False,
        )
        assert current_stream_fixture is not driver.stream

        def context_identity():
            # Do not call get_stream on untouched ambient hardware: that getter
            # creates a context/stream. These are existing Warp 1.12 CPU fields.
            target_current = driver.target._stream
            ambient_current = ambient_device._stream
            return (
                wp.get_device(),
                target_current,
                None if target_current is None else target_current.cuda_stream,
                ambient_current,
                None if ambient_current is None else ambient_current.cuda_stream,
                driver.stream,
                driver.stream.cuda_stream,
            )

        frozen_context = context_identity()
        assert frozen_context[1] is current_stream_fixture
        driver.host(frozen_context, "execution_context_owners")
        context_checks = 0

        def check_context():
            nonlocal context_checks
            actual = context_identity()
            if any(
                (
                    actual[index] is not frozen_context[index]
                    if index in (0, 1, 3, 5)
                    else actual[index] != frozen_context[index]
                )
                for index in range(len(actual))
            ):
                guardian.quarantine("execution_context_drift")
                journal("context_drift_quarantined", evidence=guardian.snapshot())
                os._exit(74)
            context_checks += 1

        from optics.execution import CpuDirectLightOpticalExecutor
        from optics.scene import OpticalInstanceSnapshot, OpticalSceneSnapshot, build_cpu_bvh_acceleration
        from physics.spatial import SpatialTransform
        from sensing.optical import OpticalRaySensorSpec, build_pinhole_camera_rays

        # Independent CPU executor, no shared Warp math. Deterministic canonical
        # registry is CPU truth only; renderer allocations remain the single a2 source.
        registry = case.modules.workloads._asymmetric_registry()
        material_indices = {name: index for index, name in enumerate(registry.materials)}
        instances = tuple(
            OpticalInstanceSnapshot(
                instance_id=instance.instance_id,
                numeric_instance_id=instance.numeric_instance_id,
                geometry_id=instance.geometry_id,
                material=registry.materials[instance.material_id],
                material_index=material_indices[instance.material_id],
                geometry=registry.geometry[instance.geometry_id],
                X_world_geometry=SpatialTransform.identity(),
                roles=instance.roles,
            )
            for instance in registry.instances
        )
        snapshot = OpticalSceneSnapshot(
            0,
            0.0,
            0,
            instances,
            tuple(registry.lights.values()),
            acceleration=build_cpu_bvh_acceleration(instances),
        )
        rays = build_pinhole_camera_rays(case.camera_spec)
        if payload["resident_rays"] is not None:
            n = payload["resident_rays"]
            rays = OpticalRaySensorSpec(
                frame_id=0,
                sim_time=0.0,
                env_idx=0,
                sensor_id="sacrificial",
                origins_world=rays.origins_world[:n],
                directions_world=rays.directions_world[:n],
                sensor_role="rgb",
                max_distance=50.0,
            )
            for name, values in (("origins", rays.origins_world), ("directions", rays.directions_world)):
                host = driver.host(np.asarray(values, dtype=np.float32).copy())
                view = driver.host(wp.array(host, dtype=wp.float32, device="cpu", copy=False))
                driver.leaf(
                    f"resident_{name}",
                    lambda name=name, view=view: wp.copy(
                        getattr(case.buffers, name), view, stream=driver.stream
                    ),
                    kind="host_to_device_copy",
                )
        truth = CpuDirectLightOpticalExecutor(
            shadows=payload["shadows"], ambient_rgb=(0.03, 0.05, 0.07), background_rgb=(0.02, 0.03, 0.07)
        ).execute(snapshot, rays)
        driver.host(truth, "cpu_oracle")
        driver.host(rays, "host_rays")
        # Warmup uses the very same production kernels and bound arguments, not
        # recorded execution before seal. Its owners are already registered.
        for index, descriptor in enumerate(case.bundle.descriptors):
            driver.leaf(
                f"warmup_{index}",
                lambda descriptor=descriptor: wp.launch(
                    descriptor.kernel,
                    dim=descriptor.dim,
                    inputs=descriptor.arguments,
                    device=driver.target,
                    stream=driver.stream,
                ),
                kind="warmup_launch",
            )
            if payload["fault"] == "warmup_failure" and index == payload["position"]:
                driver.exit_failure("warmup_failure")
        for name, value in case.buffers.arrays:
            if payload["resident_rays"] is not None and name in ("origins", "directions"):
                continue
            driver.leaf(f"sentinel_{name}", lambda value=value: value.fill_(77), kind="warmup_launch")
        host_results = {}
        host_views = {}
        for name, value in case.buffers.arrays:
            dtype = np.int32 if value.dtype is wp.int32 else np.float32
            host_results[name] = driver.host(np.empty(value.shape, dtype=dtype), "inspection_storage")
            host_views[name] = driver.host(
                wp.array(host_results[name], dtype=value.dtype, device="cpu", copy=False), "inspection_view"
            )
        first_gpu_results, first_host_results, first_host_views = {}, {}, {}
        if payload["repetitions"] == 2:
            for name, value in case.buffers.arrays:
                first_gpu_results[name] = driver.leaf(
                    f"first_snapshot_{name}",
                    lambda value=value: wp.empty(value.shape, dtype=value.dtype, device=driver.target),
                    produces_owner=True,
                )
                dtype = np.int32 if value.dtype is wp.int32 else np.float32
                first_host_results[name] = driver.host(
                    np.empty(value.shape, dtype=dtype), "first_inspection_storage"
                )
                first_host_views[name] = driver.host(
                    wp.array(first_host_results[name], dtype=value.dtype, device="cpu", copy=False),
                    "first_inspection_view",
                )
                assert first_gpu_results[name] is not value
                if value.size:
                    assert first_gpu_results[name].ptr != value.ptr
        audit_event = driver.leaf(
            "audit_event",
            lambda: wp.Event(
                device=driver.target,
                interprocess=payload["fault"] in ("audit_warp_event_import", "audit_generic_event_import"),
            ),
            produces_owner=True,
            kind="event_create",
        )
        event_controls = driver.host({}, "audit_event_controls")
        if payload["fault"] == "audit_warp_event_import":
            event_controls["warp_ipc"] = driver.leaf(
                "audit_warp_ipc", audit_event.ipc_handle, produces_owner=True, kind="event_create"
            )
        generic_control = payload["fault"].startswith("audit_generic_")
        if payload["fault"].startswith("audit_torch_") or generic_control:
            torch = driver.leaf(
                "audit_torch_import",
                lambda: __import__("torch"),
                produces_owner=True,
                kind="runtime_import_or_init",
            )
            event_controls["torch"] = torch
            event_controls["stream"] = driver.leaf(
                "audit_torch_stream",
                lambda: (
                    torch.Stream(device=f"cuda:{driver.ordinal}")
                    if generic_control
                    else torch.cuda.ExternalStream(driver.stream.cuda_stream, device=driver.ordinal)
                ),
                produces_owner=True,
            )
            event_controls["event"] = driver.leaf(
                "audit_torch_event",
                lambda: (
                    torch.Event(
                        device=f"cuda:{driver.ordinal}",
                        interprocess=payload["fault"]
                        in ("audit_generic_event_import", "audit_generic_event_export"),
                        enable_timing=payload["fault"] == "audit_generic_event_elapsed",
                    )
                    if generic_control
                    else torch.cuda.Event(
                        interprocess=payload["fault"]
                        in ("audit_torch_event_import", "audit_torch_event_export"),
                        enable_timing=payload["fault"] == "audit_torch_event_elapsed",
                    )
                ),
                produces_owner=True,
                kind="event_create",
            )
            # Both stream and event are adopted before recording. The CUDA
            # wrapper uses a retained Warp stream view; generic Torch has its
            # own explicitly target-owned stream. S quiesces the target graph.
            driver.leaf(
                "audit_torch_event_record",
                lambda: event_controls["event"].record(event_controls["stream"]),
                kind="event_record",
            )
            if payload["fault"] in ("audit_torch_event_import", "audit_generic_event_import"):
                event_controls["torch_ipc"] = driver.leaf(
                    "audit_torch_ipc",
                    # Torch 2.9 generic Event.ipc_handle is a public stub that
                    # raises NotImplementedError. Supply a genuine CUDA IPC
                    # handle from the already-owned target Warp event instead;
                    # this tests rejection, not generic IPC functionality.
                    audit_event.ipc_handle if generic_control else event_controls["event"].ipc_handle,
                    produces_owner=True,
                    kind="event_create",
                )
            if (
                (not generic_control and event_controls["stream"].cuda_stream != driver.stream.cuda_stream)
                or event_controls["stream"].device.index != driver.ordinal
                or event_controls["event"].device.index != driver.ordinal
            ):
                guardian.quarantine("audit_control_device_or_stream_drift")
                journal("audit_control_quarantined", evidence=guardian.snapshot())
                os._exit(74)
            if payload["fault"] == "audit_generic_event_cached_create":
                event_controls["event_type"] = driver.host(torch.Event, "generic_constructor_alias")
            if payload["fault"] == "audit_generic_event_cached_query":
                event_controls["cached_query"] = driver.host(
                    event_controls["event"].query, "generic_cached_native_method"
                )
        for owner in event_controls.values():
            assert guardian._guardian.contains_exact_owner(owner)
        journal("audit_event_controls_registered", names=sorted(event_controls), target=driver.ordinal)
        audit = driver.leaf(
            "prepare_leaf_audit",
            lambda: RecordedLeafAudit(
                wp,
                commands=case.bundle.commands,
                stream=driver.stream,
                sample_array=case.buffers.hit_mask,
                event=audit_event,
                fault=payload["fault"],
                event_controls=event_controls,
            ),
            produces_owner=True,
            kind="runtime_import_or_init",
        )
        auxiliary_plan = ()
        physical_arrays = driver.host(dict(case.buffers.arrays), "physical_output_binding")
        if payload["repetitions"] == 2:
            auxiliary_plan = tuple(f"snapshot:{name}" for name, _ in case.buffers.arrays)
            auxiliary_plan += tuple(f"sentinel:{name}" for name, _ in case.buffers.arrays)
        case.bundle.validate_bound()
        journal("setup_import_ledger", paths=list(ledger.verify()))

        def setup_barrier():
            journal("setup_barrier_entered")
            if payload["fault"] == "setup_barrier_raise":
                raise RuntimeError("fixed setup barrier error")
            if payload["fault"] == "setup_barrier_hang":
                while True:
                    time.sleep(60)
            wp.synchronize_device(driver.target)

        guardian.setup_barrier(channel, setup_barrier)
        guardian.seal(
            command_count=len(case.bundle.commands),
            repetitions=payload["repetitions"],
            auxiliary_plan=auxiliary_plan,
        )
        driver.assert_owner_retention()
        journal("guardian_sealed", evidence=guardian.snapshot())
        driver.transport_hold("sealed")
        case.bundle.validate_bound()
        frozen_bundle = case.bundle
        frozen_commands = case.bundle.commands
        for index in range(len(frozen_commands) * payload["repetitions"]):
            if index == len(frozen_commands):
                for auxiliary_index, name in enumerate(auxiliary_plan):
                    guardian.before_auxiliary(name)
                    role, channel_name = name.split(":")
                    value = physical_arrays[channel_name]
                    check_context()
                    try:
                        with wp.ScopedStream(driver.stream, sync_enter=False, sync_exit=False):
                            if role == "snapshot":
                                wp.copy(first_gpu_results[channel_name], value, stream=driver.stream)
                            else:
                                value.fill_(-113)
                        if payload["fault"] == "auxiliary_submit_then_raise" and auxiliary_index == 0:
                            raise RuntimeError("fixed auxiliary copy submitted then raised")
                    finally:
                        check_context()
                    guardian.auxiliary_returned(name)
                assert case.bundle is frozen_bundle and case.bundle.commands is frozen_commands
                case.bundle.validate_bound()
            command = frozen_commands[index % len(frozen_commands)]
            if payload["fault"] == "pre_latch_failure" and index == payload["position"]:
                if index == 0:
                    journal("release_begin", evidence=guardian.snapshot())
                    guardian.release()
                    journal("clean_cancel", evidence=guardian.snapshot())
                    return 0
                driver.exit_failure("pre_latch_after_previous_submission")
            guardian.before_launch(index)
            journal("launch_latched", position=index, attempted=guardian.snapshot()["attempted_launch_count"])
            if (
                payload["fault"] in ("post_latch_failure", "emergency_raise", "emergency_hang")
                and index == payload["position"]
            ):
                driver.exit_failure("post_latch_failure")
            check_context()
            try:
                audit.run(command)
                if payload["fault"] == "context_stream_drift":
                    driver.target.set_stream(driver.stream, sync=False)
                elif payload["fault"] == "context_ambient_drift":
                    # Deliberate CPU-metadata drift, no foreign GPU initialization.
                    ambient_device._stream = driver.stream
            finally:
                journal("recorded_leaf_audit", evidence=audit.evidence())
                check_context()
                journal("execution_context_checked", checks=context_checks)
            if index == 0 and payload["fault"] == "journal_encoding_failure":
                journal("fixed_encoding_failure", invalid_value=object())
            if index == 0 and payload["fault"] == "journal_write_failure":
                os.close(journal_fd)
                journal("fixed_write_failure")
            if index == 0:
                driver.transport_hold("submit")
            if (payload["fault"] == "submit_then_raise" and index == payload["position"]) or (
                payload["fault"] == "second_pass_submit_then_raise"
                and index == len(frozen_commands) + payload["position"]
            ):
                raise RuntimeError("fixed native launch submitted then raised before return confirmation")
            guardian.launch_returned(index)

        def validation_drain():
            journal("validation_drain_entered")
            if payload["fault"] == "validation_raise":
                raise RuntimeError("fixed validation error")
            if payload["fault"] == "validation_hang":
                while True:
                    time.sleep(60)
            wp.synchronize_device(driver.target)
            # Only after render quiescence; destinations were retained before seal.
            for name, value in case.buffers.arrays:
                wp.copy(host_views[name], value, stream=driver.stream)
                if payload["repetitions"] == 2:
                    wp.copy(first_host_views[name], first_gpu_results[name], stream=driver.stream)
            wp.synchronize_device(driver.target)

        guardian.validation_drain(channel, validation_drain)
        driver.assert_owner_retention()
        driver.transport_hold("validated")
        if payload["repetitions"] == 2:
            for name in host_results:
                np.testing.assert_array_equal(first_host_results[name], host_results[name])
        assert audit.native_launches == len(frozen_commands) * payload["repetitions"]
        assert audit.entry_calls == audit.native_launches == audit.restoration_checks
        assert audit.native_events == audit.native_queries == 0 and not audit.rejected
        assert audit.native_event_guard.calls == 0
        observations = (host_results,) if payload["repetitions"] == 1 else (first_host_results, host_results)
        for host_results in observations:
            for name in (
                "hit_mask",
                "range_m",
                "position_world",
                "normal_world",
                "numeric_instance_id",
                "material_index",
                "rgb",
                "intensity",
            ):
                actual, expected = host_results[name], truth.channel(name)
                if name in ("hit_mask", "numeric_instance_id", "material_index"):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
            np.testing.assert_allclose(host_results["origins"], rays.origins_world, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(
                host_results["directions"], rays.directions_world, atol=1e-6, rtol=1e-6
            )
            scalar_counts = {
                name: int(value[0])
                for name, value in host_results.items()
                if name.startswith(("bvh_", "shadow_"))
            }
            assert all(value >= 0 for value in scalar_counts.values())
            assert (
                scalar_counts["bvh_stack_overflow_count"] == scalar_counts["shadow_stack_overflow_count"] == 0
            )
            for name in ("bvh_max_stack_depth", "shadow_max_stack_depth"):
                assert scalar_counts[name] <= case.modules.native_render._MAX_BVH_STACK
            for name in scalar_counts:
                assert host_results[name].dtype == np.int32 and host_results[name].shape == (1,)
            if case.buffers.num_rays == 0:
                assert all(value == 0 for value in scalar_counts.values())
            if not payload["shadows"]:
                assert all(value == 0 for name, value in scalar_counts.items() if name.startswith("shadow_"))
            if payload["diagnostics"]:
                total = scalar_counts["shadow_traversal_ray_count"]
                assert (
                    total
                    == scalar_counts["shadow_traversal_directional_ray_count"]
                    + scalar_counts["shadow_traversal_point_ray_count"]
                )
                assert (
                    total
                    == scalar_counts["shadow_traversal_occluded_count"]
                    + scalar_counts["shadow_traversal_unoccluded_count"]
                )
                assert total <= int(host_results["hit_mask"].sum()) * len(registry.lights)
                assert (
                    scalar_counts["shadow_traversal_leaf_visit_count"]
                    <= scalar_counts["shadow_traversal_node_visit_count"]
                )
                assert scalar_counts["shadow_traversal_node_visit_count"] <= total * len(
                    case.source.arrays["bvh_left"]
                )
                assert scalar_counts["shadow_traversal_triangle_test_count"] <= total * len(
                    case.source.arrays["triangle_v0_world"]
                )
                assert scalar_counts["shadow_traversal_plane_test_count"] <= total * len(
                    case.source.arrays["plane_normal_world"]
                )
                if total == 0:
                    assert scalar_counts["shadow_max_stack_depth"] == 0
        guardian.inspection_completed()
        journal(
            "result_inspected",
            hit_count=int(host_results["hit_mask"].sum()),
            ray_count=case.buffers.num_rays,
            distinct_hit_ids=int(
                np.unique(host_results["numeric_instance_id"][host_results["hit_mask"] != 0]).size
            ),
            distinct_hit_materials=int(
                np.unique(host_results["material_index"][host_results["hit_mask"] != 0]).size
            ),
            ambient_device_ordinal=ambient_device.ordinal,
            recorded_repetitions=payload["repetitions"],
            context_checks=context_checks,
            target_current_is_execution_stream=frozen_context[1] is driver.stream,
            ambient_stream_initialized=frozen_context[3] is not None,
            leaf_audit=audit.evidence(),
            scalar_counts=scalar_counts,
            source_manifest_sha256=case.prepared.manifest.sha256,
            bundle_evidence=case.bundle.evidence.to_payload(),
            import_ledger=list(ledger.verify()),
        )
        journal("release_begin", evidence=guardian.snapshot())
        guardian.release()
        journal("released", evidence=guardian.snapshot())
        return 0
    except BaseException as exc:
        journal("driver_exception", exception_type=type(exc).__name__, message=str(exc))
        if guardian.state in (State.VALIDATED, State.INSPECTED):
            guardian.quarantine("inspection_failed")
            journal("quarantined", evidence=guardian.snapshot())
            os._exit(74)
        driver.exit_failure("driver_exception")


if __name__ == "__main__":
    raise SystemExit(main())
