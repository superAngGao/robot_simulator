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
            if name in ("origins", "directions"):
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
        guardian.seal(command_count=len(case.bundle.commands))
        driver.assert_owner_retention()
        journal("guardian_sealed", evidence=guardian.snapshot())
        driver.transport_hold("sealed")
        case.bundle.validate_bound()
        for index, command in enumerate(case.bundle.commands):
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
            command.launch(stream=driver.stream)
            assert wp.get_device() is ambient_device
            if index == 0 and payload["fault"] == "journal_encoding_failure":
                journal("fixed_encoding_failure", invalid_value=object())
            if index == 0 and payload["fault"] == "journal_write_failure":
                os.close(journal_fd)
                journal("fixed_write_failure")
            if index == 0:
                driver.transport_hold("submit")
            if payload["fault"] == "submit_then_raise" and index == payload["position"]:
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
            wp.synchronize_device(driver.target)

        guardian.validation_drain(channel, validation_drain)
        driver.assert_owner_retention()
        driver.transport_hold("validated")
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
        np.testing.assert_allclose(host_results["directions"], rays.directions_world, atol=1e-6, rtol=1e-6)
        scalar_counts = {
            name: int(value[0])
            for name, value in host_results.items()
            if name.startswith(("bvh_", "shadow_"))
        }
        assert all(value >= 0 for value in scalar_counts.values())
        assert scalar_counts["bvh_stack_overflow_count"] == scalar_counts["shadow_stack_overflow_count"] == 0
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
