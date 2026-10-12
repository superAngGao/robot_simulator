"""CPU reference checks for the fixed C4 worker validation driver.

This is temporary acceptance instrumentation, removed with the paired driver.
All camera, scene, replay and rendering math comes from the existing owners.
It cannot authorize native execution, resolution, or release.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from optical_worker.cases import canonical_json_bytes
from optical_worker.render_parameters import bind_render_parameters, decode_native_scalars
from optical_worker.workloads import build_canonical_transform_replay
from optics.execution import CpuDirectLightOpticalExecutor
from optics.output_contract import (
    OUTPUT_PROFILE_REQUIRED_CHANNELS,
    PRIMARY_DIAGNOSTIC_CHANNEL_NAMES,
    SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES,
)
from optics.scene import OpticalSceneCache
from physics.publish import CpuPublishedFrame
from physics.spatial import SpatialTransform
from sensing.optical import OpticalPinholeCameraSpec, build_pinhole_camera_rays


class FinalWorkerNumericRun:
    """Precompute CPU truth before spawning the GPU worker; compare before release.

    Arrays are staged on disk one sample at a time, so full Go2 frames do not
    accumulate in RAM. Digests and parameter records stay in this parent object.
    RESOLVE-only diagnostics are deliberately not whole D1 acceptance.
    """

    def __init__(self, directory, request):
        from optical_worker.arena import select_native_worker_row
        from optical_worker.assets import preflight_case_asset
        from optical_worker.render_parameters import resolve_render_source
        from tests.support.native_runtime_process import _write, _write_bytes, final_readback_sample_indices

        self.directory, self.request = directory, dict(request)
        self.row = select_native_worker_row(
            **{key: request[key] for key in ("row_id", "case_id", "sample_count", "pass_mode")}
        ).row
        model_dir = Path(request["go2_model_dir"]) if request["go2_model_dir"] is not None else None
        asset = preflight_case_asset(
            request["case_id"],
            repo_root=model_dir.parent if model_dir else Path(request["root"]),
            model_dir=model_dir,
        )
        source = resolve_render_source(request["case_id"], go2_model_dir=model_dir, asset_evidence=asset)
        self.reference = FinalWorkerCpuReference(
            source,
            profile=self.row["output_profile"],
            shadows=self.row["shadows"],
            pass_mode=request["pass_mode"],
        )
        self.selected = final_readback_sample_indices(request["sample_count"])
        self.samples = {}
        # Reproduction verifies the complete frozen sequence before this point;
        # only the approved bounded native comparison generations are rendered.
        for index in sorted(self.selected):
            parameters, channels = self.reference.render(index)
            stream = io.BytesIO()
            np.savez(stream, **channels)
            raw = stream.getvalue()
            filename = f"cpu_final_sample_{index:04d}.npz"
            _write_bytes(directory, filename, raw)
            self.samples[index] = {
                "parameters": parameters,
                "file": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            del channels, stream, raw
        _write(
            directory,
            "cpu_final_reference.json",
            {
                "coverage": "numeric_reference_only",
                "revision": request["revision"],
                "case_id": request["case_id"],
                "row_id": request["row_id"],
                "pass_mode": request["pass_mode"],
                "scene_sha256": source.packed_scene.scene_contract_sha256,
                "sequence_sha256": source.sequence.sha256,
                "samples": self.samples,
            },
        )

    def check(self, sample_index, resolve_sequence, manifest, worker_evidence):
        from tests.support.native_runtime_process import _write

        stem = f"final_sample_{sample_index:04d}"
        artifact_bytes = (self.directory / (stem + ".json")).read_bytes()
        artifact = json.loads(artifact_bytes)
        parameters = self.reference.parameters(sample_index)
        expected_identity = {
            "schema_version": "q54.closed-worker.final-native-observation.v1",
            "coverage": "resolve_only",
            "complete": False,
            "failure": None,
            "revision": self.request["revision"],
            "catalog_sha256": self.request["catalog_sha256"],
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "runtime_manifest_sha256": manifest.sha256,
            "row_id": self.request["row_id"],
            "case_id": self.request["case_id"],
            "pass_mode": self.request["pass_mode"],
            "lane": self.request["lane"],
            "worker_pid": worker_evidence.process_id,
            "worker_start_nonce": worker_evidence.process_start_nonce,
            "sample_index": sample_index,
            "resolve_request_sequence": resolve_sequence,
            "parameters": parameters,
            "sample_state": "RESOLVED_HELD",
        }
        for key, value in expected_identity.items():
            if key not in artifact or canonical_json_bytes(artifact[key]) != canonical_json_bytes(value):
                raise ValueError(f"final readback identity differs: {key}")
        if sample_index not in self.selected:
            if artifact["channels"] != {}:
                raise ValueError("unexpected final readback outside selected generations")
            return
        desc = self.samples[sample_index]
        raw = (self.directory / desc["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != desc["sha256"] or parameters != desc["parameters"]:
            raise ValueError("CPU final reference changed after preflight")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            expected = {name: archive[name] for name in archive.files}
        actual = _read_final_channels(self.directory, stem, artifact["channels"], parameters)
        node_count = artifact["native_node_count"]
        if type(node_count) is not int or node_count < 0:
            raise ValueError("invalid final native node count")
        result = compare_final_channels(
            actual,
            expected,
            shadows=self.row["shadows"],
            diagnostics=parameters["diagnostics"],
            packed_scene=self.reference.source.packed_scene,
            native_node_count=node_count,
        )
        # This write must complete before the request client can send RELEASE.
        _write(
            self.directory,
            stem + "_numeric.json",
            {
                "coverage": "numeric_only",
                "complete_final_observation": False,
                "sample_index": sample_index,
                "resolve_request_sequence": resolve_sequence,
                "worker_pid": worker_evidence.process_id,
                "worker_start_nonce": worker_evidence.process_start_nonce,
                "runtime_manifest_sha256": manifest.sha256,
                "cpu_reference_sha256": desc["sha256"],
                "readback_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                **result,
            },
        )


def _read_final_channels(directory, stem, channels, parameters):
    """Read only the fixed physical channel ABI, including miss and counter data."""
    from optics.output_contract import DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS

    count = parameters["width"] * parameters["height"]
    # Both native renderers store float32 results and int32 IDs. Camera query
    # buffers preserve the approved float64 ABI; host delivery uses other types.
    layout = {
        spec.name: ("float32" if spec.dtype == "float64" else "int32", spec.expected_shape(count))
        for spec in DIRECT_LIGHT_FULL_HOST_PER_RAY_CHANNEL_SPECS
    }
    layout.update(origins=("float64", (count, 3)), directions=("float64", (count, 3)))
    names = PRIMARY_DIAGNOSTIC_CHANNEL_NAMES + (
        SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES if parameters["diagnostics"] else ()
    )
    layout.update({name: ("int32", (1,)) for name in names})
    if type(channels) is not dict or channels.keys() != layout.keys():
        raise ValueError("final physical channel set differs")
    result = {}
    for name, (dtype, shape) in layout.items():
        desc = channels[name]
        filename = stem + "_" + name + ".bin"
        expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if (
            type(desc) is not dict
            or set(desc) != {"file", "dtype", "shape", "bytes", "sha256"}
            or desc["file"] != filename
            or desc["dtype"] != dtype
            or desc["shape"] != list(shape)
            or any(type(value) is not int for value in desc["shape"])
            or type(desc["bytes"]) is not int
            or desc["bytes"] != expected_bytes
        ):
            raise ValueError(f"final physical channel metadata differs: {name}")
        raw = (directory / filename).read_bytes()
        if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != desc["sha256"]:
            raise ValueError(f"final physical channel bytes differ: {name}")
        result[name] = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return result


class FinalWorkerCpuReference:
    """One canonical materialization, using the original CPU renderer."""

    def __init__(self, source, *, profile, shadows, pass_mode):
        self.source = source
        self.case_id = source.sequence.to_payload()["case_id"]
        self.profile, self.shadows, self.pass_mode = profile, shadows, pass_mode
        self.scene = OpticalSceneCache(source.registry)
        self.replay = (
            build_canonical_transform_replay(self.case_id)
            if self.case_id == "synthetic_dynamic_rigid_960x640_v1"
            else None
        )

    def parameters(self, sample_index):
        return bind_render_parameters(
            self.source.sequence,
            sample_index=sample_index,
            profile=self.profile,
            shadows=self.shadows,
            pass_mode=self.pass_mode,
            role_to_mask=self.source.packed_scene.role_to_mask,
        ).to_payload()

    def inputs(self, sample_index):
        parameters = self.parameters(sample_index)
        scalars = decode_native_scalars(parameters)
        transforms = []
        if self.replay is not None:
            rotations = self.replay.arrays["replay_x_world_R"]
            translations = self.replay.arrays["replay_x_world_r"]
            index = sample_index % len(rotations)
            transforms = [
                SpatialTransform(rotation.astype(np.float64), translation.astype(np.float64))
                for rotation, translation in zip(rotations[index], translations[index], strict=True)
            ]
        frame = CpuPublishedFrame(
            frame_id=parameters["frame_id"],
            sim_time=float.fromhex(parameters["sim_time_f64"]),
            step_index=sample_index,
            env_mask=None,
            q=None,
            qdot=None,
            X_world=transforms,
            v_bodies=None,
            contact_count=None,
            contacts=None,
            telemetry=None,
        )
        camera = OpticalPinholeCameraSpec(
            frame.frame_id,
            frame.sim_time,
            parameters["env_idx"],
            parameters["sensor_id"],
            parameters["width"],
            parameters["height"],
            *scalars[:4],
            SpatialTransform(np.asarray(scalars[7:16]).reshape(3, 3), np.asarray(scalars[4:7])),
            scalars[16],
            self.source.sequence.to_payload()["frames"][parameters["camera_index"]]["sensor_role"],
        )
        return parameters, frame, camera

    def render(self, sample_index):
        parameters, frame, camera = self.inputs(sample_index)
        scalars = decode_native_scalars(parameters)
        snapshot = self.scene.snapshot_from_published_frame(frame, acceleration="cpu_bvh")
        rays = build_pinhole_camera_rays(camera)
        truth = CpuDirectLightOpticalExecutor(
            shadows=self.shadows,
            ambient_rgb=scalars[17:20],
            background_rgb=scalars[20:23],
            shadow_bias=scalars[23],
        ).execute(snapshot, rays, output_profile=self.profile)
        channels = {
            name: np.asarray(truth.channel(name))
            for name in OUTPUT_PROFILE_REQUIRED_CHANNELS[self.profile]
            if name not in PRIMARY_DIAGNOSTIC_CHANNEL_NAMES
        }
        channels.update(origins=rays.origins_world, directions=rays.directions_world)
        return parameters, channels


def compare_final_channels(actual, expected, *, shadows, diagnostics, packed_scene, native_node_count):
    """Original C1 tolerances and exact diagnostic invariants, no GPU calls.

    Profile filtering is already applied by the original CPU executor. The
    physical native arrays may contain additional internal channels.
    """
    errors = {}
    for name, reference in expected.items():
        observed = actual[name]
        if observed.shape != reference.shape:
            raise ValueError(f"final channel shape differs: {name}")
        if name in {"hit_mask", "numeric_instance_id", "material_index"}:
            np.testing.assert_array_equal(observed, reference, err_msg=name)
            errors[name] = {"unequal_elements": 0}
        else:
            tolerance = 1e-6 if name in {"origins", "directions"} else 1e-5
            np.testing.assert_allclose(
                observed,
                reference,
                atol=tolerance,
                rtol=tolerance,
                equal_nan=True,
                err_msg=name,
            )
            finite = np.isfinite(observed) & np.isfinite(reference)
            errors[name] = {
                "max_abs": float(np.max(np.abs(observed[finite] - reference[finite]), initial=0)),
                "finite_elements": int(np.count_nonzero(finite)),
                "atol": tolerance,
                "rtol": tolerance,
            }
    names = PRIMARY_DIAGNOSTIC_CHANNEL_NAMES + (
        SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES if diagnostics else ()
    )
    counters = {}
    for name in names:
        value = actual[name]
        if value.shape != (1,) or value.dtype != np.dtype("int32") or int(value[0]) < 0:
            raise ValueError(f"invalid final diagnostic counter: {name}")
        counters[name] = int(value[0])
    if counters["bvh_stack_overflow_count"] or counters["shadow_stack_overflow_count"]:
        raise ValueError("final traversal stack overflow")
    if max(counters["bvh_max_stack_depth"], counters["shadow_max_stack_depth"]) > 32:
        raise ValueError("final traversal exceeds the existing stack capacity")
    if not shadows and any(value for name, value in counters.items() if name.startswith("shadow_")):
        raise ValueError("shadows-off counters must all be zero")
    if diagnostics:
        total = counters["shadow_traversal_ray_count"]
        if (
            total
            != counters["shadow_traversal_directional_ray_count"]
            + counters["shadow_traversal_point_ray_count"]
            or total
            != counters["shadow_traversal_occluded_count"] + counters["shadow_traversal_unoccluded_count"]
            or total > np.count_nonzero(actual["hit_mask"]) * packed_scene.num_lights
            or counters["shadow_traversal_leaf_visit_count"] > counters["shadow_traversal_node_visit_count"]
            or counters["shadow_traversal_node_visit_count"] > total * native_node_count
            or counters["shadow_traversal_triangle_test_count"] > total * packed_scene.num_triangles
            or counters["shadow_traversal_plane_test_count"] > total * packed_scene.num_planes
            or (total == 0 and counters["shadow_max_stack_depth"] != 0)
        ):
            raise ValueError("final shadow traversal counters violate their exact invariants")
    return {"errors": errors, "counters": counters}
