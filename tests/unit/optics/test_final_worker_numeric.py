"""CPU oracle integration and rejection of stale/corrupt native channels."""

import numpy as np
import pytest

from optical_worker.render_parameters import decode_native_scalars, resolve_render_source
from optics.output_contract import PRIMARY_DIAGNOSTIC_CHANNEL_NAMES, SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES
from tests.support.final_worker_numeric import FinalWorkerCpuReference, compare_final_channels

ASYMMETRIC = "asymmetric_synthetic_7x3_v1"
DYNAMIC = "synthetic_dynamic_rigid_960x640_v1"


def reference(case=ASYMMETRIC, profile="direct_light_full", shadows=False):
    return FinalWorkerCpuReference(
        resolve_render_source(case),
        profile=profile,
        shadows=shadows,
        pass_mode="diagnostic_profiled",
    )


def physical_channels(expected):
    actual = {
        name: value.astype(
            "int32"
            if name in {"hit_mask", "numeric_instance_id", "material_index"}
            else "float64"
            if name in {"origins", "directions"}
            else "float32"
        )
        for name, value in expected.items()
    }
    for name in (*PRIMARY_DIAGNOSTIC_CHANNEL_NAMES, *SHADOW_TRAVERSAL_DIAGNOSTIC_CHANNEL_NAMES):
        actual[name] = np.zeros(1, dtype="int32")
    # render_only has no public hit channel, but the native internal channel is
    # still needed to bound shadow work. All counters below are zero.
    actual.setdefault("hit_mask", np.zeros(len(expected["origins"]), dtype="int32"))
    return actual


@pytest.mark.parametrize("profile", ["direct_light_full", "rgb_preview", "render_only"])
@pytest.mark.parametrize("shadows", [False, True])
def test_reference_delegates_original_renderer_and_preserves_profile(monkeypatch, profile, shadows):
    from tests.support import final_worker_numeric as module

    cpu = reference(profile=profile, shadows=shadows)
    original = module.CpuDirectLightOpticalExecutor.execute
    calls = []

    def execute(executor, snapshot, rays, **kwargs):
        result = original(executor, snapshot, rays, **kwargs)
        calls.append((snapshot, rays, kwargs, result))
        return result

    monkeypatch.setattr(module.CpuDirectLightOpticalExecutor, "execute", execute)
    parameters, expected = cpu.render(3)
    assert len(calls) == 1 and calls[0][2] == {"output_profile": profile}
    assert parameters["frame_id"] == calls[0][0].frame_id == 4
    assert calls[0][0].instances[0].geometry is cpu.source.registry.geometry["tri_00"]
    actual = physical_channels(expected)
    result = compare_final_channels(
        actual,
        expected,
        shadows=shadows,
        diagnostics=True,
        packed_scene=cpu.source.packed_scene,
        native_node_count=41,
    )
    assert result["errors"].keys() == expected.keys()
    assert ("rgb" in expected) == (profile != "render_only")
    assert ("range_m" in expected) == (profile == "direct_light_full")
    for name, values in expected.items():
        if name not in {"origins", "directions"}:
            assert values is calls[0][3].channel(name)


def test_all_120_dynamic_camera_and_replay_values_include_last_and_wrap():
    cpu = reference(DYNAMIC)
    for index in range(121):
        parameters, frame, camera = cpu.inputs(index)
        scalars = decode_native_scalars(parameters)
        assert parameters["camera_index"] == index % 120
        assert frame.frame_id == camera.frame_id == index + 1
        assert frame.sim_time == camera.sim_time == index / 30
        assert (camera.fx, camera.fy, camera.cx, camera.cy) == scalars[:4]
        np.testing.assert_array_equal(camera.X_world_camera.R.reshape(-1), scalars[7:16])
        np.testing.assert_array_equal(camera.X_world_camera.r, scalars[4:7])
        np.testing.assert_array_equal(
            frame.X_world[0].R, cpu.replay.arrays["replay_x_world_R"][index % 120, 0]
        )
        np.testing.assert_array_equal(
            frame.X_world[0].r, cpu.replay.arrays["replay_x_world_r"][index % 120, 0]
        )
    assert cpu.inputs(120)[0]["native_camera_scalars_f64"] == cpu.inputs(0)[0]["native_camera_scalars_f64"]
    assert cpu.inputs(0)[1].X_world[0].r[2] != cpu.inputs(1)[1].X_world[0].r[2]


@pytest.mark.parametrize("fault", ["id", "rgb", "shape", "nan", "overflow", "off", "counter_sum", "nodes"])
def test_numeric_or_counter_corruption_rejects(fault):
    cpu = reference()
    _, expected = cpu.render(0)
    actual = physical_channels(expected)
    shadows = fault in {"counter_sum", "nodes"}
    if fault == "id":
        actual["numeric_instance_id"][0] += 1
    elif fault == "rgb":
        actual["rgb"][0, 0] += 1e-2
    elif fault == "shape":
        actual["rgb"] = actual["rgb"].reshape(-1)
    elif fault == "nan":
        actual["origins"][0, 0] = np.nan
    elif fault == "overflow":
        actual["bvh_stack_overflow_count"][0] = 1
    elif fault == "off":
        actual["shadow_max_stack_depth"][0] = 1
    elif fault == "counter_sum":
        actual["shadow_traversal_ray_count"][0] = 1
    else:
        for name in ("ray", "directional_ray", "unoccluded"):
            actual[f"shadow_traversal_{name}_count"][0] = 1
        actual["shadow_traversal_node_visit_count"][0] = 42
    with pytest.raises((ValueError, AssertionError)):
        compare_final_channels(
            actual,
            expected,
            shadows=shadows,
            diagnostics=True,
            packed_scene=cpu.source.packed_scene,
            native_node_count=41,
        )
