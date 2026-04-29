from __future__ import annotations

import numpy as np
import pytest

from physics.terrain import FlatTerrain, HalfSpaceTerrain, HeightmapTerrain
from sensing import CpuPlaneSurfaceQueryExecutor, SurfaceQueryResult, SurfaceQuerySpec


class TestSurfaceQuerySpec:
    def test_normalizes_directions_and_exports_ray_count(self):
        spec = SurfaceQuerySpec(
            frame_id=2,
            sim_time=0.02,
            env_idx=0,
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -2.0]],
        )

        assert spec.num_rays == 1
        np.testing.assert_allclose(spec.directions_world, [[0.0, 0.0, -1.0]])

    @pytest.mark.parametrize(
        ("origins", "directions", "match"),
        [
            ([[0.0, 0.0]], [[0.0, 0.0, -1.0]], "origins_world"),
            ([[0.0, 0.0, 1.0]], [[0.0, 0.0]], "directions_world"),
            ([[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], "same ray count"),
            ([[0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0]], "zero-length"),
        ],
    )
    def test_rejects_malformed_arrays(self, origins, directions, match):
        with pytest.raises(ValueError, match=match):
            SurfaceQuerySpec(
                frame_id=0,
                sim_time=0.0,
                env_idx=0,
                origins_world=origins,
                directions_world=directions,
            )

    def test_rejects_non_positive_max_distance(self):
        with pytest.raises(ValueError, match="max_distance"):
            SurfaceQuerySpec(
                frame_id=0,
                sim_time=0.0,
                env_idx=0,
                origins_world=[[0.0, 0.0, 1.0]],
                directions_world=[[0.0, 0.0, -1.0]],
                max_distance=0.0,
            )

    def test_coerces_string_max_distance(self):
        spec = SurfaceQuerySpec(
            frame_id=0,
            sim_time=0.0,
            env_idx=0,
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -1.0]],
            max_distance="inf",
        )

        assert np.isinf(spec.max_distance)


class TestCpuPlaneSurfaceQueryExecutor:
    def test_flat_terrain_downward_rays_hit_plane(self):
        executor = CpuPlaneSurfaceQueryExecutor.from_terrain(FlatTerrain(z=0.25))
        spec = SurfaceQuerySpec(
            frame_id=4,
            sim_time=0.04,
            env_idx=1,
            origins_world=[
                [0.0, 0.0, 1.25],
                [2.0, -1.0, 0.75],
            ],
            directions_world=[
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ],
        )

        result = executor.execute(spec)

        assert isinstance(result, SurfaceQueryResult)
        assert result.frame_id == 4
        assert result.sim_time == 0.04
        assert result.env_idx == 1
        np.testing.assert_array_equal(result.hit_mask, [True, True])
        np.testing.assert_allclose(result.distance, [1.0, 0.5])
        np.testing.assert_allclose(result.position_world, [[0.0, 0.0, 0.25], [2.0, -1.0, 0.25]])
        np.testing.assert_allclose(result.normal_world, [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    def test_default_max_distance_allows_finite_hit_but_not_parallel_infinite_hit(self):
        executor = CpuPlaneSurfaceQueryExecutor.from_terrain(FlatTerrain(z=0.0))
        spec = SurfaceQuerySpec(
            frame_id=0,
            sim_time=0.0,
            env_idx=0,
            origins_world=[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            directions_world=[
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
        )

        result = executor.execute(spec)

        np.testing.assert_array_equal(result.hit_mask, [True, False])
        np.testing.assert_allclose(result.distance[0], 1.0)
        assert np.isinf(result.distance[1])
        assert np.all(np.isnan(result.position_world[1]))

    def test_ray_pointing_away_from_plane_misses(self):
        executor = CpuPlaneSurfaceQueryExecutor.from_terrain(FlatTerrain(z=0.0))
        spec = SurfaceQuerySpec(
            frame_id=0,
            sim_time=0.0,
            env_idx=0,
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, 1.0]],
        )

        result = executor.execute(spec)

        np.testing.assert_array_equal(result.hit_mask, [False])
        assert np.isinf(result.distance[0])
        assert np.all(np.isnan(result.position_world[0]))

    def test_flat_terrain_reports_misses_for_parallel_away_and_too_far_rays(self):
        executor = CpuPlaneSurfaceQueryExecutor.from_terrain(FlatTerrain(z=0.0))
        spec = SurfaceQuerySpec(
            frame_id=0,
            sim_time=0.0,
            env_idx=0,
            origins_world=[
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            directions_world=[
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            max_distance=0.5,
        )

        result = executor.execute(spec)

        np.testing.assert_array_equal(result.hit_mask, [False, False, False])
        assert np.all(np.isinf(result.distance))
        assert np.all(np.isnan(result.position_world))
        assert np.all(np.isnan(result.normal_world))

    def test_halfspace_terrain_uses_plane_normal_and_point(self):
        terrain = HalfSpaceTerrain(
            normal=np.array([0.0, -1.0, 1.0], dtype=np.float64),
            point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        executor = CpuPlaneSurfaceQueryExecutor.from_terrain(terrain)
        spec = SurfaceQuerySpec(
            frame_id=1,
            sim_time=0.01,
            env_idx=0,
            origins_world=[[0.0, 0.0, 1.0]],
            directions_world=[[0.0, 0.0, -1.0]],
        )

        result = executor.execute(spec)

        np.testing.assert_array_equal(result.hit_mask, [True])
        np.testing.assert_allclose(result.distance, [1.0])
        np.testing.assert_allclose(result.position_world, [[0.0, 0.0, 0.0]], atol=1e-12)
        np.testing.assert_allclose(result.normal_world, [terrain.normal_world])

    def test_rejects_unsupported_terrain(self):
        terrain = HeightmapTerrain(
            heightmap=np.zeros((2, 2), dtype=np.float64),
            resolution=1.0,
            origin=np.zeros(2, dtype=np.float64),
        )

        with pytest.raises(NotImplementedError, match="HeightmapTerrain"):
            CpuPlaneSurfaceQueryExecutor.from_terrain(terrain)

    def test_rejects_invalid_plane(self):
        with pytest.raises(ValueError, match="normal_world"):
            CpuPlaneSurfaceQueryExecutor(normal_world=[0.0, 0.0, 0.0], point_world=[0.0, 0.0, 0.0])
