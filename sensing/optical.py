"""Optical sensor specs consumed by the optical execution layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OpticalRaySensorSpec:
    """World-frame ray batch for first optical executors.

    This spec lives in `sensing/` because it describes the question a sensor is
    asking. The executable scene and material bindings live in `optics/`.
    """

    frame_id: int
    sim_time: float
    env_idx: int
    sensor_id: str
    origins_world: object
    directions_world: object
    max_distance: float = np.inf
    sensor_role: str = "depth"

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins_world, dtype=np.float64)
        directions = np.asarray(self.directions_world, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_world must have shape (num_rays, 3)")
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_world must have shape (num_rays, 3)")
        if origins.shape[0] != directions.shape[0]:
            raise ValueError("origins_world and directions_world must have the same ray count")

        max_distance = float(self.max_distance)
        if max_distance <= 0.0:
            raise ValueError("max_distance must be > 0")
        sensor_role = str(self.sensor_role)
        if not sensor_role:
            raise ValueError("sensor_role must be non-empty")

        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("directions_world must not contain zero-length directions")

        self.origins_world = origins.copy()
        self.directions_world = (directions / norms[:, None]).copy()
        self.max_distance = max_distance
        self.sensor_role = sensor_role

    @property
    def num_rays(self) -> int:
        return int(self.origins_world.shape[0])
