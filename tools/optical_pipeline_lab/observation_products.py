"""Observation frame products for tick-based Optical Pipeline Lab workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from rl_env.managers import ObsManager
from rl_env.obs import ObsSchema, obs_cfg_from_schema
from sensing import StateSampleView, build_state_sample_view

from .frame_products import FrameProductResult
from .frame_tick import SimulationFrameTick


@dataclass
class PublishedStateObservationProduct:
    """Build an RL observation vector from published/sensing frame data."""

    engine: object
    schema: ObsSchema
    root_body_idx: int = 0
    root_q_slice: slice = field(default_factory=lambda: slice(0, 7))
    actuated_q_indices: object | None = None
    actuated_v_indices: object | None = None
    contact_body_names: Sequence[str] = ()
    product_name: str = "observation"
    records: list[FrameProductResult] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        schema_names = set(self.schema.names)
        if "joint_pos" in schema_names and self.actuated_q_indices is None:
            raise ValueError("actuated_q_indices are required for schema field 'joint_pos'")
        if "joint_vel" in schema_names and self.actuated_v_indices is None:
            raise ValueError("actuated_v_indices are required for schema field 'joint_vel'")
        if "contact_mask" in schema_names:
            expected_dim = self._schema_field_dim("contact_mask")
            if len(self.contact_body_names) != expected_dim:
                raise ValueError("contact_body_names length must match schema field 'contact_mask' dim")

    def begin_run(self) -> object | None:
        self.records.clear()
        return None

    def consume(self, tick: SimulationFrameTick) -> FrameProductResult:
        view = build_state_sample_view(
            self.engine,
            frame=tick.published_frame,
            env_idx=tick.env_idx,
        )
        observation = self._build_observation(view)
        metadata = {
            "schema_names": self.schema.names,
            "schema_dim": self.schema.dim,
        }
        result = FrameProductResult.from_tick(
            product_name=self.product_name,
            tick=tick,
            payload={
                "observation": observation,
                "state_sample": view,
            },
            metadata=metadata,
        )
        self.records.append(result)
        return result

    def end_run(self) -> object | None:
        return tuple(self.records)

    def _build_observation(self, view: StateSampleView):
        self._validate_view_for_schema(view)
        env = _StateSampleObservationEnv(
            view=view,
            root_body_idx=self.root_body_idx,
            root_q_slice=self.root_q_slice,
            actuated_q_indices=self.actuated_q_indices,
            actuated_v_indices=self.actuated_v_indices,
            contact_body_names=tuple(self.contact_body_names),
        )
        manager = ObsManager(obs_cfg_from_schema(self.schema), env)
        manager.eval()
        observation = manager.compute()
        if observation.shape[0] != self.schema.dim:
            raise RuntimeError("observation vector length does not match schema dim")
        return observation

    def _validate_view_for_schema(self, view: StateSampleView) -> None:
        schema_names = set(self.schema.names)
        if {"base_lin_vel_body", "base_ang_vel_body"} & schema_names and view.v_bodies is None:
            raise ValueError("published frame is missing body velocities required by observation schema")
        if {"base_orientation_quat_wxyz", "joint_pos"} & schema_names and view.q is None:
            raise ValueError("published frame is missing q required by observation schema")
        if "joint_vel" in schema_names and view.qdot is None:
            raise ValueError("published frame is missing qdot required by observation schema")
        if "contact_mask" in schema_names and view.contact_mask is None:
            raise ValueError("published frame is missing contact_mask required by observation schema")

    def _schema_field_dim(self, name: str) -> int:
        for field_spec in self.schema.fields:
            if field_spec.name == name:
                return field_spec.dim
        raise KeyError(name)


class _StateSampleObservationEnv:
    def __init__(
        self,
        *,
        view: StateSampleView,
        root_body_idx: int,
        root_q_slice: slice,
        actuated_q_indices: object,
        actuated_v_indices: object,
        contact_body_names: tuple[str, ...],
    ) -> None:
        self.q = view.q
        self.qdot = view.qdot
        self.v_bodies = view.v_bodies
        self.root_body_idx = int(root_body_idx)
        self.root_q_slice = root_q_slice
        self.actuated_q_indices = actuated_q_indices
        self.actuated_v_indices = actuated_v_indices
        self.contact_body_names = list(contact_body_names)
        self.active_contacts = self._active_contacts_from_published_mask(view)

    def _active_contacts_from_published_mask(self, view: StateSampleView) -> list[tuple[str, None]]:
        if view.contact_mask is None:
            return []
        mask = np.asarray(view.contact_mask).astype(bool).reshape(-1)
        if len(mask) != len(self.contact_body_names):
            raise ValueError("published contact_mask length must match contact_body_names")
        return [
            (name, None) for name, is_active in zip(self.contact_body_names, mask, strict=True) if is_active
        ]
