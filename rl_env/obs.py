"""Observation schema contracts for RL-facing vector observations.

This module is intentionally small: it records field order, dimensions, and
normalization semantics without making the CPU debug ``Env`` the final
manager-based GPU RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

from . import obs_terms
from .cfg import NoiseCfg, ObsTermCfg


@dataclass(frozen=True)
class ObsFieldSpec:
    """One contiguous field in an observation vector."""

    name: str
    dim: int
    term: Callable | None = None
    scale: float | tuple[float, ...] = 1.0
    unit: str = ""
    frame: str = ""
    convention: str = ""
    source: str = ""
    params: Mapping[str, object] = field(default_factory=dict)
    phase2_requirement: str = ""

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"Observation field {self.name!r} must have non-negative dim")
        if isinstance(self.scale, tuple) and len(self.scale) != self.dim:
            raise ValueError(
                f"Observation field {self.name!r} scale length {len(self.scale)} "
                f"does not match dim {self.dim}"
            )


@dataclass(frozen=True)
class ObsSchema:
    """Ordered observation-vector contract.

    Normalization convention:
        observation[field] = raw_term(field) * field.scale

    The default schema uses raw physical units for most fields because robust
    ranges are robot/task specific. Task configs may override scales later, but
    the field order and semantic names should remain stable.
    """

    fields: tuple[ObsFieldSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def dim(self) -> int:
        return sum(field.dim for field in self.fields)

    @property
    def slices(self) -> dict[str, slice]:
        offset = 0
        result: dict[str, slice] = {}
        for field_spec in self.fields:
            result[field_spec.name] = slice(offset, offset + field_spec.dim)
            offset += field_spec.dim
        return result

    @property
    def phase2_requirements(self) -> tuple[str, ...]:
        return tuple(field.phase2_requirement for field in self.fields if field.phase2_requirement)

    def field_slice(self, name: str) -> slice:
        try:
            return self.slices[name]
        except KeyError as exc:
            raise KeyError(f"Unknown observation field {name!r}") from exc


def locomotion_obs_schema(
    *,
    num_actuated_joints: int,
    num_contact_bodies: int = 0,
    include_contact_mask: bool = False,
) -> ObsSchema:
    """Return the default phase-3 locomotion observation schema.

    Field order is fixed as:
        base linear velocity, base angular velocity, base orientation,
        joint positions, joint velocities, optional contact mask.

    Quaternion convention is scalar-first ``[w, x, y, z]`` to match
    ``physics.spatial`` and ``FreeJoint``.
    """

    fields = [
        ObsFieldSpec(
            name="base_lin_vel_body",
            dim=3,
            term=obs_terms.base_lin_vel,
            unit="m/s",
            frame="base body",
            source="root body velocity",
        ),
        ObsFieldSpec(
            name="base_ang_vel_body",
            dim=3,
            term=obs_terms.base_ang_vel,
            unit="rad/s",
            frame="base body",
            source="root body velocity",
        ),
        ObsFieldSpec(
            name="base_orientation_quat_wxyz",
            dim=4,
            term=obs_terms.base_orientation,
            convention="unit quaternion [w, x, y, z], scalar-first",
            source="FreeJoint q[:4] or rot_to_quat(IMUReading.orientation_world_R)",
            phase2_requirement=(
                "If RL observations are built from sensing readings, derive "
                "base_orientation_quat_wxyz with physics.spatial.rot_to_quat "
                "from IMUReading.orientation_world_R."
            ),
        ),
        ObsFieldSpec(
            name="joint_pos",
            dim=num_actuated_joints,
            term=obs_terms.joint_pos,
            unit="rad or m",
            source="actuated joint q indices",
        ),
        ObsFieldSpec(
            name="joint_vel",
            dim=num_actuated_joints,
            term=obs_terms.joint_vel,
            unit="rad/s or m/s",
            source="actuated joint v indices",
        ),
    ]

    if include_contact_mask:
        fields.append(
            ObsFieldSpec(
                name="contact_mask",
                dim=num_contact_bodies,
                term=obs_terms.contact_mask,
                convention="binary 0.0/1.0 in contact_body_names order",
                source="published contact mask or contact-pair block",
                phase2_requirement=(
                    "Contact-mask observations require a backend-neutral "
                    "published per-body mask or contact-pair block; do not "
                    "infer masks from private contact scratch."
                ),
            )
        )

    return ObsSchema(fields=tuple(fields))


def obs_cfg_from_schema(
    schema: ObsSchema,
    *,
    noise_by_field: Mapping[str, NoiseCfg] | None = None,
) -> dict[str, ObsTermCfg]:
    """Build an ordered ``EnvCfg.obs_cfg`` from an observation schema."""

    noise_by_field = noise_by_field or {}
    cfg: dict[str, ObsTermCfg] = {}
    for field_spec in schema.fields:
        if field_spec.term is None:
            raise ValueError(f"Observation field {field_spec.name!r} has no term function")
        cfg[field_spec.name] = ObsTermCfg(
            func=_scaled_term(field_spec.term, field_spec.scale),
            params=dict(field_spec.params),
            noise=noise_by_field.get(field_spec.name),
        )
    return cfg


def _scaled_term(term: Callable, scale: float | tuple[float, ...]) -> Callable:
    def compute(env, **params):
        vec = term(env, **params)
        if isinstance(scale, tuple):
            return vec * torch.tensor(scale, dtype=vec.dtype, device=vec.device)
        if scale == 1.0:
            return vec
        return vec * float(scale)

    compute.__name__ = getattr(term, "__name__", "obs_term")
    return compute
