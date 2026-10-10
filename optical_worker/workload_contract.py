"""Frozen logical-content identities for Q54 P12.4 optical workloads.

Case definitions describe intent.  These digests bind that intent to the
actual packed scene, replay sequence, and fixed R0 fixture that may enter a
result-capable lane.  This module is dependency-light so gate validation does
not materialize large assets or import a GPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

WORKLOAD_CONTENT_SCHEMA_VERSION = "q54.p12.4.workload-content.v1"
GO2_FROZEN_SOURCE_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
GO2_FROZEN_ASSET_AGGREGATE_SHA256 = "479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0"


@dataclass(frozen=True)
class FrozenWorkloadContent:
    """Exact logical source accepted for one frozen benchmark case."""

    scene_contract_sha256: str
    transform_replay_sha256: str | None
    fixed_render_fixture_sha256: str | None
    asset_source_revision: str | None = None
    asset_aggregate_sha256: str | None = None

    def to_payload(self) -> dict[str, str | None]:
        return {
            "scene_contract_sha256": self.scene_contract_sha256,
            "transform_replay_sha256": self.transform_replay_sha256,
            "fixed_render_fixture_sha256": self.fixed_render_fixture_sha256,
            "asset_source_revision": self.asset_source_revision,
            "asset_aggregate_sha256": self.asset_aggregate_sha256,
        }


FROZEN_WORKLOAD_CONTENT: Mapping[str, FrozenWorkloadContent] = MappingProxyType(
    {
        "asymmetric_synthetic_7x3_v1": FrozenWorkloadContent(
            scene_contract_sha256=("22796e667d52a0dec276fba787502ad8e06e09e8fec7dbb3254af40e06641a53"),
            transform_replay_sha256=None,
            fixed_render_fixture_sha256=("a0d37d3845fd24707a7e8d329e7dba923a384e9acc9ef7ced3e8581d52c125fb"),
        ),
        "go2_static_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("ddc5049a137c97b1924248687702c91740a7adf27f821b7be5c4eb58631317c9"),
            transform_replay_sha256=None,
            fixed_render_fixture_sha256=("c8fcc11d586a544568eddca317707a7485950ef97d887d969894d13f81eab843"),
            asset_source_revision=GO2_FROZEN_SOURCE_REVISION,
            asset_aggregate_sha256=GO2_FROZEN_ASSET_AGGREGATE_SHA256,
        ),
        "go2_static_1920x1080_v1": FrozenWorkloadContent(
            scene_contract_sha256=("910a0795752d809e053ce2f9e87e77903c621378e5245cb440b5af768dcd1481"),
            transform_replay_sha256=("2b1ecfb401476a9737400b401a72377c880bc9e289370f2d3bd5600cc287a216"),
            fixed_render_fixture_sha256=("ab1127b93b5dd1e3d6a98e122078701a3699bca536fe059a95a9c35a0bfaab5e"),
            asset_source_revision=GO2_FROZEN_SOURCE_REVISION,
            asset_aggregate_sha256=GO2_FROZEN_ASSET_AGGREGATE_SHA256,
        ),
        "synthetic_dynamic_rigid_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("61178ca1085929f47bad2d166dce594ad5547e6a0edbd229cec3072c1c60ae72"),
            transform_replay_sha256=("d62076e56e4e0e15b0481f8a38b596cffdba6b7b23033c1b0dae14d335c59557"),
            fixed_render_fixture_sha256=None,
        ),
        "physics_body_triangle_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("9bab39d32f7131ef3e74bfa14fb522f95625e97975a4a73470733ec3d1d550ac"),
            transform_replay_sha256=None,
            fixed_render_fixture_sha256=None,
        ),
    }
)


def frozen_workload_content(case_id: str) -> FrozenWorkloadContent:
    try:
        return FROZEN_WORKLOAD_CONTENT[case_id]
    except KeyError as exc:
        raise KeyError(f"no frozen workload content for {case_id!r}") from exc


def frozen_workload_payloads() -> dict[str, dict[str, str | None]]:
    return {case_id: content.to_payload() for case_id, content in sorted(FROZEN_WORKLOAD_CONTENT.items())}
