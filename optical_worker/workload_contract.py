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
            scene_contract_sha256=("ab3b06cd48e0315ce5d33c0f81a880e71fe5df7369c805a2ae85f5c0197b9e35"),
            transform_replay_sha256=None,
            fixed_render_fixture_sha256=("4fcece950264cb2ebb78cf1ec06ad52f5a151b3d2cf2970314f1cae5a2aa7907"),
        ),
        "go2_static_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f"),
            transform_replay_sha256=None,
            fixed_render_fixture_sha256=("fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182"),
            asset_source_revision=GO2_FROZEN_SOURCE_REVISION,
            asset_aggregate_sha256=GO2_FROZEN_ASSET_AGGREGATE_SHA256,
        ),
        "go2_static_1920x1080_v1": FrozenWorkloadContent(
            scene_contract_sha256=("57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923"),
            transform_replay_sha256=("2b1ecfb401476a9737400b401a72377c880bc9e289370f2d3bd5600cc287a216"),
            fixed_render_fixture_sha256=("8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0"),
            asset_source_revision=GO2_FROZEN_SOURCE_REVISION,
            asset_aggregate_sha256=GO2_FROZEN_ASSET_AGGREGATE_SHA256,
        ),
        "synthetic_dynamic_rigid_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("a13f77e42e409cd8f929d01177845fe8c1ee9d90c79fe14eb59639ed845a5ce9"),
            transform_replay_sha256=("d62076e56e4e0e15b0481f8a38b596cffdba6b7b23033c1b0dae14d335c59557"),
            fixed_render_fixture_sha256=None,
        ),
        "physics_body_triangle_960x640_v1": FrozenWorkloadContent(
            scene_contract_sha256=("04a10a8414cae5f8e620684a4068810cc0813e4c15bbf51ba1080862bc36b056"),
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
