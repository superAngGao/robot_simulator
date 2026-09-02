from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from optics.device_bvh import (
    WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION,
    WARP_BVH_REFIT_OPERATION_ID,
    WarpBvhRefitCommandBundle,
    WarpBvhRefitCommandEvidence,
    WarpBvhRefitCommandFactory,
)
from optics.device_scene import (
    _DEVICE_SCENE_ARRAY_NAMES,
    WARP_SCENE_COMMAND_SCHEMA_VERSION,
    WARP_SCENE_UPDATE_OPERATION_ID,
    WarpSceneUpdateCommandBundle,
    WarpSceneUpdateCommandEvidence,
    WarpSceneUpdateCommandFactory,
)
from optics.scene_contract import SCENE_ARRAY_SEMANTICS


def test_warp_timed_enqueue_only_launches_prebound_commands():
    for bundle_type in (WarpSceneUpdateCommandBundle, WarpBvhRefitCommandBundle):
        source = inspect.getsource(bundle_type.enqueue)
        assert "command.launch(stream=self.stream)" in source
        for forbidden in (
            "wp.launch",
            "wp.empty",
            "wp.zeros",
            "record_event",
            "wait_event",
            "synchronize",
            "record_cmd",
        ):
            assert forbidden not in source

    for factory_type in (WarpSceneUpdateCommandFactory, WarpBvhRefitCommandFactory):
        source = inspect.getsource(factory_type.seal)
        assert "record_cmd=True" in source
        assert "record_tape=False" in source


def test_warp_c3b_evidence_cannot_claim_c4_lifecycle_or_result_capability():
    forbidden = {"slot_id", "generation", "lease", "ring", "ring_depth", "result_capable"}
    for evidence_type in (WarpSceneUpdateCommandEvidence, WarpBvhRefitCommandEvidence):
        assert forbidden.isdisjoint(field.name for field in fields(evidence_type))


def test_warp_output_disjointness_manifest_covers_every_local_scene_array():
    assert set(_DEVICE_SCENE_ARRAY_NAMES) == set(SCENE_ARRAY_SEMANTICS)


def test_warp_scene_bundle_is_immutable_and_poison_is_fail_closed():
    bundle = WarpSceneUpdateCommandBundle(
        commands=(_FailingCommand(),),
        evidence=_scene_evidence(),
        local_scene=object(),
        transforms=object(),
        outputs=object(),
        stream=object(),
    )
    with pytest.raises(RuntimeError, match="injected launch failure"):
        bundle.enqueue()
    assert bundle.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bundle.enqueue()
    with pytest.raises(AttributeError, match="immutable"):
        bundle.stream = object()


def test_warp_refit_bundle_is_immutable_and_poison_is_fail_closed():
    bundle = WarpBvhRefitCommandBundle(
        commands=(_FailingCommand(),),
        evidence=_refit_evidence(),
        world=object(),
        bvh=object(),
        stream=object(),
    )
    with pytest.raises(RuntimeError, match="injected launch failure"):
        bundle.enqueue()
    assert bundle.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bundle.enqueue()
    with pytest.raises(AttributeError, match="immutable"):
        bundle.bvh = object()


class _FailingCommand:
    def launch(self, *, stream):
        del stream
        raise RuntimeError("injected launch failure")


def _scene_evidence() -> WarpSceneUpdateCommandEvidence:
    return WarpSceneUpdateCommandEvidence(
        schema_version=WARP_SCENE_COMMAND_SCHEMA_VERSION,
        operation_id=WARP_SCENE_UPDATE_OPERATION_ID,
        device_ordinal=0,
        stream_handle=1,
        native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64,
        input_contract="unit_replay_v1",
        transform_replay_sha256="c" * 64,
        transform_frame_index=0,
        input_storage_sha256="d" * 64,
        output_storage_sha256="e" * 64,
        dispatch_count=1,
        dispatch_plan_sha256="f" * 64,
    )


def _refit_evidence() -> WarpBvhRefitCommandEvidence:
    return WarpBvhRefitCommandEvidence(
        schema_version=WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION,
        operation_id=WARP_BVH_REFIT_OPERATION_ID,
        device_ordinal=0,
        stream_handle=1,
        native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64,
        input_contract="unit_replay_v1",
        native_input_topology_sha256="c" * 64,
        topology_sha256="d" * 64,
        primitive_count=1,
        node_count=1,
        leaf_count=1,
        input_storage_sha256="e" * 64,
        output_storage_sha256="f" * 64,
        dispatch_count=1,
        dispatch_plan_sha256="0" * 64,
    )
