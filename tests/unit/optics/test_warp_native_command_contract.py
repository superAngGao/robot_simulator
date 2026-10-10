from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from optics.device_bvh import (
    WARP_BVH_REFIT_ACCELERATION_ID,
    WARP_BVH_REFIT_COMMAND_SCHEMA_VERSION,
    WARP_BVH_REFIT_OPERATION_ID,
    WARP_BVH_TOPOLOGY_ORIGIN_ID,
    WarpBvhRefitCommandBundle,
    WarpBvhRefitCommandEvidence,
    WarpBvhRefitCommandFactory,
    build_host_median_bvh,
    host_bvh_topology_sha256,
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
        from optical_worker.native_setup import _warp_command_recording

        assert "_warp_command_recording" in source
        assert "wp.launch" not in source
        source = inspect.getsource(_warp_command_recording)
        assert "record_cmd=True" in source
        assert "record_tape=False" in source


def test_warp_c3b_evidence_cannot_claim_c4_lifecycle_or_result_capability():
    forbidden = {"slot_id", "generation", "lease", "ring", "ring_depth", "result_capable"}
    for evidence_type in (WarpSceneUpdateCommandEvidence, WarpBvhRefitCommandEvidence):
        assert forbidden.isdisjoint(field.name for field in fields(evidence_type))


def test_warp_output_disjointness_manifest_covers_every_local_scene_array():
    assert set(_DEVICE_SCENE_ARRAY_NAMES) == set(SCENE_ARRAY_SEMANTICS)


def test_warp_topology_digest_binds_leaf_size_and_split_strategy():
    bounds_min = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    bounds_max = bounds_min + np.float32(0.25)
    source_order = np.arange(4, dtype=np.int64)
    variants = (
        build_host_median_bvh(
            bounds_min,
            bounds_max,
            source_order,
            leaf_size=1,
            split_strategy="sort",
        ),
        build_host_median_bvh(
            bounds_min,
            bounds_max,
            source_order,
            leaf_size=4,
            split_strategy="sort",
        ),
        build_host_median_bvh(
            bounds_min,
            bounds_max,
            source_order,
            leaf_size=4,
            split_strategy="partition",
        ),
    )
    assert len({host_bvh_topology_sha256(value) for value in variants}) == len(variants)


def test_warp_scene_bundle_is_immutable_and_poison_is_fail_closed():
    bundle = WarpSceneUpdateCommandBundle(
        commands=(_FailingCommand(),),
        evidence=_scene_evidence(),
        local_scene=object(),
        transforms=object(),
        outputs=object(),
        stream=SimpleNamespace(cuda_stream=1),
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
        acceleration_id=WARP_BVH_REFIT_ACCELERATION_ID,
        device_ordinal=0,
        stream_handle=1,
        native_input_manifest_sha256="a" * 64,
        scene_contract_sha256="b" * 64,
        input_contract="unit_replay_v1",
        native_input_topology_sha256="c" * 64,
        topology_origin_id=WARP_BVH_TOPOLOGY_ORIGIN_ID,
        topology_sha256="d" * 64,
        primitive_count=1,
        node_count=1,
        leaf_count=1,
        input_storage_sha256="e" * 64,
        output_storage_sha256="f" * 64,
        dispatch_count=1,
        dispatch_plan_sha256="0" * 64,
    )
