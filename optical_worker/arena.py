"""Frozen row selection and the sole native arena specification builder.

Reading row metadata is CPU-only. Descriptive specifications do not grant an
execution lease or replace observed input/readiness and final barrier authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .cases import canonical_json_bytes

_FROZEN_GATE_PATH = "benchmarks/results/q54_p12_4/gate.json"
_FROZEN_GATE_SHA256 = "a46d22540ee79d280f3c0e1d3a7482cc86f098eafbdb41462deea68a7062390c"
_CAPTURED_ROWS = {}


def _unique_fields(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("frozen gate contains duplicate fields")
        result[key] = value
    return result


@dataclass(frozen=True, eq=False, slots=True)
class _FrozenNativeRow:
    row_bytes: bytes
    case_bytes: bytes
    sample_count: int
    pass_mode: str

    @property
    def row(self):
        return json.loads(self.row_bytes)

    @property
    def case(self):
        return json.loads(self.case_bytes)


def _admit_fixed_row(session, *, row_id, case_id, sample_count, pass_mode):
    from .native_sources import _read_contained

    if session.phase != "materializing" or session in _CAPTURED_ROWS:
        raise ValueError("row admission must precede materialization and runtime initialization, once")
    if (
        type(row_id) is not str
        or type(case_id) is not str
        or type(sample_count) is not int
        or not 1 <= sample_count <= 1_000_000
        or pass_mode not in {"primary_unprofiled", "diagnostic_profiled"}
    ):
        raise ValueError("invalid fixed-driver row request")
    raw = _read_contained(Path(__file__).resolve().parents[1], _FROZEN_GATE_PATH)
    gate = json.loads(raw, object_pairs_hook=_unique_fields)
    if hashlib.sha256(canonical_json_bytes(gate)).hexdigest() != _FROZEN_GATE_SHA256:
        raise ValueError("row gate differs from the reviewed frozen data")
    row = next((item for item in gate["mandatory_rows"] if item["row_id"] == row_id), None)
    if row is None or row["case_id"] != case_id or row["boundary"] != "R0":
        raise ValueError("fixed final graph accepts only its frozen R0 row/case")
    admission = _FrozenNativeRow(
        canonical_json_bytes(row), canonical_json_bytes(gate["cases"][case_id]), sample_count, pass_mode
    )
    session._hold_runtime_owner(admission, "frozen_row_admission")
    _CAPTURED_ROWS[session] = admission
    return admission


def _spec_for_final_graph(session, *, sample_count):
    admission = _CAPTURED_ROWS.get(session)
    if (
        type(admission) is not _FrozenNativeRow
        or session.fixed_row is not admission
        or type(sample_count) is not int
        or sample_count != admission.sample_count
    ):
        raise ValueError("final graph requires its original pre-runtime row admission")
    provisional = session.fixed_render_graph
    row = admission.row
    if (
        row["output_profile"] != provisional.profile
        or row["shadows"] != provisional.shadows
        or admission.pass_mode != provisional.pass_mode
    ):
        raise ValueError("warmup modes differ from the originally admitted row")
    if session.lane == "cuda":
        from optics.cuda_arena import cuda_execution_stream_token

        context = session.cuda_factories[0]
        stream_token = cuda_execution_stream_token(context)
        first = session.fixed_render_storage[1]
        primitive_count, node_capacity = int(first.triangle_v0_world.shape[0]), first.num_bvh_nodes
    else:
        from optics.warp_arena import warp_execution_stream_token

        factory = provisional.factory
        stream_token = warp_execution_stream_token(factory.device, factory.stream)
        bvh = session.fixed_render_storage[2]
        primitive_count, node_capacity = bvh.num_primitives, bvh.num_nodes
    return native_optical_arena_spec_from_row(
        row=row,
        case_contract=admission.case,
        prepared=session.prepared_input,
        readiness=provisional.readiness,
        execution_stream_token=stream_token,
        primitive_count=primitive_count,
        node_capacity=node_capacity,
        diagnostic_mode=provisional.pass_mode,
    )


def native_optical_arena_spec_from_row(
    *,
    row,
    case_contract,
    prepared,
    readiness,
    execution_stream_token,
    primitive_count,
    node_capacity,
    diagnostic_mode="primary_unprofiled",
):
    """Build descriptive policy from an already validated frozen row."""
    from optics.native_arena import (
        NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
        NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        NativeOpticalArenaSpec,
    )

    from .cases import get_case
    from .lane_contract import EXACT_LANE_BINDINGS, INPUT_CONTRACT_BY_BOUNDARY
    from .native_contract import NativeReadinessEvidence

    boundary = str(row["boundary"])
    if boundary not in {"R0", "R1", "R2"}:
        raise ValueError("C4a native arena adapter only accepts R0/R1/R2 rows")
    if not prepared.manifest.result_capable:
        raise ValueError("native optical arena requires result-capable prepared input")
    manifest = prepared.manifest
    canonical_readiness = NativeReadinessEvidence.from_payload(readiness.to_payload())
    expected_case = case_contract
    expected_input_contract = INPUT_CONTRACT_BY_BOUNDARY[boundary]
    expected_lane = prepared.backend
    expected_binding = EXACT_LANE_BINDINGS[expected_lane]
    expected_acceleration = expected_binding["acceleration_by_class"][row["acceleration_class"]]

    checks = {
        "case_id": (manifest.case_id, row["case_id"]),
        "case_sha256": (manifest.case_sha256, expected_case["sha256"]),
        "input_contract": (manifest.input_contract, expected_input_contract),
        "scene_contract_sha256": (
            manifest.scene_contract_sha256,
            expected_case["scene_contract_sha256"],
        ),
        "readiness backend": (canonical_readiness.backend, expected_lane),
        "readiness device": (canonical_readiness.device_ordinal, prepared.device_ordinal),
        "readiness manifest": (canonical_readiness.manifest_sha256, manifest.sha256),
        "readiness content": (canonical_readiness.content_sha256, manifest.content_sha256),
        "readiness topology": (
            canonical_readiness.topology_sha256,
            manifest.topology_sha256,
        ),
        "readiness input contract": (
            canonical_readiness.input_contract,
            manifest.input_contract,
        ),
        "readiness scene contract": (
            canonical_readiness.scene_contract_sha256,
            manifest.scene_contract_sha256,
        ),
        "readiness replay": (
            canonical_readiness.transform_replay_sha256,
            manifest.transform_replay_sha256,
        ),
        "readiness fixture": (
            canonical_readiness.fixed_render_fixture_sha256,
            manifest.fixed_render_fixture_sha256,
        ),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"native optical arena {label} does not match its frozen row")

    if boundary == "R0":
        if manifest.fixed_render_fixture_sha256 != expected_case["fixed_render_fixture_sha256"]:
            raise ValueError("R0 arena fixture does not match frozen workload content")
        payload = "native_only"
        delivery_policy = "native_only"
        payload_contract_id = None
        ring_depth = 1
    else:
        if manifest.transform_replay_sha256 != expected_case["transform_replay_sha256"]:
            raise ValueError("dynamic arena replay does not match frozen workload content")
        if boundary == "R1":
            payload = "native_only"
            delivery_policy = "native_only"
            payload_contract_id = None
            ring_depth = 1
        else:
            payload = str(row["payload"])
            delivery_policy = str(row["delivery_policy"])
            payload_contract_id = str(row["payload_contract_id"])
            ring_depth = int(row["ring_depth"])

    case = get_case(str(row["case_id"]))
    return NativeOpticalArenaSpec(
        schema_version=NATIVE_OPTICAL_ARENA_SPEC_SCHEMA_VERSION,
        row_id=str(row["row_id"]),
        case_id=manifest.case_id,
        case_sha256=manifest.case_sha256,
        boundary=boundary,
        lane=expected_lane,
        device_ordinal=int(prepared.device_ordinal),
        execution_stream_token=execution_stream_token,
        input_contract=expected_input_contract,
        native_input_manifest_sha256=manifest.sha256,
        native_readiness_sha256=canonical_readiness.sha256,
        scene_contract_sha256=str(manifest.scene_contract_sha256),
        transform_replay_sha256=manifest.transform_replay_sha256,
        transform_frame_count=manifest.transform_frame_count,
        fixed_render_fixture_sha256=manifest.fixed_render_fixture_sha256,
        renderer_id=str(expected_binding["renderer_id"]),
        acceleration_class=str(row["acceleration_class"]),
        acceleration_id=str(expected_acceleration),
        width=case.width,
        height=case.height,
        num_rays=case.width * case.height,
        primitive_count=primitive_count,
        node_capacity=node_capacity,
        output_profile=str(row["output_profile"]),
        shadows=bool(row["shadows"]),
        diagnostic_mode=diagnostic_mode,
        payload=payload,
        delivery_policy=delivery_policy,
        payload_contract_id=payload_contract_id,
        ring_depth=ring_depth,
        ordered_delivery=True,
        arena_layout_schema_version=NATIVE_OPTICAL_ARENA_LAYOUT_SCHEMA_VERSION,
    )
