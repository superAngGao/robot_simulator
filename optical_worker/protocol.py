"""Strict, object-free protocol contracts for the native optical worker."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

CLOSED_WORKER_PROTOCOL_VERSION: Final = "q54.p12.4.closed-worker-protocol.v1"
CLOSED_WORKER_TRUST_MODEL_VERSION: Final = "q54.p12.4.closed-worker-trust-model.v2"
CLOSED_WORKER_RUNTIME_MANIFEST_VERSION: Final = "q54.p12.4.closed-worker-runtime-manifest.v2"
CLOSED_WORKER_EVIDENCE_VERSION: Final = "q54.p12.4.closed-worker-evidence.v2"
CLOSED_WORKER_STATE_MACHINE_VERSION: Final = "q54.p12.4.closed-worker-state-machine.v2"

MAX_REQUEST_BYTES: Final = 64 * 1024
MAX_RESPONSE_BYTES: Final = 4 * 1024 * 1024
MAX_NESTING_DEPTH: Final = 8
MAX_OBJECT_FIELDS: Final = 64
MAX_STRING_BYTES: Final = 4096
FRAME_HEADER_BYTES: Final = 4

LaneId = Literal["warp", "cuda"]
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}\Z")
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_RELATIVE_PATH_RE = re.compile(r"(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9_.\-/]{1,512}\Z")


class ClosedWorkerProtocolError(RuntimeError):
    """A request, response, frame, or runtime manifest is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode the protocol's canonical JSON subset.

    Floats are deliberately absent.  Timings cross the worker boundary as
    integer nanoseconds; accepting a Python float would introduce alternate
    spellings and non-finite values into the control protocol.
    """

    _validate_canonical_value(value, depth=0)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_canonical_value(value: object, *, depth: int) -> None:
    if depth > MAX_NESTING_DEPTH:
        raise ClosedWorkerProtocolError("canonical JSON nesting exceeds the protocol limit")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if isinstance(value, bool) or not -(2**63) <= value < 2**63:
            raise ClosedWorkerProtocolError("canonical JSON integer is outside signed 64-bit range")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise ClosedWorkerProtocolError("canonical JSON string exceeds the protocol limit")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_OBJECT_FIELDS:
            raise ClosedWorkerProtocolError("canonical JSON object has too many fields")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ClosedWorkerProtocolError("canonical JSON object keys must be strings")
            _validate_canonical_value(key, depth=depth + 1)
            _validate_canonical_value(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_canonical_value(item, depth=depth + 1)
        return
    raise ClosedWorkerProtocolError(f"canonical JSON rejects values of type {type(value).__name__}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ClosedWorkerProtocolError(f"duplicate canonical JSON key {key!r}")
        result[key] = value
    return result


def decode_canonical_message(encoded: bytes, *, max_bytes: int) -> object:
    if not isinstance(encoded, bytes):
        raise TypeError("encoded canonical message must be bytes")
    if not encoded or len(encoded) > max_bytes:
        raise ClosedWorkerProtocolError("canonical message size is outside the protocol limit")
    try:
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=lambda _: (_ for _ in ()).throw(
                ClosedWorkerProtocolError("canonical JSON floats are forbidden")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                ClosedWorkerProtocolError("canonical JSON constants are forbidden")
            ),
        )
    except UnicodeDecodeError as exc:
        raise ClosedWorkerProtocolError("canonical message is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ClosedWorkerProtocolError("canonical message is not valid JSON") from exc
    _validate_canonical_value(value, depth=0)
    if canonical_json_bytes(value) != encoded:
        raise ClosedWorkerProtocolError("message is not in canonical JSON encoding")
    return value


def encode_framed_message(value: object, *, max_bytes: int) -> bytes:
    payload = canonical_json_bytes(value)
    if not payload or len(payload) > max_bytes:
        raise ClosedWorkerProtocolError("framed message exceeds the protocol limit")
    return struct.pack(">I", len(payload)) + payload


def decode_frame_header(header: bytes, *, max_bytes: int) -> int:
    if len(header) != FRAME_HEADER_BYTES:
        raise ClosedWorkerProtocolError("frame header is truncated")
    (size,) = struct.unpack(">I", header)
    if size <= 0 or size > max_bytes:
        raise ClosedWorkerProtocolError("frame payload size is outside the protocol limit")
    return size


@dataclass(frozen=True)
class FrozenSourceDigest:
    relative_path: str
    sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "FrozenSourceDigest":
        if not isinstance(payload, Mapping) or set(payload) != {"relative_path", "sha256"}:
            raise ClosedWorkerProtocolError("frozen source digest fields are invalid")
        path = payload["relative_path"]
        digest = payload["sha256"]
        if not isinstance(path, str) or _RELATIVE_PATH_RE.fullmatch(path) is None:
            raise ClosedWorkerProtocolError("frozen source path is invalid")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ClosedWorkerProtocolError("frozen source digest is invalid")
        return cls(relative_path=path, sha256=digest)

    def to_payload(self) -> dict[str, object]:
        return {"relative_path": self.relative_path, "sha256": self.sha256}


@dataclass(frozen=True)
class ClosedWorkerRuntimeManifest:
    schema_version: str
    protocol_version: str
    trust_model_version: str
    state_machine_version: str
    interpreter_executable: str
    python_version: str
    environment: Mapping[str, str | None]
    environment_sha256: str
    site_package_paths: tuple[str, ...]
    repository_root: str
    repository_revision: str
    repository_clean: bool
    worker_script_relative_path: str
    lane_id: LaneId
    device_ordinal: int
    gate_relative_path: str
    gate_sha256: str
    workload_set_sha256: str
    source_digests: tuple[FrozenSourceDigest, ...]
    source_manifest_sha256: str
    ast_policy_sha256: str
    control_idle_timeout_ms: int
    response_write_timeout_ms: int
    abort_watchdog_ms: int
    gpu_preparation_enabled: bool

    @classmethod
    def from_payload(cls, payload: object) -> "ClosedWorkerRuntimeManifest":
        fields = {
            "schema_version",
            "protocol_version",
            "trust_model_version",
            "state_machine_version",
            "interpreter_executable",
            "python_version",
            "environment",
            "environment_sha256",
            "site_package_paths",
            "repository_root",
            "repository_revision",
            "repository_clean",
            "worker_script_relative_path",
            "lane_id",
            "device_ordinal",
            "gate_relative_path",
            "gate_sha256",
            "workload_set_sha256",
            "source_digests",
            "source_manifest_sha256",
            "ast_policy_sha256",
            "control_idle_timeout_ms",
            "response_write_timeout_ms",
            "abort_watchdog_ms",
            "gpu_preparation_enabled",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ClosedWorkerProtocolError("closed-worker runtime manifest fields are invalid")
        versions = {
            "schema_version": CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
            "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
            "trust_model_version": CLOSED_WORKER_TRUST_MODEL_VERSION,
            "state_machine_version": CLOSED_WORKER_STATE_MACHINE_VERSION,
        }
        for name, expected in versions.items():
            if payload[name] != expected:
                raise ClosedWorkerProtocolError(f"runtime manifest {name} is invalid")
        for name in (
            "interpreter_executable",
            "python_version",
            "repository_root",
            "repository_revision",
        ):
            if not isinstance(payload[name], str) or not payload[name]:
                raise ClosedWorkerProtocolError(f"runtime manifest {name} is invalid")
        environment = payload["environment"]
        expected_environment_fields = {
            "python_prefix",
            "python_base_prefix",
            "platform",
            "implementation",
            "conda_prefix",
            "cuda_visible_devices",
            "nvidia_visible_devices",
        }
        if not isinstance(environment, Mapping) or set(environment) != expected_environment_fields:
            raise ClosedWorkerProtocolError("runtime manifest environment fields are invalid")
        for name, value in environment.items():
            if value is not None and (not isinstance(value, str) or not value):
                raise ClosedWorkerProtocolError(f"runtime manifest environment {name} is invalid")
        if payload["environment_sha256"] != canonical_sha256(dict(environment)):
            raise ClosedWorkerProtocolError("runtime manifest environment digest is invalid")
        site_paths = payload["site_package_paths"]
        if (
            not isinstance(site_paths, list)
            or any(not isinstance(path, str) or not path for path in site_paths)
            or len(set(site_paths)) != len(site_paths)
            or site_paths != sorted(site_paths)
        ):
            raise ClosedWorkerProtocolError("runtime manifest site-package paths are invalid")
        for name in ("worker_script_relative_path", "gate_relative_path"):
            if not isinstance(payload[name], str) or _RELATIVE_PATH_RE.fullmatch(payload[name]) is None:
                raise ClosedWorkerProtocolError(f"runtime manifest {name} is invalid")
        if payload["lane_id"] not in {"warp", "cuda"}:
            raise ClosedWorkerProtocolError("runtime manifest lane_id is invalid")
        ordinal = payload["device_ordinal"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal <= 1023:
            raise ClosedWorkerProtocolError("runtime manifest device_ordinal is invalid")
        for name in ("gate_sha256", "workload_set_sha256", "ast_policy_sha256"):
            if not isinstance(payload[name], str) or _SHA256_RE.fullmatch(payload[name]) is None:
                raise ClosedWorkerProtocolError(f"runtime manifest {name} is invalid")
        if payload["repository_clean"] is not True:
            raise ClosedWorkerProtocolError("closed worker requires a clean repository")
        if payload["gpu_preparation_enabled"] is not False:
            raise ClosedWorkerProtocolError(
                "C4c0 runtime manifest must keep product GPU preparation disabled"
            )
        for name in (
            "control_idle_timeout_ms",
            "response_write_timeout_ms",
            "abort_watchdog_ms",
        ):
            value = payload[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ClosedWorkerProtocolError(f"runtime manifest {name} must be positive")
        raw_sources = payload["source_digests"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ClosedWorkerProtocolError("runtime manifest source_digests is invalid")
        sources = tuple(FrozenSourceDigest.from_payload(item) for item in raw_sources)
        paths = tuple(item.relative_path for item in sources)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ClosedWorkerProtocolError("runtime manifest sources must be unique and sorted")
        source_hash = canonical_sha256([item.to_payload() for item in sources])
        if payload["source_manifest_sha256"] != source_hash:
            raise ClosedWorkerProtocolError("runtime manifest source digest is invalid")
        return cls(
            schema_version=CLOSED_WORKER_RUNTIME_MANIFEST_VERSION,
            protocol_version=CLOSED_WORKER_PROTOCOL_VERSION,
            trust_model_version=CLOSED_WORKER_TRUST_MODEL_VERSION,
            state_machine_version=CLOSED_WORKER_STATE_MACHINE_VERSION,
            interpreter_executable=payload["interpreter_executable"],
            python_version=payload["python_version"],
            environment=MappingProxyType(dict(environment)),
            environment_sha256=payload["environment_sha256"],
            site_package_paths=tuple(site_paths),
            repository_root=payload["repository_root"],
            repository_revision=payload["repository_revision"],
            repository_clean=True,
            worker_script_relative_path=payload["worker_script_relative_path"],
            lane_id=payload["lane_id"],
            device_ordinal=ordinal,
            gate_relative_path=payload["gate_relative_path"],
            gate_sha256=payload["gate_sha256"],
            workload_set_sha256=payload["workload_set_sha256"],
            source_digests=sources,
            source_manifest_sha256=source_hash,
            ast_policy_sha256=payload["ast_policy_sha256"],
            control_idle_timeout_ms=payload["control_idle_timeout_ms"],
            response_write_timeout_ms=payload["response_write_timeout_ms"],
            abort_watchdog_ms=payload["abort_watchdog_ms"],
            gpu_preparation_enabled=False,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "trust_model_version": self.trust_model_version,
            "state_machine_version": self.state_machine_version,
            "interpreter_executable": self.interpreter_executable,
            "python_version": self.python_version,
            "environment": dict(self.environment),
            "environment_sha256": self.environment_sha256,
            "site_package_paths": list(self.site_package_paths),
            "repository_root": self.repository_root,
            "repository_revision": self.repository_revision,
            "repository_clean": self.repository_clean,
            "worker_script_relative_path": self.worker_script_relative_path,
            "lane_id": self.lane_id,
            "device_ordinal": self.device_ordinal,
            "gate_relative_path": self.gate_relative_path,
            "gate_sha256": self.gate_sha256,
            "workload_set_sha256": self.workload_set_sha256,
            "source_digests": [item.to_payload() for item in self.source_digests],
            "source_manifest_sha256": self.source_manifest_sha256,
            "ast_policy_sha256": self.ast_policy_sha256,
            "control_idle_timeout_ms": self.control_idle_timeout_ms,
            "response_write_timeout_ms": self.response_write_timeout_ms,
            "abort_watchdog_ms": self.abort_watchdog_ms,
            "gpu_preparation_enabled": self.gpu_preparation_enabled,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class ClosedNativeLaneWorkerEvidence:
    schema_version: str
    protocol_version: str
    trust_model_version: str
    state_machine_version: str
    runtime_manifest_sha256: str
    source_manifest_sha256: str
    ast_policy_sha256: str
    gate_sha256: str
    workload_set_sha256: str
    lane_id: LaneId
    device_ordinal: int
    process_id: int
    process_start_nonce: str
    worker_session_id: str
    interpreter_executable_sha256: str
    callback_surface_enabled: bool
    object_deserialization_enabled: bool
    gpu_preparation_enabled: bool
    cuda_capable_module_names: tuple[str, ...]
    repository_module_paths: tuple[str, ...]
    repository_import_ledger_sha256: str

    @classmethod
    def from_payload(cls, payload: object) -> "ClosedNativeLaneWorkerEvidence":
        fields = {
            "schema_version",
            "protocol_version",
            "trust_model_version",
            "state_machine_version",
            "runtime_manifest_sha256",
            "source_manifest_sha256",
            "ast_policy_sha256",
            "gate_sha256",
            "workload_set_sha256",
            "lane_id",
            "device_ordinal",
            "process_id",
            "process_start_nonce",
            "worker_session_id",
            "interpreter_executable_sha256",
            "callback_surface_enabled",
            "object_deserialization_enabled",
            "gpu_preparation_enabled",
            "cuda_capable_module_names",
            "repository_module_paths",
            "repository_import_ledger_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ClosedWorkerProtocolError("closed-worker evidence fields are invalid")
        if (
            payload["schema_version"] != CLOSED_WORKER_EVIDENCE_VERSION
            or payload["protocol_version"] != CLOSED_WORKER_PROTOCOL_VERSION
            or payload["trust_model_version"] != CLOSED_WORKER_TRUST_MODEL_VERSION
            or payload["state_machine_version"] != CLOSED_WORKER_STATE_MACHINE_VERSION
        ):
            raise ClosedWorkerProtocolError("closed-worker evidence version is invalid")
        for name in (
            "runtime_manifest_sha256",
            "source_manifest_sha256",
            "ast_policy_sha256",
            "gate_sha256",
            "workload_set_sha256",
            "interpreter_executable_sha256",
            "repository_import_ledger_sha256",
        ):
            value = payload[name]
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ClosedWorkerProtocolError(f"closed-worker evidence {name} is invalid")
        if payload["lane_id"] not in {"warp", "cuda"}:
            raise ClosedWorkerProtocolError("closed-worker evidence lane is invalid")
        for name in ("device_ordinal", "process_id"):
            value = payload[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ClosedWorkerProtocolError(f"closed-worker evidence {name} is invalid")
        for name in ("process_start_nonce", "worker_session_id"):
            value = payload[name]
            if not isinstance(value, str) or not value:
                raise ClosedWorkerProtocolError(f"closed-worker evidence {name} is invalid")
        for name in (
            "callback_surface_enabled",
            "object_deserialization_enabled",
            "gpu_preparation_enabled",
        ):
            if payload[name] is not False:
                raise ClosedWorkerProtocolError(f"closed-worker evidence {name} must remain false")
        modules = payload["cuda_capable_module_names"]
        if not isinstance(modules, list) or modules:
            raise ClosedWorkerProtocolError("C4c0 worker imported CUDA-capable modules before activation")
        repository_paths = payload["repository_module_paths"]
        if (
            not isinstance(repository_paths, list)
            or repository_paths != sorted(set(repository_paths))
            or any(
                not isinstance(path, str) or _RELATIVE_PATH_RE.fullmatch(path) is None
                for path in repository_paths
            )
            or canonical_sha256(repository_paths) != payload["repository_import_ledger_sha256"]
        ):
            raise ClosedWorkerProtocolError("closed-worker repository import ledger is invalid")
        return cls(
            schema_version=CLOSED_WORKER_EVIDENCE_VERSION,
            protocol_version=CLOSED_WORKER_PROTOCOL_VERSION,
            trust_model_version=CLOSED_WORKER_TRUST_MODEL_VERSION,
            state_machine_version=CLOSED_WORKER_STATE_MACHINE_VERSION,
            runtime_manifest_sha256=payload["runtime_manifest_sha256"],
            source_manifest_sha256=payload["source_manifest_sha256"],
            ast_policy_sha256=payload["ast_policy_sha256"],
            gate_sha256=payload["gate_sha256"],
            workload_set_sha256=payload["workload_set_sha256"],
            lane_id=payload["lane_id"],
            device_ordinal=payload["device_ordinal"],
            process_id=payload["process_id"],
            process_start_nonce=payload["process_start_nonce"],
            worker_session_id=payload["worker_session_id"],
            interpreter_executable_sha256=payload["interpreter_executable_sha256"],
            callback_surface_enabled=False,
            object_deserialization_enabled=False,
            gpu_preparation_enabled=False,
            cuda_capable_module_names=(),
            repository_module_paths=tuple(repository_paths),
            repository_import_ledger_sha256=payload["repository_import_ledger_sha256"],
        )

    def to_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def _require_exact_fields(value: object, fields: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ClosedWorkerProtocolError(f"{label} fields are invalid")
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ClosedWorkerProtocolError(f"{label} is invalid")
    return value


def _token(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise ClosedWorkerProtocolError(f"{label} is invalid")
    return value


def validate_request(value: object) -> dict[str, object]:
    request = _require_exact_fields(
        value,
        {"protocol_version", "sequence", "operation", "arguments"},
        label="closed-worker request",
    )
    if request["protocol_version"] != CLOSED_WORKER_PROTOCOL_VERSION:
        raise ClosedWorkerProtocolError("request protocol version is invalid")
    sequence = request["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or not 0 <= sequence < 2**63:
        raise ClosedWorkerProtocolError("request sequence is invalid")
    operation = request["operation"]
    if operation not in {
        "HELLO",
        "OPEN_FROZEN_SESSION",
        "PREPARE_FROZEN_ROW",
        "CANCEL_PREPARED_SAMPLE",
        "EXECUTE_PREPARED_SAMPLE",
        "RESOLVE_PREPARED_SAMPLE",
        "RELEASE_RESOLVED_SAMPLE",
        "CLOSE_SESSION",
    }:
        raise ClosedWorkerProtocolError("request operation is invalid")
    arguments = request["arguments"]
    if operation == "HELLO":
        args = _require_exact_fields(
            arguments, {"manifest_sha256", "lane_id", "device_ordinal"}, label="HELLO arguments"
        )
        if (
            not isinstance(args["manifest_sha256"], str)
            or _SHA256_RE.fullmatch(args["manifest_sha256"]) is None
        ):
            raise ClosedWorkerProtocolError("HELLO manifest_sha256 is invalid")
        if args["lane_id"] not in {"warp", "cuda"}:
            raise ClosedWorkerProtocolError("HELLO lane_id is invalid")
        ordinal = args["device_ordinal"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 0 <= ordinal <= 1023:
            raise ClosedWorkerProtocolError("HELLO device_ordinal is invalid")
    elif operation == "OPEN_FROZEN_SESSION":
        args = _require_exact_fields(
            arguments, {"gate_sha256", "workload_set_sha256"}, label="OPEN arguments"
        )
        for name in ("gate_sha256", "workload_set_sha256"):
            if not isinstance(args[name], str) or _SHA256_RE.fullmatch(args[name]) is None:
                raise ClosedWorkerProtocolError(f"OPEN {name} is invalid")
    elif operation == "PREPARE_FROZEN_ROW":
        args = _require_exact_fields(
            arguments,
            {"row_id", "case_id", "repeat_index", "sample_count"},
            label="PREPARE arguments",
        )
        _identifier(args["row_id"], label="PREPARE row_id")
        _identifier(args["case_id"], label="PREPARE case_id")
        for name, maximum in (("repeat_index", 1_000_000), ("sample_count", 1_000_000)):
            number = args[name]
            lower = 0 if name == "repeat_index" else 1
            if not isinstance(number, int) or isinstance(number, bool) or not lower <= number <= maximum:
                raise ClosedWorkerProtocolError(f"PREPARE {name} is invalid")
    elif operation in {
        "CANCEL_PREPARED_SAMPLE",
        "EXECUTE_PREPARED_SAMPLE",
        "RESOLVE_PREPARED_SAMPLE",
        "RELEASE_RESOLVED_SAMPLE",
    }:
        field = {
            "CANCEL_PREPARED_SAMPLE": "prepared_token",
            "EXECUTE_PREPARED_SAMPLE": "prepared_token",
            "RESOLVE_PREPARED_SAMPLE": "pending_token",
            "RELEASE_RESOLVED_SAMPLE": "resolved_token",
        }[operation]
        args = _require_exact_fields(arguments, {field}, label=f"{operation} arguments")
        _token(args[field], label=f"{operation} {field}")
    else:
        _require_exact_fields(arguments, set(), label="CLOSE_SESSION arguments")
    return {
        "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
        "sequence": sequence,
        "operation": operation,
        "arguments": dict(arguments),
    }


def validate_response_envelope(value: object) -> dict[str, object]:
    response = _require_exact_fields(
        value,
        {"protocol_version", "sequence", "status", "payload"},
        label="closed-worker response",
    )
    if response["protocol_version"] != CLOSED_WORKER_PROTOCOL_VERSION:
        raise ClosedWorkerProtocolError("response protocol version is invalid")
    if response["status"] not in {"ok", "error"}:
        raise ClosedWorkerProtocolError("response status is invalid")
    sequence = response["sequence"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ClosedWorkerProtocolError("response sequence is invalid")
    _validate_canonical_value(response["payload"], depth=1)
    return dict(response)
