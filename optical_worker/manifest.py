"""Frozen source/environment validation for the closed worker."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .protocol import (
    CLOSED_WORKER_PROTOCOL_VERSION,
    CLOSED_WORKER_STATE_MACHINE_VERSION,
    CLOSED_WORKER_TRUST_MODEL_VERSION,
    MAX_NESTING_DEPTH,
    MAX_OBJECT_FIELDS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_STRING_BYTES,
    ClosedWorkerProtocolError,
    ClosedWorkerRuntimeManifest,
    canonical_sha256,
)

_RUNTIME_CONTRACT_RELATIVE_PATH = "benchmarks/results/q54_p12_4/closed_worker_runtime_contract.json"
_INTEGRATED_RUNTIME_CONTRACT_PATH = "optical_worker/data/integrated_worker_runtime_contract_v3.json"
_INTEGRATED_RUNTIME_CONTRACT_SHA256 = "b3fc6ca25c2af42ceb7e8d509ed4c073a777e39512ba9cf6096c44a7f01e313c"

AST_POLICY = {
    "schema_version": "q54.p12.4.closed-worker-ast-policy.v1",
    "forbidden_import_roots": ["cloudpickle", "ctypes", "importlib", "pickle", "runpy"],
    "forbidden_calls": ["__import__", "compile", "eval", "exec"],
    "forbidden_attributes": [
        "__closure__",
        "__wrapped__",
        "wp_cuda_event_record",
        "wp_cuda_launch_kernel",
    ],
    "source_suffix": ".py",
    "bytecode_only_allowed": False,
}
AST_POLICY_SHA256 = canonical_sha256(AST_POLICY)


def current_environment_identity() -> dict[str, str | None]:
    """Return frozen process metadata without importing a CUDA runtime."""

    return {
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def workload_set_sha256(gate: dict[str, object]) -> str:
    """One workload identity for both legacy and source-bound native manifests."""
    return canonical_sha256({
        "cases": gate["cases"], "boundaries": gate["boundaries"], "mandatory_rows": gate["mandatory_rows"],
    })


def build_integrated_worker_runtime_manifest(source, *, pass_mode, go2_model_dir):
    """Describe the actual captured session before native imports, without discovery.

    This is a manifest builder, not admission. The source session and guarded
    controller must separately register the worker which consumes this record.
    """
    from .arena import _FROZEN_GATE_PATH, _FROZEN_GATE_SHA256
    from .cases import canonical_json_bytes as canonical_gate_bytes
    from .native_source_policy import NATIVE_AST_POLICY_SHA256
    from .native_sources import CATALOG_PATH, _read_contained
    from .protocol import (
        INTEGRATED_WORKER_RUNTIME_MANIFEST_VERSION,
        FrozenSourceDigest,
        IntegratedWorkerRuntimeManifest,
    )

    source.check_live()
    if source.phase != "imported" or source.runtime_observation.environment is None:
        raise ClosedWorkerProtocolError("native manifest requires CPU definitions and frozen runtime custody")
    encoded = _read_contained(source.root, _INTEGRATED_RUNTIME_CONTRACT_PATH)
    if hashlib.sha256(encoded).hexdigest() != _INTEGRATED_RUNTIME_CONTRACT_SHA256:
        raise ClosedWorkerProtocolError("native runtime contract differs from its frozen release")
    contract = json.loads(encoded)
    configuration = source.bound_controller()._guard_client.configuration.to_payload()
    expected_limits = {
        "schema_version": "q54.p12.4.closed-worker-runtime-contract.v3",
        "guard_protocol_version": configuration["protocol_version"],
        "source_catalog_relative_path": CATALOG_PATH,
        "request_limit_bytes": MAX_REQUEST_BYTES, "response_limit_bytes": MAX_RESPONSE_BYTES,
        "nesting_depth_limit": MAX_NESTING_DEPTH, "object_field_limit": MAX_OBJECT_FIELDS,
        "string_limit_utf8_bytes": MAX_STRING_BYTES,
    }
    if any(contract[key] != value for key, value in expected_limits.items()):
        raise ClosedWorkerProtocolError("native runtime contract differs from protocol implementation")
    gate_bytes = _read_contained(source.root, _FROZEN_GATE_PATH)
    gate = json.loads(gate_bytes)
    if hashlib.sha256(canonical_gate_bytes(gate)).hexdigest() != _FROZEN_GATE_SHA256:
        raise ClosedWorkerProtocolError("native manifest gate differs from the frozen workload")
    environment = current_environment_identity()
    deployment = source.runtime_observation.environment
    sites = tuple(sorted(row["path"] for row in deployment.to_payload()["roots"] if row["kind"] == "site"))
    sources = tuple(FrozenSourceDigest(item.relative_path, item.sha256) for item in source.snapshot.sources)
    manifest = IntegratedWorkerRuntimeManifest(
        schema_version=INTEGRATED_WORKER_RUNTIME_MANIFEST_VERSION,
        protocol_version=contract["protocol_version"], trust_model_version=contract["trust_model_version"],
        state_machine_version=contract["state_machine_version"],
        interpreter_executable=str(Path(sys.executable).resolve()), python_version=platform.python_version(),
        environment=environment, environment_sha256=canonical_sha256(environment), site_package_paths=sites,
        repository_root=str(source.root), repository_revision=source.revision, repository_clean=True,
        worker_script_relative_path=contract["worker_script_relative_path"], lane_id=source.lane,
        device_ordinal=configuration["device_ordinal"], gate_relative_path=_FROZEN_GATE_PATH,
        gate_sha256=hashlib.sha256(gate_bytes).hexdigest(), workload_set_sha256=workload_set_sha256(gate),
        source_digests=sources,
        source_manifest_sha256=canonical_sha256([item.to_payload() for item in sources]),
        ast_policy_sha256=NATIVE_AST_POLICY_SHA256,
        control_idle_timeout_ms=contract["control_idle_timeout_ms"],
        response_write_timeout_ms=contract["response_write_timeout_ms"],
        abort_watchdog_ms=contract["abort_watchdog_ms"],
        gpu_preparation_enabled=contract["gpu_preparation_enabled"],
        guard_configuration=configuration, source_catalog_sha256=source.catalog_sha256,
        runtime_environment_sha256=deployment.sha256, case_id=source.runtime_observation.inputs["case_id"],
        pass_mode=pass_mode, go2_model_dir=go2_model_dir,
    )
    return IntegratedWorkerRuntimeManifest.from_payload(manifest.to_payload())


def validate_integrated_worker_runtime_manifest(manifest, source):
    """Bind all launch metadata to the current source, process and frozen contract."""
    expected = build_integrated_worker_runtime_manifest(
        source, pass_mode=manifest.pass_mode, go2_model_dir=manifest.go2_model_dir,
    )
    if manifest.to_payload() != expected.to_payload():
        raise ClosedWorkerProtocolError("native manifest differs from its actual process and frozen contract")


def validate_repository_import_ledger(
    manifest: ClosedWorkerRuntimeManifest,
) -> tuple[str, ...]:
    """Reject every loaded repository module outside the frozen source set."""

    root = Path(manifest.repository_root).resolve()
    allowed = {source.relative_path for source in manifest.source_digests}
    loaded: set[str] = set()
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            continue
        candidate = Path(origin).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.suffix in {".pyc", ".pyo"}:
            spec = getattr(module, "__spec__", None)
            source_origin = getattr(spec, "origin", None)
            if isinstance(source_origin, str):
                candidate = Path(source_origin).resolve()
                try:
                    relative = candidate.relative_to(root)
                except ValueError as exc:
                    raise ClosedWorkerProtocolError(
                        "repository bytecode module has no in-repository source"
                    ) from exc
        path = relative.as_posix()
        if path not in allowed:
            raise ClosedWorkerProtocolError(
                f"worker loaded repository module outside source allowlist: {path!r}"
            )
        loaded.add(path)
    return tuple(sorted(loaded))


def validate_trusted_source_ast(path: Path) -> None:
    if path.suffix != ".py":
        raise ClosedWorkerProtocolError("trusted worker source must be a Python source file")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ClosedWorkerProtocolError(f"trusted worker source {path.name!r} cannot be parsed") from exc
    forbidden_imports = set(AST_POLICY["forbidden_import_roots"])
    forbidden_calls = set(AST_POLICY["forbidden_calls"])
    forbidden_attributes = set(AST_POLICY["forbidden_attributes"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
            if roots & forbidden_imports:
                raise ClosedWorkerProtocolError("trusted worker source imports a forbidden module")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").partition(".")[0]
            if root in forbidden_imports:
                raise ClosedWorkerProtocolError("trusted worker source imports a forbidden module")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ClosedWorkerProtocolError("trusted worker source uses a forbidden dynamic call")
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
            raise ClosedWorkerProtocolError("trusted worker source reaches a forbidden runtime attribute")


def validate_runtime_manifest_against_process(
    manifest: ClosedWorkerRuntimeManifest,
    *,
    require_clean_repository: bool = True,
) -> None:
    """Validate code/environment identity before any CUDA-capable import."""

    root = Path(manifest.repository_root).resolve()
    if not root.is_dir():
        raise ClosedWorkerProtocolError("runtime manifest repository root does not exist")
    if Path(sys.executable).resolve() != Path(manifest.interpreter_executable).resolve():
        raise ClosedWorkerProtocolError("runtime interpreter does not match the manifest")
    if platform.python_version() != manifest.python_version:
        raise ClosedWorkerProtocolError("runtime Python version does not match the manifest")
    environment = current_environment_identity()
    if environment != dict(manifest.environment) or canonical_sha256(environment) != (
        manifest.environment_sha256
    ):
        raise ClosedWorkerProtocolError("runtime environment does not match the manifest")
    if manifest.ast_policy_sha256 != AST_POLICY_SHA256:
        raise ClosedWorkerProtocolError("runtime AST policy does not match the implementation")
    for site_path in manifest.site_package_paths:
        candidate = Path(site_path)
        if not candidate.is_absolute() or not candidate.resolve().is_dir():
            raise ClosedWorkerProtocolError("runtime site-package path is invalid")

    contract_path = root / _RUNTIME_CONTRACT_RELATIVE_PATH
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosedWorkerProtocolError("closed-worker runtime contract cannot be read") from exc
    expected_contract = {
        "schema_version": "q54.p12.4.closed-worker-runtime-contract.v2",
        "protocol_version": CLOSED_WORKER_PROTOCOL_VERSION,
        "trust_model_version": CLOSED_WORKER_TRUST_MODEL_VERSION,
        "state_machine_version": CLOSED_WORKER_STATE_MACHINE_VERSION,
        "control_idle_timeout_ms": manifest.control_idle_timeout_ms,
        "response_write_timeout_ms": manifest.response_write_timeout_ms,
        "abort_watchdog_ms": manifest.abort_watchdog_ms,
        "gpu_preparation_enabled": False,
        "worker_script_relative_path": manifest.worker_script_relative_path,
        "supervisor_script_relative_path": "tools/optical_closed_worker_supervisor.py",
        "trusted_source_paths": [source.relative_path for source in manifest.source_digests],
        "request_limit_bytes": MAX_REQUEST_BYTES,
        "response_limit_bytes": MAX_RESPONSE_BYTES,
        "nesting_depth_limit": MAX_NESTING_DEPTH,
        "object_field_limit": MAX_OBJECT_FIELDS,
        "string_limit_utf8_bytes": MAX_STRING_BYTES,
    }
    if contract != expected_contract:
        raise ClosedWorkerProtocolError("runtime manifest does not match the frozen C4c0 contract")

    actual_sources: list[dict[str, object]] = []
    for source in manifest.source_digests:
        candidate = (root / source.relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ClosedWorkerProtocolError("trusted worker source escapes the repository") from exc
        if not candidate.is_file() or sha256_file(candidate) != source.sha256:
            raise ClosedWorkerProtocolError(
                f"trusted worker source {source.relative_path!r} does not match the manifest"
            )
        validate_trusted_source_ast(candidate)
        actual_sources.append(source.to_payload())
    if canonical_sha256(actual_sources) != manifest.source_manifest_sha256:
        raise ClosedWorkerProtocolError("trusted worker source manifest does not replay")

    gate_path = (root / manifest.gate_relative_path).resolve()
    try:
        gate_path.relative_to(root)
    except ValueError as exc:
        raise ClosedWorkerProtocolError("runtime gate path escapes the repository") from exc
    if not gate_path.is_file() or sha256_file(gate_path) != manifest.gate_sha256:
        raise ClosedWorkerProtocolError("runtime gate does not match the manifest")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != manifest.repository_revision:
        raise ClosedWorkerProtocolError("runtime repository revision does not match the manifest")
    if require_clean_repository:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ClosedWorkerProtocolError("runtime repository is not clean")
