"""CPU-only source checks for the not-yet-admitted integrated worker.

This is a byte/AST verifier, not a loader, execution capability or Python
sandbox. The eventual trusted bootstrap must supply its reviewed literal
per-lane manifest and enforce the actual import ledger. C0 does not use this
module and retains its v1 policy and closed preparation gate.
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path, PurePosixPath

from .manifest import AST_POLICY
from .protocol import ClosedWorkerProtocolError, canonical_sha256

NATIVE_AST_POLICY_VERSION = "q54.p12.4.closed-worker-ast-policy.v2"
_SNAPSHOT_PATH = "optics/warp_native_render.py"
_SNAPSHOT_FUNCTION = "_render_launch_snapshot"
# Exact LF Git/source-release bytes reviewed in C2c4 (4b7bf4c). Deliberately
# not universal-newline-normalized at verification time. A CRLF checkout is
# not this POSIX worker release; packaging must supply the reviewed bytes.
_SNAPSHOT_SOURCE_SHA256 = "636019f65594ac83dce1b386f4b3f684053e419a8cf81ba5feddce237cf318e4"
_SNAPSHOT_AST_SHA256 = "aece3ed687339523c475e1d145ef0edd4214af1cc2a70750c54a5741106bc410"
_CTYPES_ATTRIBUTES = frozenset(("addressof", "c_float", "c_int64", "c_int32"))
_FORBIDDEN_IMPORTS = frozenset(AST_POLICY["forbidden_import_roots"]) | {
    "benchmarks", "examples", "tests", "tools",
}
_FORBIDDEN_NAMES = frozenset(AST_POLICY["forbidden_calls"])
_FORBIDDEN_ATTRIBUTES = frozenset(AST_POLICY["forbidden_attributes"])
_SOURCE_ROOTS = frozenset(("optical_worker", "optics", "physics", "robot", "sensing"))
_LAUNCHER_PATHS = frozenset((
    "tools/optical_closed_worker.py",
    "tools/optical_closed_worker_supervisor.py",
))
NATIVE_AST_POLICY_SHA256 = canonical_sha256({
    "schema_version": NATIVE_AST_POLICY_VERSION,
    "forbidden_import_roots": sorted(_FORBIDDEN_IMPORTS),
    "forbidden_names": sorted(_FORBIDDEN_NAMES),
    "forbidden_attributes": sorted(_FORBIDDEN_ATTRIBUTES),
    "source_roots": sorted(_SOURCE_ROOTS),
    "launcher_paths": sorted(_LAUNCHER_PATHS),
    "ctypes_exception": {
        "path": _SNAPSHOT_PATH,
        "source_sha256": _SNAPSHOT_SOURCE_SHA256,
        "function": _SNAPSHOT_FUNCTION,
        "function_ast_sha256": _SNAPSHOT_AST_SHA256,
        "attributes": sorted(_CTYPES_ATTRIBUTES),
    },
    "source_encoding": "utf-8",
    "bytecode_only_allowed": False,
})


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    if type(relative_path) is not str or not relative_path or "\\" in relative_path:
        raise ClosedWorkerProtocolError("native source needs a canonical relative POSIX path")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute() or path.as_posix() != relative_path
        or any(part in {".", ".."} or ":" in part for part in path.parts)
        or path.suffix != ".py"
        or (path.parts[0] not in _SOURCE_ROOTS and relative_path not in _LAUNCHER_PATHS)
    ):
        raise ClosedWorkerProtocolError("native source path is outside the production source scope")
    return path


def _validate_tree(relative_path: str, tree: ast.Module) -> None:
    snapshots = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == _SNAPSHOT_FUNCTION]
    snapshot = snapshots[0] if relative_path == _SNAPSHOT_PATH and len(snapshots) == 1 else None
    if relative_path == _SNAPSHOT_PATH:
        if snapshot is None or hashlib.sha256(
            ast.dump(snapshot, include_attributes=False).encode("utf-8")
        ).hexdigest() != _SNAPSHOT_AST_SHA256:
            raise ClosedWorkerProtocolError("read-only packed snapshot function has drifted")
    snapshot_nodes = set(ast.walk(snapshot)) if snapshot is not None else set()
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    ctypes_imports = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                if root == "ctypes" and relative_path == _SNAPSHOT_PATH:
                    if (alias.name != "ctypes" or alias.asname is not None
                            or len(node.names) != 1 or node not in tree.body):
                        raise ClosedWorkerProtocolError("ctypes import must be the sole plain module import")
                    ctypes_imports += 1
                elif root in _FORBIDDEN_IMPORTS:
                    raise ClosedWorkerProtocolError("native source imports a forbidden module")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").partition(".")[0] in _FORBIDDEN_IMPORTS:
                raise ClosedWorkerProtocolError("native source imports a forbidden module")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise ClosedWorkerProtocolError("native source uses or aliases a forbidden dynamic name")
            if node.id == "ctypes":
                parent = parents.get(node)
                if not (
                    node in snapshot_nodes and isinstance(node.ctx, ast.Load)
                    and isinstance(parent, ast.Attribute) and parent.value is node
                    and isinstance(parent.ctx, ast.Load) and parent.attr in _CTYPES_ATTRIBUTES
                ):
                    raise ClosedWorkerProtocolError("ctypes reference escapes the read-only snapshot")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES | _FORBIDDEN_NAMES:
            raise ClosedWorkerProtocolError("native source reaches a forbidden runtime attribute")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Reject literal reflection spelling too, not just obj.attr syntax.
            parent = parents.get(node)
            if node.value in _FORBIDDEN_ATTRIBUTES and (
                isinstance(parent, ast.Subscript)
                or (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                    and parent.func.id in {"getattr", "setattr", "delattr", "hasattr"})
            ):
                raise ClosedWorkerProtocolError("native source reflects a forbidden runtime attribute")
    if relative_path == _SNAPSHOT_PATH and ctypes_imports != 1:
        raise ClosedWorkerProtocolError("packed snapshot needs exactly one plain ctypes import")


def validate_native_source_bytes(
    relative_path: str, source: bytes, *, expected_sha256: str,
) -> None:
    """Check exact bytes supplied by a trusted manifest owner; grant no authority.

    A caller-computed expected hash proves only consistency, NOT admission.
    The integrated bootstrap/source manifest remains a separate required gate.
    The ctypes exception additionally pins the independently reviewed renderer.
    """
    _validate_relative_path(relative_path)
    if type(source) is not bytes or type(expected_sha256) is not str or not re.fullmatch(
        "[0-9a-f]{64}", expected_sha256
    ):
        raise ClosedWorkerProtocolError("native source bytes/digest are invalid")
    actual = hashlib.sha256(source).hexdigest()
    if actual != expected_sha256:
        raise ClosedWorkerProtocolError("native source bytes do not match the manifest")
    try:
        tree = ast.parse(source.decode("utf-8"), filename=relative_path)
    except (UnicodeError, SyntaxError, ValueError) as exc:
        raise ClosedWorkerProtocolError("native source cannot be parsed as UTF-8 Python") from exc
    _validate_tree(relative_path, tree)
    if relative_path == _SNAPSHOT_PATH and actual != _SNAPSHOT_SOURCE_SHA256:
        raise ClosedWorkerProtocolError("ctypes exception source is not the reviewed release")


def validate_native_source_file(
    repository_root: Path, relative_path: str, *, expected_sha256: str,
) -> None:
    """Read-only untimed check, rejecting symlinked components and escaped paths.

    No import is performed. This does not solve a later loader's TOCTOU or
    bytecode-cache policy; that loader must consume/verify its own exact bytes.
    """
    relative = _validate_relative_path(relative_path)
    root = Path(repository_root).resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ClosedWorkerProtocolError("native source contains a symlinked component")
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ClosedWorkerProtocolError("native source is not a contained file")
        source = resolved.read_bytes()
    except OSError as exc:
        raise ClosedWorkerProtocolError("native source file is unavailable") from exc
    validate_native_source_bytes(relative_path, source, expected_sha256=expected_sha256)
