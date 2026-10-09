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

NATIVE_AST_POLICY_VERSION = "q54.p12.4.closed-worker-ast-policy.v16"
_BOOTSTRAP_PATH = "optical_worker/native_bootstrap.py"
_BOOTSTRAP_SOURCE_SHA256 = "8448dc8aab0b99d16830db3bb10243f0beb71f0cfa72fc649e85da52e6310877"
_BOOTSTRAP_CLASS_PINS = {
    "_RuntimeObservation": "cfb64f25a4bf1774a5e67128ac7e1b3b5a06ab9800ddd2f9b51d39e2b9537b3e",
    "_CapturedSourceLoader": "92da4e467035fe308c4ec6af32f2d05831adc7e5015f5f400545d1541ad5556b",
    "_CapturedSourceFinder": "67bbea24396197e51d2acfb4076c98d9b514dc34d5d416e16f81d88624105e92",
    "_ExternalSourceLoader": "a52ecdc72961066c61f5341522e764153210dce1cb59a1fc69d8d83f07790f53",
    "_ExternalAdmission": "157107d2ca7210f687ec3898d072231ccb993f061647cc740982262c8b004070",
    "_ExternalNamespaceLoader": "524801ff222139e86fa76fa918b42a20ee61d6bcfe9e783542c880da3cad46bf",
    "_RuntimeSetupBinding": "85bf4bb6a12a3ab17807393c27993ca319e5b614eda9c14d4f1d6c0b042f6112",
    "_SourceSession": "424e3176c10680ad090bf54878d93f528964a0d6c9c9de6a37e7d83a2dc5e911",
}
_BOOTSTRAP_IMPORTS = frozenset(("importlib.abc", "importlib.machinery", "importlib.util"))
_BOOTSTRAP_USES = frozenset((
    "importlib.abc.SourceLoader", "importlib.abc.MetaPathFinder", "importlib.abc.Loader",
    "importlib.machinery.SourceFileLoader", "importlib.machinery.ExtensionFileLoader",
    "importlib.machinery.PathFinder.find_spec", "importlib.util.spec_from_loader",
    "importlib.machinery.BuiltinImporter", "importlib.machinery.FrozenImporter",
    "importlib.machinery.PathFinder", "importlib.machinery.ModuleSpec",
))
_SNAPSHOT_PATH = "optics/warp_native_render.py"
_SNAPSHOT_FUNCTION = "_render_launch_snapshot"
# Exact LF Git/source-release bytes reviewed in C2c4 (4b7bf4c). Deliberately
# not universal-newline-normalized at verification time. A CRLF checkout is
# not this POSIX worker release; packaging must supply the reviewed bytes.
_SNAPSHOT_SOURCE_SHA256 = "86eee027aab7f47168e40fb6e1301d8ca3e5a5730bfc068158359905e2a567ea"
_SNAPSHOT_FUNCTION_SHA256 = "725c7f315fdba4ad6df78e31a149865fd97e26da3c95df24678dbadba0459199"
# Existing CPU regex construction is not Python code compilation. Keep this
# exception release-pinned; do not generally admit arbitrary obj.compile().
_REGEX_SOURCE_PINS = {
    "optical_worker/protocol.py": "38ff7919ee1b41656a50cf1d6c4fa3280a1f867aac07baaf4a4adbc4189a6470",
    "optical_worker/lifecycle.py": "2ca76eac3b72fc99f22d6c671e51a9a86724609575b0d61c5b101d3f7a6334dc"
}
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
    "forbidden_builtin_imports": sorted(_FORBIDDEN_NAMES | {"*"}),
    "literal_lookup_policy": "reject forbidden name/attribute literals in calls, keywords and subscripts",
    "existing_regex_source_pins": _REGEX_SOURCE_PINS,
    "source_loader_exception": {
        "path": _BOOTSTRAP_PATH,
        "source_sha256": _BOOTSTRAP_SOURCE_SHA256,
        "class_source_pins": _BOOTSTRAP_CLASS_PINS,
        "imports": sorted(_BOOTSTRAP_IMPORTS),
        "uses": sorted(_BOOTSTRAP_USES),
    },
    "source_roots": sorted(_SOURCE_ROOTS),
    "launcher_paths": sorted(_LAUNCHER_PATHS),
    "ctypes_exception": {
        "path": _SNAPSHOT_PATH,
        "source_sha256": _SNAPSHOT_SOURCE_SHA256,
        "function": _SNAPSHOT_FUNCTION,
        "function_source_sha256": _SNAPSHOT_FUNCTION_SHA256,
        "function_source_encoding": "exact physical UTF-8 lines including final newline",
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


def _validate_tree(relative_path: str, tree: ast.Module, source: bytes) -> None:
    class_pins = _BOOTSTRAP_CLASS_PINS if relative_path == _BOOTSTRAP_PATH else {}
    importlib_imports = _BOOTSTRAP_IMPORTS if relative_path == _BOOTSTRAP_PATH else frozenset()
    importlib_uses = _BOOTSTRAP_USES if relative_path == _BOOTSTRAP_PATH else frozenset()
    if class_pins:
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        for name, digest in class_pins.items():
            node = classes.get(name)
            if node is None or hashlib.sha256(b"".join(
                source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
            )).hexdigest() != digest:
                raise ClosedWorkerProtocolError("source-only loader definitions have drifted")
    snapshots = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == _SNAPSHOT_FUNCTION]
    snapshot = snapshots[0] if relative_path == _SNAPSHOT_PATH and len(snapshots) == 1 else None
    if relative_path == _SNAPSHOT_PATH:
        # ast.dump differs across Python3.12/3.14. Use AST only to locate the
        # function, then pin its actual bytes, with no source normalization.
        if snapshot is None or hashlib.sha256(b"".join(
            source.splitlines(keepends=True)[snapshot.lineno - 1:snapshot.end_lineno]
        )).hexdigest() != _SNAPSHOT_FUNCTION_SHA256:
            raise ClosedWorkerProtocolError("read-only packed snapshot function has drifted")
    snapshot_nodes = set(ast.walk(snapshot)) if snapshot is not None else set()
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    ctypes_imports = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.partition(".")[0]
                if root == "importlib" and importlib_imports:
                    if (alias.name not in importlib_imports or alias.asname is not None
                            or len(node.names) != 1 or node not in tree.body):
                        raise ClosedWorkerProtocolError("source bootstrap importlib import has drifted")
                elif root == "ctypes" and relative_path == _SNAPSHOT_PATH:
                    if (alias.name != "ctypes" or alias.asname is not None
                            or len(node.names) != 1 or node not in tree.body):
                        raise ClosedWorkerProtocolError("ctypes import must be the sole plain module import")
                    ctypes_imports += 1
                elif root in _FORBIDDEN_IMPORTS:
                    raise ClosedWorkerProtocolError("native source imports a forbidden module")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").partition(".")[0] in _FORBIDDEN_IMPORTS:
                raise ClosedWorkerProtocolError("native source imports a forbidden module")
            if node.level == 0 and node.module == "builtins" and any(
                alias.name in _FORBIDDEN_NAMES or alias.name == "*" for alias in node.names
            ):
                raise ClosedWorkerProtocolError("native source imports a forbidden dynamic builtin")
        elif isinstance(node, ast.Name):
            if node.id in _FORBIDDEN_NAMES:
                raise ClosedWorkerProtocolError("native source uses or aliases a forbidden dynamic name")
            if node.id == "importlib":
                chain, current = [node.id], node
                while isinstance(parents.get(current), ast.Attribute) and parents[current].value is current:
                    current = parents[current]
                    chain.append(current.attr)
                if not (isinstance(node.ctx, ast.Load)
                        and isinstance(current.ctx, ast.Load) and ".".join(chain) in importlib_uses):
                    raise ClosedWorkerProtocolError("importlib reference escapes the source bootstrap")
            if node.id == "ctypes":
                parent = parents.get(node)
                if not (
                    node in snapshot_nodes and isinstance(node.ctx, ast.Load)
                    and isinstance(parent, ast.Attribute) and parent.value is node
                    and isinstance(parent.ctx, ast.Load) and parent.attr in _CTYPES_ATTRIBUTES
                ):
                    raise ClosedWorkerProtocolError("ctypes reference escapes the read-only snapshot")
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTES | _FORBIDDEN_NAMES:
            if not (
                (relative_path in _REGEX_SOURCE_PINS and node.attr == "compile"
                and isinstance(node.value, ast.Name) and node.value.id == "re"
                and isinstance(node.ctx, ast.Load))
            ):
                raise ClosedWorkerProtocolError("native source reaches a forbidden runtime attribute")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Reject literal reflection spelling too, not just obj.attr syntax.
            parent = parents.get(node)
            if node.value in _FORBIDDEN_ATTRIBUTES | _FORBIDDEN_NAMES and (
                isinstance(parent, ast.Subscript)
                or isinstance(parent, (ast.Call, ast.keyword))
            ):
                # Conservatively reject dangerous literal lookup spellings in
                # calls too, independent of direct/qualified/imported alias
                # naming. This is a restricted source dialect, not general
                # Python dataflow analysis; policy-definition strings in data
                # containers remain valid and do not execute a lookup.
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
    _validate_tree(relative_path, tree, source)
    if relative_path == _SNAPSHOT_PATH and actual != _SNAPSHOT_SOURCE_SHA256:
        raise ClosedWorkerProtocolError("ctypes exception source is not the reviewed release")
    if relative_path in _REGEX_SOURCE_PINS and actual != _REGEX_SOURCE_PINS[relative_path]:
        raise ClosedWorkerProtocolError("regex exception source is not the reviewed release")
    if relative_path == _BOOTSTRAP_PATH and actual != _BOOTSTRAP_SOURCE_SHA256:
        raise ClosedWorkerProtocolError("source bootstrap exception is not the reviewed release")


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
