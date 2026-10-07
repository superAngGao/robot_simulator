"""CPU-only deployment input for guarded runtime admission.

This validates a trusted launcher's frozen inventory, not execution or readiness.
Parsing/hashing a manifest cannot authorize an import. The captured session must
separately bind it to the launch configuration, custody and guarded importer.
No site-package discovery, resolver, backend import or CUDA query occurs here.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from .protocol import canonical_json_bytes, decode_canonical_message

RUNTIME_ENVIRONMENT_VERSION = "q54.p12.4.runtime-environment.v2"
# Literal policy, not a caller-extensible import denylist. File hashes still
# belong to the one trusted deployment inventory (not a duplicate hash table).
_DISABLED_OPTIONAL_ADAPTERS = (
    ("trimesh.path.polygons", "trimesh/path/polygons.py", "trimesh.path.path", "shapely", "polygons"),
    ("trimesh.ray.ray_pyembree", "trimesh/ray/ray_pyembree.py", "trimesh.ray", "embreex", "ray_pyembree"),
)
_PHASES = frozenset({"materialization", "runtime"})
_FILE_KINDS = frozenset({"python", "extension", "native", "metadata", "data"})
_REPOSITORY_ROOTS = frozenset(
    {
        "optical_worker",
        "optics",
        "physics",
        "robot",
        "sensing",
        "benchmarks",
        "examples",
        "tests",
        "tools",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fields(value: object, expected: set[str], label: str) -> dict:
    _require(type(value) is dict and set(value) == expected, f"invalid {label} fields")
    return value


def _text(value: object, label: str) -> str:
    _require(type(value) is str and bool(value) and "\x00" not in value, f"invalid {label}")
    return value


def _digest(value: object) -> str:
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "invalid runtime environment SHA-256",
    )
    return value


def _module(value: object) -> str:
    value = _text(value, "module name")
    _require(all(part.isidentifier() for part in value.split(".")), "invalid module name")
    _require(value.partition(".")[0] not in _REPOSITORY_ROOTS, "repository is not an external module")
    return value


def _relative(value: object) -> str:
    value = _text(value, "relative deployment path")
    path = PurePosixPath(value)
    _require(
        bool(path.parts)
        and not path.is_absolute()
        and path.as_posix() == value
        and "\\" not in value
        and all(part not in {".", ".."} and ":" not in part for part in path.parts),
        "deployment path must be a normalized contained relative path",
    )
    return value


def _absolute(value: object) -> str:
    value = _text(value, "absolute deployment path")
    path = Path(value)
    _require(path.is_absolute() and path.as_posix() == value, "deployment path must be absolute/normalized")
    _require(all(part not in {".", ".."} for part in path.parts), "deployment path contains traversal")
    return value


def _rows(value: object, key, label: str) -> list:
    _require(type(value) is list, f"{label} must be an ordered list")
    keys = [key(row) for row in value]
    _require(keys == sorted(keys) and len(keys) == len(set(keys)), f"{label} order or duplicate")
    return value


def _phase(value: object) -> None:
    _require(type(value) is str and value in _PHASES, "invalid runtime admission phase")


def _validate(payload: object) -> None:
    p = _fields(
        payload,
        {
            "schema_version",
            "interpreter",
            "roots",
            "files",
            "distributions",
            "modules",
            "absent_probes",
            "disabled_optional_modules",
            "publications",
            "namespaces",
            "settings",
        },
        "runtime environment",
    )
    _require(p["schema_version"] == RUNTIME_ENVIRONMENT_VERSION, "runtime environment version")
    interpreter = _fields(p["interpreter"], {"path", "sha256", "implementation", "version"}, "interpreter")
    _absolute(interpreter["path"])
    _digest(interpreter["sha256"])
    _require(interpreter["implementation"] == "cpython", "only frozen CPython is admitted")
    version = interpreter["version"]
    _require(
        type(version) is list and len(version) == 3 and all(type(v) is int and v >= 0 for v in version),
        "invalid interpreter version",
    )
    roots = {}
    root_kinds = {}
    for row in _rows(p["roots"], lambda r: _fields(r, {"id", "path", "kind"}, "root")["id"], "roots"):
        identity = _text(row["id"], "root ID")
        _require(identity.isidentifier(), "invalid deployment root ID")
        roots[identity] = _absolute(row["path"])
        _require(row["kind"] in {"site", "native"}, "invalid deployment root kind")
        root_kinds[identity] = row["kind"]
    _require(bool(roots) and len(set(roots.values())) == len(roots), "missing/duplicate root paths")

    def file_key(row):
        r = _fields(row, {"root", "path", "kind", "sha256", "size"}, "inventory file")
        return _text(r["root"], "file root"), _relative(r["path"])

    files = {}
    for row in _rows(p["files"], file_key, "files"):
        _require(row["root"] in roots, "file has unknown root")
        _require(type(row["kind"]) is str and row["kind"] in _FILE_KINDS, "invalid file kind")
        _digest(row["sha256"])
        _require(type(row["size"]) is int and row["size"] >= 0, "invalid file size")
        files[(row["root"], row["path"])] = row
    _require(bool(files), "empty deployment inventory")
    paths = [Path(roots[r["root"]]) / r["path"] for r in files.values()]
    _require(len(paths) == len(set(paths)), "multiple inventory names for the same physical path")

    def referenced_file(root, path, kinds):
        key = _text(root, "file root"), _relative(path)
        _require(key in files and files[key]["kind"] in kinds, "unlisted/wrong-kind referenced file")

    for row in _rows(
        p["distributions"],
        lambda r: _fields(
            r,
            {"name", "version", "root", "metadata"},
            "distribution",
        )["name"],
        "distributions",
    ):
        _text(row["name"], "distribution name")
        _text(row["version"], "distribution version")
        referenced_file(row["root"], row["metadata"], {"metadata"})

    modules = {}
    for row in _rows(
        p["modules"],
        lambda r: _fields(
            r,
            {"name", "root", "path", "is_package", "phase"},
            "file module",
        )["name"],
        "modules",
    ):
        name = _module(row["name"])
        referenced_file(row["root"], row["path"], {"python", "extension"})
        _require(root_kinds[row["root"]] == "site", "native library root cannot supply imported modules")
        _require(type(row["is_package"]) is bool, "is_package must be boolean")
        _phase(row["phase"])
        modules[name] = row

    namespaces = set()
    for row in _rows(
        p["namespaces"],
        lambda r: _fields(
            r,
            {"name", "locations", "phase"},
            "namespace",
        )["name"],
        "namespaces",
    ):
        name = _module(row["name"])
        _require(name not in modules, "namespace collides with a file module")
        _phase(row["phase"])
        locations = _rows(
            row["locations"],
            lambda r: (
                _text(_fields(r, {"root", "path"}, "namespace location")["root"], "namespace root"),
                _relative(r["path"]),
            ),
            "namespace locations",
        )
        _require(bool(locations), "namespace must have frozen locations")
        for location in locations:
            _require(location["root"] in roots, "namespace has unknown root")
            _require(root_kinds[location["root"]] == "site", "namespace requires a site root")
        namespaces.add(name)

    publications = {}
    for row in _rows(
        p["publications"],
        lambda r: _fields(
            r,
            {"name", "kind", "creator", "publisher", "attributes", "phase"},
            "publication",
        )["name"],
        "publications",
    ):
        name = _module(row["name"])
        _require(name not in modules and name not in namespaces, "publication name collision")
        _require(row["kind"] in {"created_module", "module_alias"}, "invalid publication kind")
        # Roots must be file-loaded, preventing creator cycles or self-certified children.
        _require(_module(row["creator"]) in modules, "creator must be an admitted file module")
        _require(_module(row["publisher"]) in modules, "publisher must be an admitted file module")
        attributes = row["attributes"]
        _require(
            type(attributes) is list
            and bool(attributes)
            and all(type(a) is str and a.isidentifier() for a in attributes),
            "publication requires a fixed nonempty attribute path",
        )
        _phase(row["phase"])
        publications[name] = row

    def probe_key(row):
        r = _fields(row, {"name", "requester", "phase"}, "absent probe")
        return _module(r["name"]), _module(r["requester"]), _text(r["phase"], "probe phase")

    for row in _rows(p["absent_probes"], probe_key, "absent probes"):
        _require(row["requester"] in modules, "absent probe requires an admitted requester")
        _require(
            row["name"] not in modules and row["name"] not in publications and row["name"] not in namespaces,
            "absent probe collides with an admitted module",
        )
        _phase(row["phase"])

    disabled = set()
    policy = {row[0]: row[1:] for row in _DISABLED_OPTIONAL_ADAPTERS}
    executable_paths = {(row["root"], row["path"]) for row in modules.values()}
    absent_names = {row["name"] for row in p["absent_probes"]}

    def disabled_key(row):
        r = _fields(
            row,
            {
                "name",
                "root",
                "path",
                "requester",
                "phase",
                "absent_prerequisite",
                "fallback_attribute",
            },
            "disabled optional module",
        )
        return _module(r["name"]), _module(r["requester"]), _text(r["phase"], "disabled phase")

    for row in _rows(p["disabled_optional_modules"], disabled_key, "disabled optional modules"):
        name = row["name"]
        _require(name in policy, "disabled optional module is not in the fixed policy")
        _require(
            (row["path"], row["requester"], row["absent_prerequisite"], row["fallback_attribute"])
            == policy[name]
            and row["phase"] == "materialization",
            "disabled optional mapping differs from the fixed policy",
        )
        _require(
            name not in modules
            and name not in publications
            and name not in namespaces
            and name not in absent_names,
            "disabled optional name collides with an import rule",
        )
        referenced_file(row["root"], row["path"], {"python"})
        _require(root_kinds[row["root"]] == "site", "disabled adapter needs a site root")
        _require(
            (row["root"], row["path"]) not in executable_paths,
            "disabled path is executable under another name",
        )
        for owner in (row["requester"], "trimesh.exceptions"):
            _require(owner in modules, "disabled adapter requires an admitted fallback owner")
            owner_rule = modules[owner]
            _require(
                owner_rule["phase"] == "materialization"
                and files[(owner_rule["root"], owner_rule["path"])]["kind"] == "python",
                "disabled fallback owner must be a materialization Python module",
            )
        prerequisite = row["absent_prerequisite"]
        _require(
            not any(
                n == prerequisite or n.startswith(prerequisite + ".")
                for n in modules | publications | dict.fromkeys(namespaces)
            ),
            "disabled prerequisite cannot also be admitted",
        )
        disabled.add(name)

    # Any qualified name has an explicitly admitted package parent. Optional
    # probes cannot fabricate parents; creator publications cannot bypass them.
    present = set(modules) | set(publications) | namespaces
    for name in present | absent_names | disabled:
        parent = name.rpartition(".")[0]
        if parent:
            _require(parent in present, f"unlisted module parent: {parent}")
            _require(
                name in publications or parent not in modules or modules[parent]["is_package"],
                "file parent is not a package",
            )

    settings = _fields(p["settings"], {"TORCH_DEVICE_BACKEND_AUTOLOAD", "warp_quiet"}, "runtime settings")
    _require(
        settings["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0" and settings["warp_quiet"] is True,
        "unsafe runtime settings",
    )


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        while data := handle.read(1024 * 1024):
            digest.update(data)
            count += len(data)
    return digest.hexdigest(), count


def _contained(root: str, relative: str, *, directory: bool = False) -> Path:
    base = Path(root)
    _require(base.resolve(strict=True) == base and base.is_dir(), "deployment root changed")
    candidate = base
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        _require(not candidate.is_symlink(), "symlinked deployment file component")
    _require(candidate.resolve(strict=True).is_relative_to(base), "deployment file escaped root")
    _require(candidate.is_dir() if directory else candidate.is_file(), "deployment file kind changed")
    return candidate


@dataclass(frozen=True)
class RuntimeEnvironmentContract:
    """Immutable canonical configuration; not a runtime capability.

    Direct construction is validated too. verify_files must be given the digest
    from the trusted launch configuration, never a hash copied from this object.
    It checks listed files, not the absence of unlisted imports: the guarded
    resolver/ledger must enforce those separately at actual import time.
    """

    encoded: bytes

    def __post_init__(self) -> None:
        _require(type(self.encoded) is bytes, "environment encoding must be bytes")
        _validate(decode_canonical_message(self.encoded, max_bytes=len(self.encoded)))

    @classmethod
    def from_payload(cls, payload: object) -> RuntimeEnvironmentContract:
        _validate(payload)
        return cls(canonical_json_bytes(payload))

    def to_payload(self) -> dict:
        return decode_canonical_message(self.encoded, max_bytes=len(self.encoded))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encoded).hexdigest()

    @property
    def required_go2_disabled_names(self) -> tuple[str, ...]:
        """Fixed canonical Go2 obligation, not a field supplied in a receipt."""
        return tuple(row[0] for row in _DISABLED_OPTIONAL_ADAPTERS)

    def verify_files(self, *, expected_sha256: str) -> None:
        """Verify CPU filesystem facts without importing a single inventoried module."""
        _digest(expected_sha256)
        _require(type(self) is RuntimeEnvironmentContract, "exact environment contract required")
        # Repeat structural validation: frozen dataclasses are not an authority
        # boundary against object.__setattr__ in a trusted-process negative test.
        payload = self.to_payload()
        _validate(payload)
        _require(self.sha256 == expected_sha256, "environment differs from trusted launch digest")
        interpreter = payload["interpreter"]
        executable = Path(interpreter["path"])
        _require(executable.resolve(strict=True) == executable, "interpreter path drift")
        _require(Path(sys.executable).resolve(strict=True) == executable, "wrong running interpreter")
        _require(sys.implementation.name == interpreter["implementation"], "interpreter implementation drift")
        _require(list(sys.version_info[:3]) == interpreter["version"], "interpreter version drift")
        _require(_hash_file(executable)[0] == interpreter["sha256"], "interpreter file digest drift")
        roots = {entry["id"]: entry["path"] for entry in payload["roots"]}
        for entry in payload["files"]:
            path = _contained(roots[entry["root"]], entry["path"])
            _require(_hash_file(path) == (entry["sha256"], entry["size"]), "deployment file bytes changed")
        for entry in payload["distributions"]:
            path = _contained(roots[entry["root"]], entry["metadata"])
            metadata = BytesParser().parsebytes(path.read_bytes(), headersonly=True)
            _require(
                metadata.get_all("Name") == [entry["name"]]
                and metadata.get_all("Version") == [entry["version"]],
                "distribution identity differs from its frozen METADATA",
            )
        for entry in payload["namespaces"]:
            for location in entry["locations"]:
                _contained(roots[location["root"]], location["path"], directory=True)
