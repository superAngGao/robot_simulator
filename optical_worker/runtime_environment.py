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
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

from .protocol import canonical_json_bytes, decode_canonical_message

# Existing callers/fixtures still author the unchanged v5 input contract.
RUNTIME_ENVIRONMENT_VERSION = "q54.p12.4.runtime-environment.v5"
RUNTIME_ENVIRONMENT_COMPILATION_VERSION = "q54.p12.4.runtime-environment.v6"
NATIVE_GLOBAL_OBSERVATIONS_VERSION = "q54.p12.4.native-global-observations.v1"
MODULE_REPLACEMENT_OBSERVATIONS_VERSION = "q54.p12.4.module-replacement-observations.v1"
INTERPRETER_ALIAS_OBSERVATIONS_VERSION = "q54.p12.4.interpreter-alias-observations.v1"
# Literal policy, not a caller-extensible import denylist. File hashes still
# belong to the one trusted deployment inventory (not a duplicate hash table).
_DISABLED_OPTIONAL_ADAPTERS = (
    ("trimesh.path.polygons", "trimesh/path/polygons.py", "trimesh.path.path", "shapely", "polygons"),
    ("trimesh.ray.ray_pyembree", "trimesh/ray/ray_pyembree.py", "trimesh.ray", "embreex", "ray_pyembree"),
)
# Intrinsic metadata only: neither value is an import/alias permission. Exact
# file identity and version remain solely in the bound deployment inventory.
_NATIVE_MODULE_NAMES = (
    ("scipy._lib._uarray._uarray", "uarray._uarray"),
)
_TORCH_MODULE_REPLACEMENTS = (
    ("torch._VF", "VFModule", "native_attribute", "torch._C", ("_VariableFunctions",)),
    ("torch.backends", "GenericModule", "original_backlink", None, ()),
    ("torch.backends.cudnn", "CudnnModule", "original_backlink", None, ()),
    ("torch.backends.miopen", "MiopenModule", "original_backlink", None, ()),
    ("torch.backends.mkldnn", "MkldnnModule", "original_backlink", None, ()),
    ("torch.backends.opt_einsum", "OptEinsumModule", "original_backlink", None, ()),
    ("torch.backends.quantized", "QuantizedEngine", "original_backlink", None, ()),
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


def _expected_file_module_name(row: dict, file: dict) -> str:
    for import_name, intrinsic_name in _NATIVE_MODULE_NAMES:
        if row["name"] == import_name:
            _require(
                file["kind"] == "extension" and row["is_package"] is False,
                "intrinsic native name requires a nonpackage extension",
            )
            return intrinsic_name
    return row["name"]


@lru_cache(maxsize=1)
def _file_module_names(encoded: bytes) -> tuple[tuple[str, str], ...]:
    """Pure immutable metadata index, keyed by complete canonical bytes.

    A worker binds one environment. Never cache file verification, source/session
    liveness, loader/module identity or an execution observation here.
    """
    _require(type(encoded) is bytes, "environment encoding must be bytes")
    payload = decode_canonical_message(encoded, max_bytes=len(encoded))
    _validate(payload)
    files = {(entry["root"], entry["path"]): entry for entry in payload["files"]}
    return tuple(
        (row["name"], _expected_file_module_name(row, files[(row["root"], row["path"])]))
        for row in payload["modules"]
    )


def _validate(payload: object) -> None:
    _require(type(payload) is dict, "invalid runtime environment fields")
    version = payload.get("schema_version")
    _require(version in (RUNTIME_ENVIRONMENT_VERSION, RUNTIME_ENVIRONMENT_COMPILATION_VERSION),
             "runtime environment version")
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
            "native_globals",
            "module_replacements",
            "namespaces",
            "settings",
        } | ({"setuptools_initialization"} if version == RUNTIME_ENVIRONMENT_COMPILATION_VERSION else set()),
        "runtime environment",
    )
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
        _expected_file_module_name(row, files[(row["root"], row["path"])])
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
            type(attributes) is list and all(type(a) is str and a.isidentifier() for a in attributes),
            "publication requires a fixed attribute path",
        )
        _phase(row["phase"])
        if not attributes:
            creator = modules[row["creator"]]
            _require(
                row["kind"] == "module_alias"
                and row["creator"] == row["publisher"]
                and not creator["is_package"]
                and row["phase"] == creator["phase"],
                "creator-self alias requires one non-package native creator/publisher and phase",
            )
            referenced_file(creator["root"], creator["path"], {"extension"})
        publications[name] = row

    native_globals = {}
    for row in _rows(
        p["native_globals"],
        lambda r: _fields(r, {"name", "observer", "operation", "phase"}, "native global")["name"],
        "native globals",
    ):
        name = _module(row["name"])
        _require(name not in modules and name not in publications and name not in namespaces,
                 "native global name collision")
        _require(_module(row["observer"]) in modules, "native global observer missing")
        observer = modules[row["observer"]]
        _require(row["operation"] == "exec" and not observer["is_package"]
                 and row["phase"] == observer["phase"], "invalid native global interval")
        referenced_file(observer["root"], observer["path"], {"extension"})
        _phase(row["phase"])
        native_globals[name] = row

    replacement_policy = {row[0]: row[1:] for row in _TORCH_MODULE_REPLACEMENTS}
    replacements = {}
    for row in _rows(
        p["module_replacements"],
        lambda r: _fields(r, {"name", "phase", "class_attribute", "relation", "native_root",
                              "native_attributes"}, "module replacement")["name"],
        "module replacements",
    ):
        name = _module(row["name"])
        _require(name in replacement_policy and name in modules, "unlisted replacement file")
        _require(type(row["native_attributes"]) is list, "replacement path must be a list")
        _require((row["class_attribute"], row["relation"], row["native_root"],
                  tuple(row["native_attributes"])) == replacement_policy[name],
                 "replacement differs from literal policy")
        original = modules[name]
        _require(row["phase"] == original["phase"] == "runtime"
                 and original["is_package"] is (name != "torch._VF"), "replacement file phase/package")
        referenced_file(original["root"], original["path"], {"python"})
        replacements[name] = row
    _require(not replacements or set(replacements) == set(replacement_policy),
             "replacement table must contain all seven or none")
    if replacements:
        _require("torch" in modules and modules["torch"]["is_package"]
                 and modules["torch"]["phase"] == "runtime"
                 and "torch._C" in modules and not modules["torch._C"]["is_package"]
                 and modules["torch._C"]["phase"] == "runtime", "replacement runtime roots missing")
        native = modules["torch._C"]
        referenced_file(native["root"], native["path"], {"extension"})

    def probe_key(row):
        r = _fields(row, {"name", "requester", "phase"}, "absent probe")
        return _module(r["name"]), _module(r["requester"]), _text(r["phase"], "probe phase")

    for row in _rows(p["absent_probes"], probe_key, "absent probes"):
        _require(row["requester"] in modules, "absent probe requires an admitted requester")
        _require(
            row["name"] not in modules and row["name"] not in publications
            and row["name"] not in namespaces and row["name"] not in native_globals,
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
            and name not in native_globals
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
                for n in modules | publications | native_globals | dict.fromkeys(namespaces)
            ),
            "disabled prerequisite cannot also be admitted",
        )
        disabled.add(name)

    # Any qualified name has an explicitly admitted package parent. Optional
    # probes cannot fabricate parents; creator publications cannot bypass them.
    present = set(modules) | set(publications) | set(native_globals) | namespaces
    if p.get("setuptools_initialization") is not None:
        _validate_setuptools_profile(p, roots, root_kinds, files, modules, present | absent_names | disabled)
        present.add("distutils")
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


def _validate_setuptools_profile(p, roots, root_kinds, files, modules, declared):
    """One pinned package layout; presence of facts is not import authority."""
    from .historical.compilation_contract import setuptools_initialization_descriptor

    expected = setuptools_initialization_descriptor()
    _require(p["setuptools_initialization"] == expected, "unknown setuptools initialization descriptor")
    _require(p["interpreter"]["version"] == [3, 12, 13], "setuptools profile requires pinned interpreter")
    _require({key for key in roots if root_kinds[key] == "site"} == {"site", "setuptools_vendor"}
             and roots["setuptools_vendor"] == (Path(roots["site"]) / "setuptools" / "_vendor").as_posix(),
             "setuptools vendor root layout")
    _require("distutils" not in declared, "setuptools alias collides with another rule")
    _require(not any(row["name"].startswith("distutils.")
                     for key in ("publications", "native_globals", "module_replacements") for row in p[key]),
             "distutils children require ordinary file or namespace rules")
    metadata = expected["metadata"]
    _require({"name": "setuptools", "version": "83.0.0", "root": metadata["root"],
              "metadata": metadata["path"]} in p["distributions"], "pinned setuptools distribution missing")
    for kind, row in (("metadata", metadata), ("data", expected["resource"])):
        file = files.get((row["root"], row["path"]))
        _require(file is not None and file["kind"] == kind, "setuptools metadata/resource file missing")
    for name, root, path, package in (
        ("setuptools", "site", "setuptools/__init__.py", True),
        ("setuptools._distutils", "site", "setuptools/_distutils/__init__.py", True),
        ("setuptools.version", "site", "setuptools/version.py", False),
        ("setuptools.command", "site", "setuptools/command/__init__.py", True),
        ("setuptools.command.build_ext", "site", "setuptools/command/build_ext.py", False),
        ("_distutils_hack", "site", "_distutils_hack/__init__.py", True),
        ("_distutils_hack.override", "site", "_distutils_hack/override.py", False),
        ("distutils.core", "site", "setuptools/_distutils/core.py", False),
        ("jaraco.text", "setuptools_vendor", "jaraco/text/__init__.py", True),
        ("torch.utils.cpp_extension", "site", "torch/utils/cpp_extension.py", False),
    ):
        _require(modules.get(name) == {"name": name, "root": root, "path": path,
                                       "is_package": package, "phase": "runtime"}
                 and files[(root, path)]["kind"] == "python", "setuptools mandatory source binding")
    for row in p["modules"]:
        name = row["name"]
        if row["root"] == "setuptools_vendor" or name.split(".")[0] in {"setuptools", "_distutils_hack", "distutils"}:
            _require(row["phase"] == "runtime", "setuptools module cannot enter the input materialization phase")
        if name.startswith("distutils."):
            relative = name.replace("distutils.", "setuptools/_distutils/", 1).replace(".", "/")
            relative += "/__init__.py" if row["is_package"] else ".py"
            _require(row["root"] == "site" and row["path"] == relative
                     and files[(row["root"], row["path"])]["kind"] == "python",
                     "distutils child source layout")
    namespace_rows = {row["name"]: row for row in p["namespaces"]}
    for name, root, path in (("jaraco", "setuptools_vendor", "jaraco"),
                             ("distutils.compilers", "site", "setuptools/_distutils/compilers"),
                             ("distutils.compilers.C", "site", "setuptools/_distutils/compilers/C")):
        _require(namespace_rows.get(name) == {"name": name, "phase": "runtime",
                                              "locations": [{"root": root, "path": path}]},
                 "setuptools mandatory namespace binding")
    for row in p["namespaces"]:
        if row["name"].startswith("distutils.") or any(r["root"] == "setuptools_vendor" for r in row["locations"]):
            _require(row["phase"] == "runtime", "setuptools namespace phase")


def validate_native_global_observations(payload, environment, *, require_complete=False):
    """Replay descriptive facts only; never register a live module or frame.

    Whole runtime custody must additionally bind the actual import and guard
    histories. This component cannot certify a worker's completed execution.
    """
    p = _fields(payload, {"schema_version", "environment_sha256", "calls", "events"},
                "native global observations")
    _require(p["schema_version"] == NATIVE_GLOBAL_OBSERVATIONS_VERSION, "native global evidence version")
    _require(type(environment) is RuntimeEnvironmentContract
             and p["environment_sha256"] == environment.sha256, "native global environment mismatch")
    config = environment.to_payload()
    rules = {row["name"]: row for row in config["native_globals"]}
    modules = {row["name"]: row for row in config["modules"]}
    _require(type(p["calls"]) is list and type(p["events"]) is list, "native global evidence lists")
    calls, seen_calls = [], set()
    for index, value in enumerate(p["calls"]):
        call = _fields(value, {"sequence", "parent_sequence", "observer", "operation", "phase",
                               "absent_names", "state"}, "native call")
        _require(type(call["sequence"]) is int and call["sequence"] == index, "native call order")
        parent = call["parent_sequence"]
        _require(parent is None or type(parent) is int and 0 <= parent < index, "native parent order")
        name, operation = call["observer"], call["operation"]
        _require(name in modules and operation in {"create", "exec"}
                 and call["phase"] == modules[name]["phase"], "native call relation")
        _require((name, operation) not in seen_calls, "duplicate native call")
        if operation == "exec":
            _require((name, "create") in seen_calls, "native exec before create")
        seen_calls.add((name, operation))
        expected = sorted(n for n, rule in rules.items() if rule["observer"] == name and operation == "exec")
        _require(type(call["absent_names"]) is list and call["absent_names"] == expected,
                 "native absence facts differ from rule")
        _require(call["state"] in {"started", "returned", "completed", "failed"}, "native call state")
        if require_complete:
            _require(call["state"] == "completed", "incomplete native call")
        if parent is not None and calls[parent]["state"] == "completed":
            _require(call["state"] == "completed", "completed parent has incomplete child")
        calls.append(call)
    pending, admitted, tokens, returned = {}, {}, set(), set()
    event_keys = {"index", "operation", "name", "phase", "observer", "observer_call", "call_sequence",
                  "parent_call_sequence", "absence_checkpoint", "checkpoint", "request_name", "object_token"}
    for index, value in enumerate(p["events"]):
        event = _fields(value, event_keys, "native global event")
        _require(type(event["index"]) is int and event["index"] == index, "native global event order")
        name, sequence = event["name"], event["call_sequence"]
        _require(name in rules and type(sequence) is int and 0 <= sequence < len(calls),
                 "native global event relation")
        call, rule = calls[sequence], rules[name]
        _require(event["phase"] == call["phase"] == rule["phase"]
                 and event["observer"] == call["observer"] == rule["observer"]
                 and event["observer_call"] == call["operation"] == "exec"
                 and type(event["parent_call_sequence"]) is type(call["parent_sequence"])
                 and event["parent_call_sequence"] == call["parent_sequence"]
                 and event["absence_checkpoint"] == "observer_exec_entry" and name in call["absent_names"],
                 "native global interval mismatch")
        token = event["object_token"]
        _require(type(token) is int and token > 0, "native global object token")
        # An exec-return fact closes the delegated interval, even if fallible
        # wrapper work has not yet emitted its admission. No nested event can
        # occur after that fact, including in a failed prefix.
        parent = call["parent_sequence"]
        while parent is not None:
            _require(parent not in returned, "parent return preceded nested event")
            parent = calls[parent]["parent_sequence"]
        if event["checkpoint"] == "request_entry":
            _require(sequence not in returned, "native request observation after return")
        if event["checkpoint"] == "exec_return":
            _require(call["state"] in {"returned", "completed", "failed"},
                     "native return observation in started call")
            returned.add(sequence)
        if event["operation"] == "observed_pending":
            _require(name not in pending and name not in admitted and token not in tokens,
                     "duplicate native global observation")
            _require(all(old["call_sequence"] != sequence for old in admitted.values()),
                     "native observation after same-call admission")
            checkpoint, request = event["checkpoint"], event["request_name"]
            _require((checkpoint == "exec_return" and request is None)
                     or (checkpoint == "request_entry" and type(request) is str and bool(request)),
                     "native global observation checkpoint")
            # An ancestor cannot already have completed admission while a new
            # descendant publication is still being observed.
            parent = call["parent_sequence"]
            while parent is not None:
                _require(all(old["call_sequence"] != parent for old in admitted.values()),
                         "parent admission preceded nested child")
                parent = calls[parent]["parent_sequence"]
            pending[name] = event
            tokens.add(token)
        elif event["operation"] == "admitted":
            original = pending.pop(name, None)
            _require(original is not None and original["call_sequence"] == sequence
                     and original["object_token"] == token and name not in admitted
                     and event["checkpoint"] == "exec_return" and event["request_name"] is None
                     and call["state"] == "completed", "invalid native global admission")
            for child in pending.values():
                parent = calls[child["call_sequence"]]["parent_sequence"]
                while parent is not None:
                    _require(parent != sequence, "parent admitted with pending child")
                    parent = calls[parent]["parent_sequence"]
            admitted[name] = event
        else:
            raise ValueError("unknown native global event")
    for call in calls:
        if call["state"] == "completed":
            _require(all(name in admitted for name in call["absent_names"]),
                     "completed native observer lacks admissions")
    if require_complete:
        _require(not pending, "native global pending at completed boundary")
    return None


def validate_interpreter_alias_observations(payload, *, require_observed=False):
    """Parse descriptive stdlib observations; confer no import or exec authority.

    Paths are lexical so a CPU reader can inspect either deployment platform.
    The final custody record must bind this descriptor to its source session.
    """
    p = _fields(payload, {"schema_version", "interpreter_identity", "source_path", "source_sha256",
                          "resolution", "events"}, "interpreter alias observations")
    _require(p["schema_version"] == INTERPRETER_ALIAS_OBSERVATIONS_VERSION, "interpreter alias version")
    identity = p["interpreter_identity"]
    _require(type(identity) is list and len(identity) == 7, "interpreter alias identity")
    executable, version, abi, platform, multiarch, directory, os_name = identity
    _require(type(version) is list and len(version) == 5
             and all(type(v) is int and v >= 0 for v in (*version[:3], version[4]))
             and version[3] in {"alpha", "beta", "candidate", "final"}, "interpreter alias version facts")
    _require(all(type(v) is str for v in (executable, abi, platform, multiarch, directory, os_name))
             and executable and platform and directory and os_name in {"nt", "posix"},
             "interpreter alias topology facts")
    path_type = PureWindowsPath if os_name == "nt" else PurePosixPath
    source = path_type(_text(p["source_path"], "interpreter alias source path"))
    base = path_type(directory)
    _require(base.is_absolute() and path_type(executable).is_absolute()
             and str(source) == p["source_path"] and ".." not in base.parts
             and source == base / "multiprocessing" / "__init__.py", "interpreter alias source topology")
    _digest(p["source_sha256"])
    resolution = p["resolution"]
    if resolution is not None:
        r = _fields(resolution, {"name", "origin", "loader", "search_locations"}, "interpreter alias resolution")
        _require(r == {"name": "multiprocessing", "origin": p["source_path"],
                       "loader": "SourceFileLoader", "search_locations": [str(source.parent)]},
                 "interpreter alias resolution differs from source")
    events = p["events"]
    _require(type(events) is list and len(events) <= 1, "interpreter alias event count")
    if events:
        e = _fields(events[0], {"index", "operation", "checkpoint", "alias", "target", "publisher",
                               "main_token", "publisher_token"}, "interpreter alias event")
        _require(resolution is not None and type(e["index"]) is int and e["index"] == 0
                 and e["operation"] == "interpreter_alias_observed"
                 and e["checkpoint"] in {"request_entry", "exec_return", "audit"}
                 and e["alias"] == "__mp_main__" and e["target"] == "__main__"
                 and e["publisher"] == "multiprocessing"
                 and type(e["main_token"]) is int and e["main_token"] == 1
                 and type(e["publisher_token"]) is int and e["publisher_token"] == 2,
                 "interpreter alias observation relation")
    _require(not require_observed or len(events) == 1, "interpreter alias observation missing")


def validate_module_replacement_observations(payload, environment, import_observations, *, require_complete=False):
    """Descriptive replacement facts referencing the one actual file-call ledger."""
    validate_native_global_observations(import_observations, environment, require_complete=require_complete)
    p = _fields(payload, {"schema_version", "environment_sha256", "events"}, "replacement observations")
    _require(p["schema_version"] == MODULE_REPLACEMENT_OBSERVATIONS_VERSION
             and p["environment_sha256"] == environment.sha256, "replacement evidence identity")
    _require(type(p["events"]) is list, "replacement events must be a list")
    rules = {row["name"]: row for row in environment.to_payload()["module_replacements"]}
    calls = import_observations["calls"]
    keys = {"index", "operation", "name", "phase", "call_sequence", "checkpoint", "request_name",
            "relation", "original", "replacement", "class", "native_root", "native_value"}
    pending, admitted, tokens, returned = {}, {}, set(), set()
    for index, value in enumerate(p["events"]):
        event = _fields(value, keys, "replacement event")
        name, sequence = event["name"], event["call_sequence"]
        _require(type(event["index"]) is int and event["index"] == index and name in rules
                 and type(sequence) is int and 0 <= sequence < len(calls), "replacement event order/reference")
        call, rule = calls[sequence], rules[name]
        _require(call["observer"] == name and call["operation"] == "exec"
                 and call["phase"] == event["phase"] == rule["phase"] == "runtime"
                 and event["relation"] == rule["relation"], "replacement own file interval")
        native = rule["relation"] == "native_attribute"
        object_fields = ("original", "replacement", "class", * (("native_root", "native_value") if native else ()))
        values = tuple(event[key] for key in object_fields)
        _require(all(type(token) is int and token > 0 for token in values)
                 and len(set(values)) == len(values), "replacement distinct object tokens")
        if not native:
            _require(event["native_root"] is None and event["native_value"] is None,
                     "backlink has invented native owner")
        else:
            _require(any(c["observer"] == rule["native_root"] and c["operation"] == "exec"
                         and c["state"] == "completed" for c in calls[:sequence]),
                     "replacement native root has no completed file call")
        parent = call["parent_sequence"]
        while parent is not None:
            _require(parent not in returned, "replacement child observed after parent return")
            parent = calls[parent]["parent_sequence"]
        checkpoint, request = event["checkpoint"], event["request_name"]
        _require((checkpoint == "request_entry" and type(request) is str and bool(request)
                  and sequence not in returned)
                 or (checkpoint == "exec_return" and request is None
                     and call["state"] in {"returned", "completed", "failed"}),
                 "replacement observation checkpoint")
        if checkpoint == "exec_return":
            returned.add(sequence)
        if event["operation"] == "observed_pending":
            _require(name not in pending and name not in admitted and not tokens.intersection(values),
                     "replacement duplicate observation/token")
            pending[name] = event
            tokens.update(values)
        elif event["operation"] == "admitted":
            original = pending.pop(name, None)
            _require(original is not None and name not in admitted and checkpoint == "exec_return"
                     and call["state"] == "completed"
                     and all(original[key] == event[key] for key in keys -
                             {"index", "operation", "checkpoint", "request_name"}),
                     "replacement admission differs from observation")
            for child in pending.values():
                parent = calls[child["call_sequence"]]["parent_sequence"]
                while parent is not None:
                    _require(parent != sequence, "replacement parent admitted with pending child")
                    parent = calls[parent]["parent_sequence"]
            admitted[name] = event
        else:
            raise ValueError("unknown replacement event")
    for call in calls:
        if call["observer"] in rules and call["operation"] == "exec" and call["state"] == "completed":
            _require(call["observer"] in admitted, "completed file lacks required replacement")
    if require_complete:
        _require(not pending, "replacement pending at completed boundary")


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


def _freeze_environment_value(value):
    if type(value) is dict:
        return ("object", tuple((key, _freeze_environment_value(item)) for key, item in value.items()))
    if type(value) is list:
        return ("array", tuple(_freeze_environment_value(item) for item in value))
    return value


def _copy_environment_value(value):
    if type(value) is not tuple:
        return value
    kind, entries = value
    if kind == "object":
        return {key: _copy_environment_value(item) for key, item in entries}
    return [_copy_environment_value(item) for item in entries]


@lru_cache(maxsize=1)
def _immutable_environment_payload(encoded: bytes):
    """Cache only validated immutable configuration, keyed by complete bytes.

    No mutable dictionary/list escapes this cache. File verification, live
    admission indexes, object identity, execution and custody are never cached.
    Changing encoded via object.__setattr__ still selects/revalidates new bytes.
    """
    _require(type(encoded) is bytes, "environment encoding must be bytes")
    payload = decode_canonical_message(encoded, max_bytes=len(encoded))
    _validate(payload)
    return _freeze_environment_value(payload)


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
        _immutable_environment_payload(self.encoded)

    @classmethod
    def from_payload(cls, payload: object) -> RuntimeEnvironmentContract:
        _validate(payload)
        return cls(canonical_json_bytes(payload))

    def to_payload(self) -> dict:
        return _copy_environment_value(_immutable_environment_payload(self.encoded))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.encoded).hexdigest()

    @property
    def required_go2_disabled_names(self) -> tuple[str, ...]:
        """Fixed canonical Go2 obligation, not a field supplied in a receipt."""
        return tuple(row[0] for row in _DISABLED_OPTIONAL_ADAPTERS)

    def expected_file_module_name(self, name: str) -> str:
        """One declared file's intrinsic name; does not resolve or authorize it."""
        _require(type(self) is RuntimeEnvironmentContract, "exact environment contract required")
        _module(name)
        for import_name, intrinsic_name in _file_module_names(self.encoded):
            if import_name == name:
                return intrinsic_name
        raise ValueError("intrinsic name requires a declared file module")

    @property
    def setuptools_initialization(self) -> dict | None:
        return self.to_payload().get("setuptools_initialization")

    def external_search_roots(self, *, setuptools_active: bool = False) -> tuple[str, ...]:
        """Pure search-layout projection; only the live session may activate it."""
        _require(type(setuptools_active) is bool, "invalid setuptools activation projection")
        payload = self.to_payload()
        roots = {r["id"]: str(Path(r["path"])) for r in payload["roots"] if r["kind"] == "site"}
        if payload.get("setuptools_initialization") is None:
            _require(not setuptools_active, "setuptools initialization is not declared")
            return tuple(roots.values())
        return (roots["site"], roots["setuptools_vendor"]) if setuptools_active else (roots["site"],)

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
