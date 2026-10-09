"""Cold deployment artifact validation; never grants live module authority.

The render worker consumes the derived ordinary runtime environment, not this
build record. Compatibility callers may use the same cold validation before
ordinary fixed-name imports. No function here imports a backend or builds code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from optical_worker.protocol import canonical_json_bytes
from optical_worker.runtime_environment import RuntimeEnvironmentContract, _hash_file
from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from tools.optical_cuda_build_recipe import build_service_argv, validate_build_environment
from tools.optical_cuda_build_status import require_normal_exit


VERSION = "q54.p12.4.cuda-artifact-set.v1"
MANIFEST_NAME = "cuda-artifacts.json"
SOURCE_FILES = tuple(sorted((
    "tools/optical_cuda_deploy.py", "tools/optical_cuda_build_recipe.py",
    "tools/optical_cuda_build_status.py", "tools/optical_cuda_artifacts.py",
    "tools/optical_cuda_toolchain_inventory.py", "optics/cuda_extension_recipes.py",
    "optics/_cuda_scene_source.py", "optics/_cuda_lbvh_source.py", "optics/_cuda_direct_light_source.py",
)))
_PLATFORM_FIELDS = {
    "system", "machine", "python_version", "soabi", "ext_suffix", "torch_version",
    "torch_cuda_version", "cxx11_abi", "cuda_arch_list",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields):
    _require(type(value) is dict and set(value) == set(fields), "artifact record fields differ")
    return value


def _digest(value, size=64):
    _require(type(value) is str and len(value) == size
             and all(c in "0123456789abcdef" for c in value), "invalid artifact digest")


def _relative(value):
    _require(type(value) is str and value != "", "empty artifact path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and path.as_posix() == value
             and not any(p in {".", ".."} for p in path.parts)
             and ":" not in value and "\\" not in value, "non-contained artifact path")
    return path


def _file(row):
    _fields(row, {"path", "size", "sha256"})
    _relative(row["path"])
    _require(type(row["size"]) is int and row["size"] >= 0, "invalid artifact size")
    _digest(row["sha256"])
    return row


def file_row(root: Path, path: Path) -> dict:
    root = root.resolve(strict=True)
    _require(path.is_absolute(), "artifact file must be absolute")
    relative = path.relative_to(root).as_posix()
    checked = contained_file(root, relative)
    digest, size = _hash_file(checked)
    return {"path": relative, "size": size, "sha256": digest}


def contained_file(root: Path, relative: str) -> Path:
    parts = _relative(relative).parts
    _require(root.is_absolute() and root.resolve(strict=True) == root
             and not any(p.is_symlink() for p in (root, *root.parents)), "artifact root is not canonical")
    path = root
    for part in parts:
        path /= part
        _require(not path.is_symlink(), "symlink in artifact file path")
    _require(path.is_file() and path.resolve(strict=True).is_relative_to(root), "artifact file missing or outside root")
    return path


def verify_file(root: Path, row: dict) -> None:
    _file(row)
    _require(file_row(root, root / row["path"]) == row, "artifact file bytes changed")


def recipe_rows() -> list[dict]:
    return [{"name": r.name, "entries": list(r.entries),
             "cpp_sha256": hashlib.sha256(r.source_module.CPP_SOURCE.encode("utf-8")).hexdigest(),
             "cuda_sha256": hashlib.sha256(r.source_module.CUDA_SOURCE.encode("utf-8")).hexdigest(),
             "cflags": list(r.cflags), "cuda_cflags": list(r.cuda_cflags)} for r in CUDA_EXTENSION_RECIPES]


def _validate_toolchain(value):
    _fields(value, {"schema_version", "roots", "files", "symlinks", "tools"})
    _require(value["schema_version"] == "q54.p12.4.cuda-toolchain-inventory.v1", "toolchain inventory version")
    _require(type(value["files"]) is list and bool(value["files"]), "missing toolchain files")
    previous, files = "", set()
    for row in value["files"]:
        _fields(row, {"path", "size", "sha256"})
        _require(type(row["path"]) is str and row["path"].startswith("/")
                 and row["path"] > previous, "toolchain file order/path")
        _require(type(row["size"]) is int and row["size"] >= 0, "toolchain file size")
        _digest(row["sha256"])
        previous = row["path"]
        files.add(previous)
    _require(type(value["tools"]) is dict and bool(value["tools"])
             and all(type(k) is str and type(v) is str and v in files for k, v in value["tools"].items()),
             "toolchain executable not covered")
    _require(type(value["roots"]) is list and bool(value["roots"]), "missing toolchain roots")
    for row in value["roots"]:
        _fields(row, {"path", "kind", "present", "resolved"})
        _require(type(row["path"]) is str and row["path"].startswith("/")
                 and row["kind"] in {"tree", "files", "file"} and type(row["present"]) is bool,
                 "toolchain root declaration")
        _require(type(row["resolved"]) is str and row["resolved"].startswith("/")
                 if row["present"] else row["resolved"] is None, "toolchain root resolution")
    _require(type(value["symlinks"]) is list, "toolchain symlink table")
    for row in value["symlinks"]:
        _fields(row, {"path", "target"})
        _require(type(row["path"]) is str and row["path"].startswith("/")
                 and type(row["target"]) is str and bool(row["target"]), "toolchain symlink declaration")


def validate_manifest(value) -> dict:
    """Check the frozen descriptive schema; no hashes register live owners."""
    _fields(value, {"schema_version", "source", "recipes", "platform", "environment",
                    "toolchain", "builds", "artifacts"})
    _require(value["schema_version"] == VERSION, "artifact manifest version")
    source = _fields(value["source"], {"revision", "files"})
    _digest(source["revision"], 40)
    _require(type(source["files"]) is list, "artifact source file table")
    _require(tuple(_file(row)["path"] for row in source["files"]) == SOURCE_FILES, "artifact source closure differs")
    _require(value["recipes"] == recipe_rows(), "artifact recipes differ from sole source definitions")
    platform = _fields(value["platform"], _PLATFORM_FIELDS)
    _require((platform["system"], platform["machine"], platform["python_version"], platform["torch_version"],
              platform["torch_cuda_version"], platform["cuda_arch_list"])
             == ("Linux", "x86_64", [3, 12, 13], "2.9.1+cu128", "12.8", "9.0"), "artifact platform differs")
    _require(type(platform["cxx11_abi"]) is bool and type(platform["soabi"]) is str
             and platform["soabi"].startswith("cpython-312-")
             and platform["ext_suffix"] == "." + platform["soabi"] + ".so", "artifact Python/C++ ABI")
    environment = value["environment"]
    _require(type(environment) is dict and all(type(k) is str and type(v) is str
                                             for k, v in environment.items()), "artifact environment types")
    validate_build_environment(environment)
    _validate_toolchain(value["toolchain"])
    _require(type(value["artifacts"]) is list and len(value["artifacts"]) == 3, "artifact set is partial")
    for recipe, row in zip(CUDA_EXTENSION_RECIPES, value["artifacts"], strict=True):
        _fields(row, {"name", "path", "size", "sha256"})
        _require(row["name"] == recipe.name and row["path"] == recipe.name + platform["ext_suffix"],
                 "artifact module/file order or ABI differs")
        _file({k: v for k, v in row.items() if k != "name"})
        _require(row["size"] > 0, "empty extension output")
    _require(type(value["builds"]) is list and len(value["builds"]) == 3, "artifact build set is partial")
    invocations, units = set(), set()
    for index, (recipe, row) in enumerate(zip(CUDA_EXTENSION_RECIPES, value["builds"], strict=True)):
        _fields(row, {"name", "unit", "argv", "waiter_returncode", "termination", "cgroup", "stdout", "stderr", "inputs"})
        _require(row["name"] == recipe.name, "build recipe order differs")
        terminal = _fields(row["termination"], {"invocation_id", "service_result", "exit_code", "exit_status", "record"})
        group = _fields(row["cgroup"], {"path", "invocation_id", "outcome"})
        _require(group["outcome"] in {"empty", "absent"}
                 and group["invocation_id"] == terminal["invocation_id"], "build group is not closed or changed invocation")
        require_normal_exit({"unit": row["unit"], "cgroup_path": group["path"],
                             **{k: terminal[k] for k in ("invocation_id", "service_result", "exit_code", "exit_status")}},
                            unit=row["unit"], waiter_returncode=row["waiter_returncode"])
        _require(terminal["invocation_id"] not in invocations and row["unit"] not in units, "reused build invocation")
        invocations.add(terminal["invocation_id"])
        units.add(row["unit"])
        _require(type(row["argv"]) is list and bool(row["argv"])
                 and all(type(v) is str and bool(v) for v in row["argv"]), "build argv is absent or malformed")
        _require(len(row["argv"]) >= 11, "build argv is incomplete")
        job = PurePosixPath(row["argv"][-3])
        script = PurePosixPath(row["argv"][-7])
        _require(len(script.parents) >= 2, "build script lacks repository root")
        _require(job == PurePosixPath(environment["TMPDIR"]).parent / "jobs" / str(index)
                 and row["unit"].endswith(f"-{index}.service"), "build directory/recipe differs")
        _require(row["argv"] == build_service_argv(script.parents[1], job, row["unit"], environment),
                 "build argv differs from the fixed command")
        for file in (terminal["record"], row["stdout"], row["stderr"]):
            _file(file)
        _require((terminal["record"]["path"], row["stdout"]["path"], row["stderr"]["path"])
                 == tuple(f"jobs/{index}/{name}" for name in ("termination.json", "stdout.log", "stderr.log")),
                 "build logs/status refer to another job")
        _require(type(row["inputs"]) is list and bool(row["inputs"]), "build inputs absent")
        paths = [_file(file)["path"] for file in row["inputs"]]
        _require(paths == [f"jobs/{index}/{path}" for path in ("source/cuda.cu", "source/main.cpp", "temporary/build.ninja")],
                 "build inputs missing, duplicated or reordered")
    return json.loads(canonical_json_bytes(value))


def verify_deployment(root: Path, expected_sha256: str, *, repository: Path, expected_platform: dict) -> dict:
    """Validate an explicitly selected immutable set before launching a worker."""
    _digest(expected_sha256)
    path = contained_file(root, MANIFEST_NAME)
    encoded = path.read_bytes()
    _require(hashlib.sha256(encoded).hexdigest() == expected_sha256, "artifact manifest identity differs")
    manifest = validate_manifest(json.loads(encoded))
    _require(canonical_json_bytes(manifest) == encoded, "artifact manifest is not canonical")
    _require(manifest["platform"] == expected_platform, "artifact deployment ABI differs from trusted launch")
    for row in manifest["source"]["files"]:
        verify_file(repository, row)
    for row in manifest["artifacts"]:
        verify_file(root, {k: v for k, v in row.items() if k != "name"})
    for build in manifest["builds"]:
        for row in (build["stdout"], build["stderr"], build["termination"]["record"], *build["inputs"]):
            verify_file(root, row)
        raw = json.loads(contained_file(root, build["termination"]["record"]["path"]).read_bytes())
        expected = {"unit": build["unit"], "cgroup_path": build["cgroup"]["path"],
                    **{k: build["termination"][k] for k in ("invocation_id", "service_result", "exit_code", "exit_status")}}
        _require(raw == expected, "artifact terminal facts differ from actual footer")
        for row in build["inputs"][:2]:
            recipe = next(r for r in manifest["recipes"] if r["name"] == build["name"])
            expected_hash = recipe["cuda_sha256" if row["path"].endswith("cuda.cu") else "cpp_sha256"]
            _require(row["sha256"] == expected_hash, "actual build source differs from the selected recipe")
    return manifest


def extend_runtime_environment(environment: RuntimeEnvironmentContract, *, deployment: Path,
                               manifest_sha256: str, repository: Path, expected_platform: dict) -> RuntimeEnvironmentContract:
    """Bind exactly three artifact modules and their manifest into ordinary v5."""
    manifest = verify_deployment(deployment, manifest_sha256, repository=repository, expected_platform=expected_platform)
    payload = environment.to_payload()
    _require(payload["schema_version"] == "q54.p12.4.runtime-environment.v5", "artifact worker requires ordinary v5 environment")
    _require(all(row["id"] != "cuda_artifacts" and row["path"] != deployment.as_posix() for row in payload["roots"]),
             "artifact environment root already exists")
    names = {row["name"] for row in payload["modules"]}
    _require(not names.intersection(r.name for r in CUDA_EXTENSION_RECIPES), "artifact module already declared")
    payload["roots"].append({"id": "cuda_artifacts", "path": deployment.as_posix(), "kind": "site"})
    payload["files"].append({"root": "cuda_artifacts", "kind": "metadata", **file_row(deployment, deployment / MANIFEST_NAME)})
    for row in manifest["artifacts"]:
        payload["files"].append({"root": "cuda_artifacts", "kind": "extension", **{k: v for k, v in row.items() if k != "name"}})
        payload["modules"].append({"name": row["name"], "root": "cuda_artifacts", "path": row["path"],
                                   "is_package": False, "phase": "runtime"})
    payload["roots"].sort(key=lambda r: r["id"])
    payload["files"].sort(key=lambda r: (r["root"], r["path"]))
    payload["modules"].sort(key=lambda r: r["name"])
    return RuntimeEnvironmentContract.from_payload(payload)
