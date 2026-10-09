"""Deploy the one fixed CUDA artifact set; requires reviewed execution preflight.

No worker invokes this command. Failed staging is retained as evidence, never
resumed or published. The existing user manager owns each bounded compiler job.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath


def _run_readonly(argv):
    return subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()


def observe_group_closure(path: str, *, unit: str, manager: str) -> str:
    """Observe only the known nondelegated job; absence differs from read failure."""
    group, parent = PurePosixPath(path), PurePosixPath(manager)
    if (not group.is_absolute() or group.as_posix() != path or ".." in group.parts
            or group.name != unit or not group.is_relative_to(parent) or group == parent):
        raise ValueError("cgroup is not the owned user-manager service")
    directory = Path("/sys/fs/cgroup") / path.lstrip("/")
    try:
        events = directory.joinpath("cgroup.events").read_text().splitlines()
        populated = dict(line.split() for line in events)["populated"]
        if populated != "0" or directory.joinpath("cgroup.procs").read_text().strip():
            raise RuntimeError("build cgroup is still populated")
    except FileNotFoundError:
        # Do not turn a missing interface in a still-existing group into closure.
        try:
            directory.stat()
        except FileNotFoundError:
            return "absent"
        raise RuntimeError("existing build cgroup lacks its required interfaces") from None
    return "empty"


def deploy(output: Path, *, dependency_profile: Path, profile_sha256: str) -> str:
    repository = Path(__file__).resolve(strict=True).parents[1]
    sys.path.insert(0, str(repository))
    from optical_worker.protocol import canonical_json_bytes
    from optical_worker.runtime_environment import RuntimeEnvironmentContract
    from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
    from tools.optical_cuda_artifacts import (
        MANIFEST_NAME, SOURCE_FILES, VERSION, file_row, recipe_rows, validate_manifest, verify_deployment,
    )
    from tools.optical_cuda_build_recipe import _BUILD_ENVIRONMENT, _PREFIX, _create_directory, build_service_argv
    from tools.optical_cuda_build_status import require_normal_exit, validate_status
    from tools.optical_cuda_toolchain_inventory import _toolchain_inventory

    if (sys.platform != "linux" or tuple(sys.version_info[:3]) != (3, 12, 13)
            or Path(sys.prefix) != _PREFIX or platform.machine() != "x86_64"
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode):
        raise RuntimeError("deployment requires the fixed isolated Linux interpreter")
    if not output.is_absolute() or output.parent.resolve(strict=True) != output.parent:
        raise ValueError("deployment output parent is not canonical")
    if os.path.lexists(output):
        raise FileExistsError(output)
    # All authorized publishers use this exclusive reservation. Existing output
    # and abandoned reservations are never overwritten or silently resumed.
    with output.with_name(output.name + ".reserve").open("xb") as handle:
        handle.write(b"fixed CUDA deployment reservation\n")
    identifier = uuid.uuid4().hex
    staging = _create_directory(output.parent / (".cuda-stage-" + identifier))
    jobs = _create_directory(staging / "jobs")
    temporary = _create_directory(staging / "tmp")
    revision = _run_readonly(["git", "-C", str(repository), "rev-parse", "HEAD"])
    if _run_readonly(["git", "-C", str(repository), "status", "--porcelain"]):
        raise RuntimeError("deployment source checkout is dirty")
    profile = RuntimeEnvironmentContract.from_payload(json.loads(dependency_profile.read_bytes()))
    profile.verify_files(expected_sha256=profile_sha256)
    profile_payload = profile.to_payload()
    required_modules = {"torch", "torch.utils.cpp_extension", "setuptools", "setuptools.command.build_ext"}
    if not required_modules.issubset(row["name"] for row in profile_payload["modules"]):
        raise ValueError("build dependency profile lacks the fixed build modules")
    sources = {"revision": revision, "files": [file_row(repository, repository / path) for path in SOURCE_FILES]}
    inventory = {"schema_version": "q54.p12.4.cuda-toolchain-inventory.v1", **_toolchain_inventory()}
    original_toolchain_files = inventory["files"]
    # Add the already enumerated installed Python dependency/interpreter bytes to
    # the same file table; no new recursive dependency discovery is performed.
    roots = {row["id"]: Path(row["path"]) for row in profile_payload["roots"]}
    covered = {row["path"]: row for row in inventory["files"]}
    extra = [{"path": str(roots[row["root"]] / row["path"]), "size": row["size"], "sha256": row["sha256"]}
             for row in profile_payload["files"]]
    interpreter = profile_payload["interpreter"]
    extra.append({"path": str(Path(interpreter["path"]).resolve(strict=True)), "size": Path(interpreter["path"]).stat().st_size,
                  "sha256": interpreter["sha256"]})
    for row in extra:
        if row["path"] in covered and covered[row["path"]] != row:
            raise ValueError("dependency and toolchain inventories disagree")
        covered[row["path"]] = row
    inventory["files"] = [covered[path] for path in sorted(covered)]
    environment = {**_BUILD_ENVIRONMENT, "PATH": str(_PREFIX / "bin") + ":/usr/bin:/bin",
                   "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""), "TMPDIR": str(temporary), "LC_ALL": "C"}
    manager = _run_readonly(["/usr/bin/systemctl", "--user", "show", "--property=ControlGroup", "--value"])
    if manager != f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service":
        raise RuntimeError("unexpected user manager cgroup")
    builds, artifacts, observed_platform = [], [], None
    for index, recipe in enumerate(CUDA_EXTENSION_RECIPES):
        job = _create_directory(jobs / str(index))
        unit = f"robot-sim-cuda-{identifier}-{index}.service"
        argv = build_service_argv(repository, job, unit, environment)
        with (job / "invocation.json").open("xb") as handle:
            handle.write(canonical_json_bytes({"unit": unit, "argv": argv}))
            handle.flush()
            os.fsync(handle.fileno())
        print(f"Building {recipe.name}: {unit}; retained job {job}", flush=True)
        try:
            with (job / "waiter.log").open("xb") as log:
                result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=False)
            raw_status = validate_status(json.loads((job / "termination.json").read_bytes()), unit=unit)
            closure = observe_group_closure(raw_status["cgroup_path"], unit=unit, manager=manager)
            terminal = require_normal_exit(raw_status, unit=unit, waiter_returncode=result.returncode)
        except BaseException:
            print(f"Deployment stopped; retained job {job}; manager unit {unit}. No artifact published.", file=sys.stderr, flush=True)
            raise
        details = json.loads((job / "build-details.json").read_bytes())
        if details["name"] != recipe.name:
            raise ValueError("builder returned another recipe")
        built = job / details["path"]
        actual = file_row(job, built)
        if actual != {k: details[k] for k in ("path", "size", "sha256")}:
            raise ValueError("build output changed after main exit")
        current = {"system": "Linux", "machine": "x86_64", "python_version": [3, 12, 13],
                   "soabi": details["soabi"], "ext_suffix": details["ext_suffix"],
                   "torch_version": "2.9.1+cu128", "torch_cuda_version": "12.8",
                   "cxx11_abi": details["cxx11_abi"], "cuda_arch_list": "9.0"}
        if observed_platform is not None and observed_platform != current:
            raise ValueError("three builds disagree on ABI")
        observed_platform = current
        if details["path"] != "output/" + recipe.name + details["ext_suffix"]:
            raise ValueError("builder output filename differs")
        destination = staging / built.name
        with built.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        artifacts.append({"name": recipe.name, **file_row(staging, destination)})
        inputs = [job / "source/main.cpp", job / "source/cuda.cu", job / "temporary/build.ninja"]
        builds.append({"name": recipe.name, "unit": unit, "argv": argv, "waiter_returncode": result.returncode,
                       "termination": {**{k: terminal[k] for k in ("invocation_id", "service_result", "exit_code", "exit_status")},
                                       "record": file_row(staging, job / "termination.json")},
                       "cgroup": {"path": terminal["cgroup_path"], "invocation_id": terminal["invocation_id"], "outcome": closure},
                       "stdout": file_row(staging, job / "stdout.log"), "stderr": file_row(staging, job / "stderr.log"),
                       "inputs": [file_row(staging, path) for path in sorted(inputs)]})
    profile.verify_files(expected_sha256=profile_sha256)
    after = _toolchain_inventory()
    if any(after[key] != inventory[key] for key in ("roots", "symlinks", "tools")):
        raise ValueError("toolchain changed during deployment")
    if after["files"] != original_toolchain_files:
        raise ValueError("toolchain bytes changed during deployment")
    if sources["files"] != [file_row(repository, repository / path) for path in SOURCE_FILES]:
        raise ValueError("build source changed during deployment")
    payload = validate_manifest({"schema_version": VERSION, "source": sources, "recipes": recipe_rows(),
                                 "platform": observed_platform, "environment": environment, "toolchain": inventory,
                                 "builds": builds, "artifacts": artifacts})
    encoded = canonical_json_bytes(payload)
    with (staging / MANIFEST_NAME).open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    import hashlib
    digest = hashlib.sha256(encoded).hexdigest()
    verify_deployment(staging, digest, repository=repository, expected_platform=observed_platform)
    if os.path.lexists(output):
        raise FileExistsError(output)
    staging.rename(output)
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    args = parser.parse_args()
    print(deploy(args.output, dependency_profile=args.dependency_profile, profile_sha256=args.profile_sha256))


if __name__ == "__main__":
    main()
