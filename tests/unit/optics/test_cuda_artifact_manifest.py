"""Descriptive CPU fixtures only: dummy bytes are not compiled/native evidence."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from optical_worker.protocol import canonical_json_bytes
from optical_worker.runtime_environment import RuntimeEnvironmentContract
from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES
from tools.optical_cuda_artifacts import (
    MANIFEST_NAME, SOURCE_FILES, VERSION, extend_runtime_environment, file_row, recipe_rows, validate_manifest, verify_deployment,
)
from tools.optical_cuda_build_recipe import _BUILD_ENVIRONMENT, build_service_argv
from tests.unit.optics.test_runtime_environment import deployment


REPOSITORY = Path(__file__).resolve().parents[3]


@pytest.fixture
def artifact(tmp_path):
    environment = {**_BUILD_ENVIRONMENT, "PATH": "/fixed/bin", "LD_LIBRARY_PATH": "",
                   "TMPDIR": "/data/staging/tmp", "LC_ALL": "C"}
    platform = {"system": "Linux", "machine": "x86_64", "python_version": [3, 12, 13],
                "soabi": "cpython-312-x86_64-linux-gnu", "ext_suffix": ".cpython-312-x86_64-linux-gnu.so",
                "torch_version": "2.9.1+cu128", "torch_cuda_version": "12.8", "cxx11_abi": True,
                "cuda_arch_list": "9.0"}

    def write(relative, content):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return file_row(tmp_path, path)

    builds, outputs = [], []
    for i, recipe in enumerate(CUDA_EXTENSION_RECIPES):
        unit = "robot-sim-cuda-" + "a" * 32 + f"-{i}.service"
        group = "/user.slice/user-1006.slice/user@1006.service/app.slice/" + unit
        status = {"unit": unit, "invocation_id": f"{i+1:032x}", "service_result": "success",
                  "exit_code": "exited", "exit_status": "0", "cgroup_path": group}
        record = write(f"jobs/{i}/termination.json", canonical_json_bytes(status))
        builds.append({
            "name": recipe.name, "unit": unit,
            "argv": build_service_argv(PurePosixPath("/data/repo"), PurePosixPath(f"/data/staging/jobs/{i}"), unit, environment),
            "waiter_returncode": 0,
            "termination": {**{k: status[k] for k in ("invocation_id", "service_result", "exit_code", "exit_status")}, "record": record},
            "cgroup": {"path": group, "invocation_id": status["invocation_id"], "outcome": "absent"},
            "stdout": write(f"jobs/{i}/stdout.log", b"synthetic stdout"),
            "stderr": write(f"jobs/{i}/stderr.log", b""),
            "inputs": [write(f"jobs/{i}/source/cuda.cu", recipe.source_module.CUDA_SOURCE.encode()),
                       write(f"jobs/{i}/source/main.cpp", recipe.source_module.CPP_SOURCE.encode()),
                       write(f"jobs/{i}/temporary/build.ninja", b"synthetic emitted input")],
        })
        outputs.append({"name": recipe.name, **write(recipe.name + platform["ext_suffix"], b"NOT AN EXTENSION")})
    payload = {
        "schema_version": VERSION,
        "source": {"revision": "1" * 40, "files": [file_row(REPOSITORY, REPOSITORY / path) for path in SOURCE_FILES]},
        "recipes": recipe_rows(), "platform": platform, "environment": environment,
        "toolchain": {"schema_version": "q54.p12.4.cuda-toolchain-inventory.v1",
                      "roots": [{"path": "/fixed/tool", "kind": "file", "present": True, "resolved": "/fixed/tool"}],
                      "files": [{"path": "/fixed/tool", "size": 1, "sha256": "a" * 64}],
                      "symlinks": [], "tools": {"cxx": "/fixed/tool"}},
        "builds": builds, "artifacts": outputs,
    }
    return tmp_path, payload


def _publish_fixture(root, payload):
    encoded = canonical_json_bytes(payload)
    (root / MANIFEST_NAME).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_descriptive_manifest_roundtrip_does_not_import_dummy_outputs(artifact):
    import sys
    root, payload = artifact
    before = dict(sys.modules)
    digest = _publish_fixture(root, payload)
    assert verify_deployment(root, digest, repository=REPOSITORY, expected_platform=payload["platform"]) == payload
    assert all(recipe.name not in sys.modules for recipe in CUDA_EXTENSION_RECIPES)
    assert all(sys.modules[name] is module for name, module in before.items())


@pytest.mark.parametrize("field,value", [
    ("service_result", "timeout"), ("exit_code", "killed"), ("exit_status", "TERM"), ("exit_status", "1"),
])
def test_artifact_manifest_cannot_promote_failed_main_to_success(artifact, field, value):
    _, payload = artifact
    payload["builds"][0]["termination"][field] = value
    with pytest.raises(ValueError):
        validate_manifest(payload)


@pytest.mark.parametrize("mutation", [
    "partial", "entry", "other_status", "other_argv", "other_source", "old_invocation", "populated", "wrong_abi",
])
def test_cross_build_and_partial_artifact_claims_reject(artifact, mutation):
    _, payload = artifact
    first, second = payload["builds"][:2]
    if mutation == "partial":
        payload["artifacts"].pop()
    elif mutation == "entry":
        payload["recipes"][0]["entries"].reverse()
    elif mutation == "other_status":
        first["termination"]["record"] = copy.deepcopy(second["termination"]["record"])
    elif mutation == "other_argv":
        first["argv"] = list(second["argv"])
    elif mutation == "other_source":
        first["inputs"] = copy.deepcopy(second["inputs"])
    elif mutation == "old_invocation":
        second["termination"]["invocation_id"] = first["termination"]["invocation_id"]
        second["cgroup"]["invocation_id"] = first["termination"]["invocation_id"]
    elif mutation == "populated":
        first["cgroup"]["outcome"] = "populated"
    else:
        payload["platform"]["ext_suffix"] = ".cpython-311-x86_64-linux-gnu.so"
    with pytest.raises(ValueError):
        validate_manifest(payload)


@pytest.mark.parametrize("target", ["output", "footer", "source"])
def test_file_changes_are_checked_against_selected_manifest(artifact, target):
    root, payload = artifact
    digest = _publish_fixture(root, payload)
    relative = (payload["artifacts"][0]["path"] if target == "output" else
                payload["builds"][0]["termination"]["record"]["path"] if target == "footer" else
                payload["builds"][0]["inputs"][0]["path"])
    (root / relative).write_bytes(b"changed after publication")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_deployment(root, digest, repository=REPOSITORY, expected_platform=payload["platform"])


def test_rehashed_footer_cannot_disagree_with_manifest_facts(artifact):
    root, payload = artifact
    row = payload["builds"][0]["termination"]["record"]
    (root / row["path"]).write_bytes(b'{"exit_code":"killed"}')
    payload["builds"][0]["termination"]["record"] = file_row(root, root / row["path"])
    digest = _publish_fixture(root, payload)
    with pytest.raises(ValueError, match="terminal facts"):
        verify_deployment(root, digest, repository=REPOSITORY, expected_platform=payload["platform"])


def test_existing_v5_environment_binds_exact_files_without_compilation_profile(artifact):
    root, payload = artifact
    digest = _publish_fixture(root, payload)
    profile = json.loads((REPOSITORY / "tests/data/runtime_environments/183_cuda_go2_author_v1.json").read_bytes())
    # The environment contract intentionally validates native filesystem paths.
    # Rebase only fixture paths for this Windows CPU schema check; do not claim
    # that the target's installed dependency bytes were verified here.
    profile["interpreter"]["path"] = (root / "fixture-python").as_posix()
    for row in profile["roots"]:
        row["path"] = (root / "fixture-dependencies" / row["id"]).as_posix()
    base = RuntimeEnvironmentContract.from_payload(profile)
    original = base.to_payload()
    derived = extend_runtime_environment(base, deployment=root, manifest_sha256=digest,
                                         repository=REPOSITORY, expected_platform=payload["platform"])
    result = derived.to_payload()
    assert base.to_payload() == original
    assert result["schema_version"] == original["schema_version"]
    assert "setuptools_initialization" not in result
    assert len(result["files"]) == len(original["files"]) + 4
    assert [row["name"] for row in result["modules"] if row["root"] == "cuda_artifacts"] == sorted(
        r.name for r in CUDA_EXTENSION_RECIPES)
    assert next(row for row in result["files"] if row["root"] == "cuda_artifacts" and row["kind"] == "metadata")["sha256"] == digest
    with pytest.raises(ValueError, match="already exists"):
        extend_runtime_environment(derived, deployment=root, manifest_sha256=digest,
                                   repository=REPOSITORY, expected_platform=payload["platform"])


@pytest.mark.parametrize("attack", ("base_digest", "launcher_abi"))
def test_cold_composer_rejects_wrong_selection_without_writing_inventory(artifact, deployment, attack):
    root, payload = artifact
    _, base_payload = deployment
    base = RuntimeEnvironmentContract.from_payload(base_payload)
    base_path, platform_path, output = root / "base.json", root / "platform.json", root / "derived.json"
    base_path.write_bytes(base.encoded)
    selected_platform = copy.deepcopy(payload["platform"])
    # Deliberately impossible ABI on every host, including the actual Linux target.
    selected_platform["soabi"] = "wrong-reviewed-platform"
    platform_path.write_bytes(canonical_json_bytes(selected_platform))
    digest = _publish_fixture(root, payload)
    result = subprocess.run([
        sys.executable, "-I", "-S", "-B", str(REPOSITORY / "tools/optical_cuda_environment.py"),
        "--base-profile", str(base_path), "--base-sha256", "0" * 64 if attack == "base_digest" else base.sha256,
        "--deployment", str(root), "--manifest-sha256", digest,
        "--expected-platform", str(platform_path), "--output", str(output),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and not output.exists()
    assert ("selected digest" if attack == "base_digest" else "cold launcher ABI") in result.stderr
