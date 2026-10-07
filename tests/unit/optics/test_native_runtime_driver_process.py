"""Runtime driver wiring fails closed before dependency/GPU entry.

Real POSIX supervisor and captured production source; deliberately incomplete
external rules reject NumPy before its loader. Not a successful native setup test.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optical_worker.guarded_evidence import validate_custody_artifact
from optical_worker.native_sources import CATALOG_PATH
from optical_worker.runtime_environment import RUNTIME_ENVIRONMENT_VERSION, RuntimeEnvironmentContract
from tests.unit.optics.test_native_sources import release as _release_fixture

release = _release_fixture
pytestmark = pytest.mark.skipif(os.name != "posix", reason="actual inherited POSIX supervisor descriptors")
HARNESS = Path(__file__).resolve().parents[2] / "support/guarded_custody_process.py"


@pytest.mark.parametrize("lane", ["warp", "cuda"])
@pytest.mark.parametrize("fault", ["environment_digest", "unlisted_numpy"])
def test_fixed_runtime_driver_quarantines_without_fallback_or_ordinary_cleanup(
    release, tmp_path, lane, fault
):
    root, revision = release
    site = tmp_path / "site"
    site.mkdir()
    canary = b"raise RuntimeError('must_not_execute')\n"
    (site / "unlisted.py").write_bytes(canary)
    executable = Path(sys.executable).resolve(strict=True)
    environment = RuntimeEnvironmentContract.from_payload(
        {
            "schema_version": RUNTIME_ENVIRONMENT_VERSION,
            "interpreter": {
                "path": executable.as_posix(),
                "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "implementation": sys.implementation.name,
                "version": list(sys.version_info[:3]),
            },
            "roots": [{"id": "site", "path": site.as_posix(), "kind": "site"}],
            "files": [
                {
                    "root": "site",
                    "path": "unlisted.py",
                    "kind": "python",
                    "size": len(canary),
                    "sha256": hashlib.sha256(canary).hexdigest(),
                }
            ],
            "distributions": [],
            "modules": [],
            "absent_probes": [],
            "disabled_optional_modules": [],
            "publications": [],
            "namespaces": [],
            "settings": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "warp_quiet": True},
        }
    )
    environment_path = tmp_path / "environment.json"
    environment_path.write_bytes(environment.encoded)
    request = {
        "root": str(root),
        "revision": revision,
        "lane": lane,
        "catalog_sha256": hashlib.sha256((root / CATALOG_PATH).read_bytes()).hexdigest(),
        "guard_configuration": {
            "protocol_version": "q54.closed-worker.guarded-wait.v1",
            "session": "e" * 32,
            "device_ordinal": 1,
            "ack_timeout_ms": 1000,
            "operation_timeout_ms": 5000,
        },
        "environment_path": str(environment_path),
        "environment_sha256": "0" * 64 if fault == "environment_digest" else environment.sha256,
        "case_id": "asymmetric_synthetic_7x3_v1",
        "go2_model_dir": None,
    }
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            str(HARNESS),
            "supervisor",
            "runtime_success",
            str(read_fd),
            str(tmp_path),
        ],
        pass_fds=(read_fd,),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(read_fd)
    try:
        output, error = process.communicate(timeout=30)
        assert process.returncode == 0, (output, error)
        result = json.loads(output)
        validate_custody_artifact(result["artifact"])
        assert result["worker_returncode"] == 74 and result["parent_alive"] is True
        worker = result["artifact"]["worker"]
        assert worker["summary"]["custody"] == "full_quarantine"
        assert worker["summary"]["release_status"] == "not_authorized"
        assert worker["summary"]["emergency_sync_attempt_count"] == 0
        assert worker["summary"]["terminal_arm_attempt_count"] == 0
        assert worker["summary"]["normal_complete_count"] == (1 if fault == "environment_digest" else 2)
        failure = json.loads((tmp_path / "failure.json").read_text())
        assert failure["phase"] == (
            "external_environment" if fault == "environment_digest" else "canonical_materialization"
        )
        assert (
            "digest" in failure["message"]
            if fault == "environment_digest"
            else "unlisted external import" in failure["message"]
        )
        assert not (tmp_path / "ordinary_atexit.json").exists()
        assert not (tmp_path / "terminal_sync_entered.json").exists()
        assert not (tmp_path / "runtime_initialization.json").exists()
        assert not (tmp_path / "input_validation_report.json").exists()
        if fault == "unlisted_numpy":
            observed = json.loads((tmp_path / "external_failure.json").read_text())
            assert observed["failed"] is True
            assert observed["events"] == [
                {
                    "index": 0,
                    "operation": "failed",
                    "name": "numpy",
                    "kind": "failure",
                    "phase": "materialization",
                }
            ]
        (tmp_path / "custody_artifact.json").write_text(json.dumps(result["artifact"]), encoding="utf-8")
        print(
            json.dumps(
                {
                    "lane": lane,
                    "fault": fault,
                    "supervisor_pid": process.pid,
                    "worker_pid": result["worker_pid"],
                    "worker_returncode": result["worker_returncode"],
                }
            )
        )
    finally:
        os.close(write_fd)
        if process.poll() is None:
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)
