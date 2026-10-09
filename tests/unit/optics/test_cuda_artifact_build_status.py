"""CPU counterexamples for deployment termination facts, not systemd OS proof."""

import pytest

from tools.optical_cuda_build_status import require_normal_exit, validate_status
from tools.optical_cuda_build_recipe import validate_build_environment


UNIT = "robot-sim-cuda-" + "a" * 32 + "-0.service"


def _status():
    return {
        "unit": UNIT,
        "invocation_id": "b" * 32,
        "service_result": "success",
        "exit_code": "exited",
        "exit_status": "0",
        "cgroup_path": "/user.slice/user-1006.slice/user@1006.service/app.slice/" + UNIT,
    }


def test_normal_exit_is_necessary_but_does_not_assert_group_closure():
    assert require_normal_exit(_status(), unit=UNIT, waiter_returncode=0) == _status()


def test_manager_fixture_keeps_production_service_boundary():
    from pathlib import PurePosixPath
    from tests.support.cuda_artifact_service_acceptance import fixture_argv

    repository, job = PurePosixPath("/data/repo"), PurePosixPath("/data/acceptance/descendant")
    argv = fixture_argv(repository, job, UNIT)
    assert argv[:6] == ["/usr/bin/systemd-run", "--user", "--no-ask-password", "--wait", "--collect", "--service-type=exec"]
    for setting in ("ExitType=cgroup", "Restart=no", "RemainAfterExit=no", "KillMode=control-group",
                    "SendSIGKILL=yes", "Delegate=no", "RuntimeMaxSec=6s", "TimeoutStartSec=6s", "TimeoutStopSec=1s"):
        assert "--property=" + setting in argv
    assert next(value for value in argv if value.startswith("--property=ExecStopPost=")).endswith(
        f"/data/repo/tools/optical_cuda_build_status.py --output {job}/termination.json --unit {UNIT}")
    assert argv[-5:] == [str(repository / "tests/support/cuda_artifact_service_acceptance.py"),
                         "--fixture", "descendant", "--output", str(job)]
    assert str(repository / "tools/optical_cuda_build_recipe.py") not in argv


@pytest.mark.parametrize("signal", ["TERM", "HUP", "INT", "PIPE"])
def test_systemd_clean_signal_cannot_become_completed_build(signal):
    record = {**_status(), "exit_code": "killed", "exit_status": signal}
    # The footer must preserve the manager's true success/killed facts.
    assert validate_status(record, unit=UNIT) == record
    with pytest.raises(ValueError, match="normal main-process exit"):
        require_normal_exit(record, unit=UNIT, waiter_returncode=0)


@pytest.mark.parametrize("changes,waiter", [
    ({"exit_status": "1", "service_result": "exit-code"}, 1),
    ({"service_result": "timeout"}, 0),
    ({"exit_code": "dumped", "exit_status": "ABRT"}, 0),
    ({"exit_code": None, "exit_status": None}, 0),
    ({"invocation_id": None}, 0),
    ({"cgroup_path": None}, 0),
    ({}, 1),
    ({}, False),
])
def test_incomplete_or_failed_termination_does_not_publish(changes, waiter):
    with pytest.raises(ValueError):
        require_normal_exit({**_status(), **changes}, unit=UNIT, waiter_returncode=waiter)


@pytest.mark.parametrize("changes", [
    {"unit": UNIT.replace("-0.", "-1.")},
    {"invocation_id": "not-an-invocation"},
    {"cgroup_path": "/user.slice/../" + UNIT},
    {"cgroup_path": _status()["cgroup_path"] + "/.control"},
    {"cgroup_path": _status()["cgroup_path"].replace("-0.", "-1.")},
    {"exit_status": 0},
    {"extra": "unrecognized"},
])
def test_footer_cannot_substitute_unit_group_or_fact_shape(changes):
    with pytest.raises(ValueError):
        validate_status({**_status(), **changes}, unit=UNIT)


@pytest.mark.parametrize("changes", [
    {"CUDA_VISIBLE_DEVICES": "0"},
    {"CUDA_VISIBLE_DEVICES": None},
    {"PYTORCH_NVML_BASED_CUDA_CHECK": "0"},
    {"PYTORCH_NVML_BASED_CUDA_CHECK": None},
    {"TORCH_CUDA_ARCH_LIST": "native"},
    {"TORCH_DEVICE_BACKEND_AUTOLOAD": "1"},
    {"NVCC_APPEND_FLAGS": ""},
    {"LD_PRELOAD": "/tmp/override.so"},
])
def test_build_environment_cannot_reenable_native_discovery_or_override_compiler(changes):
    baseline = {
        "CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "TORCH_CUDA_ARCH_LIST": "9.0",
        "MAX_JOBS": "1", "CUDA_HOME": "/home/ang.gao/miniforge3/envs/robot_sim",
        "CXX": "/usr/bin/x86_64-linux-gnu-g++-13", "CC": "/usr/bin/x86_64-linux-gnu-gcc-13",
    }
    validate_build_environment(baseline)
    with pytest.raises(RuntimeError):
        validate_build_environment({**baseline, **changes})
