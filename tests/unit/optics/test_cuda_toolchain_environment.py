"""CPU environment admission only; no complete inventory or compiler proof."""

from pathlib import Path

import pytest

from optical_worker import cuda_compilation as compilation
from optical_worker.protocol import canonical_json_bytes


@pytest.fixture
def toolchain(tmp_path, monkeypatch):
    environment = {
        "CUDA_HOME": compilation._CONDA, "CXX": "/usr/bin/x86_64-linux-gnu-g++-13",
        "CC": "/usr/bin/x86_64-linux-gnu-gcc-13", "TORCH_CUDA_ARCH_LIST": "9.0",
        "MAX_JOBS": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        "TMPDIR": str(tmp_path / "tmp"), "TORCH_EXTENSIONS_DIR": str(tmp_path / "torch"),
        "PATH": "fixture-path-never-executed", "LD_LIBRARY_PATH": None,
    }
    for key in compilation._INJECTION_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    value = object.__new__(compilation._Toolchain)
    value.root = tmp_path
    value.payload = {"environment": environment, "tools": {"cxx": environment["CXX"], "cc": environment["CC"]},
                     "pch_paths": [str(tmp_path / "absent.gch"), str(tmp_path / "absent.sign")]}
    value.encoded = canonical_json_bytes(value.payload)
    monkeypatch.setattr(compilation, "_private_directory", Path)
    value.audit()
    return value


@pytest.mark.parametrize("variable", ["CUDA_INC_PATH", "CUDNN_HOME", "CUDNN_PATH",
                                      "TORCH_DONT_CHECK_COMPILER_ABI", "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES",
                                      "GCC_EXEC_PREFIX", "COMPILER_PATH", "C_INCLUDE_PATH", "DEPENDENCIES_OUTPUT",
                                      "SUNPRO_DEPENDENCIES", "GCC_COMPARE_DEBUG"])
def test_unfrozen_build_overrides_reject_before_dependency_use(toolchain, monkeypatch, variable):
    monkeypatch.setenv(variable, "1" if variable.startswith("TORCH_") else "/unlisted/build/input")
    with pytest.raises(RuntimeError, match="injected"):
        toolchain.audit()


def test_absent_library_path_is_distinct_from_present_empty_path(toolchain, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    with pytest.raises(RuntimeError, match="environment changed"):
        toolchain.audit()
