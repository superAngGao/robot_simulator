"""Build one fixed deployment recipe; never import or publish the result.

The deployment launcher runs this with -I -S -B inside its systemd service.
This command is not a rendering-worker entrypoint or a standalone publisher.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import re
import sys
import sysconfig
from pathlib import Path, PurePosixPath


_PREFIX = PurePosixPath("/home/ang.gao/miniforge3/envs/robot_sim")
_BUILD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    "TORCH_CUDA_ARCH_LIST": "9.0",
    "MAX_JOBS": "1",
    "CUDA_HOME": str(_PREFIX),
    "CXX": "/usr/bin/x86_64-linux-gnu-g++-13",
    "CC": "/usr/bin/x86_64-linux-gnu-gcc-13",
}
_INJECTION_VARIABLES = frozenset({
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "PYTORCH_NVCC", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
    "LDFLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_PRELOAD",
    "CUDA_INC_PATH", "CUDNN_HOME", "CUDNN_PATH", "TORCH_DONT_CHECK_COMPILER_ABI",
    "TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "GCC_EXEC_PREFIX", "COMPILER_PATH",
    "C_INCLUDE_PATH", "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES", "GCC_COMPARE_DEBUG",
})


def validate_build_environment(environment) -> None:
    if any(environment.get(key) != value for key, value in _BUILD_ENVIRONMENT.items()):
        raise RuntimeError("fixed build environment is absent or changed")
    if _INJECTION_VARIABLES.intersection(environment):
        raise RuntimeError("compiler override or injection variable is present")


def build_service_argv(repository: Path, job: Path, unit: str, environment: dict) -> list[str]:
    """One fixed systemd invocation, shared by launch and provenance validation."""
    from tools.optical_cuda_build_status import validate_status
    from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES

    validate_build_environment(environment)
    if set(environment) != set(_BUILD_ENVIRONMENT) | {"PATH", "LD_LIBRARY_PATH", "TMPDIR", "LC_ALL"}:
        raise ValueError("unexpected build environment keys")
    if any(type(v) is not str or "\x00" in v or "\n" in v for v in environment.values()):
        raise ValueError("invalid build environment value")
    validate_status({"unit": unit, "invocation_id": None, "service_result": None,
                     "exit_code": None, "exit_status": None, "cgroup_path": None}, unit=unit)
    index = int(unit[-9])
    for path in (repository, job):
        if not path.is_absolute() or re.fullmatch(r"/[A-Za-z0-9_/.-]+", path.as_posix()) is None:
            raise ValueError("deployment command paths must be absolute literal Linux paths")
    python = str(_PREFIX / "bin/python")
    footer = [python, "-I", "-S", "-B", str(repository / "tools/optical_cuda_build_status.py"),
              "--output", str(job / "termination.json"), "--unit", unit]
    return [
        "/usr/bin/systemd-run", "--user", "--no-ask-password", "--wait", "--collect",
        "--service-type=exec", "--expand-environment=no", "--unit=" + unit,
        "--property=ExitType=cgroup", "--property=Restart=no", "--property=RemainAfterExit=no",
        "--property=RuntimeMaxSec=120s", "--property=TimeoutStartSec=120s", "--property=TimeoutStopSec=10s",
        "--property=KillMode=control-group", "--property=SendSIGKILL=yes", "--property=Delegate=no",
        "--property=WorkingDirectory=" + str(job),
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--property=ExecStopPost=" + " ".join(footer),
        "--", "/usr/bin/env", "-i", *(key + "=" + environment[key] for key in sorted(environment)),
        python, "-I", "-S", "-B", str(repository / "tools/optical_cuda_build_recipe.py"),
        "--recipe", CUDA_EXTENSION_RECIPES[index].name, "--job-directory", str(job), "--unit", unit,
    ]


def _check_environment() -> None:
    # This precedes every repository or third-party import. In particular,
    # cpp_extension imports must not reach the default CUDA availability probe.
    if (sys.platform != "linux" or sys.implementation.name != "cpython"
            or tuple(sys.version_info[:3]) != (3, 12, 13)
            or Path(sys.prefix) != _PREFIX
            or not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode):
        raise RuntimeError("builder requires the fixed isolated deployment interpreter")
    validate_build_environment(os.environ)
    if any(name == root or name.startswith(root + ".")
           for name in sys.modules for root in ("torch", "warp", "setuptools")):
        raise RuntimeError("build dependency was imported before environment validation")


def _private_directory(path: Path) -> Path:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("noncanonical build directory")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlink in build directory")
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise ValueError("build directory is not private and owned")
    return path


def _create_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)  # Existing/stale staging is never a cache hit.
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fchmod(descriptor, 0o700)  # Shared parent may carry setgid.
    finally:
        os.close(descriptor)
    return _private_directory(path)


def _observe_builder_driver_state(stage: str, torch=None) -> dict:
    """Observe this builder without initializing CUDA or loading a driver.

These checkpoints accompany the fixed installed-source/environment proof; they
are not a syscall history or an assertion about another process's past activity.
"""
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                     if "/libcuda.so" in line or "/libnvidia-ml.so" in line})
    gpu_fds = []
    for path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue  # The directory iterator's own descriptor may have closed.
        if target.startswith(("/dev/nvidia", "/dev/dri/")):
            gpu_fds.append(target)
    initialized = None if torch is None else torch.cuda.is_initialized()
    drivers = [path for path in mapped if "/libcuda.so" in path]
    driver_status, current_context = None, None
    if len(drivers) == 1:
        # Torch may map libcuda without calling cuInit. RTLD_NOLOAD reuses that
        # mapping; this getter returns CUDA_ERROR_NOT_INITIALIZED (3) before
        # driver initialization. CUDA_SUCCESS with a null context is NOT enough.
        library = ctypes.CDLL(drivers[0], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
        query = library.cuCtxGetCurrent
        query.argtypes, query.restype = [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int
        context = ctypes.c_void_p()
        driver_status = query(ctypes.byref(context))
        current_context = context.value
    record = {"stage": stage, "mapped_driver_libraries": mapped,
              "gpu_device_fds": sorted(gpu_fds), "torch_initialized": initialized,
              "driver_status": driver_status, "current_context": current_context}
    print(json.dumps({"builder_driver_observation": record}, sort_keys=True, separators=(",", ":")), flush=True)
    if (len(drivers) > 1 or any("/libnvidia-ml.so" in path for path in mapped)
            or (drivers and (driver_status != 3 or current_context is not None))
            or gpu_fds or initialized not in (None, False)):
        raise RuntimeError("fixed CPU builder acquired a driver/GPU runtime")
    return record


def build_recipe(name: str, job: Path, unit: str) -> None:
    _check_environment()
    job = _private_directory(job)
    repository = Path(__file__).resolve(strict=True).parents[1]
    sys.path.insert(0, str(repository))
    from tools.optical_cuda_build_status import validate_status
    from optics.cuda_extension_recipes import CUDA_EXTENSION_RECIPES, cuda_extension_recipe

    recipe = cuda_extension_recipe(name)
    if not unit.endswith(f"-{CUDA_EXTENSION_RECIPES.index(recipe)}.service"):
        raise ValueError("build unit selected another recipe")
    groups = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    if len(groups) != 1 or not groups[0].startswith("0::"):
        raise RuntimeError("builder is outside the unified service hierarchy")
    validate_status({"unit": unit, "invocation_id": None, "service_result": None,
                     "exit_code": None, "exit_status": None, "cgroup_path": groups[0][3:]}, unit=unit)
    source = _create_directory(job / "source")
    temporary = _create_directory(job / "temporary")
    output = _create_directory(job / "output")
    for filename, value in (("main.cpp", recipe.source_module.CPP_SOURCE),
                            ("cuda.cu", recipe.source_module.CUDA_SOURCE)):
        with (source / filename).open("xb") as handle:
            handle.write(value.encode("utf-8"))

    # -S avoids .pth/sitecustomize execution. Only the selected deployment's
    # installed packages are added, and setuptools handles its ordinary alias.
    _observe_builder_driver_state("before_dependencies")
    sys.path.append(str(_PREFIX / "lib/python3.12/site-packages"))
    import setuptools
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    _observe_builder_driver_state("after_dependencies", torch)

    if str(torch.__version__) != "2.9.1+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("builder Torch/CUDA ABI differs from the deployment")
    extension = CUDAExtension(
        name=recipe.name,
        sources=[str(source / "main.cpp"), str(source / "cuda.cu")],
        extra_compile_args={"cxx": list(recipe.cflags), "nvcc": list(recipe.cuda_cflags)},
    )
    setuptools.setup(
        name=recipe.name,
        version="0.0.0",
        packages=[],
        py_modules=[],
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        script_args=["--no-user-cfg", "build_ext", "--build-temp", str(temporary), "--build-lib", str(output)],
    )
    _observe_builder_driver_state("after_build", torch)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if type(suffix) is not str or not suffix.endswith(".so"):
        raise RuntimeError("deployment extension suffix is not a Python shared object")
    artifact = output / (recipe.name + suffix)
    if artifact.is_symlink() or not artifact.is_file() or sorted(output.iterdir()) != [artifact]:
        raise RuntimeError("builder did not produce exactly its fixed extension")
    if any(item.name in sys.modules for item in CUDA_EXTENSION_RECIPES):
        raise RuntimeError("a build output was imported")
    # Descriptive output only. The launcher must still verify service termination,
    # cgroup closure, every source/input/output and the complete artifact set.
    with artifact.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    details = {"name": recipe.name, "path": str(artifact.relative_to(job)),
               "size": artifact.stat().st_size,
               "sha256": digest,
               "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
               "soabi": sysconfig.get_config_var("SOABI"), "ext_suffix": suffix}
    with (job / "build-details.json").open("xb") as handle:
        handle.write(json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--job-directory", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    build_recipe(args.recipe, args.job_directory, args.unit)


if __name__ == "__main__":
    main()
