"""Bind an explicitly selected deployment into the cold worker's v5 inventory.

The reviewed launcher supplies the independent expected platform, base profile
and deployment identities. This command imports no backend and launches no job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig


def compose(*, base_profile, base_sha256, deployment, manifest_sha256, expected_platform, output):
    repository = Path(__file__).resolve(strict=True).parents[1]
    sys.path.insert(0, str(repository))
    from optical_worker.runtime_environment import RuntimeEnvironmentContract
    from tools.optical_cuda_artifacts import extend_runtime_environment

    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise RuntimeError("cold environment composition requires python -I -S -B")
    if os.path.lexists(output):
        raise FileExistsError(output)
    base = RuntimeEnvironmentContract.from_payload(json.loads(base_profile.read_bytes()))
    if base.to_payload()["schema_version"] != "q54.p12.4.runtime-environment.v5":
        raise ValueError("worker composition requires the ordinary v5 base profile")
    if base.sha256 != base_sha256:
        raise ValueError("base environment differs from the selected digest")
    selected_platform = json.loads(expected_platform.read_bytes())
    running = {"system": platform.system(), "machine": platform.machine(),
               "python_version": list(sys.version_info[:3]), "soabi": sysconfig.get_config_var("SOABI"),
               "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX")}
    if any(selected_platform.get(key) != value for key, value in running.items()):
        raise ValueError("selected CUDA platform differs from the cold launcher ABI")
    derived = extend_runtime_environment(base, deployment=deployment, manifest_sha256=manifest_sha256,
                                         repository=repository, expected_platform=selected_platform)
    # Check the final bytes/paths too, using the same verifier the worker uses.
    derived.verify_files(expected_sha256=derived.sha256)
    with output.open("xb") as handle:
        handle.write(derived.encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return derived.sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-platform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(compose(**vars(args)))


if __name__ == "__main__":
    main()
