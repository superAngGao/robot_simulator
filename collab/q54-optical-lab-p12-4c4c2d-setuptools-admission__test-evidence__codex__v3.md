# Installed compilation profile: source and read-only target evidence

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: CPU/source inventory evidence; dependency execution remains closed
Author: codex
Version: v3
Date: 2026-10-09
Status: draft
Related Files: setuptools-admission implementation-note v5; design-request/design-review v2
Owner Summary: The explicit v6 target profile passes production schema/file verification, complete file hashes, new module-source resolution, namespace-directory checks and finite absence probes. No actual setuptools, Torch, Warp, compiler or GPU code executed.

## Inputs and provenance

Local worktree base `d69c281b647f2c101eb340516d2019e2a2fde341`; candidate source
changes are described in implementation-note v5. Target isolated tree was clean
detached `553038f11c24716d23be279dc08ac9e9257efc27`. Its runtime-environment and
compilation-contract implementation is identical to this candidate. Target:
tileops-dev, `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, 3.12.13.

Source inputs in `C:\Users\AG\新建文件夹\c4_assets` (copies under
`/data2/shared/ang.gao` where previously collected):

| Input | SHA256 |
| --- | --- |
| cuda-compilation-helper-sources-v1.zip | 8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3 |
| cuda-compilation-packaging-sources-v1.zip | 52db25f3a79ef2df681dc4c49149d788610237ce9527c10899f3f496ff72bbde |
| cuda-compilation-target-importlib-sources-v1.zip | 3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289 |
| cuda-compilation-setuptools-conda-facts-v1.json | 45203dcfcd97a7b925b8c72061b71b38187328e7205e08ced3f28c98a097f403 |
| c2d3-offline-wheel-inventory-v1.json | 5e7e954d4c79ce323c9712d97c883d9311dc7182f9fb72891912ab007ab9099a |
| cuda-compilation-stdlib-312-facts-v1.json | 739e34613c16da9afc15d323790c38de198820d9735c5278b86a9566a95891e7 |

The exploratory AST graph alone is not a complete closure or permission source.
Authoring adds the source-audited lazy roots and selects the actual Python3.12/
Linux branches described in implementation-note v5. It matches every added
Python source byte against the separate installed Conda/wheel facts and emits
explicit rows before any dependency execution. The resulting profile is tracked
at `tests/data/runtime_environments/183_cuda_compilation_author_v1.json`.

The one-time authoring scripts, full audit and target evidence are archived in
`cuda-compilation-profile-author-evidence-v1.zip` in the local scratch directory.
Archive SHA256 `17ebe27d160499e8e971d51b246fd6d2e9cfb0cb99dc7198ef4bad746f2766de`,
224,309bytes.
Individual exact contents:

| File | SHA256 |
| --- | --- |
| build_compilation_profile_candidate_v1.py | ca21f11b70ffe354e753ae97ced60da469e23e50a3e99255ec5183f0e2fa1c2a |
| inspect_compilation_python_closure_v1.py | cc0e56707a5ae6d6bfd3454ccd02b81a82e98c06ff71773ba11f35d82733cf3b |
| verify_compilation_profile_files_v1.py | 2fda30e28877954f3a0b8d73f94e3ffe7d9e9ff085a2abdd5a25203c393aec62 |
| cuda-compilation-profile-source-audit-v1.json | 787338845926ccb9f89b2fee20d6028b8c5555518b53a7a937ab29d568184500 |
| cuda-compilation-profile-files-proof-v1.json | c4be78edaf190bf28808e62819b3a763e626112b5537d3cc0e7441f0161d4635 |
| 183_cuda_compilation_author_candidate_v1.json | 1725bfd714059d60010644e9160ef9022d27ce4d32ee89e2bae229979cb8dea8 |

## Actual target command and result

Uploaded the explicit profile, source audit and read-only verifier to
`/data2/shared/ang.gao`, then ran:

```
cd /data2/shared/ang.gao/robot_sim_cuda_compilation_wip
git rev-parse HEAD
git status --short
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/verify_compilation_profile_files_v1.py
```

Final verifier exit0, PID397061, 8.086642558686435s. Production
`RuntimeEnvironmentContract.from_payload` and `verify_files` passed. A second
stat-stable streaming hash sweep observed all1,791 files /4,701,253,155 bytes.
All84 new module names resolved to the declared SourceFileLoader origins without
executing them. Cython, dl and `_distutils_system_mod` resolved absent on the
original interpreter path plus site then vendor. Egg-info METADATA was absent
including dangling links. Setuptools PKG-INFO and all25 distribution identities
passed the production metadata parser.

The first authoring verifier attempt failed at `PathFinder` namespace wrapping:
an unimported distutils parent has no sys.modules entry. The final verifier uses
standard FileFinder namespace fragments on the explicit parent directories,
aggregates those locations and compares the frozen rows, without importing or
fabricating parents. This is a directory/resolution check, not runtime namespace
publication proof. A preceding successful read-only run (PID394686) used the
same file/resolution checks before production verify_files was added. Both PIDs
were subsequently checked absent in `/proc`; no worker/compiler was launched.

Reserved dependency roots were absent from sys.modules before and after all
checks. No .pth files ran; the interpreter used -I -S -B. No installation,
toolchain mutation, subreaper/signal handling or native observer ran.

## Local CPU checks

On Windows Python3.14, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`:

```
C:\Python314\python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_compilation_environment_profile.py tests/unit/optics/test_native_sources.py::test_literal_lane_catalogs_have_no_losing_optical_modules tests/unit/optics/test_native_sources.py::test_complete_catalog_capture_is_cpu_safe_and_byte_exact
```

180passed/1skipped, 16.32s, exit0. The POSIX-only parser test was skipped; its
expected child-name assertion was subsequently corrected from source package
name to requested `distutils.core`, consistent with the approved non-alias child
contract. That POSIX assertion still needs execution on target. Target production
schema/file validation above passed independently. Prior source-only cold
bootstrap suite on the same production sources passed238/skipped9,112.86s,
before the six later helper pin tests (covered by the current focused run).

Actual dependency import, complete v2 compilation evidence, compiler process
cases, native scope/build/failure matrices and independent whole checkpoint
approval are not claimed. This evidence does not open any higher C4 gate.
