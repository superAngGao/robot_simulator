# Independent setuptools CPU execution re-preflight

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: independent code/execution re-preflight, dependency-only
Author: codex, independent reviewer /root/setuptools_cpu_preflight_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — bounded execution preflight only
Related Files: setuptools-cpu-preflight review-request v1/v2 and unchanged review v1; setuptools-admission design-request/design-review v2 and test-evidence v2/v3; compilation-admission design-request/design-review v3
Owner Summary: Approve one dependency-only execution at the exact frozen revision, stopping at RESTORED. Both original blockers are independently closed: the runner now identifies the actual released profile bytes, and real source binding/repeated audits work under the narrow Git allowance. Forbidden process/backend operations remain denied, and both Linux owned-group cleanup cases pass. No real setuptools execution or whole implementation acceptance is claimed.

## Verdict and scope

**Verdict: Approve.** No substantive blocking or non-blocking finding remains for readiness to perform the one fixed dependency-only run described below. This is not approval of the whole implementation, nor evidence that the proposed dependency run has already succeeded.

The exact approved candidate is `311f16f1cfec641957b1a688454bb834006ccbce`. The permitted execution uses the real source session, external admission, initialization owner, ordinary dependency loaders, guardian and guard transport, with the explicitly descriptive outer input/device/toolchain stand-ins in the inspected runner. It may observe RESTORED and real resource/version reads. It must not claim COMPLETE, cpp_extension completion, compilation, native scope, readiness or a validated compilation record.

The approved command is the request's command, against the still-clean exact candidate and a fresh absolute directory that does not already exist:

```text
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/robot_sim_cuda_compilation_wip/tests/support/setuptools_dependency_process.py parent /data2/shared/ang.gao/robot_sim_cuda_compilation_wip /data2/shared/ang.gao/setuptools-cpu-311f16f1cfec641957b1a688454bb834006ccbce-v1
```

**That command was not executed during this review.** Neither installed setuptools/_distutils_hack/distutils nor any backend was imported. The runner module was loaded only for its inert constants/helpers by permitted CPU tests/probes; neither `parent()` nor `child()` ran. Source parse/compile did not execute the compiled runner code.

The complete negative dependency matrix, actual alias/resource/version integration, full v2 compilation replay, compiler/native preflight, real three builds, native/process failure matrices, whole implementation and C4 acceptance, systematic refactor, timing and product/result/baseline gates remain open work and closed authority. The subsequent real CPU result must be evaluated as new evidence, not presumed successful from this approval.

## Independence and candidate identity

I received a fresh task packet without the implementation conversation, did not author or modify the candidate, and did not delegate this review. I read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, both execution requests, the unchanged prior report and original R1/R2 reproductions, the accepted designs/reviews and referenced source/evidence. Author claims and prior verdicts were checked rather than adopted as proof.

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Local branch: `implementation/c4-supervised-waits`.
- Original implementation base, confirmed by merge-base: `52fb36a8960596ddb2ac3decd8d129188e4b1eb4`.
- Previous reviewed HEAD: `32fba6096f6425e70a4b671d2e14800dd10c2f8f`.
- Current reviewed HEAD: `311f16f1cfec641957b1a688454bb834006ccbce`.
- Current tree: `f993ae42541a1b7884380e151f6d9fa04498f65d`.
- Primary target: SSH alias `tileops-dev`, `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`, same HEAD/tree. No fallback host was used.
- Executed target interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12`, Python 3.12.13, always with `-I -S -B`.
- Initial and final local and target porcelain status were empty. HEAD/tree remained unchanged. `git diff --check 52fb36a8960596ddb2ac3decd8d129188e4b1eb4 HEAD` passed.
- The delta from the previous reviewed HEAD contains only the revised runner, new runner regression tests, v2 request and unchanged v1 report. Production bootstrap/source/profile/initialization code is unchanged by this repair.

Independently observed SHA256 identities:

| Artifact | SHA256 |
| --- | --- |
| Preserved review v1, local bytes | `ceaba8a08a3d8c676d890182725be6157629c004914f402b871d2080612d6774` |
| Accepted setuptools design v2, local bytes | `e8343ed7c9217a47ad45da5f055801cf43537a67f8059240453ed14078e27db8` |
| Accepted compilation design v3, local bytes | `028b52c466ef46433235d4ec2495cd47d00200822cc64dfaea95614fd5bf26a2` |
| Current target runner bytes | `845ede36cb292ac62521bc45753736255f2ea1b3c9fcd7dea9cf604b2a026c8e` |
| Profile Git blob = target working file = runner pin | `0cb6288345fc3a2644e61bb01dc6e5897ee58d52eb71050c31b2e76f8ea60753` |
| Source catalog v13, checked by real source capture | `17387af6b8188c40707da7e1835f639e87d1464ffa664b7cdcdb301eedd9d399` |
| Helper source archive | `8f5a7f25d4e7654cea2e157d44af708cc14b912edcdedbcc68e7852dcffdaac3` |
| Packaging source archive | `52db25f3a79ef2df681dc4c49149d788610237ce9527c10899f3f496ff72bbde` |
| Target importlib archive | `3a41be7920961a96cb36e0aa442ecbc0ec0ff2f805ceadae9fadba917fab8289` |
| Historical profile-author evidence archive | `17ebe27d160499e8e971d51b246fd6d2e9cfb0cb99dc7198ef4bad746f2766de` |
| Author's current eight-test log | `c0e9e2abe01db311c2b6b06a313a38eaccfa42297a9f9816df4cfd0cec3f162c` |

## Prior findings and counterexamples

### R1 — closed: actual release pin

At `tests/support/setuptools_dependency_process.py:24,75`, the one accepted digest is now the actual LF Git release. I independently read `git show HEAD:<profile>` on the target, compared it byte-for-byte with the working file, and compared their SHA256 with the imported runner constant. All three match. The original CRLF assertion still rejects those LF bytes, confirming the original mismatch rather than erasing its history. No runtime normalization or alternative digest was introduced.

Using this exact target profile, production `RuntimeEnvironmentContract.from_payload` and `verify_files(expected_sha256=environment.sha256)` passed all 1,791 files and the profile's interpreter/distribution/namespace checks. The new test joins Git release bytes to the runner pin and on POSIX to the working file. Both clean-tree identity and byte identity were tested; neither was substituted for the other.

### R2 — closed: required source audits and narrow process allowance

At runner lines 32–47 and its hook installation after `_verify`, the restriction permits exactly `git rev-parse HEAD` and `git show <frozen-revision>:optical_worker/data/native_sources/v13.json`, with executable `git`, list argv, exact root path type/value and `env=None`. The fixed Linux caller supplies an exact PosixPath root. There is no arbitrary path-like coercion, alternate environment, revision, command or cwd branch. Backend imports and the four other process/library events remain denied. Production source audits are not bypassed.

I independently reproduced the original cold-source failure at the revised exact target using a separate source-only probe and the old blanket-denial behavior. Actual binding attempted `git rev-parse HEAD`, was rejected, and set `session.failed=True`, before dependency initialization. The probe then ran in a fresh isolated process with the current real hook: `_install`, `_verify`, actual custody binding, two explicit session audits and snapshot revalidation succeeded with `session.failed=False`. It observed ten permitted Popen audit events, all with exact PosixPath cwd. The first pair was:

```text
git ['git', 'rev-parse', 'HEAD'] PosixPath /data2/shared/ang.gao/robot_sim_cuda_compilation_wip None
git ['git', 'show', '311f16f1cfec641957b1a688454bb834006ccbce:optical_worker/data/native_sources/v13.json'] PosixPath /data2/shared/ang.gao/robot_sim_cuda_compilation_wip None
```

Actual attempted `subprocess.run(['git','status'], cwd=root)` and `os.system('true')` were rejected before launch. Explicit `sys.audit` probes for fork, posix_spawn, ctypes.dlopen and all five prohibited backend roots were rejected by the installed hook. These latter probes test real hook dispatch without performing those operations or importing those packages; they are not claims of actual native-library execution. The candidate's finite negative test additionally rejects wrong executable, revision, command, cwd, string cwd and non-None env. No reserved module remained in the probe or test launchers.

The `_load_catalog` production chain still synchronously runs its two Git calls with 20-second subprocess timeouts, compares HEAD and committed catalog bytes, and is reached by binding and subsequent snapshot/source audits. The prior claim that Git had finished permanently before hook installation is replaced with the truthful continuing allowance.

### Owned CPU descendant lifetime

Runner lines 50–57 and 200–240 now create the direct child with `start_new_session=True`, signal only its private process group, and wait its exact direct child with a ten-second bound. Timeout, failure and final cleanup reach that group helper. After successful timeout cleanup, `group_reaped` prevents a second group cleanup. Normal leader exit still leads to group cleanup, covering a live Git descendant if the leader dies first. No name-wide kill, unrelated process scan, device reset or subreaper is used.

Both actual Linux stdlib tests passed: the leader still running with a sleeping child, and the leader already waited/reaped while its child remains. Tests verify the descendant's PGID equals the owned leader PID and then require disappearance or zombie state, rather than treating any `/proc` entry as necessarily running. Direct child wait completes. This proves the bounded CPU group's termination behavior, not grandchild waitpid ownership, kernel exit timing guarantees, compiler descendants in separate groups, or the later compiler sole-wait/subreaper contract. Those distinctions are stated correctly in v2.

No substantive new finding was identified in this bounded review.

## Inspected affected integration

- Followed real source capture, custody binding and repeated audits; checked v13/catalog, v14 AST pin exceptions and CUDA-only initialization continuation against the shared prelude. Target source-only/bootstrap and policy tests exercise actual production entrypoints.
- Read the initialization owner, bootstrap finder/file-loader hooks, environment v6, shared action/role table, compiler coupling and affected replay joins. Actual issuance is session-bound; retention occurs through the existing root/token/guardian path; partial owners remain held; failure is sticky. Restoration is owned by the initialization helper and the file-loader finally path. No second importer or token registry is introduced by this repair.
- Checked the actual pinned `_distutils_hack` spec producer and custom create/exec route, same-object alias metadata transitions, ordinary distutils child loading, singleton finder and P+vendor transition, original-P stdlib lookup, resource handle interface and named metadata reader. The real installed alias/resource/version execution is intentionally deferred to the approved run.
- Independently joined all 84 added module rows to the helper/packaging archive bytes, sizes and hashes and parsed those sources with AST. On the current target, all 84 names resolved to their declared SourceFileLoader origins without executing packages. The three new absence probes (`Cython`, `_distutils_system_mod`, `dl`), absent egg-info METADATA including dangling-link detection, and the three new namespace locations passed.
- Checked the source-audited lazy paths `monkey.patch_all -> _patch_distribution_metadata -> _core_metadata` and `_jit_compile -> hipify_python/GeneratedFileCleaner`; the latter remains outside this run. Python 3.12 selects stdlib tarfile, and the finite profile excludes older-importlib/backports and other-platform branches. Predecessor file/module/namespace/absence/publication/native-global/replacement declarations remain preserved by the passing profile tests.
- The runner begins the real dependency entry at guard6, with clearly labeled outer input/device/toolchain stand-ins. It constructs no `_Compilation` or fake cpp_extension. Its assertions demand RESTORED, actual alias identity, version83.0.0, resource digest and closed read handles, completed ordinary records, unchanged hooks/topology, source/admission/retention audits, no backend modules and no completed initialization action. The descriptive CPU result is not fed to the compilation parser as accepted evidence.
- The inspected HOST_OWNER terminal path supplies no synchronization callback and hard-exits after writing the result. Parent stdout/stderr and exact-child exit/timeout evidence are retained. A failed run cannot satisfy parent success. The existing real CPU Peer remains the transport helper and joins its thread on close; this is not the compiler supervisor or native-scope proof.

## Executed verification

Read-only inspection used UTF-8 Get-Content, rg, git status/rev-parse/merge-base/diff/show, Get-FileHash, local stdlib zipfile/hashlib/json/ast and target isolated Python. Some large combined reads were truncated and were followed by bounded section reads. No candidate mutation, dependency installation, environment change, checkout, reset or real materialization occurred.

Target launchers were sent on stdin using:

```text
ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12 -I -S -B -
```

Pytest launchers set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, changed to the exact target tree and inserted only the repository and robot_sim site-packages path for the test framework. They used `-q -p no:cacheprovider` and an extra reviewer hook rejecting installed dependency/backend imports. The new cold-source and resolution probes keep the normal isolated stdlib path; they do not add site-packages or execute .pth files.

| Independent execution | Observed result |
| --- | --- |
| Exact Git/profile/runner digest equality and production schema/file verification | Passed; 1,791 files |
| `test_setuptools_dependency_runner.py`, `test_compilation_environment_profile.py`, `test_setuptools_admission.py`, `test_setuptools_contract.py`, `test_setuptools_replay.py`, `test_runtime_environment.py`, `test_native_source_policy.py` | 313 passed in 10.68s; exit0; reserved modules absent |
| Source-only `test_native_bootstrap.py` plus `test_native_sources.py::test_literal_lane_catalogs_have_no_losing_optical_modules` and `::test_complete_catalog_capture_is_cpu_safe_and_byte_exact` | 80 passed, 1 skipped in 84.41s; exit0; reserved modules absent |
| Exact-target cold source probe, old blanket hook | Original R2 reproduced, sticky failure true; expected result exit0 |
| Exact-target cold source probe, repaired hook | Real binding/repeated audits passed; ten allowed Git events; tested denials passed; exit0 |
| Current target finite source resolution/AST/namespace/absence probe | 84 modules, three namespaces, three absent probes and absent METADATA passed; reserved modules absent; runner compile-only passed |
| Local raw archive/hash/size/AST joins | All 84 added rows matched |
| Final target/local identity/status and base-to-HEAD whitespace check | Passed, same clean HEAD/tree |

The single skip is the bootstrap test for rejecting unsupported platforms, which skips on this supported Linux interpreter. It is not a missing Linux process-group test: both group cases ran and passed. Source-test fixtures create temporary scratch repositories/files and owned children; no tracked candidate changes resulted. Test child source processes use `-I -S -B`; no bytecode or pytest cache was requested.

One initial reviewer launcher completed `verify_files` but then failed while trying to print `len(environment.files)`, an attribute the production contract does not expose. This was a reviewer reporting error after verification, not a product failure. The saved corrected launcher counts the parsed payload rows and reran verification plus the full selected tests successfully. No code fix or dependency execution was used to resolve it.

The supplied author log was read and its digest verified; its eight-test result is not counted as an independent run. Author PIDs 549345/550099/550102 and recorded reviewer PIDs 562939/564777/567245/569733/569939/578025 were independently absent in `/proc` at final checking. Group tests themselves establish descendant cessation. This is scoped process hygiene, not exhaustive OS/GPU attestation or a claim to reap grandchildren. No compiler/GPU worker or persistent service was started.

## Review artifacts and limitations

Artifacts are outside the tracked tree under `C:/Users/AG/新建文件夹/c4_assets`. Exact scratch probes make the old/repaired hook behavior and target resolution reproducible without running the dependency entry:

| Artifact | SHA256 |
| --- | --- |
| `setuptools_cpu_preflight_v2_independent_tests.py` | `b7c6fe1514516cf0c3a2e1671008d25644d200d350a6664067e558c8cd07173b` |
| `setuptools_cpu_preflight_v2_cold_probe.py` | `98aca79faed136d1492f2d419b8d1cd5fea18eff4c3e6b6176e0f2bce72ac432` |
| `setuptools_cpu_preflight_v2_resolution.py` | `a3cdf6a41c4141442221a6edd0c2c21f7534ded93785ad870df062e54622eb02` |
| `setuptools_cpu_preflight_v2_independent_tests.log` | `15df2a6ab09e3a6643980363b5fc4872c43c8992916cfb1e76f879538f03eea9` |
| `setuptools_cpu_preflight_v2_bootstrap.log` | `0b3c0605ac9f8bb70da2c9467db973e406894661ee084e3f4990b05c1bf88c6f` |
| `setuptools_cpu_preflight_v2_old_hook.log` | `c998856aea4120467a5aa4a09d378f75340d2ac563b7f46f82eb89a07d0a6008` |
| `setuptools_cpu_preflight_v2_repaired_hook.log` | `ea1c697ced550aac30c49ad6b15cd0e9aa5c4226a8b8bb845377c3976db14328` |
| `setuptools_cpu_preflight_v2_resolution.log` | `c6607a3570a39f5a92c8c6201a74cd33012d0f712ad65e5a9b78f35e4391d322` |

Run the cold probe on stdin with argument `old` or `repaired` after the final `-`. Logs are new local Tee-Object captures of this review's target stdout, not reused author outputs.

The environment is the fixed trusted deployment and Linux ABI in the packet; the audit hook is a bounded test restriction, not a hostile-code sandbox. Windows Popen compatibility, arbitrary processes/daemon escape, compiler-tree ownership and native safety are not granted by it. Static finite source checks cannot establish that every runtime import/resource call will succeed. Actual dependency initialization and the full parent success/timeout wrapper were intentionally not executed; the helper/source tests cover their permitted preflight mechanics only.

Approval binds the exact candidate and bounded single-run scope above. Any implementation or operative workflow change needs review of its delta. The unchanged report can be archived as evidence without changing its verdict or opening later gates. The report's whole-file SHA256 is recorded in the adjacent `.sha256` sidecar and review handoff, avoiding a self-referential digest inside the hashed report.
