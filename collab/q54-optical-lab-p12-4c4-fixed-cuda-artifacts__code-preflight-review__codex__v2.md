# Independent code preflight and manager evidence review: fixed CUDA artifacts v2

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: actual manager evidence review and bounded builder repair preflight
Author: Codex independent reviewer /root/fixed_artifacts_preflight_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted - manager evidence and bounded continuation only
Related Files: fixed-cuda-artifacts design-request/design-review v2; code-preflight-request v1/v2; code-preflight-review v1
Owner Summary: Approve the actual six manager cases and the small builder correction for the exact fresh-output continuation. The failed first build remains failed evidence. No successful build, driver-state execution, artifact/GPU/numerical acceptance, C4 completion or product gate is approved by this report.

## Verdict and independence

**Verdict: Approve.** No substantive blocking or non-blocking finding is raised for the reviewed delta and bounded continuation.

This is a fresh independent session, `/root/fixed_artifacts_preflight_v2`, without the implementation conversation. I did not author or edit the candidate, delegate implementation, or launch subagents. I read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, both preflight requests, the prior report and the accepted fixed-artifact design. Prior approval and author summaries were treated as claims, not proof. The two separate conclusions are:

1. **Actual manager evidence accepted:** the six installed-manager cases executed at a43f0c1 satisfy their bounded fixture requirements, including separate-PGID descendant ownership, timeout after waiter loss, clean-signal rejection, cgroup closure and sentinel isolation.
2. **Repair/continuation preflight approved:** the builder correction at 66240e4 may be exercised by the exact v2 continuation. Real execution of the new getter and builds is still pending. This is not acceptance of the fixed-artifact feature or its later native campaign.

No service, build, GPU import, native CUDA getter, worker or fault campaign was executed by this reviewer. Remote work consisted of bounded read-only SSH, stdlib file/hash/Git/proc inspection and installed-source/header reads.

## Exact identities and scope

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Branch: `implementation/c4-supervised-waits`.

- Whole replacement base: `dcfa7a7120690a4c79bfbc5baa36a116c641ab28`.
- Previously reviewed code and actual v1 execution: `a43f0c1e292e586b2740bfb2142de142e5284639`.
- Repair/code execution revision: `66240e4fedc24ad8a5109690e3b1e78b450e2f62`.
- Frozen reviewed packet HEAD: `762f7c3fa66062f7b37ea29c3f5b6177d9978cfc`.
- Actual initial and final porcelain status: empty. HEAD stayed unchanged.
- Ancestry checks passed. Base-to-packet `git diff --check` passed.
- Since a43f0c1, the only production/test code changes are `tools/optical_cuda_build_recipe.py` and `tests/unit/optics/test_cuda_artifact_build_status.py`. Other changes are records/PROGRESS. Code-to-packet changes are only PROGRESS.md and the v2 request.
- Accepted design byte SHA256 independently matched `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.

Fresh read-only SSH observed the primary isolated checkout `tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` clean at **a43f0c1e292e586b2740bfb2142de142e5284639**, not yet at the proposed repair. I did not transfer code or change remote HEAD. The original repository and fallback host were not used.

This review independently examines the delta, actual manager/failed-build records and affected deployment/source/environment/command invariants. It does not claim a fresh repetition of the prior whole-replacement worker review or its 793 CPU tests. The absence of worker/catalog/ABI changes was verified from Git.

## Actual manager evidence

I inspected the unchanged fixture, production service argv builder, termination recorder, normal-exit validator and cgroup closure implementation. I then recomputed all local raw-file hashes and compared them to freshly read remote originals:

- Manager directory: all **65 files** match by relative path, size and SHA256.
- Failed build job0: all **7 files** match.
- Combined archive SHA256 matches `100270980dece1bb0dab48ca8c51f51879eef4999e21fb806408fc5a2c115afa`.

Local directories are `C:/Users/AG/新建文件夹/c4_assets/c4-fixed-artifact-manager-v1` and `C:/Users/AG/新建文件夹/c4_assets/fixed-artifacts-build-v1-job0`. Remote originals are the packet's manager directory and `.cuda-stage-a59890bf38eb469696857fc48b0b9097/jobs/0`.

I independently joined results.json to every per-case result, raw termination footer, unit, main cgroup and invocation argv. All six argv lists exactly equal the current unchanged `fixture_argv` output for their recorded paths/units. They use production properties with only the documented fixture command and 6s/6s/1s bounds substituted. The failed build argv separately exactly equals production `build_service_argv`, including its fixed env -i selection and 120s/120s/10s bounds.

| Case | Waiter | Manager main/result | Normal-exit gate | Cgroup |
| --- | ---: | --- | --- | --- |
| normal | 0 | success / exited / 0 | accepts | absent |
| descendant | 0 | success / exited / 0 | accepts | absent |
| timeout | 1 | timeout / exited / 0 | rejects | absent |
| launcher_loss | -9 | timeout / exited / 0 | rejects | absent |
| signal | 0 | success / killed / TERM | rejects | absent |
| nonzero | 7 | exit-code / exited / 7 | rejects | absent |

The main/child identity records show different PGIDs within the same service cgroup in all three descendant cases. The fixture checks the main has exited while the waiter and child remain alive before writing after-main-exit.json. Record times are consistent with that sequence. The released child subsequently writes child-complete.json before success. The timeout log reports 7.509s service runtime under the short runtime plus stop bound. The launcher-loss waiter log ends after its invocation line, as expected; its later timeout footer and independently observed closure provide terminal service evidence instead of pretending the killed waiter observed it.

Every case records the same separately owned sentinel alive. Its final record is PID2045207, returncode -15 after explicit terminate/wait. I independently reran the real normal-exit validator over every raw footer/recorded waiter status and obtained exactly the eligibility above. In particular systemd success and waiter0 do not admit the SIGTERM main.

Fresh read-only /proc and cgroup inspection found all six exact recorded fixture groups absent, and all **16** recorded fixture main/child/waiter/sentinel PIDs absent:
2045207, 2045208, 2045209, 2045211, 2045212, 2045213, 2045217, 2045218, 2045221, 2045400, 2045401, 2045402, 2045647, 2045648, 2045650, 2045651.
This is a current scoped observation, not historical PID attestation or proof from intended kill commands.

These records, their provenance and their producing code are sufficient to accept the requested actual six-case manager evidence. Repeating that unchanged fixture is unnecessary for the builder-only correction.

## Failed first build

The real first scene build is correctly retained as a failure, not relabeled:

- Unit: `robot-sim-cuda-a59890bf38eb469696857fc48b0b9097-0.service`.
- Invocation: `24aaf6c8592d4a46bd691a19ab49b7e3`.
- Raw footer: result exit-code, main exited/status1.
- Waiter log: service runtime1.479s, matching unit/invocation and failure.
- Stderr traceback: rejection at after_dependencies, before CUDAExtension construction or setuptools compilation.
- Stdout: before_dependencies had no recorded driver mapping or GPU FDs; after_dependencies had one mapped `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`, no recorded GPU FDs, Torch initialized=false.
- Source inputs exist; no build-details, Ninja input or extension output exists in the seven-file retained job evidence.

The old observer treated any mapping as driver use. These logs establish that condition, **not** that cuInit had successfully initialized the driver, nor that it had never been attempted. The precise dependency loading call is not established and is not needed to attribute this specific rejection.

Fresh remote inspection found PID2047742 and the exact failed-build cgroup absent. The v1 deployment path remains absent; v2 deployment/environment/runtime output paths were also absent when inspected. The failed private staging and reservation must remain preserved. The repair does not resume them.

## Correction, API behavior and provenance

At tools/optical_cuda_build_recipe.py:116-155 the observer still reads its own maps/FDs and Torch flag at the existing checkpoints. It now distinguishes a mapped driver from an initialized one:

- No mapped libcuda: no CDLL or getter call.
- Exactly one mapped libcuda: reopen only that absolute mapped path using RTLD_NOLOAD|RTLD_LOCAL, bind fixed cuCtxGetCurrent with CUcontext* / integer result, and query into a zero-initialized pointer.
- Accept only status3 with a null pointer. Status0 with a null current context is rejected, as is any nonnull context, other status, multiple distinct driver paths, NVML mapping, GPU FD or initialized Torch flag.
- Load, symbol or getter exceptions propagate; there is no fallback cuInit, library discovery, device enumeration, NVML call, context creation or exception-to-success conversion.
- The original manager deadline also bounds a hung or failed getter; no new timeout/process layer was introduced.

NVIDIA's CUDA12.8.1 initialization contract specifies the uninitialized-driver error before cuInit; the context getter lists that error and separately describes successful null-context results. Thus a null pointer alone cannot establish an uninitialized driver. This implementation makes the correct distinction. Sources: [CUDA initialization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__INITIALIZE.html), [cuCtxGetCurrent](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html).

The installed `targets/x86_64-linux/include/cuda.h` independently confirms the exact CUcontext* signature and CUDA_ERROR_NOT_INITIALIZED=3. Its definition explicitly includes initialization failure. Therefore status3 is a current-state observation, **not a history proving no initialization attempt**. This limit is correctly disclosed in v2 and must remain in the eventual evidence review.

RTLD_NOLOAD does not load an absent object; reopening an already loaded object does not rerun its constructors. Python's POSIX ctypes adds RTLD_NOW, which can resolve previously lazy symbols; the call is therefore not claimed to have zero dynamic-linker bookkeeping. This does not add an initializing CUDA API or relax the pinned-library/trusted-dependency boundary. Sources: [Linux dlopen contract](https://man7.org/linux/man-pages/man3/dlopen.3.html), [Python ctypes mode behavior](https://docs.python.org/3.12/library/ctypes.html).

Fresh remote driver file observation matches the existing selected build profile:
path `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`, size91501576, SHA256
`76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6`.
The launcher verifies profile files before and after deployment. The new getter consumes that installed trusted dependency rather than a caller-selected probe module. No test helper is imported into production; worker source admission and its ABI are unchanged.

I freshly fetched the installed Torch sources read-only:
`cpp_extension.py` SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`;
`torch/cuda/__init__.py` SHA256 `9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99`.
The selected availability path still parses empty CUDA_VISIBLE_DEVICES to an empty list, returns zero before NVML discovery, and avoids the negative-result CUDA fallback. Explicit architecture9.0 avoids device/capability discovery. The fixed standard build path and environment are unchanged.

The source/environment argument plus the corrected checkpoints is sufficient for this bounded retry preflight. Neither the old logs nor CPU mocks establish the new native getter's actual result, no-GPU-init history across the whole build, or compiler-child behavior. The next real evidence review must inspect all actual before/after checkpoints and raw build/provenance records. There is no permission to suppress another rejection, reinterpret unexpected native status, change dependencies, or weaken the no-GPU-init requirement.

## Fresh verification

Executed from the frozen repository on Windows:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_cuda_artifact_build_status.py tests/unit/optics/test_cuda_artifact_manifest.py tests/unit/optics/test_cuda_extension_recipes.py
```

Result: **62 passed in17.49s**, terminal exit0, tool session35831.

Additional independent CPU checks exercised the real observer with mocked OS/CDLL inputs: status3/null accepts; status0/null, status0/non-null, status3/non-null, deinitialized4, unknown999, multiple mappings, simulated load/symbol failures and getter exception all reject; absent mapping makes no native-library call. **11 branch checks passed.** These checks never called a native CDLL/getter. They supplement the committed tests' GPU-FD/NVML/Torch/injection and manifest/recipe negatives without modifying the candidate.

Independent evidence checks validated the seven complete argv lists, footer/result/unit/cgroup relations, child identity/time ordering and real normal-exit gate. Profile canonical digests match:
build `a9f42cc833088f308545749e4f239c246f787d9ec39096825815ae9557bb4829`;
CUDA base `79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919`;
Warp base `88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8`.

Other checks were Git revision/status/ancestry/diff/diff-check, UTF8 source reads, rg searches, stdlib JSON/hash checks, official API documentation and bounded SSH reads. The first remote read attempted the non-existent prefix/include/cuda.h and exited1 before returning its JSON. The corrected targets/x86_64-linux/include/cuda.h read completed exit0. An initial local root REVIEW_PRINCIPLES.md read missed its collab location; the required collab file was then read in full. One Windows rg glob-path invocation failed and was replaced by a directory search. These were inspection-path errors, not candidate/runtime failures.

All reviewer tool sessions reached terminal completion. No owned service, GPU process or compiler was started. Only outside-candidate source copies, read-only observation JSON and this report were written. The candidate remained clean.

## Exact authorized continuation and closed gates

Approval authorizes clean Git transfer of **66240e4fedc24ad8a5109690e3b1e78b450e2f62** to the named isolated primary checkout, preserving history and checking exact clean HEAD, then **v1 steps2,3,4 in order** with only the four substitutions specified by v2:

- revision: 66240e4fedc24ad8a5109690e3b1e78b450e2f62;
- deployment: /data2/shared/ang.gao/c4-fixed-cuda-artifacts-v2;
- environment: /data2/shared/ang.gao/c4-fixed-cuda-environment-v2.json;
- runtime evidence: /data2/shared/ang.gao/c4-fixed-artifact-runtime-v2.

The v1 packet's exact env -i prefix, `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B`, fixed profile digests and independent expected-platform file remain mandatory. Each three-recipe build retains 120s runtime/start and10s stop. Derive manifest/environment digests only from successful new outputs. Inspect terminal results, actual cgroups, checkpoints and logs before advancing. No overwrite or reuse of v1 failed output/reservation/staging; no automatic retry after the next unexpected result.

The runtime continuation remains the same ten fixed sequential cases and private-copy artifact negatives, with normal worker GPU visibility and target1/ambient0, original 120000ms operation/1000ms ACK and1235s fixture wait. Approval adds no fault selector, dependency installation, budget increase, alternate process system, broad signaling or product activation. The independent expected CXX11 ABI cannot be rewritten to fit an unexpected build.

The real getter/build/artifact/GPU/numerical acceptance remains pending and needs independent evidence review after execution. Fixed-artifact preflight is not C2d3 or C4 acceptance. Warp NVRTC/cache/ModuleExec custody, parent timeline/arena, one-slot graph, ten warmups, final readiness/graph/seal/barrier, C4c3/c4, C4d/B3 and the owner-required whole-C4 structural refactor/review remain. Product PREPARE, benchmarks/results/baselines and backend selection stay closed.

The complete unchanged report may be archived under the evidence-only exception. Any operative code or command delta requires review of that delta.

