# Independent whole fixed-artifact evidence review v1

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: independent whole fixed-artifact replacement evidence/acceptance review
Author: Codex independent reviewer /root/fixed_artifacts_acceptance_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted
Related Files: fixed-cuda-artifacts design-request/design-review v2; code-preflight-request/review v1/v2; test-evidence v1
Owner Summary: Approve the whole bounded fixed-artifact replacement at code66240e4, frozen packet0df3e260, based on independently checked manager/build/file evidence, all ten actual native cases and596 fresh passing CPU tests. Numerical rendering, remaining C2d3/C4 work and product/performance gates remain unproved and closed.

## Independence and exact scope

Fresh session `/root/fixed_artifacts_acceptance_v1` received the requirements and frozen candidate without authoring the implementation. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the accepted design and both preflight request/review rounds, then checked actual code, raw evidence and fresh CPU executions. Earlier approvals and author summaries were claims to verify, not acceptance proof. I did not edit candidate files, delegate implementation, launch services/builds/native imports/getters/faults, or signal/restart/take ownership of the matrix. Scratch scripts, downloaded source copies and observations are outside the candidate in `C:/Users/AG/新建文件夹/c4_assets/`.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Whole replacement base: `dcfa7a7120690a4c79bfbc5baa36a116c641ab28`.
- Frozen local packet HEAD: `0df3e2609a7af03c789be819714f8d79f104ce63`.
- Actual remote execution/source revision: `66240e4fedc24ad8a5109690e3b1e78b450e2f62`.
- Named remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip`.
- Read-only SSH independently observed that exact clean remote HEAD; local status was empty. Code-to-packet delta is four documentation/evidence files, 386 insertions. Whole base-to-packet diff is 76 files, 4559 insertions and 5383 deletions.
- Base-to-execution-code diff-check passed. Base-to-packet diff-check reports only an extra final blank line in the verbatim archived v2 review; this is nonoperative archival formatting, not a requested repair to the reviewer report.

Accepted design SHA256 independently matched `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`; current catalog SHA256 matched `3a21fcc175518b0e8bdbdfe3e484278f4d97056c03d28ae91e7525fbdc028b79`.

The Windows working copy of `optics/_cuda_lbvh_source.py` has CRLF while its Git blob and deployed source have LF. The initial scratch working-byte comparison therefore rejected; comparison against the exact committed Git bytes passed for every deployment source row, and remote `verify_deployment` independently checked the actual LF files. No source was rewritten to make the check pass.

## Source and integration assessment

Inspected the deployment manifest/launcher/builder/status/cold-environment path; fixed recipe and source changes; compatibility resolution; ordinary external create/exec admission; issued extension-set identity; factory preparation; native-scope and runtime-custody replay; source policy/catalog and deletion/historical boundaries; both actual acceptance fixtures and their native observer/fault paths.

The deployment has fixed recipes and bounded user-manager argv, unique staging/reservation, ordinary CUDAExtension/BuildExtension, raw manager footer plus separately observed cgroup closure, and complete-set publication only after successful outputs/provenance checks. A manager-success killed main remains ineligible. Cold selection checks exact artifact/source/platform bytes and derives the existing ordinary v5 environment. Compatibility and captured preparation share recipes, entry checks, fixed output resolution and factory/kernel logic; neither has a JIT fallback.

CUDA imports use the existing ordinary file loader in original guard3 after runtime binding. Create returns enter retained admission state inside indexed target/stream contexts before fallible journal/adoption/unwind. Entry and set ownership must join the exact session/controller/runtime/admission and named prior guardian adoptions. Factories revalidate the live issued set and actual context, and LBVH sizing and construction use the same resolved module. Partial create/exec/adoption failures cannot issue a usable set or complete runtime initialization; factory failure cannot complete factory preparation or issue final readiness.

V3 replay joins each of the six ordered native create/exec pairs to its actual file frame, entered leaf, original guard and retained create return, then matches module/spec/loader and separately named entries/set adoptions. V1/v2 meanings remain distinct. Direct comparison of relocated historical readers found documentation/import-path changes, shared unchanged input phases, and the explicit historical-v2 constant; no live worker compiler route is retained for archive compatibility. The live compiler/setuptools/supervision and special source-policy allowances are removed. Search hits in historical readers/catalogs and inert build-profile declarations are not worker execution routes. The parent whole-C4 structural refactor remains a separate obligation.

## Real installed-manager and build evidence

Independently inspected the actual six manager fixture directories and production normal-exit gate. Normal and separately grouped descendant cases are eligible; timeout, launcher loss, output-then-SIGTERM and exit7 are not. Raw descendant rows show different PGIDs in the same service cgroup and the waiter still live after main exit. The released child completed before success. Timeout/lost-launcher rows preserve manager timeout, waiter1/-9 respectively; SIGTERM preserves success/killed/TERM with waiter0, correctly rejected; nonzero preserves exit-code/exited/7 with waiter7. All six recorded cgroups and recorded main/child/waiter PIDs were independently absent on read-only SSH. Sentinel remained outside each service and alive during the campaign; retained terminal record reports PID2045207, waited returncode-15. No fixture was rerun.

Published deployment: `/data2/shared/ang.gao/c4-fixed-cuda-artifacts-v2`.

- Canonical manifest SHA256: `9981192f6c3874777755425fbe28e257c3b85d582be8303a665f2d2209d3c71a`.
- Selective local raw build archive SHA256: `35caf5fe21dc14b358829a5e72e9e4e9b41985d607385a074f429c443ab49ba6`.
- Canonical environment SHA256: `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`.

The independent archive audit checked canonical manifest schema, committed source bytes, all three extension bytes, emitted source/Ninja inputs, stdout/stderr/footers, invocation argv, build-details output identity and ABI. The remote production verifier independently accepted the published deployment against current source and the independent expected-platform file. Deriving v5 anew from the selected base returned byte-for-byte the canonical environment. All three existing private invalid copies independently rejected again: missing/partial file and changed bytes; no worker was launched by these CPU checks.

| Artifact | Manager invocation | Recorded runtime | Size | SHA256 |
| --- | --- | --- | --- | --- |
| scene_v1 | 34581134aa3944d4b5ba155c49f170db | 50.195s | 313640 | 344af1bb7c6465f200f68a0db50f89bcb795f1a7a99dfb67e674d18d4ac089c9 |
| lbvh_native_v3 | 6b64a4ce47f449f1afca4fe66325adae | 54.493s | 745288 | 870073bdf92abc8e46ba488014f0ea2d46d4db66cec60eaf115aeaa29a36bf7f |
| direct_light_v8 | a56994121a7f4f4189051b25083739cd | 51.561s | 497160 | 98e42e77f5aa80987fbc2cb2b6d7d01e05ad48d11b535a27093ac56f709d7b22 |

Each raw footer is success/exited/0 with the exact unit and invocation; manifest closure is absent, freshly reobserved absent. Builder PIDs2074763/2077412/2079106 were absent. Actual CXX11 ABI=true, Python3.12.13 SOABI/suffix, Torch2.9.1+cu128, CUDA12.8 and explicit sm_90-only build flags match selection. Both recorded warnings (no visible CUDA runtime and absent compiler-version bounds) remain preserved, with normal successful builds.

All nine builder checkpoints passed independent inspection: before-dependency no driver mappings/FDs; later checkpoints one mapped libcuda, status3/null context, no GPU FDs and Torch initialized=false. I freshly fetched the installed source read-only: cpp_extension SHA256 `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`; torch/cuda SHA256 `9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99`. Source inspection confirms the selected NVML-based availability branch reaches an empty visible-device list and returns zero before NVML discovery/CUDA fallback; explicit architecture bypasses device discovery. The standard build route does not enter load_inline/PCH/output-import code. Combined with actual emitted inputs and checkpoints this supports the accepted trusted-source no-GPU-initialization build argument. It is not syscall history, proof that no earlier failed initialization call was attempted, or compiler-child tracing. No stronger history claim is made.

## Fresh CPU verification

From the frozen Windows repository, interpreter `C:/Python314/python.exe -B`, all pytest commands use `-m pytest -q -p no:cacheprovider`, test paths under `tests/unit/optics/`:

1. `test_cuda_artifact_build_status.py test_cuda_artifact_manifest.py test_cuda_artifact_runtime.py test_cuda_artifact_replay.py test_cuda_extension_loading.py test_cuda_extension_recipes.py test_runtime_custody.py test_guarded_lifecycle.py`: **249 passed in22.97s**, terminal exit0, session49318.
2. `test_native_source_policy.py test_native_sources.py test_runtime_environment.py test_runtime_native_scope_codec.py test_guarded_runtime_binding.py test_setuptools_replay.py test_cuda_compilation_codec.py test_compiler_job.py`: **347 passed,47 skipped in50.33s**, terminal exit0, session19849.

Total **596 passed,47 skipped**. POSIX/archive/native-dependent skips are unexecuted coverage, not native passes. The tests cover malformed and partial manifest inputs, wrong session/context/entry and replaced/copy authority, return/adoption failure retention, named ownership and scope/file replay attacks, source closure/deletion, and affected lifecycle/version handling. Mocks establish CPU relations only.

Independent outside-candidate commands:

- `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/fixed_artifacts_acceptance_v1_build_audit.py` validates the local actual build archive and inspects manager raw records; final exit0. Output retained in the matching `_build_audit.json`.
- `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/fixed_artifacts_acceptance_v1_run_remote.py initial` and `... expanded` send the retained read-only CPU script to `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`; both exit0. JSONL/stderr retained with those suffixes. The script validates only finalized case directories, verifies deployment/environment and original file negatives, and runs public custody selection/parser/attachment. It asserts no Torch/Warp module import occurred in the reviewer process.
- With independent actual lifecycle/file facts fixed, four mutations of the real cuda_fresh record rejected: same-recipe/cross-recipe entry exchange (`runtime adoption operation mismatch`), set replaced by entry (`runtime adoption role mismatch`), and wrong scope leaf (`native operation lacks entered leaf`). No evidence file was modified.
- Other read-only commands: Git status/revision/diff/numstat/diff-check/Git blob comparisons, rg/source reads, SSH cat of pinned installed sources, and stdlib JSON/hash/process-path/cgroup-path inspection.

Initial root REVIEW_PRINCIPLES.md lookup was incorrect and was corrected to the required collab path. The initial local source-byte scratch assertion failed solely on the checked CRLF/LF difference noted above. Neither was a candidate/runtime failure. Some broad tool displays were truncated; focused reads and retained full machine-readable observations supplied the relevant facts.

## Native matrix and final verdict

Actual raw matrix root: `tileops-dev:/data2/shared/ang.gao/c4-fixed-artifact-runtime-v2`. Published build raw files remain at the deployment root above; the local selective build archive is `C:/Users/AG/新建文件夹/c4_assets/c4-fixed-artifacts-build-evidence-v2.tar.gz`.

The coordinator notified this reviewer that the original root-owned session94514 reached terminal exit0 with all ten cases accepted. Only then did I perform the final whole-matrix audit and issue this verdict. Original parent PID2083102 was independently absent. I did not rely on live/incomplete sidecars or take over that run.

Read-only command `C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/fixed_artifacts_acceptance_v1_run_remote.py terminal` completed exit0, session88436. Full output is retained in `fixed_artifacts_acceptance_v1_remote_terminal.jsonl`, with empty matching stderr. Public `validate_custody_artifact`, runtime selection, `RuntimeCustodyRecord.from_payload` and attachment validation passed for every actual finalized directory. Every request matches the frozen execution revision, actual lane/case, selected environment, device1 and unchanged120000ms operation/1000ms ACK budgets. The final results file exactly equals the ten per-case accepted rows in prescribed order, revision66240e4; SHA256 **a82fe3dbb5ae47e0210fb35128746d700101fdb46b0314a7430bd957660b11d5**. The listed supervisor/worker PIDs are all absent and every supervisor is complete/EXITED with the recorded worker returncode.

| Case | Supervisor / worker PID | Worker exit | Independently observed result |
| --- | --- | ---: | --- |
| cuda_fresh | 2083343 / 2083344 | 0 | original terminal sync and release |
| cuda_reused_deployment | 2091709 / 2091711 | 0 | original terminal sync and release |
| cuda_full_resolution | 2099118 / 2099119 | 0 | original terminal sync and release |
| warp_go2 | 2106907 / 2106910 | 0 | original terminal sync and release |
| warp_full_resolution | 2111487 / 2111489 | 0 | original terminal sync and release |
| create_after_raise | 2115739 / 2115741 | 74 | original guard3, full UNKNOWN quarantine, no sync/release |
| exec_after_raise | 2121544 / 2121555 | 74 | original guard3, full UNKNOWN quarantine, no sync/release |
| entry_adoption_raise | 2127213 / 2127214 | 74 | original guard3, full UNKNOWN quarantine, no sync/release |
| factory_after_raise | 2133648 / 2133649 | 74 | original guard6, full UNKNOWN quarantine, no sync/release |
| factory_hang | 2141175 / 2141176 | -9 | original guard6 deadline and supervisor kill |

The five positives have original input guards0..5, all42 arrays uploaded/read back and validated, matching content SHA256 `8e79f7061cf08e8778fd4fca57aac7ad6936ddf3e9ef67f0f6c074faf69c3c18` and topology SHA256 `3ef81f6fc01bfda47983cb77ddb7523d45b8b0291010956f400652ccb7561e69`. The960x640 manifest is `ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12`;1920x1080 is `4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e`. CUDA has seven normally completed setup guards then its original emergency disposal; Warp has six then disposal. All positives use exactly one successful terminal synchronization and completed release, without deadline/kill or ordinary Python atexit cleanup. This proves input lifecycle/readback, not image rendering at either resolution.

Each CUDA positive has six complete ordered artifact native pairs in guard3, real file/module/entry ownership joins, and actual factory preparation in guard6. The distinct fresh/reused workers bind the same immutable deployment while their live session and stream identities differ. Factory observations assert actual issued module and entry identity, original prepared input and context/stream, exact same-module LBVH sizing, all owners adopted, and equal before/after native state. Only primary context1 is active; the held stream is on1. Logical ambient device0 is restored while raw runtime device remains1 under the existing lazy-restoration contract; the result does not claim raw device0. For376139 primitives the real workspace query reports node_capacity752277, alignment256 and required_bytes4590335. Sample submissions remain0 and final execution readiness remainsfalse.

Create/exec/adoption failures reached the intended original guard3 boundaries, preserve their original RuntimeError, and retain partial ownership before the failure. Their worker terminal evidence records full UNKNOWN quarantine, zero emergency syncs and zero new terminal-arm attempts, release not_authorized and exit74; runtime initialization never completes. The factory exception reached actual LBVH factory return/adoption in guard6 after input validation, then follows the same quarantine/no-sync/no-release terminal path. This is supported by the producing wrappers and production retention path as well as the marker/journal/lifecycle joins; the marker's boolean alone is not treated as ownership proof.

The hang marker is `cuda_lbvh_factory.after_adoption`, original setup guard6, returned_owner_held=true. The actual supervisor records deadline_expired=true, kill_requested=true, kill_error_kind=null, observed_returncode=-9, terminal_reason=deadline_expired and complete EXITED. The durable worker record is only the preceding input-validation prefix: RUNNING/CONSTRUCTING, six completed guards, no terminal disposition, no release authorization. Public attachment validates that bounded prefix against the later supervisor facts. It must not be described as worker-reported quarantine, successful cleanup, factory completion or final readiness. Actual kill/exit plus process absence establishes process termination; it does not retroactively invent a missing worker terminal record. Raw custody SHA256 is `bdac1e3d9797945d14b85d5208eeaa1fa7761b9fe825175abc4a9c2668948e61`.

Intermediate read-only audits (`fault_prefix`, `nine_cases`) checked newly finalized cases without treating pending cases as passed; their commands exited0. All reviewer tool sessions are terminal. Final local Git checks show unchanged clean0df3e260 on implementation/c4-supervised-waits and both base/execution commits remain ancestors. No implementation changes occurred during review.

## Verdict, findings and accepted boundary

**Verdict: Approve.** No substantive blocking or non-blocking finding, or unresolved evidence gap, remains for this bounded whole fixed-artifact replacement acceptance. The archived review's final blank line is retained unchanged; it requires no implementation follow-up.

Approval binds implementation **66240e4fedc24ad8a5109690e3b1e78b450e2f62**, frozen packet **0df3e2609a7af03c789be819714f8d79f104ce63**, the selected manifest/environment identities above and the actual bounded manager/build/native evidence. It covers the integrated replacement of live worker JIT with the three deployed fixed extensions: manager-owned deployment/publication, cold selection/file rejection, ordinary captured loading and live ownership, source-backed target/stream restoration, reuse in fresh workers, actual factory/workspace preparation, both-lane Go2 input/upload/readback, failure/hang disposal and public replay, plus removal/factory integration. This is a whole review of that replacement, not a helper-only approval.

The no-GPU-init build argument retains its trusted-source/checkpoint limits; actual runtime numerical kernel/render equivalence remains unproved. No new service/build/GPU/fault campaign or speculative infrastructure is authorized by this evidence verdict. The unchanged complete report may be archived under the repository's evidence-only exception; operative changes require independent review of their delta.


## Scope limits that remain closed

No numerical optical render, sample submission, warmup, final execution readiness, one-slot graph, timeline/arena completion, seal/barrier completion, performance or benchmark result is established by artifact loading/factory/input evidence. Warp NVRTC/module-cache/ModuleExec custody, remaining parent C2d3/C2e/f work, C4c3/c4, C4d/B3 joint reclaim/drain, required whole-C4 structural refactor/review, product PREPARE, result/baseline publication and backend selection remain closed. Fixed-artifact acceptance cannot substitute for their real correctness/failure matrices.
