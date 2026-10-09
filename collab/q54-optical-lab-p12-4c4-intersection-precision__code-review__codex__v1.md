# Independent code review: native intersection precision v1

Initiative: Q54 / P12.4 / C4 intersection precision
Stage: bounded independent code review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request/review codex v3; intersection-precision implementation-note codex v1; optics/warp_native_render.py; optics/_cuda_direct_light_source.py; tests/unit/optics/test_native_intersection_precision.py
Owner Summary: Request changes. The double-intermediate repair is within the existing design and preserves the storage/parameter/output ABI and numeric constants, but primary traversal still discards a candidate accepted by its unchanged float32 tie comparator. A fresh two-face reproduction fails in both actual source bodies despite all 51 submitted checks passing. Native compilation and optical/performance acceptance remain unproved.

Reviewer session: `/root/intersection_precision_code_review_v1`, fresh independent Codex session provided the task packet and repository, without implementation history. No implementation authorship, candidate edits, subagents, native GPU jobs, installs, or changes to original evidence.

Verdict: **Request changes** for this exact bounded candidate. F1 below is a blocking P1 correctness finding. This is not an approval of the implementation, native compilation/runtime behavior, numerical acceptance, performance, or whole C4. Native validation is separately mandatory after a repaired and independently reviewed source revision; passing host tests cannot supply it.

## Exact source and scope

- Candidate: `C:/Users/AG/新建文件夹/c4_intersection_precision`, branch `implementation/c4-intersection-precision`.
- Base: `8cce978b73901eeac42a9e2fdc7e861a23a57890`.
- HEAD: `c7d10114ccf9d29e688fa9f6a6387c6fc6509de8`; tree `1eadf79f4d921223fa950ac026099204ed880f83`.
- Initial and final tracked/untracked status were clean. Full base-to-HEAD diff contains only two production source files, one regression test, its JSON fixture, and the implementation note (813 additions, 58 removals). `git diff --check base HEAD` passed. Candidate remained unchanged throughout review.
- Original native revision remains `5a5a17cdd4e75c01016e1e71738aed99d9dab8fe`. The local original archive independently hashes to `619167158ae9da9a99112a6ce22a191fb033645dbcf2bb714a4c810ae33e8f3a`. The prior complete native review hashes to `467491b087a1ffb1bed59d13d0b13ccfa5ab34d39f857250235d857385d47aeb`. I read that review; I did not rerun its native lifecycle or original full-array analysis.

Read AGENTS.md, full collab/README.md and REVIEW_PRINCIPLES.md, implementation note, accepted design/review v3, affected production helpers, primary/shadow callers, unchanged comparator, AABB arithmetic, CPU scalar oracle and all new test code. The v3 design document has no Git content delta from its reviewed revision `674af14eb55e34e3a249b2ee58b3e6ede54621a9` to this candidate. Its working-file hash differs from the historical report's byte hash because working-file representation is not used as the Git content comparison.

The accepted design section 9.2 requires independent CPU rays/geometry/IDs/material/RGB/intensity, real Go2, unchanged existing C1 parity tolerances, and explicitly says binary32 input agreement does not imply GPU arithmetic agreement. Together with the shared workflow's bounded-fix exception, this permits repair of existing numerical implementation without a new renderer/design phase, provided no new tolerance, acceptance policy, geometry, camera, ABI or gate is introduced. The inspected diff stays within that kind of repair. It does not authorize calling the current discrepancy closed or changing the existing oracle/tolerances to accommodate remaining errors.

## F1 — P1: a rounded tie search bound rejects an admissible float32 hit

Locations: `optics/warp_native_render.py:922` (double distance rejection), `:1372` (primary tie bound passed as max_t); `optics/_cuda_direct_light_source.py:383` (double distance rejection), `:555` (primary tie bound). These interact with unchanged `_is_better_hit` at Warp `:777` and `better_hit` at CUDA `:276`.

The new triangle helpers calculate `t` in double, test it against a float32 `max_t`, and only then cast the accepted distance to float32. Primary traversal supplies `min(max_distance, float32(best_t + epsilon))` as that max_t. The comparator subsequently works on the returned float32 distance. A true double distance slightly above the rounded search cap can round to exactly that cap and satisfy the existing comparator, yet is discarded before reaching it. Therefore the comment promising to preserve every candidate admitted by the comparator is false at this rounding boundary.

Exact reproduction uses valid finite binary32 inputs and an axis-aligned unit ray:

- Origin `(-float32(1e-9), 0, 0)`, direction `(1, 0, 0)`, physical max distance `10`.
- Both faces have v0 `(x,-1,-1)`, e1 `(0,2,0)`, e2 `(0,0,2)`.
- Near face: `x=float32(.1)=0.10000000149011612`, source key and instance20.
- Far face: `x=float32(float32(.1)+float32(1e-5))=0.10001000016927719`, source key and instance10.
- Far double intersection distance: `0.10001000116927716`.
- Returned float32 distance when tested against physical range10: `0.10001000016927719`.
- Rounded difference from near best is `0.000009998679161071777`, inside the unchanged tie epsilon. The actual extracted Warp comparator returns true for the lower key.
- Calling either actual triangle helper with the current tie cap returns miss. Calling it with range10 returns the admissible hit.

Full actual primary kernels confirm the consequence in **both lanes**:

| Traversal fixture | Near face stored first | Far face stored first |
| --- | --- | --- |
| One leaf | instance20, wrong under existing tie comparator | instance10 |
| Separate leaves | instance20 | instance20 (near leaf is visited first) |

This is not a speculative GPU-only issue: it reproduces in the actual Python bodies and actual host-compiled CUDA C++ kernel. It is also not evidence that the CPU product oracle's narrower tie rule should be widened: the defect is failure to honor the existing native comparator that this candidate explicitly promises to preserve.

Why existing checks miss it: their synthetic distances are1 and1.000005, well inside the tie window, with zero origin offset. They never put an unrounded double t just above a rounded float32 search bound. The captured real-edge cases test the helper with max10 or a strict range below the expected distance, not this best-hit-dependent bound.

Direction for repair: distinguish strict physical validity from conservative search pruning. Preserve the existing float32 comparator while ensuring its admitted returned distances survive primary search clipping. Do not globally round t before all validity checks or indiscriminately expand the physical range: true double `t<0`, `t>max_distance`, and shadow cutoff rejection must stay strict. The primary pruning bound and physical/shadow cutoff have different jobs even though this candidate passes both through one helper parameter. The author chooses the implementation; this reviewer has not authored a fix.

Required proof: retain this exact counterexample in both source paths with both storage orders and leaf layouts; add boundary coverage for true distances just behind origin, just above the physical maximum and shadow cutoff that round onto their boundary; preserve strict rejection there. Exercise cases at and just outside the existing tie threshold, and confirm revised node/child pruning remains conservative under the unchanged physical maximum. Re-run the existing captured-edge checks and native validation below on the revised frozen source.

## Other inspected invariants and test limits

Both helpers widen the ray direction and encoded vertex/edge components before cancellation-prone products/subtractions, and keep determinant, barycentric and distance arithmetic in binary64. Inputs, arrays, kernel signatures, hit distance outputs and normal calculation remain float32. CUDA widens the existing float kBuildEps in comparison; Warp explicitly widens `wp.float32(_BUILD_EPS)`. No literal epsilon or comparison operator was relaxed. The shared helpers are used by both primary and shadow paths, so precision/performance effects extend to shadow rays even though shadow policy itself is unchanged. Historical non-BVH Warp bodies remain unchanged and are explicitly excluded from the numerical claim.

Root, node-stack, child and leaf primary bounds were all changed consistently to the intended capped tie expression. This repairs the coarse premature-pruning example but not F1. No upload, allocation, synchronization, new variant selector, runtime gate, source-key rule, material lookup, geometry or camera change exists in the diff. This review establishes no additional concrete precision defect beyond F1; it does not prove watertight intersection or all possible BVH float-rounding boundaries.

The tests execute actual extracted bodies instead of maintaining a copied intersection formula. The existing scalar CPU oracle is genuinely called. All15 fixture cases were independently matched to the supplied lane-specific arithmetic JSON vertices/edges and native-outlier JSON origins/directions. Archive and prior review hashes match the packet. I did not regenerate all Go2 assets or re-extract every raw native array in this code review; that evidence's earlier independent review remains its own bounded record.

Host extraction removes Warp decorators/annotations and substitutes NumPy scalar operations; CUDA host compilation substitutes thread and atomic stubs. Constants are injected by the harness (including stack64, while CUDA production stack is32); the shallow regression trees do not test capacity or device stack behavior. These tests do not validate Warp type lowering, generated device code, CUDA contraction/fast-math behavior, device allocation, concurrency or performance. The oracle also retains its own pre-existing determinant/tie constants; agreement on these15 cases is not proof that all oracle/native boundary semantics are identical. No constants were changed to make this review pass.

## Fresh verification and artifacts

Environment: local `C:/Python314/python.exe`, existing NumPy/pytest and `C:/msys64/ucrt64/bin/g++.exe`; no installation. Commands ran from the candidate unless noted, with `-B` and pytest `-p no:cacheprovider`.

1. `python -B -m pytest -q -p no:cacheprovider --basetemp=C:/Users/AG/AppData/Local/Temp/c4_intersection_review_v1 tests/unit/optics/test_native_intersection_precision.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/optics/test_native_warmup_support.py`: **51 passed in23.77s**, exit0, tool session10828.
2. `python -B ../c4_assets/intersection_review_boundary_v1.py .`: exact F1 reproduction above, both lanes, exit0 (script prints diagnostic failures rather than returning pytest failure). Repeated once under a logged runner to preserve output in `intersection_review_boundary_v1.log`. Script SHA-256 `bcf0fdbe887098d9f9c10ac16e25d0c5d32c949850d9caa02f9637ed0ef64a0e`.
3. `python -B ../c4_assets/intersection_review_checks_v1.py`: runs the logged boundary probe, matches15 fixture cases, hashes original archive and records child PIDs/exits in `intersection_review_checks_v1.json`.
4. `python -B ../c4_assets/intersection_review_before_runner_v1.py`: verifies both scratch production bodies exactly equal `git show base:optics/<file>`, loads those old bodies with unchanged current package dependencies for the oracle, runs the same38 numerical tests from `c4_assets/intersection_before_v1`: **28 failed,10 passed in2.25s**, exit1 as expected. Log `intersection_review_before_v2.log`. This reproduces that the submitted tests reject original errors while missing F1.

Tool/environment corrections, not candidate failures: an initial pytest run used a Chinese-character compiler temporary path and g++ failed to open it;19 CUDA fixture-dependent checks failed and32 passed in27.60s. Moving only basetemp to the ASCII path above resolved compilation. An initial direct invocation of the deliberately partial old-source scratch tree failed collection because `optics.execution` is absent there (exit2, log `intersection_review_before_v1.log`); the explicit old-body/current-unchanged-dependency runner above fixed the invocation and verified source identity before testing. Some broad reads were truncated and relevant ranges reread; initial PowerShell Chinese decoding was corrected to UTF-8; a few wrong-root reads failed before corrected paths; an unquoted Git tree suffix was misparsed by PowerShell and rerun quoted successfully. None changed candidate source or masks a code-test failure.

## Required native continuation and scope still closed

No native build/runtime verification occurred in this review. Parent reports that a separate remote candidate was transferred and offline artifact preparation is planned; those actions are not independent native proof and do not close F1. Keep source frozen during repair review; rebuild fixed CUDA artifacts and source-frozen Warp compilation from the final reviewed source using existing approved deployment tooling, with new provenance rather than stale original caches/artifacts. New double arithmetic and explicit Warp casts require actual compiler validation; host C++ success does not establish either lane's device compilation.

After repair, supervised original real Go2 replays at960/1920 must preserve the original input, camera, geometry, acceptance tolerances and terminal lifecycle. Compare complete geometry/RGB arrays, including hit masks, range/position, IDs/materials and normals, lane-specific actual binary32 rays, all six original RGB outliers and the two extra normal-only pixels1116499/1116503. Distinguish finite hit data from existing nonfinite miss sentinels. Re-run CPU diagnostics on both lanes' actual rays/encoded geometry, and investigate residuals without an RGB-only filter or new epsilon. All affected profiles/shadow modes and broader accepted R0/R1 correctness matrix remain required in their integration checkpoint. The original lifecycle approval does not approve this new source's optical behavior.

Double math can materially affect GPU throughput, especially on hardware with low FP64 rate and on shadow-heavy frames. No performance claim follows from host timings; eventual timing must use equal native-ready boundaries and valid render durations, not supervisor elapsed time or old unresolved timing fields. No new performance threshold is proposed by this review.

Original P1 optical discrepancy remains open. Full C4 integration/fault matrices, later systematic review/refactor, numerical and performance acceptance, activation, result/baseline/backend selection, host delivery/R2/S and whole-C4 completion remain outside this review and closed as before.

## Process hygiene and candidate preservation

Direct boundary probe: Python36628, parent45772, terminal exit0. Logged runner37192, parent39996; child41800 boundary exit0 and16876 initial scratch invocation exit2, both synchronously waited. JSON records wall-clock launch/end times. Corrected old-source runner5052, parent41428, terminal exit1 expected. The initial pytest driver shell45936 had parent10808 and creation time2026-10-10 05:18:02 local; both full-suite tool sessions reached terminal exits (initial1, corrected0). Individual Python/compiler PIDs for those two direct pytest calls were not separately sampled; no retroactive start-identity claim is made. Compiler subprocesses are synchronously waited by the actual test fixture with a timeout. Read-only inspection shells completed terminally.

At2026-10-10T05:20:46.8102311+08:00, scoped Win32_Process inspection found no recorded review PID or Python/g++/cc1plus/collect2 command referencing these review scripts, fixture test, or review compilation temporaries. The earlier broad command-line query matched only its own inspection shell; the final query excluded that shell. No process was killed. No remote/SSH/GPU process was launched, so no new remote GPU cleanup claim is made. The reviewer's identity is this agent session, not a separately launched CLI process.

Only scratch scripts/logs/JSON, compiler/test temporary directories and this report were written outside the frozen candidate. Initial failed toolchain temporary artifacts are retained as evidence; candidate status remains clean at c7d1011. Archive this complete report verbatim; evidence-only recording cannot alter the verdict or reopen any gate.
