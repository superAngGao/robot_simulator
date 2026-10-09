# Independent code re-review: native intersection precision v2

Initiative: Q54 / P12.4 / C4 intersection precision
Stage: bounded independent code re-review
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request/review codex v3; intersection-precision implementation-note v1, code-review v1 and review-followup v2; optics/warp_native_render.py; optics/_cuda_direct_light_source.py; tests/unit/optics/test_native_intersection_precision.py
Owner Summary: Request changes. The exact original F1 counterexample is repaired without weakening the shared helper's strict physical cutoff. However, a new oblique-ray counterexample proves that rounded AABB entry distances still prune a hit accepted by the unchanged comparator. Both actual primary source bodies select the wrong instance/material when the faces occupy separate leaves, despite all 81 targeted checks passing. No native or later gate is approved.

Reviewer session: `/root/intersection_precision_code_review_v2`, fresh independent Codex session supplied the task packet and repository without implementation conversation. No implementation authorship, source edits, subagents, installs, remote access, native build or GPU jobs.

Verdict: **Request changes** for the exact bounded candidate below. Original F1's exact leaf-intersection reproduction is closed, but its broader conservative-search requirement is not satisfied: F2 is a blocking P1 correctness finding. This is neither code approval nor approval to proceed on the assumption that this revision has passed the code checkpoint. Native compilation and real-device validation remain mandatory after repair and fresh independent review. Optical/performance/full-C4 gates remain closed.

## Revision, evidence and isolation

- Candidate: `C:/Users/AG/新建文件夹/c4_intersection_precision`, branch `implementation/c4-intersection-precision`.
- Original base: `8cce978b73901eeac42a9e2fdc7e861a23a57890`.
- Rejected v1: `c7d10114ccf9d29e688fa9f6a6387c6fc6509de8`.
- Reviewed HEAD: `69c53d72a605e4ebd7d6530ab46e3ef2ca93f2c3`; tree `9642de5c2985ee51ded29402ae6f472e121ae21e`.
- Initial and final tracked/untracked status were clean; candidate was unchanged. Base-to-HEAD changes are two production source files, one test file, one JSON fixture and three collaboration records. The v1-to-v2 executable delta passes physical max_distance to primary triangle helpers and adds comments; tests add boundary cases and derive constants from source. Deployment tools and approved design content are unchanged. `git diff --check 8cce978 HEAD` passed.
- Archived complete code-review v1 independently hashes to `fb5ff4def3d21e5b8d709dcbdbaa68701021faaf3c40c5c12a763f870e54002a`. Its original repro independently retains SHA-256 `bcf0fdbe887098d9f9c10ac16e25d0c5d32c949850d9caa02f9637ed0ef64a0e`.
- Original native evidence archive independently hashes to `619167158ae9da9a99112a6ce22a191fb033645dbcf2bb714a4c810ae33e8f3a`; its complete native review hashes to `467491b087a1ffb1bed59d13d0b13ccfa5ab34d39f857250235d857385d47aeb`. Original native source remains the separate `5a5a17cdd4e75c01016e1e71738aed99d9dab8fe` record. I read that report but did not re-extract/replay its native arrays or claim its execution for this candidate.

Read AGENTS.md and full collab/README.md and REVIEW_PRINCIPLES.md; accepted runtime-integration design/review v3, especially section 9.2 and the scope/validation/continuation boundaries; full archived code-review v1; implementation-note v1 and followup v2. The design file has no Git content difference from its approved revision `674af14eb55e34e3a249b2ee58b3e6ede54621a9`. The earlier note's statement that primary triangles receive the tie cap is historical and is explicitly superseded by followup v2 and actual code.

Inspected both complete production diffs, actual shared triangle helpers and primary/shadow callers, unchanged comparators, AABB arithmetic, BVH bound construction/refit, scalar CPU oracle and complete current numerical test code. The existing design permits a bounded repair under unchanged ABI, input geometry/camera and numerical acceptance requirements; it does not permit widening product tolerances or treating host arithmetic as device acceptance.

## Original F1 disposition

The original unchanged `intersection_review_boundary_v1.py` was run against this exact HEAD. Both lanes now select instance10 in both storage orders and both single/separate-leaf layouts. The helper with physical range10 accepts the farther triangle; the same helper with the rounded bound treated as a physical cutoff rejects it. The comparator accepts the rounded farther hit. Thus the original exact failure is repaired by separating primary triangle validity from best-hit pruning.

Static call flow and fresh submitted tests also confirm that true double `t < 0` and `t > max_t` are still tested before float32 output conversion. Shared shadow callers still pass their original max_distance. The helper boundary tests cover just below/at/above a cutoff with the same rounded output and a tiny negative distance. This is CPU/source-level evidence; actual GPU subnormal handling and compiler lowering are not established here.

The complete promised conservative primary-search behavior remains incomplete because of F2 below. Closing the original exact example does not close all of the original review's required node/child pruning proof.

## F2 — P1: AABB entry rounding still drops a comparator-admissible hit

Locations: `optics/warp_native_render.py:1353` and `optics/_cuda_direct_light_source.py:528` (popped-node rejection using the new capped tie bound). The entry values come from Warp `_intersect_aabb_axis` at `:796` and CUDA `intersect_aabb_axis` at `:293`; root/child tests at Warp `:1341/:1414/:1430` and CUDA `:516/:588/:592` use the same insufficiently conservative cap.

The triangle helper now computes a double distance and returns its correctly rounded float32 result. The AABB helper still computes float32 `(bound - origin) * (1 / direction)`, with a separately rounded reciprocal. Its entry can be one float32 step larger than that returned triangle distance. When the far hit lies at the rounded tie cap, `node_t > cap` discards it before the unchanged comparator can accept its lower source key. This is a concrete failure of the newly promised tie-preserving traversal, not a request to replace the oracle's different pre-existing tie policy.

Exact finite binary32 reproduction, retained in `c4_assets/intersection_review_aabb_v2.py`:

- Origin `(0, 0, 0)`, direction `(0.8446270823478699, 0.5353550910949707, 0)`; this is a rounded unit direction with an oblique nonzero x component. Physical maximum10.
- Near face v0 `(0.1518450379371643, -2, -2)`, source key/instance20, material2.
- Far face v0 `(0.15185348689556122, -2, -2)`, source key/instance10, material1.
- Both faces e1 `(0, 8, 0)`, e2 `(0, 0, 8)`, normal `(-1, 0, 0)`. Ray intersections are in their interiors. Bounds are ordinary component min/max of these vertices, with no artificial shrunken box.
- Both actual triangle helpers accept near distance `0.1797776073217392` and far distance `0.17978760600090027`.
- The unchanged comparator accepts the farther lower-key hit. Rounded tie cap is `0.17978760600090027`.
- Actual extracted Warp AABB entry is `0.17978762090206146`, exactly the next binary32 value above that cap. The host-compiled CUDA primary traversal reproduces the same exclusion.

Full actual primary kernels produce:

| Lane | One leaf, near-first / far-first | Separate leaves, either storage order |
| --- | --- | --- |
| Warp Python source body | instance10, material1 / instance10, material1 | instance20, material2 |
| CUDA host-compiled source body | instance10, material1 / instance10, material1 | instance20, material2 |

Separate-leaf traversal visits the nearer leaf first and later rejects the stored farther entry. Both are well inside physical range10. No physical cutoff or shadow-policy relaxation is needed to expose or repair this issue. Inspection of `optics/host_bvh.py:90`, `optics/device_bvh.py:805` and `optics/_cuda_lbvh_source.py:316` confirms that the builders/refit aggregate or copy min/max bounds; they do not add padding that would invalidate this fixture. The probe is a small admitted traversal fixture, not a claim to have run a native Go2 builder.

Why current tests miss it: all new synthetic tie/range traversal cases use direction `(1,0,0)`, so the reciprocal is exact and they do not exercise the AABB-versus-triangle rounding discrepancy. Captured Go2 regressions call the triangle helper, not complete BVH traversal. All81 targeted tests pass while this counterexample fails the required ID/material selection.

Repair direction: make primary node/child/root search clipping conservative with respect to the actual returned float32 distance and existing comparator, including the arithmetic that computes/stores entry values. A rounded tie sum alone is not a proven conservative bound. Preserve the strict physical/negative/shadow validity checks in the triangle helpers; do not globally round validity before testing, change the comparator epsilon or introduce an unexplained geometric tolerance. The author must select and justify the implementation; this reviewer made no candidate fix.

Required proof: retain this exact oblique-ray example in both actual paths and storage orders/layouts, plus the unchanged original F1 example. Cover both entry computation and stored-node rejection, and relevant root/child pruning, with oblique directions and boundary candidates. Continue checking just-outside tie candidates and strict physical/shadow cutoffs. Demonstrate the chosen conservative-search rule over its intended arithmetic domain rather than fitting only this one ULP example. Then rerun captured-edge regressions and required native compilation/validation on the independently reviewed revision.

## Other invariants and test quality

The precision repair widens original ray/vertex/edge values before cancellation-prone subtraction/multiplication. Determinant, barycentric and distance calculations are binary64; arrays, signatures, normal calculation and returned distance remain binary32. Original determinant epsilon values and strict comparison operators remain unchanged, including Warp's explicit float32-then-float64 epsilon conversion. No new precision selector, fallback implementation, upload/allocation/synchronization mechanism, camera/material policy or gate is added. Both primary and shadow paths share the same modified helper; historical non-BVH Warp bodies remain outside the numerical claim.

The harness executes actual extracted Warp functions/kernel bodies with NumPy scalars and actual CUDA helpers/primary kernel compiled as host C++; it does not copy the intersection implementation. The oracle is the existing scalar routine. Current constant extraction removes the previous duplicated CUDA stack64 substitution and derives production stack32. This improves harness fidelity but does not test real stack capacity or GPU lowering. Decorator/type removal, thread/atomic stubs, host compiler semantics and NumPy execution still limit the evidence. No fresh native compile, JIT type validation, GPU FMA/fast-math/subnormal behavior, concurrency, allocation or throughput is proven. Direct shadow traversal was inspected but not newly executed by this CPU probe; the shared predicate tests are not full shadow acceptance.

No additional concrete blocking production defect was established beyond F2. This statement is not a proof of watertight intersections, all possible float BVH boundaries, full oracle/native policy identity or complete rendering correctness. Original six RGB and two normal-only discrepancies remain unresolved pending actual native repair verification.

## Fresh commands and results

Environment: local PowerShell, `C:/Python314/python.exe`, existing NumPy/pytest, `C:/msys64/ucrt64/bin/g++.exe` added to the invocation's PATH. No installs. All commands below ran from the candidate, except the initial scratch arithmetic search. Python used `-B`; pytest disabled its cache provider. ASCII compiler temporaries avoided the prior review's toolchain path failure.

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=C:/Users/AG/AppData/Local/Temp/c4_intersection_review_v2 tests/unit/optics/test_native_intersection_precision.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/optics/test_native_warmup_support.py`: **81 passed in24.66s**, exit0, tool session45992. This includes all68 current numerical cases and13 related source/support checks; no CUDA-host skip occurred.
2. `C:/Python314/python.exe -B ../c4_assets/intersection_review_boundary_v1.py .`: original exact F1 outcomes described above, exit0. The unchanged script prints diagnostics rather than using a failing assertion for wrong selection.
3. A deterministic NumPy scalar/vector arithmetic search (seed38451, one million finite x/d candidates) identified the above reciprocal-rounding boundary. It was discovery only; conclusions rely on the subsequent actual-source execution.
4. `C:/Python314/python.exe -B ../c4_assets/intersection_review_aabb_v2.py .`: actual-source F2 reproduction above, exit0. Like the original diagnostic it reports selections instead of asserting the expected ID; exit0 is successful probe execution, not numerical acceptance. Script SHA-256 `71220582b372ed303afb2e55ff8fd293d71b68ddf38ea05189554103464d35ee`.
5. `C:/Python314/python.exe -B ../c4_assets/intersection_review_runner_v2.py`: repeated both diagnostic scripts solely to retain complete logs and process ledger. Exit0. Artifacts: `intersection_review_boundary_v1_rereview_v2.log`, `intersection_review_aabb_v2_rereview_v2.log`, `intersection_review_runner_v2.json`. They reproduce the same closure/failure observations.
6. Read-only Git status/revision/tree/diffs, `git diff --check`, source/call-site reads, bound-construction reads and SHA-256 checks above completed. The old original38-test/28-failure and rejected-v1 51-test results were read as historical evidence and were not rerun or credited as executions on this HEAD.

Inspection/tooling corrections: initial workspace-wide discovery and a few combined reads were too broad/truncated; scoped relevant code and design ranges were reread. Initial PowerShell Chinese decoding was corrected with explicit UTF-8. Speculative BVH filenames and a PowerShell wildcard argument to rg failed before actual filenames were found. These were read-only tooling issues, not code/test failures. The current test suite and actual-source repro commands had no environment failure.

## Native continuation and closed scope

No native build or execution was launched for this revision by the reviewer. Do not claim current source optical correctness from81 passing CPU checks or from the original lifecycle-only approval. Repair F2, freeze the new revision and obtain fresh independent code review before treating the bounded code checkpoint as passed.

Once repaired/reviewed, required native continuation retains the existing approved deployment tooling and original fixed three recipes, new isolated source `/data2/shared/ang.gao/robot_sim_intersection_precision_wip`, new output `c4-intersection-cuda-artifacts-v1`, original profile SHA-256 `a9f42cc833088f308545749e4f239c246f787d9ec39096825815ae9557bb4829` and existing environment/service budgets. These are packet constraints, not remote facts independently checked by this CPU-only review. Any artifacts must identify the final reviewed source and actual compiler/environment; old source/artifacts/caches cannot stand in for new compilation proof.

Actual Warp compilation and fixed CUDA device compilation, followed by supervised real Go2 replays at960/1920, remain required. Preserve original geometry, cameras, actual lane-specific binary32 rays, acceptance tolerances and terminal lifecycle. Compare full geometry/RGB arrays, hit masks, range/position, IDs/materials and normals, distinguishing nonfinite miss sentinels from finite hits. Include all six original RGB outliers and normal-only pixels1116499/1116503; retain residuals rather than filtering by RGB or changing epsilon. Re-run CPU diagnostics on both lanes' actual rays/encoded geometry. Affected profiles/shadow modes and the broader R0/R1 correctness matrix remain obligations of their integration checkpoint.

Binary64 arithmetic, including shadow intersections, may materially affect GPU throughput. Measure using equal native-ready boundaries and actual resolved render durations; host timings, supervisor elapsed time and older unresolved duration fields provide no performance acceptance. No new performance threshold is introduced here.

The original P1 optical discrepancy, native numerical/performance acceptance, full C4 integration/fault matrices, systematic refactor/whole review, activation, result/baseline/backend selection, host delivery/R2/S and whole-C4 completion remain outside this code re-review and closed.

## Process hygiene and preservation

Direct original probe: Python46680, parent47816, terminal exit0. Direct new AABB probe: Python43536, parent41188, terminal exit0. Logged runner40892, parent45148, ran from2026-10-10T05:27:45.280256+08:00 to05:27:47.850123+08:00. It launched/waited original probe21712 (05:27:45.282122 to05:27:46.606313) and new probe35864 (05:27:46.606697 to05:27:47.849819), both exit0. JSON retains command, parentage, timestamps and script digests. These are observed launch/end times, not fabricated OS creation timestamps. Compiler subprocesses are synchronously waited by the actual fixture with a60s timeout. Individual compiler PIDs and the first pytest driver's Python PID were not separately sampled; pytest tool session45992 reached terminal exit0. Short read-only shells also completed terminally.

At2026-10-10T05:28:08.8472685+08:00, scoped Win32_Process inspection found none of those recorded review PIDs or Python/g++/cc1plus/collect2 processes whose command lines referenced these review scripts, current precision tests or review compiler temporaries. The inspection excluded its own shell. No process was killed and no unresolved owned residual/hang was observed. No remote/GPU process was launched, so this report makes no new remote health/cleanup claim. The reviewer is this agent session, not a separate CLI PID.

Only new scratch probe/runner/log/JSON files, temporary compiler/test artifacts and this report were written outside the frozen candidate. No source fix, commit, source/evidence deletion or modification of original native artifacts was made. Final candidate remained clean at69c53d7. Archive this complete report verbatim under collab through the evidence-only exception; recording it cannot change the verdict, requirements, candidate identity or closed gates.
