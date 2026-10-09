# Independent code re-review: native intersection precision v3

Initiative: Q54 / P12.4 / C4 intersection precision
Stage: bounded independent code re-review
Author: codex
Version: v3
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request/review codex v3; intersection-precision implementation-note v1, code-review v1/v2 and review-followup v2/v3; optics/warp_native_render.py; optics/_cuda_direct_light_source.py; tests/unit/optics/test_native_intersection_precision.py
Owner Summary: Approve the exact bounded source repair. The original F1 and F2 counterexamples are closed in both actual source paths without relaxing physical/shadow cutoffs or the existing comparator. Fresh targeted verification gives 115 passed. This is source-level approval for proceeding to the mandatory native compilation and verification checkpoint, not optical, performance, activation or whole-C4 acceptance.

Reviewer session: `/root/intersection_precision_code_review_v3`, a fresh independent Codex session supplied the review packet and repository, without implementation conversation. I did not author or edit the candidate, spawn subagents, install dependencies, access remote hosts, compile native GPU code or launch GPU jobs.

## Verdict and exact revision

**Approve**, solely for the bounded numerical source repair at the exact revision below. No substantive blocking or non-blocking code finding remains from this review. Required later native evidence is explicitly outside this CPU/source checkpoint and remains outstanding; this approval must not be cited as that evidence.

- Candidate: `C:/Users/AG/新建文件夹/c4_intersection_precision`.
- Branch: `implementation/c4-intersection-precision`.
- Original base: `8cce978b73901eeac42a9e2fdc7e861a23a57890`.
- Rejected v1: `c7d10114ccf9d29e688fa9f6a6387c6fc6509de8`.
- Rejected v2 / current parent: `69c53d72a605e4ebd7d6530ab46e3ef2ca93f2c3`.
- Reviewed HEAD: `a42dbe0154e7c97dcd23a1de2514636551079d74`.
- Reviewed tree: `3cac4b09a794480fd0464841553342ae93a5616f`.
- Initial and final tracked/untracked working-tree status were clean; HEAD remained unchanged. `git diff --check` from the original base passed.

The full original-base diff has two production files, one numerical test file, one captured JSON fixture and five collaboration records, with 1,270 additions and 76 removals. I inspected the complete production diff, current helpers, primary and shadow call flow, comparator, bound aggregation/refit, test harness and current tests, including the v2-to-v3 test changes. No deployment tools, runtime infrastructure, camera/geometry, product tolerance, public gate or backend-selection changes are present. Historical notes describing the old tie cap are superseded by the follow-ups and actual code, not current behavior claims.

I read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md in full, the accepted runtime-integration design/review v3 requirements and boundaries, implementation note, both previous code reviews, both follow-ups and original repros. Accepted section 9.2 requires CPU-oracle checks with unchanged existing parity tolerances and explicitly does not infer GPU arithmetic from binary32 input agreement. This bounded repair remains within that requirement; it does not establish that the full requirement has passed.

The original native archive freshly hashes to `619167158ae9da9a99112a6ce22a191fb033645dbcf2bb714a4c810ae33e8f3a`. The archived v2 report freshly hashes to `0c4b3d206f12dfff33006cbcbd8c117297c406da698484e71d702bd4238a3570`. I inspected the original native report's scope and discrepancy evidence. I did not repeat its archive-member audit, full-array analysis or Go2 materialization, and do not claim its original native execution on the current source.

## Prior findings and numerical reasoning

### F1: closed for this code checkpoint

The unchanged original `intersection_review_boundary_v1.py` was executed against this HEAD. For origin `(-float32(1e-9),0,0)`, unit x direction, near x `0.10000000149011612` and far x `0.10001000016927719`, both lanes select instance10 in both storage orders and both single/separate-leaf layouts. The true farther double distance is `0.10001000116927716`; its returned float32 distance is admitted by the unchanged tie comparator.

The shared helper accepts that hit at physical maximum10 and rejects it when the rounded far distance itself is supplied as the physical cutoff. Thus closure is not obtained by rounding away the strict range check. Primary triangles receive physical max_distance; ordering is subsequently decided by the original float32 comparator. The suite also covers a one-float step outside the tie and physically invalid distances that round onto the cutoff, plus a genuinely negative near-zero distance.

### F2: closed for this code checkpoint

The unchanged original `intersection_review_aabb_v2.py` was executed against this HEAD. Its oblique direction `(0.8446270823478699,0.5353550910949707,0)`, near x `0.1518450379371643`, far x `0.15185348689556122`, and interior hits still produce triangle distances `0.1797776073217392` and `0.17978760600090027`. The Warp box entry is now `0.17978760600090027`, rather than the old next-higher float. Both full primary source bodies select instance10/material1 for all four order/layout combinations. The original probe labels the diagnostic box value `warp_aabb_entry` in both lane records; it does not independently call the CUDA box there. CUDA traversal is genuinely executed, and the submitted stress test separately calls the actual CUDA box helper.

The repair addresses distinct sources of loss: both slab helpers widen before subtracting and directly divide in double; root and child box validity use physical max_distance; popped-node pruning compares the float32 distance difference with the existing epsilon instead of constructing a rounded `best_t + epsilon` bound. The stack entry is quantized downward if nearest float32 conversion increased it. CUDA uses `nextafterf(entry,0)`; Warp uses the positive IEEE representation decrement through `wp.cast`. Since t_min starts at zero and only increases, this conditional adjustment does not subtract from a negative value or from zero. It preserves the computed nonnegative double entry through float32 stack storage without adding a geometric tolerance.

For stored entries that bound returned hit distances, monotonic float32 subtraction makes the new popped-node condition conservative relative to the existing farther-hit tie comparison. Candidates nearer than best_t pass directly. This is the relevant pruning argument; it does not assert that every conceivable double triangle/slab computation or encoded BVH bound is an exact geometric lower bound. Universal watertightness was not promised, is not proven here and is not imposed as a new approval requirement.

The half-ULP regression exercises the additional entry-versus-triangle rounding case across both full primary paths and layouts. The deterministic 5,000-case test in each lane checks entry <= true axis distance and entry <= actual returned triangle distance with varied edges and both rounding parities. Oblique outside-tie and physical-limit cases complement it. These tests address the actual previous failures and the storage boundary, while remaining explicitly non-exhaustive.

## Other inspected invariants and maintainability

Both shared triangle helpers widen original binary32 values before the cancellation-prone arithmetic; determinant, barycentric and t calculations use double. True calculated t is tested against zero and physical max_t before returning float32. The original determinant threshold is retained, including Warp's explicit binary32-then-binary64 conversion. Barycentric comparisons, comparator epsilon, source-key ordering, normal orientation and float32 outputs are unchanged.

Primary root/child/leaf calls consistently use the physical maximum; only popped-node ordering applies best_t. Plane handling, masks, material/instance updates, sentinel initialization, near-child ordering, stack capacities and overflow counters are unchanged. Existing host/device/CUDA bound builders aggregate or copy the supplied primitive bounds; this patch adds no hidden bound padding. Passing physical limits can increase traversal work, but no concrete new stack-capacity failure was established in this review.

Shadow callers retain their existing physical max_distance through root, children and triangle helpers; downward entry storage does not relax the boolean slab validity test or triangle cutoff. Shadow policy and shared normal calculation are unchanged. The precision change therefore affects both primary and shadow math. Historical non-BVH Warp triangle bodies are not converted or approved for numerical equivalence; its existing AABB wrapper does call the changed shared interval helper, so this report does not claim every historical path is behaviorally untouched.

The test harness extracts actual function/kernel bodies and compiles actual CUDA helper/primary source as host C++, instead of retaining a second intersection formula. Numeric and stack constants derive from source. Captured cases invoke the existing scalar CPU oracle. This is useful regression coverage with bounded adapters, not new production infrastructure. The oracle's own pre-existing determinant/tie policies are not silently replaced by the native comparator, and 15 captured cases cannot prove global policy equivalence.

## Fresh commands and observed results

Environment: local PowerShell, `C:/Python314/python.exe -B`, installed NumPy/pytest and existing `C:/msys64/ucrt64/bin/g++.exe`, added to this runner's PATH. No installs. ASCII compiler temporaries avoid the previously recorded Windows non-ASCII path limitation. Pytest cache provider was disabled; candidate artifacts were not generated.

Executed through new external runner `C:/Users/AG/新建文件夹/c4_assets/intersection_review_runner_v3.py`:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=C:/Users/AG/AppData/Local/Temp/c4_intersection_review_v3 tests/unit/optics/test_native_intersection_precision.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/optics/test_native_warmup_support.py
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/intersection_review_boundary_v1.py C:/Users/AG/新建文件夹/c4_intersection_precision
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/intersection_review_aabb_v2.py C:/Users/AG/新建文件夹/c4_intersection_precision
```

Results: **115 passed in 26.20s**, exit0, no skips; both original repro commands exit0 with the correct diagnostic outcomes described above. Those repro scripts print results rather than assert every selection, so I inspected the returned instances/materials rather than treating exit0 alone as proof. The count comprises 102 numerical cases and 13 source/support checks. This independently reproduces the author's pass count, not its exact elapsed time. No current test failure was masked or repaired.

Logs are `c4_assets/intersection_review_v3_suite.log`, `intersection_review_v3_boundary.log`, and `intersection_review_v3_aabb.log`; launch/end ledger is `intersection_review_v3_processes.json`. Read-only inspection used Git status/revision/diffs/check, rg, UTF-8 reads and Get-FileHash. A few combined reads were truncated and relevant ranges were reread. A read-only attempt to locate Warp under `C:/Python314/Lib/site-packages/warp` found no such directory; it is not a native compilation attempt or code failure. The author's installed Warp1.12 builtin inspection is not claimed as my independent execution.

## Limits and mandatory continuation

Host extraction strips Warp decorators/types and models `wp.cast` with a NumPy bit view. Host C++ substitutes thread/atomic stubs and uses g++ -O2. These checks do not establish Warp JIT typing or device lowering, actual wp.cast builtin behavior in the deployment toolchain, CUDA device compilation/contraction, subnormal behavior, concurrency, allocation, GPU stack behavior or throughput. Direct full shadow traversal and all accepted rendering profiles are not executed by this host suite; shared-helper tests plus call-flow inspection are only bounded evidence.

Proceed using the existing approved deployment path/tools to source-frozen Warp compilation and newly built fixed CUDA artifacts for this exact approved source. Record actual compiler/environment/source provenance; original artifacts or earlier caches cannot substitute for current native build evidence. No new build/runtime infrastructure is approved or needed by this verdict. If device compilation or results expose a defect, repair it and independently review the changed source before treating that revision as approved.

Supervised real Go2 960/1920 readbacks and CPU diagnostics remain mandatory. Preserve geometry, cameras, lane-specific actual binary32 rays, acceptance tolerances and terminal lifecycle. Compare complete geometry/RGB arrays, hit masks, distances/positions, IDs/materials and normals; distinguish finite hit values from existing nonfinite miss sentinels. Include all six original RGB outliers and normal-only pixels1116499/1116503, and retain other residuals rather than screening by RGB or changing epsilon. The affected profile/shadow combinations and broader accepted R0/R1 correctness matrix remain required at their integration checkpoint.

Double arithmetic and additional traversal can affect GPU cost, including shadow-heavy cases. Measure with equal native-ready boundaries and actual resolved durations; host elapsed time and supervisor time are not performance evidence. This review introduces no performance threshold.

Original P1 optical discrepancy closure remains pending native verification. Numerical/performance acceptance, full C4 integration/fault matrices, systematic refactor and whole review, activation, result/baseline authority, backend selection, host delivery/R2/S and whole-C4 completion remain closed or unfinished as specified by the parent design. The original native report only approved its old diagnostic/lifecycle/copy scope; this code verdict does not expand it.

## Process hygiene and preservation

Runner Python48116, parent46548, launched and synchronously waited suite Python27216 (05:40:25.738366 to05:40:52.761079 local), boundary Python43816 (05:40:52.767128 to05:40:54.084587), and AABB Python32368 (05:40:54.091609 to05:40:55.353165), all exit0. These are recorded launch/end wall times, not OS creation-time claims. Runner/tool session55958 ended exit0. Compiler subprocesses are synchronously waited by the fixture with a60s timeout; their individual PIDs were not sampled. Short inspection shells returned terminally.

At `2026-10-10T05:41:22.3344477+08:00`, checker shell40076 queried Win32_Process for recorded PIDs and Python/g++/cc1plus/collect2 command lines referring to this review's runner, repros, test file and compiler temporary path, excluding itself. It found zero residuals. No process was killed and no unresolved owned hang was observed. No remote, worker, supervisor, GPU or separate reviewer CLI process was launched; remote/GPU hygiene is therefore not claimed by this local CPU-only round.

Only the external runner, logs/JSON, compiler/test temporaries and this report were created by this reviewer. Candidate, original repro scripts and original native evidence were preserved. Record this complete report verbatim under collab using the evidence-only exception; doing so cannot change the verdict, approved source identity, required native continuation or closed gates.
