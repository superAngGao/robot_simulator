# C4c1b2 deterministic initialization and fixed shade ABI — independent code review

Initiative: q54-optical-lab-p12-4c4c1b2-fixed-shade
Stage: review
Author: codex
Version: v1
Date: 2026-09-28
Status: accepted
Related Files: optics/warp_native_render.py; optics/warp_execution.py; tests/unit/optics/test_warp_fixed_shade_abi.py; tests/unit/optics/test_warp_native_kernel_source.py; tests/gpu/test_optical_gpu_runtime.py; collab/q54-optical-lab-c4-completion__implementation-note__codex__v1.md
Owner Summary: Approve with comments for this exact deterministic-kernel/four-shade-ABI checkpoint only. No blocking code finding. Independently executed 163 repository tests and additional hand-derived kernel probes. Full C4c1b/C4c1/C4 and every activation gate remain unapproved/closed.

## Identity, independence, and frozen revision

- Reviewer session: `/root/c4_fixed_shade_review_v1`, fresh independent review session; this reviewer did not author the candidate.
- Base: `48bbc20328f4eb36c984dc87dc8c3d68f525ea57`.
- HEAD: `845dbb3d923b73669154494ebe84270ab7164775`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4c1b2_impl`.
- Isolated execution candidate: `/data3/ang.gao/robot_sim_c4_shade_review1`, host `ang.gao@192.168.80.101`.
- Both candidates reported empty `git status --porcelain=v1`; remote HEAD and parent were checked again after testing. No candidate file was edited by this reviewer.
- Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` in full. Read C4c1 design v1/v2/v3, and checked the applicable parent C4c v7 stage/gate/cleanup provisions and C4 v2 external-hold addendum. Lifecycle v3 and v2's public compatibility isolation control over conflicting v1 text.
- Author implementation notes and claimed 75-test result were treated as claims. The tests below were new executions, not reused author output.

## Verdict and findings

**Approve with comments**, limited to the exact revision and scope below. No substantive correctness, compatibility, storage-ABI, source-ownership, or gate-activation defect was found in the inspected delta. No blocking finding or required checkpoint evidence gap remains.

### C1 — Non-blocking: retain the exact counter boundary probes as regression coverage

- Severity: low, non-blocking test-hardening comment, not a demonstrated implementation defect.
- Location: `tests/gpu/test_optical_gpu_runtime.py:3588`, especially the aggregate invariants near lines 3693–3703.
- Trigger/risk: a future change can preserve `rays == directional + point == occluded + unoccluded` and nonnegativity while incorrectly changing node, leaf, triangle, or plane counts, or accidentally counting an untraced short point-light segment.
- Current checks: the new in-repository matrix validates aggregate relationships but does not freeze exact hand-derived traversal counts for these cases. Its successful CPU color comparison cannot establish diagnostic-counter semantics.
- Independent proof obtained now: the scratch probe below verifies exact counters for empty traversal, single plane, single-leaf triangle, masked triangle, no lights, degenerate directional/coincident point light, and point distance below shadow bias, for all four ABIs over two sentinel reuses. All passed. This closes the current review concern; it is not an outstanding failure.
- Suggested direction/disposition: bounded follow-up in C4c1c should retain equivalent hand-authored microfixtures in tracked tests, together with the already-required recorded reuse/zero-ray tests. Do not infer counts from observed candidate output.
- Proof for that follow-up: altering a traversal counter or counting the skipped short-point ray must fail an exact expected-count assertion; both shadow identities and diagnostics-off behavior should remain covered.

## High-risk implementation inspected

### Deterministic first-hit and reset entries

`optics/warp_native_render.py:919` now assigns all six geometry outputs before loading/intersecting the ray: hit 0, positive infinity range, NaN position/normal, zero instance/material. The existing hit selection, source-key tie-breaking, role checks, plane equations, triangle intersection, and BVH stack ordering remain unchanged in the diff. The final hit branch overwrites every geometry field. This also covers an empty scene and all-miss reuse; it does not depend on compatibility host defaults or reduced-profile initialization.

The reset entries at lines 1931, 1939, and 1947 each write exactly the declared scalar `[0]` values. Their signatures are two BVH counters, two shadow counters, or two shadow plus nine canonical diagnostic counters, respectively. They require the intended one-thread launch; they allocate nothing and contain no runtime diagnostic selection. Their presence does not mean an unfinished native command plan exists. Compatibility retains its existing fresh zero allocations, as explicitly scoped in the implementation note.

### Mathematics and traversal counters

Inspected `_ShadowTrace`, `_ShadowTotals`, `_is_occluded_for_ray`, `_direct_light_begin`, `_direct_light_sample`, `_direct_light_accumulate`, `_direct_light_write`, and both shared ray-shading functions, against the base implementation, including actual branch order and argument forwarding.

- Point and directional normalization, attenuation, positive cosine/attenuation gating, ambient/material multiplication, background handling, luminance coefficients, shadow origin bias, and point-light shadow limit preserve the original equations. The point distance is reused instead of redundantly recomputed; its scalar operations are equivalent.
- Misses do not dereference material/surface inputs and always write background RGB plus zero intensity. A scratch miss used an intentionally out-of-range material index and NaN position/normal without being read by the miss path.
- Shadow traversal preserves right-then-left stack pushes, triangle role filtering, existing within-leaf testing behavior after an occluder, and plane traversal only while not already occluded.
- Former global per-light counters are accumulated locally and then atomically reduced per ray. Overflow uses sum, maximum depth uses max, and all visit/test/ray classification totals use sum. Ray counting occurs only when a positive-length shadow segment is actually traced; zero-light, miss, degenerate-light and short-point branches were checked.
- No-shadow entries leave the explicitly reset counters at zero. The diagnostics-off entry has no overlay array arguments or aliases; computing local shared traversal statistics does not reintroduce forbidden diagnostic storage.

### Four fixed ABIs and compatibility

The four real kernels at lines 1644, 1689, 1743, and 1828 are thin adapters over shared device functions. Their actual argument order matches the immutable registry at line 2065. No-shadow signatures omit geometry/BVH/role/bias inputs. Primary signatures omit all nine overlay arrays. Both primary counters are present in all variants, and diagnostic arrays are the trailing canonical nine fields. Neither shade signature has a runtime shadow/diagnostic flag.

`warp_execution.py:865` selects these exact production kernel objects. Its compatibility-only argument dictionary/tuple construction remains outside a native measured enqueue. Disabled diagnostics now create no alias dictionary. Enabled arrays are distinct, passed in canonical order, retained by result resources, and appended only after profile filtering. Existing allocation, stream scoping, event/wait policy, public methods and result construction remain intact. No recorded bundle, factory, driver, or worker dispatch was introduced. Full, RGB-preview, and render-only physical output behavior was tested, not merely exposed channel names.

### Source ownership and scope/gates

The former runtime-flag shade kernel was removed, including its no-Warp fallback. Production device mathematics remain in `warp_native_render.py`; `warp_execution.py` imports the shared selector/kernel objects. CPU-only import remains deferred-error safe, verified with Warp imports deliberately denied in a fresh process.

The nine-file diff is limited to the two production modules, three test files, two notes, PROGRESS and REFLECTIONS. No gate, worker allowlist, dispatcher, native arena, baseline or result writer changed. Direct inspection confirmed `gpu_preparation_enabled = false` and `native_result_activation.result_capable = false`, and the worker still returns `GPU_PREPARATION_DISABLED_PENDING_C4C0_REVIEW`.

The new C4 completion ledger is consistent with the applicable scope: it schedules remaining checkpoints, preserves the compatibility/recorded separation, requires same-slice submission safety before later enabling submission, and keeps B3/joint activation/a3b/shipping/baselines separate. It does not claim this checkpoint completes C4c1 or authorize missing detailed policy. The external-hold addendum remains required; C4 completion must not be represented as proof of B3-dependent joint lifecycle or shipping activation.

## Independently executed verification

SSH transport used `ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes ang.gao@192.168.80.101`. Unless noted, commands ran in `/data3/ang.gao/robot_sim_c4_shade_review1` using `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim`.

1. `git status --porcelain=v1`, `git rev-parse HEAD`, `git rev-parse HEAD^`, `git diff --stat 48bbc20..HEAD`, full relevant diffs and source inspection: exact revision/parent above, clean tree before and after.
2. `python -m pytest tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/gpu/test_optical_gpu_runtime.py -q`: **75 passed in 5.61s**, no skips. This includes existing geometry/tie-break, camera, point/directional/no-light/multi-light/plane-shadow/triangle-shadow compatibility regressions plus the new 12 profile/mode cases and three resets.
3. `python -m pytest tests/unit/optics/test_warp_native_command_contract.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py -q`: **26 passed in 4.76s**.
4. `python -m pytest tests/unit/optics/test_warp_render_buffer_contract.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_warp_native_commands.py -q`: **62 passed in 3.11s**. These are storage/C3 regression evidence, not execution of the unfinished C4 render bundle.
5. `ruff check optics/warp_execution.py optics/warp_native_render.py tests/gpu/test_optical_gpu_runtime.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py`: **All checks passed**.
6. `git diff --check 48bbc20..HEAD`: passed locally and remotely.
7. `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python /tmp/c4-fixed-shade-review-probes.py`: **56 shade runs passed** with hand-derived RGB, intensity and exact counter expectations; **two empty-scene first-hit sentinel reuses passed**. All launches used a nondefault stream and were individual kernels, never recorded commands. Each reuse reset from a nonzero sentinel before shade and overwrote RGB/intensity from sentinels.
8. `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python /tmp/c4-fixed-shade-review-cpu-import.py`: **PASS fresh-process import with all Warp imports denied**, shared selector identity preserved, runtime error deferred to use.

Repository tests total **163 passed**. Independent scratch probe sources are retained outside the candidate:

- `C:/Users/AG/新建文件夹/c4-fixed-shade-review-probes.py`, also `/tmp/c4-fixed-shade-review-probes.py` on 101; SHA-256 `7040e8e6161827752c04bc42d2dbc396306eb56a15aa56b7818986a5350f7547`.
- `C:/Users/AG/新建文件夹/c4-fixed-shade-review-cpu-import.py`, also `/tmp/c4-fixed-shade-review-cpu-import.py` on 101.

Observed environment: Warp 1.12.0, Warp-reported CUDA Toolkit 12.9 / driver 13.3, driver package 610.43.02. Warp `cuda:0` was NVIDIA H200 NVL; two A100-SXM4-40GB devices were also visible. The review packet identifies NVCC 12.8 for the conda environment; this review did not independently rebuild all extension/toolchain artifacts or treat the Warp toolkit display as NVCC provenance. Normal shared JIT/extension caches were allowed. The scratch execution explicitly reported a cached `optics.warp_native_render` module load; assertions and launches were fresh executions.

One additional exploratory test command named nonexistent planned files (`tests/unit/optics/test_warp_native_render_contract.py` and `tests/unit/optics/test_output_contract.py`) and collected no tests. This was a reviewer path-selection error, not a code/toolchain failure. It was corrected to the actual existing buffer and native command test paths above; no nonexistent-file run is counted as evidence.

## Exact approval scope and limitations

Approved: the delta from the stated base to HEAD implementing deterministic BVH first-hit miss initialization, the three scalar reset kernels, four fixed shade entry ABIs with shared device math/counters, compatibility selection of those exact kernels, the associated tests and checkpoint/ledger documentation. The low-priority comment is retained as a C4c1c regression-hardening follow-up, not a prerequisite for this checkpoint.

Not approved or completed by this verdict: the remainder of C4c1b (camera binding, evidence, immutable factory and recorded bundle), C4c1c sacrificial guardian/setup/launch/drain fault matrix, full C4c1 acceptance, C4c2–C4c4, C4d, aggregate C4 completion, native measured execution, production worker GPU preparation, host payload/B3, joint C4/B3 activation, a3b/results/baselines, backend selection, shipping, push/merge/release, or losing-lane cleanup. Future recorded zero-ray plans, exhaustive repeated bundle reuse, target-1/ambient-0 lifecycle restoration, failure quarantine, arena lifetime and contamination requirements are still mandatory at their designated stages.

The reviewer did not run the entire repository nonslow suite, benchmarks or performance comparisons. After the independent checks, the coordinator reported an author full-suite result of 1906 passed, 39 skipped, 117 deselected and one existing warning, with log `/data3/ang.gao/c4-fixed-shade-v1-full.log`; this reviewer did not independently verify that log and does not count it among the 163 independently executed tests. There is no new performance claim. Focused tests did not inject stack-overflow/pathological BVH faults or exhaust every finite floating-point input. The source comparison and hand-derived boundary cases give proportional evidence for this bounded refactor, not a replacement for later aggregate gates.

Primary 183 was reported unavailable in the review packet; this review used the owner-authorized 101 fallback and did not independently re-test 183 transport. The earlier b1 content-recovery history is not re-approved as original commit ancestry. Reconciliation remains necessary when the original host is available. The present parent/HEAD identity and clean worktree checks establish exactly what was reviewed here.

Only scratch probe/report files outside the candidate were created. Test caches and temporary pytest artifacts may have been generated normally; no tracked or untracked candidate changes appeared in final status. No code was silently fixed.
