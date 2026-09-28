# C4c1b2 recording and preflight — independent code re-review

Initiative: q54-optical-lab-p12-4c4c1b2-recording
Stage: review
Author: codex
Version: v2
Date: 2026-09-28
Status: accepted
Related Files: optics/warp_native_render.py; tests/gpu/test_warp_render_recording.py; tests/unit/optics/test_warp_render_command_contract.py; C4c1 design requests v1/v2/v3; recording review v1
Owner Summary: Approve the exact recording/preflight checkpoint at 266fc8210020402e5a370b5a19fb279acf354140. Prior F1 is independently closed. Fresh focused execution: 203 passed. No recorded render command was executed; C4c1c and all later/product gates remain closed.

## Identity and scope

- Reviewer: `/root/c4_recording_review_v2`, a fresh independent Codex review session, not the implementation author. The handoff contained requirements, candidate identity, prior report/reproduction and repair summary, not the implementation conversation.
- Whole-checkpoint base: `4cda6ed5f51cfc75906f0728582e2ca530d08626`.
- Prior rejected HEAD: `c09d4e7cbf441d76f4aa784b112f1e5326bc8586`.
- Reviewed HEAD: `266fc8210020402e5a370b5a19fb279acf354140`.
- Reviewed tree: `82335a56e3eb318917ff494cf5474b848e3764cf`.
- Local frozen checkout: `C:/Users/AG/新建文件夹/p12_4_c4_render_binding`.
- Independent runtime checkout: `/data3/ang.gao/robot_sim_c4_recording_review2`, branch `review/c4-recording-v2`, host `ang.gao@192.168.80.101`.
- Local and remote HEAD/tree agree. Both working trees were clean before and after review. No candidate source, test, design or tracked evidence was edited by this reviewer.
- Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely. Read C4c1 design v1/v2/v3, applying v2's withdrawal of the public recorded compatibility driver and v3's later supervised execution requirements. Prior report and author notes were treated as claims to verify.
- Scope: normalized camera/query values, strict local evidence, immutable factory/binding/bundle, exact recorded Warp parameter/owner bindings, untimed recording and preflight. This is not recorded execution, lifecycle closure, full C4c1, numerical recorded-render correctness, arena composition or product approval.

## Verdict and findings

**Approve**, bound only to the exact HEAD/tree and scope above. No substantive blocking finding or required evidence gap remains for this recording/preflight checkpoint. No new non-blocking finding is raised. The conclusion is based on code inspection and fresh independent tests, not the author's pass count.

### Prior F1 — closed

The relevant repair is `optics/warp_native_render.py:2497`–`2498` and the frozen ownership fact at line 2530. `_render_factory_facts()` now requires an exact target-device Warp stream with a positive nondefault handle and `owner is True`. This check runs in factory construction before `prepare()` enters its scoped `wp.load_module`, and in every factory validation. Binding validation calls factory validation; bundle validation calls binding validation; `_seal()` validates before any recording leaf. Ownership is also included in the original frozen facts.

This is a check of Warp's actual native resource-ownership property, not a new caller-supplied evidence flag. Independently inspected installed Warp 1.12.0 `warp/_src/context.py`: `Stream.__new__` initializes `owner=False`; the external `cuda_stream=` constructor registers the handle without taking ownership; the ordinary native creation branch sets `owner=True`; the destructor destroys an owned stream and only unregisters a borrowed handle. The factory/bundle graph strongly retains that exact owning stream object. External-stream interop is not admitted.

The prior reproduction was freshly executed unchanged. It hardcodes the **v1** checkout; its actual HEAD was independently checked as `c09d4e7...`, clean. It still printed:

```text
BORROWED_STREAM_ACCEPTED False True
BORROWED_WRAPPER_FIELDS ['_cached_event', 'cuda_stream', 'device', 'owner']
STREAM_OWNERSHIP_DRIFT_ACCEPTED
NONDEFAULT_AMBIENT_STREAM_RESTORED
```

Those outputs establish the original counterexample on the rejected revision, not evidence against v2. A separate independent v2 scratch probe explicitly imports the v2 checkout and prints its module path. It repeats the genuine-owner/external-wrapper construction and same-object drift, then expands the affected checks:

- borrowed external wrapper rejected before module load or command recording, with both leaves guarded to fail if entered;
- `owner=False`, `0`, `1`, `None` and `'true'` rejected through preparation, factory validation, binding validation, bundle validation and resealing, proving exact `True`, not truthiness;
- original ownership restored in `finally` before normal resource cleanup; restored valid bundles pass preflight again;
- successful and raising preparation restore the ambient device and both real nondefault per-device streams, target `cuda:1` / ambient `cuda:0`;
- successful and raising recording preserve the same context invariants;
- dropping caller setup/buffer references and collecting garbage leaves the exact genuine stream owner retained by the bundle, still valid;
- `wp.Launch.launch` is forbidden throughout the probe, and genuine external ownership is retained throughout; no dangling-stream submission or forced native destruction is attempted.

Fresh v2 output:

```text
CANDIDATE_MODULE /data3/ang.gao/robot_sim_c4_recording_review2/optics/warp_native_render.py
BORROWED_REJECTED_BEFORE_LOAD_AND_RECORD
OWNER_DRIFT_REJECTED_ALL_ENTRIES False
OWNER_DRIFT_REJECTED_ALL_ENTRIES 0
OWNER_DRIFT_REJECTED_ALL_ENTRIES 1
OWNER_DRIFT_REJECTED_ALL_ENTRIES None
OWNER_DRIFT_REJECTED_ALL_ENTRIES 'true'
PREPARE_SUCCESS_AND_FAILURE_RESTORE_BOTH_STREAMS_AND_DEVICE
RECORD_SUCCESS_AND_FAILURE_RESTORE_BOTH_STREAMS_AND_DEVICE
BUNDLE_RETAINS_GENUINE_STREAM_OWNER_AFTER_CALLER_REFERENCES_DROP
PASS_NO_RECORDED_RENDER_EXECUTION
```

Reproduction sources remain outside the candidate:

- Original: `C:/Users/AG/新建文件夹/c4-recording-review-repro-v1.py`, `/data3/ang.gao/c4-recording-review-repro-v1.py`.
- Independent closure/attacks: `C:/Users/AG/新建文件夹/c4-recording-review-repro-v2.py`, `/data3/ang.gao/c4-recording-review-repro-v2.py`.

## Inspected high-risk areas and affected invariants

- Inspected the complete v1-to-v2 delta: three added implementation lines, 53 added GPU-test lines, the preserved prior report and the v2 implementation note. No gate, worker, compatibility, arena, result or baseline implementation changes are present. The whole-checkpoint diff contains only the render module, its two new test files, progress/reflection and recording evidence documents.
- Traced admission and revalidation through `prepare`, `__post_init__`, `_seal`, binding `_validate`, and bundle `validate_bound`. The new ownership check cannot be bypassed by choosing resident-ray sealing instead of camera sealing; both use the same `_seal` path. The added tuple element does not disturb the existing last-two storage/kernel fact checks.
- Reviewed host query/camera normalization, finite f32 conversion, detached immutable transforms, positive camera dimensions and focal lengths, exact ray count, int32/int64 limits and exact scene role masks. Tests cover host-source mutation and malformed/boundary values.
- Reviewed strict evidence fields and mutually exclusive fixed/replay and camera/resident-ray branches, expected command ordering/counts and independent reconstruction from live bound objects. Evidence is not treated as readiness, completion, lease or provenance authority by itself.
- Reviewed exact production prepared/readiness validation, original local allocation checks, original R0 world/BVH identities, R1 frame bounds and explicit separation from later C3 producer/composition authority.
- Reviewed owner snapshots, live layout/ctype checks, strong native allocation roots, descriptor reconstruction, actual packed native Launch parameters and address arrays, scalar types/values, launch dimensions/settings, command/module/hooks identities and one-way baseline seals. Executed tests exercise same-content reuploads, copied readiness/prepared objects, command replacement/reordering/deletion, scalar/array parameter edits, native address-array drift, dimensions, evidence and kernel substitution.
- Normal native owning streams continue to record across all four shade/diagnostic variants and all three output profiles. Resident rays omit raygen; zero rays retain both reset commands. This verifies recording plans, not numerical reset execution.
- Searches for `WarpDirectLightRenderCommandFactory`/`Bundle` in `optics`, `optical_worker` and `benchmarks` found no consumer outside the defining render module. The bundle has no enqueue/execute/finish/completion/terminal API. Gate files were unchanged; inspected GPU-preparation and native-result activation flags remain false.
- The integrity boundary remains supported-path validation in a trusted process. This review does not claim protection from arbitrary same-process rewriting of Warp runtime internals, native handles, code or every frozen baseline, nor does it authorize concurrent mutation during future submission.

## Independently executed commands and results

SSH: `ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`. Runtime commands used `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim` in `/data3/ang.gao/robot_sim_c4_recording_review2` unless otherwise stated.

1. `git status --short`, `git rev-parse HEAD`, `git rev-parse HEAD^{tree}`, `git branch --show-current`, whole-checkpoint diff/stat and v1-to-v2 diff: exact identities above; candidate trees clean, rechecked after tests/probes.
2. Fresh scoped suite:

   ```sh
   /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -m pytest tests/gpu/test_warp_render_recording.py tests/unit/optics/test_warp_render_command_contract.py tests/gpu/test_warp_render_buffers.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/gpu/test_optical_gpu_runtime.py -q
   ```

   **203 passed in 6.98s**, no skips. The real multi-GPU tests ran. Existing compatibility/kernel runtime tests may execute their already-existing non-bundle paths. Recording tests forbid recorded render execution; the replay fixture's existing C3 preparation occurs before its render guard, as explicitly scoped.
3. `... python /data3/ang.gao/c4-recording-review-repro-v1.py`: exit 0, original counterexample reproduced on its hardcoded rejected v1 checkout, identity checked separately.
4. `... python /data3/ang.gao/c4-recording-review-repro-v2.py`: exit 0, closure and attacks above, verified v2 module path, no recorded render execution.
5. `... python -m ruff check optics/warp_native_render.py tests/gpu/test_warp_render_recording.py tests/unit/optics/test_warp_render_command_contract.py`: **All checks passed!**
6. `... python -m ruff format --check` on the same three files: **3 files already formatted**.
7. `git diff --check 4cda6ed5f51cfc75906f0728582e2ca530d08626 HEAD`: passed locally and remotely.
8. Installed Warp stream source inspection via `sed -n 3280,3435p /data3/ang.gao/anaconda3/envs/robot_sim/lib/python3.12/site-packages/warp/_src/context.py`: native ownership semantics above confirmed directly.
9. `... nvcc --version`: CUDA compilation tools **12.8, V12.8.93**.

Observed runtime: Warp **1.12.0**, Python environment path **3.12**, Warp banner CUDA Toolkit **12.9** / Driver **13.3**, H200 NVL on `cuda:0`, A100-SXM4-40GB on `cuda:1` and `cuda:2`. The banner is not conflated with the separately observed NVCC 12.8 version. Ordinary shared Warp JIT cache was present; probes reported cached module loads. All tests/assertions above were freshly executed, not reused output. No source recompilation or clean-cache numerical equivalence claim is made.

Reviewer tooling limitations/errors, not candidate failures: the initial broad workspace listing was truncated and provided no review evidence; the first combined instruction read used the wrong decoding for Chinese and was re-read in UTF-8 completely; a guessed Python 3.11 installed-source path did not exist and was corrected by locating the actual Python 3.12 path. These were resolved without editing the candidate. The original reproduction's hardcoded v1 path is explicitly distinguished from the v2 probe.

## Limits, residual risks and closed gates

- The author-reported full non-slow regression on v1 is not independent v2 evidence and is not used as proof here. A full repository suite was not repeated for this three-line ownership repair; focused real-device tests, affected-invariant attacks, source/diff checks and static checks are the proportional validation executed.
- No recorded render bundle execution occurred, on either success or failure paths. This review proves neither recorded numerical CPU-oracle parity nor sentinel-reuse/reset execution, contamination-free enqueue, supervised S/V/E waits, setup/partial-recording guardian adoption, possible-submission latch, quarantine, terminal resolution or release. These remain C4c1c work and requirements.
- Future consumers must perform the required preflight and introduce their separately reviewed execution/lifecycle authority. A recorded bundle alone is not such authority. Replay render evidence still does not prove ordered C3 production or transaction composition.
- The primary 183 SSH transport was supplied as unavailable in the packet; this reviewer used the owner-authorized 101 fallback and did not independently retry 183. Original unrecovered primary-host ancestry is not certified. This approval binds only the recorded recovered base/HEAD/tree.
- Scratch probes and this report are outside the frozen candidate. Ordinary Python/JIT caches caused no tracked or untracked candidate worktree changes reported by Git. No output result/baseline artifact or performance measurement was produced.
- Full C4c1, C4c1c, C4c2–C4c4, C4d/B3, product-worker GPU preparation, host payload, a3b, result activation, baseline access, shipping and backend selection remain closed. Prior approvals remain bound to their own scopes; this approval does not expand them.
