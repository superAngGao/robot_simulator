# C4c1b2 recording and preflight — independent code review

Initiative: q54-optical-lab-p12-4c4c1b2-recording
Stage: review
Author: codex
Version: v1
Date: 2026-09-28
Status: in_review
Related Files: optics/warp_native_render.py; tests/gpu/test_warp_render_recording.py; tests/unit/optics/test_warp_render_command_contract.py; C4c1 design requests v1/v2/v3
Owner Summary: Request changes. The recorded/preflight path accepts a borrowed CUDA stream wrapper without retaining the actual stream owner, and does not detect stream ownership drift. Independently reran the author's focused suite: 200 passed. No recorded render command was executed.

## Identity and scope

- Reviewer session: `/root/c4_recording_review_v1`, fresh independent reviewer, not an author of this candidate.
- Base: `4cda6ed5f51cfc75906f0728582e2ca530d08626`.
- HEAD: `c09d4e7cbf441d76f4aa784b112f1e5326bc8586`.
- HEAD tree: `5478ce4983b89d598b2e40bb46660e5aa84f2ddb`.
- Local frozen candidate: `C:/Users/AG/新建文件夹/p12_4_c4_render_binding`.
- Isolated test candidate: `/data3/ang.gao/robot_sim_c4_recording_review1`, branch `review/c4-recording-v1`, host `ang.gao@192.168.80.101`.
- Local and remote working trees were clean when inspected. Remote HEAD/tree and clean status were rechecked after tests and the independent reproduction. No candidate source, tests, or tracked evidence was edited by this reviewer.
- Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely. Read the accepted C4c1 v1/v2/v3 design contracts, applying v2's withdrawal of the public recorded compatibility driver and v3's lifecycle requirements. Inspected the implementation note and relevant predecessor review records as context, not substitute evidence.
- This is camera/query normalization, strict local evidence, immutable factory/recorded bindings, retained owners and preflight only. It is not C4c1c execution/lifecycle approval.

## Verdict

**Request changes.** One substantive blocking correctness/contract finding remains for this checkpoint. No finding quota or test-count threshold was used.

### F1 — P2: the purported strong stream owner can be only a borrowed handle wrapper

**Location:** `optics/warp_native_render.py:2493`–`2496` (`_render_factory_facts` stream admission), its frozen facts at approximately `2521`–`2526`, and factory/bundle revalidation through `2589` and `3084`.

**Trigger:** create a genuine Warp-owned nondefault stream, then create the supported Warp external-handle wrapper:

```python
genuine_owner = wp.Stream(device=scene.device)
borrowed = wp.Stream(device=scene.device, cuda_stream=genuine_owner.cuda_stream)
factory = WarpDirectLightRenderCommandFactory.prepare(
    prepared_input=prepared, readiness=readiness, local_scene=scene,
    device=scene.device, stream=borrowed,
)
```

The candidate admits this factory, records a complete bundle, and accepts `bundle.validate_bound()`, although `borrowed.owner is False`. No spoofed type, forged source hash, native pointer write, or baseline replacement is needed. Separately, changing an ordinary owned stream's `owner` field to `False` after sealing is also accepted by full bundle validation.

**Code/runtime evidence:** the installed Warp 1.12.0 implementation in `warp/_src/context.py:3291` initializes `owner=False`; providing `cuda_stream=` registers the external handle and never takes ownership. Only Warp-created streams set `owner=True`. Its documented external-stream contract places destruction responsibility on the caller. The external wrapper stores only `_cached_event`, `cuda_stream`, `device`, and `owner`; it has no reference to the genuine owner. The factory retains this wrapper, not the object that controls the CUDA stream lifetime. Candidate admission checks exact `wp.Stream`, target device, and positive integer handle only. The frozen facts include wrapper identity and handle, but not ownership.

**Consequence:** the sealed graph does not establish its promised strongly owned nondefault CUDA stream. A caller can release the real external owner independently while the factory/bundle still retain and validate the borrowed wrapper. The handle then lacks the lifetime guarantee required for later supervised consumption; changing ownership after sealing also escapes the advertised immutable owner-graph preflight. This finding is about the current binding/retention contract, not an assertion that this checkpoint already launches a dangling stream. No recorded execution API exists here.

**Why existing checks/tests are insufficient:** all recording fixtures create streams with `wp.Stream(device=...)`, which owns the stream. The mutation test replaces the entire stream object, so the identity check catches it, but it never varies ownership while retaining object identity and handle. The 200 passing tests therefore do not establish the strong ownership claim. A Python strong reference to an external wrapper is not a strong reference to the native owner.

**Repair direction:** in this bounded native-only contract, reject borrowed streams before module loading/recording and freeze/revalidate the genuine ownership property as part of the stream binding. Alternatively, if external streams are intentionally supported, define and retain an exact authoritative owner graph; a caller boolean or raw handle is not enough. Do not broaden into an unreviewed interop/lifecycle API merely to admit the wrapper.

**Proof required after repair:** independently repeat the original external-wrapper reproduction and require rejection before the recording leaf. Add a same-object ownership-drift test that rejects with no `Launch.launch` call, while normal owned streams remain valid. Preserve nondefault-stream and target-one/ambient-zero restoration on successful and raising preparation/recording paths. If adopting external-owner support instead, prove exact lifetime retention and rejection of a wrapper with no retained owner, within an approved design.

**Independent reproduction:** saved outside both candidates at `C:/Users/AG/新建文件夹/c4-recording-review-repro-v1.py` and `/data3/ang.gao/c4-recording-review-repro-v1.py`. It uses the repository's canonical setup helper, patches `wp.Launch.launch` to raise, retains `genuine_owner` throughout, restores the mutated owner flag, and never deliberately destroys a stream or submits recorded render work. Fresh observed output:

```text
BORROWED_STREAM_ACCEPTED False True
BORROWED_WRAPPER_FIELDS ['_cached_event', 'cuda_stream', 'device', 'owner']
STREAM_OWNERSHIP_DRIFT_ACCEPTED
NONDEFAULT_AMBIENT_STREAM_RESTORED
```

## High-risk areas inspected

- Exact production Warp prepared/readiness authority: direct production validator, registered object identities and storage, canonical readiness parsing, original local a2 allocation identities, manifest/source counts and role table. Copied prepared/readiness objects and same-content local reuploads are rejected in the executed matrix.
- R0 and R1 separation: fixed input requires the original prepared world/BVH arrays actually used by render; replay requires an in-range independent frame index. R1 render storage/evidence is not represented as proof of C3 producer order or transaction composition. That later binding remains explicitly required.
- Host camera normalization: immutable finite float32 scalar tuples detach from mutable transform arrays, positive dimensions/focal lengths/distance, exact ray count and int32 bounds, query role mask checked against the prepared scene. No per-recorded-launch NumPy conversion path was introduced.
- Descriptor/actual native parameter relationship: reconstructed command order and argument semantics; real Warp `Launch` kernel/device/dimensions; ctypes parameter owners, addresses and packed array data/shape/strides/gradient; scalar ctypes and values; module/hook identities and launch options. Inspected installed Warp `Launch` implementation as well as candidate snapshot code. Existing tests attack command replacement/reordering/deletion, scalar/array `set_param_*`, native address-array edits, dimensions, and evidence substitution without executing a bundle.
- Array owner graph: existing output snapshots retain actual arrays, Warp view/root chains and deleters. Factory retains prepared/local roots; binding retains actual world/BVH input snapshots; bundle retains command owners and kernel/module paths. The stream exception is F1 above.
- One-way seals: frozen/slotted factory, binding and bundle reject ordinary assignment/deletion and same-object `__post_init__` baseline recreation. Single-point stream-object mutation and storage/native parameter mutations are checked. This is supported-path integrity, not protection against arbitrary same-process rewriting of all baselines or runtime classes.
- Strict evidence: fixed-fixture/replay branch exclusion, camera/device-ray exclusion, digest/type/count checks, fixed ordered command plans, unknown field rejection, evidence reconstruction from actual bound owners. No evidence hash supplies lifecycle or readiness authority.
- Empty resident rays record only the two reset commands; nonempty resident rays omit raygen. Three camera profiles bind the same physical five-command plan and fixed shade variants. Actual numerical reset execution is intentionally not claimed by this recording review.
- Scope and compatibility: the six-file diff changes only the render module, two tests and three documentation/evidence records. Searches found no factory/bundle consumer in public compatibility, worker or benchmark production paths. No worker dispatcher/allowlist, gate, arena, result writer or baseline file changed. Bundle exposes no enqueue/execute/finish/terminal/completion API.

## Independently executed verification

SSH used `ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`. Commands ran in `/data3/ang.gao/robot_sim_c4_recording_review1` unless indicated, with `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim`.

1. `git status --short`, `git rev-parse HEAD`, `git rev-parse HEAD^{tree}`, `git branch --show-current`, local base resolution and full relevant diff/source inspection: exact identity above; clean candidate before and after verification.
2. Fresh focused suite:

   ```sh
   PATH=/usr/local/cuda-12.8/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
   /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -m pytest \
     tests/gpu/test_warp_render_recording.py \
     tests/unit/optics/test_warp_render_command_contract.py \
     tests/gpu/test_warp_render_buffers.py \
     tests/unit/optics/test_warp_fixed_shade_abi.py \
     tests/unit/optics/test_warp_native_kernel_source.py \
     tests/gpu/test_optical_gpu_runtime.py -q
   ```

   Result: **200 passed in 6.78s**, no skips. The target-one/ambient-zero test executed. Recording tests prohibit `wp.Launch.launch` for the render bundle; the dynamic setup helper uses the already-existing C3 preparation path before the render no-execution guard.
3. Independent scratch reproduction: `... conda run --no-capture-output -n robot_sim python /data3/ang.gao/c4-recording-review-repro-v1.py`: exit 0, outputs above. This also separately checked restoration of a real nondefault ambient stream after recording.
4. `python -m ruff check optics/warp_native_render.py tests/gpu/test_warp_render_recording.py tests/unit/optics/test_warp_render_command_contract.py`: **All checks passed!**
5. Local `git diff --check 4cda6ed HEAD`: passed.
6. Installed Warp source inspection: stream ownership constructor/destructor and actual Launch packing/retention implementation. No internet source or recalled API behavior was used as proof.

Observed runtime: Warp 1.12.0; Warp banner reports CUDA Toolkit 12.9 / Driver 13.3, with H200 NVL on `cuda:0` and A100-SXM4-40GB on `cuda:1`/`cuda:2`. The review packet designates CUDA/NVCC 12.8; this review did not independently rebuild toolchain artifacts or interpret Warp's toolkit banner as NVCC provenance. Normal shared Warp JIT cache was present and the reproduction reported a cached module load. Tests and assertions were fresh executions, not reused cached results.

Reviewer tool/path errors, not candidate failures: the initial instruction-file lookup used the workspace root before correcting to the candidate and `collab/REVIEW_PRINCIPLES.md`; one runtime-source search attempted unavailable remote `rg` and incorrect remote quoting, then used `find`/`grep`/`sed` successfully. No failed exploratory lookup is counted as verification evidence.

## Limits and remaining gates

- The author's concurrently running full non-slow suite is not independently executed evidence in this report. The scoped tests above are this reviewer's actual observations.
- No recorded render bundle was executed, including fault paths. No independent CPU-oracle comparison of a recorded bundle, sentinel reuse execution, launch contamination audit, setup guardian, supervised S/V/E wait, latch, quarantine, terminal resolution or release proof is claimed. These remain C4c1c requirements and cannot be replaced by this binder or a fake lifecycle approval.
- The primary 183 transport was supplied as unreachable; this reviewer used the owner-authorized 101 fallback and did not independently retry 183. Recovered original b1 ancestry remains unverified, as documented. This report binds the exact recovered base/HEAD/tree above, not an unavailable original history.
- No candidate files were changed. Scratch reproduction and this report are outside the candidate. Generated ordinary Python/JIT caches did not create tracked worktree changes.
- F1 must be fixed and independently re-reviewed before this recording checkpoint can be approved. Prior kernel/storage approvals remain scoped to their own revisions; this verdict neither revokes them nor expands them.
- Full C4c1, C4c1c execution, C4c2–C4c4 arena/composition, C4d/B3, product worker GPU preparation, host payload, a3b, result activation, baseline access and shipping remain closed. No performance or backend-selection claim is made.
