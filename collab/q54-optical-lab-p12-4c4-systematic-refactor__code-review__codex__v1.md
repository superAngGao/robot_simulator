# Execution and setup consolidation: independent code review

Initiative: Q54 / P12.4 / C4
Stage: independent read-only code review, execution-consolidation checkpoint
Author: Codex, fresh reviewer session `/root`
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: in_review
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3; systematic-refactor gap-review codex v1; systematic-refactor implementation-note codex v2
Owner Summary: Request changes for the new supervised resolver's incomplete execution-context binding validation. Event initialization, relocation, frame binding and warmup refactoring have substantial positive CPU evidence. An empty LBVH launch-hook discrepancy is a non-blocking comment. No final graph, native-runtime or whole-C4 acceptance is issued.

## Identity, revision and verdict

**Verdict: Request changes.** F1 below is a P2 blocking finding for this bounded candidate. C1 is a P3 non-blocking comment.

This is a fresh independent reviewer invocation, identified by the supplied session role `/root` and local review record `execution_consolidation_review_v1`. No implementation conversation was supplied, no candidate changes were authored here, and no agents were spawned. A separate provider/CLI session UUID was not exposed to this reviewer; none is invented. Author notes and counts were treated as claims, not approval evidence.

- Candidate: `C:/Users/AG/新建文件夹/c4_final_graph`
- Branch, independently observed: `implementation/c4-final-graph`
- Exact base: `c9d537778659db8ac79ba509bdf2f73d971a8c23`
- Exact reviewed HEAD: `1934ac7673348c12ccb76a6e0b54ef883b17324b`
- HEAD tree: `223db191194bf2ac1fe63198bbbf5c887e3e3ec3`
- Initial, intermediate and final `git status --porcelain=v1`: empty; HEAD and branch remained unchanged.
- Full base..HEAD scope: 19 files, 944 insertions, 492 deletions. Included the previously uncommitted groundwork now committed in this candidate.
- Report and reviewer-created repros/logs are outside the candidate, in `C:/Users/AG/新建文件夹/c4_assets/execution_consolidation_review_v1/`.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely. Read the supplied design, particularly sections 5-8, the complete gap review, and the implementation note. Revisited UTF-8 text and smaller chunks where initial output was garbled or truncated. Reviewed every changed source/test/document and the catalog delta, using AST/blob comparisons to verify the large relocation and catalog precisely.

## F1 — P2, blocking: supervised resolution commits despite mutable device/stream binding drift

**Location:** `optics/cuda_arena.py:747-767`, especially the pre-wait scope check, `require_adopted`, and post-wait `_assert_configuration_stable`; supporting existing signature at `optics/cuda_arena.py:829-849`. The mutable context is defined at `optics/cuda_timeline.py:379-447`.

**Trigger:** Finish a valid controlled sample, then replace its execution context's stream with another already-retained stream before guarded resolve, or change `context.device` / `context.stream` during the terminal wait. These fields are ordinarily writable on the real `CudaOpticalExecutionContext`; the attack does not need to override the active sample's immutability or modify a frozen completion object.

The new resolver validates the context's current device ordinal once and verifies adoption of its current stream once. Its post-wait check freezes only `id(transaction._context)`, not that context's device, stream identity or native handle. It never invokes the context identity validator or checks the sealed template's context signature. Consequently a mutation within the same context object passes both active-sample signature checks. Adoption of a stream establishes custody, not that it is the stream sealed into this transaction.

**Observed consequence:** The physical wait and normal DISARMED exchange succeed, then the existing lease is resolved even though its execution binding has changed. In the stronger repro, all three attacks produced:

`error=None; sample/lease=PAYLOAD_READY; completion_resolution_count=1; transaction_hold_count=0; device_result_hold_count=1; final guard operation=normal_disarmed`.

This violates the checkpoint's binding-failure retention claim and the exact immutable target/stream binding required by design sections 6-8 and M5. The repro does not demonstrate a wrong CUDA event completing or actual device memory corruption; it demonstrates an invalid binding being admitted to resolution and its transaction hold being dropped. The new supervised path relies on an existing weak active-sample signature; this finding does not claim that context mutability itself was first introduced by this commit.

**Independent proof:** `reviewer_controlled_resolution.py` uses the real `CudaOpticalExecutionContext`, controlled template, prepared sample, single-use prepared capability, minted live execution authority, acquire activation, enqueue/finish transitions, C4 lease and guard controller. Only device/event/stage operations and C3 inputs are CPU fixtures; no real CUDA is used. The control passes. Stream replacement before wait, stream replacement during wait and device replacement during wait each fail the expected-rejection assertion because resolution succeeds. No active-sample state or authority registry is patched to force the post-acquire flow. See `reviewer-controlled-resolution.txt` (3 failed, 1 passed).

The earlier smaller probe using the candidate's resolution fixtures independently reproduced the two during-wait cases (`reviewer-probes.txt`).

**Why current tests miss it:** The new `binding_drift` case replaces `transaction._completion_event`, whose identity is included in `_configuration_signature`. The wrong-target test changes device before the one pre-wait ordinal check. Neither changes the contents of the retained context after validation. The new fixture also uses a namespace context without the production context validator.

**Repair direction:** In the single existing resolution commit path, validate the exact context/device/stream/native handle sealed for this transaction, terminal completion and live authority before waiting and again before committing after DISARMED. Reuse the existing template/context/authority facts; do not accept a fresh caller label or make a second resolver state machine. A changed binding must preserve the unresolved generation and holds and use the existing failure disposition. Keep the standalone wrapper and ACK-loss behavior intact.

**Required repair proof:** Run the unchanged failing repros against the repaired candidate, plus the successful control, foreign/unadopted stream, same-device alternate stream, device/handle drift during wait and ACK handling, and wait-success/lost-DISARMED cases. Assert zero resolution/reclaim and retained transaction/output ownership on rejection. Native target1/ambient0 testing remains a separate mandatory later check.

## C1 — P3, non-blocking: empty LBVH stage still reports a native-launch boundary

**Location:** `optics/cuda_lbvh.py:528-532`, with the empty-entry branch at `:550-552`.

`enqueue` unconditionally invokes `transaction.before_native_launch()` before `_launch_bound_entry()`. A valid zero-primitive LBVH has `_extension_entry=None`, so the latter performs no native call. The analogous scene-update change correctly checks each triangle/plane entry first.

Independent probe `test_lbvh_launch_hook_matches_actual_entry[False]` observes `['launch']` where no native entry exists; the populated control observes `['launch', 'entry']`. This is a conservative extra hook, not evidence of unsafe execution. The current R0 warmup driver only runs render stages, so this does not block that working path. It does contradict the broader claimed per-native-entry/empty-phase behavior and would inflate setup launch bookkeeping when the same adapter serves empty R1 acceleration stages.

The new scene test covers zero/one/two scene entries, but there is no matching empty-LBVH check. Place the hook at the actual nonempty extension-entry boundary while preserving phase/diagnostic order and the shared launch implementation. Add empty/populated and diagnostics on/off coverage. See `reviewer_probe_tests.py` and its log. Retain this comment for disposition; it is not a claim of R1 acceptance.

## Positive findings and inspected flow

- **Shared event initialization:** Guarded setup, standalone CUDA arena and direct-light compatibility helpers call `_prepare_native_events`. It creates the terminal owner first, adopts each returned event before the next fallible operation, records each diagnostic pair, then records and waits the terminal event last on the same explicit stream. CUDA unbound events are accepted only until first record; guarded metadata validation rejects foreign scope before wait. No diagnostic event is created in the off mode. R0 has three pairs and R1/R2 five. The diagnostic return mappings preserve ordering.
- **Event failure matrix:** Independent tests faulted every create, returned-owner adoption and record position, plus the wait, for 0/3/5 pairs in both lanes: 120 cases passed. They assert original exception identity, no subsequent leaf, rooted custody of every returned event, restored CPU-double ambient device/stream, full quarantine and zero successful normal completion/emergency synchronization. Existing tests separately exercise foreign events and actual standalone/compatibility wrappers. Real runtime restoration is not established by these doubles.
- **Launch ordering:** Each triangle and plane entry has its own pre-launch hook; an empty scene does not call it. Raygen/first-hit/shade each call the hook. The common timeline still checks authority and exact active-stage order, and the setup subclass declares submission independently of diagnostic recording. Existing warmup tests execute the real CUDA render stage with native doubles in both diagnostic modes, including launch/record/wait failures. C1 is the remaining empty-acceleration discrepancy.
- **Guard and hold ordering:** The resolver physically waits while the transaction/result holds remain live, completes the normal handshake, and only then calls the existing lease transition. Wait failure or lost DISARMED leaves zero resolution/reclaim, poisons the generation and retains the transaction hold; guard failure quarantines without an emergency retry. Successful resolution alone preserves the device-result hold. The new post-wait signature check catches event-field replacement, but is incomplete as described in F1.
- **Frame binder:** The extracted binder retains canonical binary32 scalar decoding, camera metadata, immutable R0 world/BVH storage borrowing, Warp recorded-command/module checks and CUDA ABI validation. Provisional frames remain indices 0..9 and readiness remains warmup0. Existing tests cover three profiles, shadows on/off, diagnostics on/off, rebound factory/module/context/device and returned-stage drift. This is preparatory reuse, not a final graph consumer.
- **Warmup bookkeeping:** Named fields replace the positional list without replacing identity registries. Issuance is reserved before fallible work; transactions are retained; active frame/ticket/operation are cleared in `finally`; completion credit occurs after the exact completed ticket check; the session registration permanently retires on success/failure. Copied authority/transaction/evidence and repeat-entry checks pass. The frame count is shared between construction and validation; the final-frame audit still uses literal `index == 9`, so the code is not entirely free of count-dependent literals. The frozen required count remains ten.
- **C3 relocation/provenance:** All ten moved function definitions remain present. Nine AST bodies are byte-structure equivalent; the only changed body is the intended camera selector expression in `build_cuda_c3_binding_from_arena_target`. All ten compatibility aliases are the exact production objects. Exact prepared array checks, replay storage identity checks, scalar ABI validation and live C3 registration remain in the production path; the old module contains aliases, not a copied implementation.
- **Selectors/storage:** Plan schema v2 includes `render_sample_index` in the digest. Keys are `(transform_frame_index, render_sample_index)` through seal, selection and validation; bool/float/negative indices reject. Existing builders retain their legacy `None` camera-selector behavior. Replay selection remains in replay evidence and the camera index affects bound render parameters. There is no new final multi-camera builder in this candidate, and the new selector unit test alone does not prove such a graph.
- **Acquire cost:** ASTs of `CudaNativeOpticalArena.acquire`, prepared fast validation, active activation, authority minting and the live-authority constructor are unchanged. Selector validation and new plan fields are untimed preparation work. The new launch hook adds fixed per-entry authority checks during enqueue. No new frame/resource traversal in acquire was found. No performance measurement or 1/120/600-frame runtime operation-count result is claimed here; those remain required for the later final graph path.
- **Source closure/gates:** Independently read exact LF Git blobs for all 88 catalog entries, checked their SHA256s and ran the unchanged source policy. Catalog SHA256 is `ad4801eb4eb0b511e0eb711c66234c382a009c53cd383557df665e012a312094`. The adapter is CUDA-only. Product PREPARE remains fail-closed in `worker.py`; no final barrier/admission or worker resolver dispatch was added. The historical extraction-test allowance refers to pre-existing input-precision bodies, not a new source modification in this diff.

## Commands and observed results

Environment: Windows PowerShell, local `C:/Python314/python.exe`; Python 3.14.0, pytest 9.1.1, NumPy 2.5.2. Review execution used `-B` and `PYTHONDONTWRITEBYTECODE=1`; pytest cache provider was disabled and each temporary base was under scratch. No installation, GPU command, SSH, Git mutation, candidate edit or process cleanup was performed.

Read-only inspection used `Get-Content -Encoding UTF8`, `rg`, `git status --porcelain=v1`, `git branch --show-current`, `git rev-parse HEAD HEAD^{tree}`, and `git diff --stat/--numstat/--check c9d5377..1934ac7`, plus file-specific full diffs and `git show` blob reads. `git diff --check` passed. Two exploratory reads used nonexistent test paths (`tests/conftest.py`, `test_cuda_arena_authority.py`); discovery corrected these. They are inspection-command errors, not candidate failures.

Executed commands, from the candidate directory (all paths below are relative to scratch where indicated):

1. `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=C:/Users/AG/新建文件夹/c4_assets/execution_consolidation_review_v1/pytest-temp-1` followed by these `tests/unit/optics/` files: `test_native_render_preparation.py`, `test_native_render_warmup.py`, `test_native_warmup_support.py`, `test_cuda_timeline.py`, `test_cuda_arena_resolution.py`, `test_cuda_arena_contract.py`, `test_native_optical_arena.py`, `test_guarded_lifecycle.py`, `test_native_contract_extraction.py`, `test_cuda_extension_recipes.py`.
   **372 passed in 120.78s**, exit 0; `targeted-tests.txt`.
2. The same Python/pytest flags with `--basetemp=.../pytest-temp-policy`, `tests/unit/optics/test_native_source_policy.py` and `tests/unit/optics/test_native_sources.py::test_literal_lane_catalogs_have_no_losing_optical_modules`.
   **170 passed in 2.60s**, exit 0; `reviewer-policy-tests.txt`.
3. `C:/Python314/python.exe -B .../reviewer_source_audit.py`.
   **88/88 exact LF Git source hashes and AST-policy validations passed; all ten moved aliases are identical**, exit 0; `reviewer-source-audit.txt`.
4. Python/pytest with `-q -p no:cacheprovider --basetemp=.../pytest-temp-probes .../reviewer_probe_tests.py`.
   **121 passed, 3 failed in 7.92s**, exit 1. The three expected-contract failures are two F1 cases and the empty-LBVH C1 case. The 120 event-boundary cases and populated-LBVH control pass; `reviewer-probes.txt`.
5. Python/pytest with `-q -s -p no:cacheprovider --basetemp=.../pytest-temp-controlled .../reviewer_controlled_resolution.py`.
   **1 passed, 3 failed in 2.01s**, exit 1: the complete controlled-resolution success control passes, all three F1 negative controls fail because invalid resolution succeeds; `reviewer-controlled-resolution.txt` includes actual hold/state/guard observations.
6. `C:/Python314/python.exe -B .../reviewer_acquire_audit.py`.
   **All five inspected acquire-path ASTs unchanged**, exit 0; `reviewer-acquire-audit.txt`.

Results above are this review's new executions, not the author's overlapping suite counts. The deliberately failing scratch assertions are candidate behavioral findings, not environment/toolchain failures. No aggregate pass count is used as approval evidence.

## Limits, process hygiene and closed scope

No GPU/native image, real target1/ambient0, device lifetime, actual CUDA event semantics, hang watchdog, numerical correctness or performance evidence was generated. Real Go2 assets and POSIX checkout acceptance were not run and are not represented as passing through mocks. Most of `test_native_sources.py` uses a scratch Git init/add/commit release fixture; it was not invoked under this review's no-Git-mutation constraint. Exact committed LF blobs were checked independently instead. This proves catalog/source-policy consistency, not the full cold worker import campaign or native runtime closure.

The reviewer-owned long pytest process was observed as PID 42152, parent PID 13320, creation time 2026-10-10 09:48:47 local, with the exact `pytest-temp-1` command. Tool session 24519 returned exit 0. A subsequent scoped `Win32_Process` query for that PID and its direct children returned no rows. Other reviewer commands returned synchronously with the exit statuses above; no reviewer tool session remains running. The guarded lifecycle suite internally checks and reaps its CPU subprocesses; successful child PIDs were captured by pytest rather than printed in this quiet log, so this report does not claim a complete per-child OS launch ledger. A pre-existing Python `run.py` process in the shared scratch directory was observed and was not treated as reviewer-owned or touched. No GPU process query or broad cleanup was performed. No reviewer-owned unresolved hang was observed.

The candidate remained frozen and clean. Only scratch repros, logs and this report were written. Existing parent artifacts were preserved.

This verdict concerns execution/event/setup consolidation and the committed groundwork only. It does not reject the checkpoint merely for excluding later integration, and it does not accept that later work. Final R_setup/R_final origin and graph/arena construction, immutable coverage, acknowledged final target barrier, the Warp arena, R1 composition, common integrated worker resolution/token lifecycle, post-reclaim close/ACK/observed exit and mandatory native verification remain open. Whole C4/C2d/C2e/C2f acceptance, product dispatch/activation, later R2/S/host delivery gates, results/baselines/a3b, backend selection and shipping remain closed. Recording this report unchanged cannot open them.
