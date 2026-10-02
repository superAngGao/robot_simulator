# C2c4 render-binding migration independent code re-review

Initiative: q54-optical-lab-p12-4c4c2c-render-binding-migration
Stage: review
Author: codex
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: runtime-integration design v3; render-binding-migration implementation-note v1/v2 and independent review v1; optics/cuda_direct_light.py; optics/warp_native_render.py; tests/support/sacrificial_sources.py
Owner Summary: Approve with comments for exact C2c4 candidate 4b7bf4c. All four prior P2 findings are independently closed; no blocking code issue or required native evidence gap remains for this bounded migration. Retain one non-blocking P3 cleanup-ledger follow-up. C2/C4 completion and activation remain closed.

## Identity and scope

Reviewer session: `/root/c4_c2c4_render_binding_review_v2`, fresh independent session, not the author. The coordinator supplied the review packet and subsequently asked about same-object cached-evidence mutation and cleanup-ledger coverage; conclusions below come from independent source inspection and tests.

Verdict: **Approve with comments**, exclusively for the revision and bounded scope below. No substantive blocking code findings remain.

- Overall base: `9c49722d5defef6acf7add78aee730ee0fd65e5b`.
- Previous candidate: `8b1d13c88499a8b2ddac4279e8d41008dd860869`.
- Reviewed candidate: `4b7bf4cee0d1671e5d60c16c697d37caeec6d8a3`.
- Tree: `027623ff65495caddc6283ff8da8c7aa7ac7ed85`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote detached checkout: `/data3/ang.gao/robot_sim_c2c4_4b7bf4c`, authorized fallback `ang.gao@192.168.80.101`.
- Initial local and remote trees were clean and both HEADs matched. Candidate files have not been edited. Scratch scripts and this report are outside the candidate.

Read AGENTS.md, collab/README.md, and collab/REVIEW_PRINCIPLES.md fully, accepted runtime-integration design v3, implementation notes v1/v2, and prior independent report/repros. Inspected the overall migration and repair delta, actual binding/authority/dataflow chains, source admission and relevant lifecycle transitions. Author pass counts are not independent validation evidence.

Scope is only C2c4 v13/v8 render binding, stage v3/C3 v6/receipt parameter chain, native ABI validation and repair of its regressions. This is not complete C2/C4 or runtime activation. Product PREPARE/admission, C2d/e/f, Warp arena integration, C4c3/c4, C4d/B3, a3b, results/baselines, host payloads, shipping, depth-two delivery and backend selection remain closed or outstanding. No performance result is approved.

## Static/local observations

1. F1 repair compares actual native `c_float` bytes against the descriptor's little-endian binary32 encoding at initial seal; integer ABI checks remain exact. The independent real-device probe rejects initial and post-seal wrong-sign zeros; the complete native recording matrix passes.
2. F2 repair builds render evidence at stage sealing, then returns the same frozen object. Enqueue still calls live stage stability validation. The signature now includes inline first-hit distance, role mask, plane/BVH counts, shade light count, evidence identity and existing physical tensor signatures. Full validation/storage-disjointness/scalar serialization no longer occurs via evidence retrieval. The independent timed probe and eight post-acquire drift attacks pass as detailed below.
3. F3 test-only source closure adds the four actual transitive camera/render modules, not the product allowlist. Cold schema import and negative unlisted-module checks pass locally, together with stale-cache and before-exec verification controls. The complete 135-case real sacrificial regression passes, reaching actual oracle/leaf/fault/hold states rather than early import rejection.
4. F4 changes are narrowly scoped: exactly three old provenance definitions may evolve and one new helper is allowed; all other moved AST definitions and old-import object identities remain protected. Historical gate validation now uses the actual historical revision; canonical workload comparison excludes exactly the two new catalog fields. The former three failing assertions pass locally.
5. Re-executed the independent historical AST/preimage probe against this candidate. All 20 actual v12 row preimages/hashes and all v13 preimages/hashes match, with every semantic digest changed, all old preimage keys retained, unchanged mandatory row objects/case/content fields/thresholds/activation gates. Gate and both preimage fixture files are unchanged by the repair. No live v12 fallback is added.
6. Stage -> aggregate binding -> receipt remains a one-way parameter chain. Actual role tables are checked in native adapters; descriptive parsing is not scene authority. Runner request frame/pass validation still precedes unconditional C3 timed-result rejection. Product worker/manifest/protocol/runtime source contract remain unchanged.

## Non-blocking comment C1 — P3: reconcile exact cleanup-ledger paths

Location: `benchmarks/results/q54_p12_4/cleanup_ledger.json`, entries list; runtime-integration design v3 section 10.

The ledger remains unchanged and has no entries covering `optical_worker/render_parameters.py`, `optical_worker/render_sequence_contract.py`, the four `optical_worker/data/render_sequences/v1/*.jsonl` files, or the v12/v13 migration fixtures and their comparison test. The implementation notes explicitly state durable selected-lane canonical records versus temporary comparison artifacts, with the P12.4e cleanup obligation, so the lifetime policy is not missing. However, the design requested per-slice ledger upkeep; a future exact-path cleanup audit can omit these artifacts if it reads only the ledger.

Follow-up: reconcile exact paths, both backend-selection dispositions, and the already-declared P12.4e deadline in the ledger, retaining selected-lane canonical validation and historical evidence. Validate ledger schema/source coverage without changing gates. This is non-blocking for this bounded ABI migration; it is not a claim that cleanup is complete. Earlier C2c3 artifacts are mentioned only to make the shared follow-up complete, not to reopen its accepted code scope.

## Executed local validation

Environment: `C:/Python314/python.exe`, Python 3.14 with NumPy/pytest/Ruff; no Torch/Warp/trimesh. No package or asset installation.

```text
python -B C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-static-v1.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_render_parameters.py tests/unit/benchmarks/test_optical_render_migration.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_resolved_camera_contract.py
python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_contract_extraction.py::test_single_definitions_preserve_all_bodies_and_alias_identities tests/unit/optics/test_native_contract_extraction.py::test_historical_gate_and_closed_product_source_boundary_are_preserved
git diff --check 9c49722 HEAD
python -B -m ruff check benchmarks/optical/cuda_arena.py benchmarks/optical/runner.py benchmarks/optical/schema.py optical_worker/provenance.py optical_worker/render_parameters.py optics/cuda_arena.py optics/cuda_direct_light.py optics/warp_native_render.py tests/unit/benchmarks/test_optical_render_migration.py tests/gpu/test_cuda_native_arena.py tests/gpu/test_warp_render_recording.py tests/support/sacrificial_sources.py tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_sacrificial_sources.py
```

Observed: independent 20-row proof passed; **244 passed in 7.75s**, then **6 passed in 1.30s**; diff check and Ruff clean. These CPU runs are not native-runtime proof. Local commands completed synchronously; a scoped subsequent Win32 process query found no reviewer script/test residual. The short local commands did not record a separate child PID launch ledger; remote native rounds use the explicit identity ledger below.

## Independent native validation

GPU clearance was explicitly received only after the author's separate round and residual checks completed. Its pass count is not used here. Remote environment: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, same prefix for CUDA_HOME, bin prepended to PATH, lib prepended to LD_LIBRARY_PATH, MAX_JOBS=4, PYTHONDONTWRITEBYTECODE=1, exact candidate PYTHONPATH. CUDA_VISIBLE_DEVICES in order:

```text
GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989,GPU-aa67d96f-c377-7912-8efb-2ac8f0ddcfe4,GPU-3cf5c216-4235-4739-5ed0-818737fae8d7
```

Observed Warp 1.12.0, H200 NVL logical 0 and two real A100 logical 1/2. Ordinary installed JIT/Warp caches were used (Warp native render module `73d56b1`); no stale execution output is used as proof. Primary183 transport failure was supplied in the packet; the reviewer used the authorized101 fallback and checked exact detached HEAD/clean status before execution. No dirty shared worktree was touched.

All remote invocations use:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c4-review-runner-v2.py <mode>
```

Scratch sources are `c4_assets/c4-c2c4-review-runner-v2.py` and `c4_assets/c4-c2c4-review-probe-v2.py`, copied to `/tmp/`. Fresh logs and exact process-start metadata are in `/tmp/c4-c2c4-independent-review-v2`. The runner bounds native/core rounds to 300 seconds, full sacrificial regression to 1800 seconds, and records descendant PID/parent/start identity. Pytest is fail-fast (`-x`). If the whole-round deadline expires, it first interrupts pytest so its reviewed sacrificial supervisor can close/reap children; fallback cleanup targets only exact recorded live PID/start identities. No deadline has fired in the completed rounds.

### Independent counterexample probe

Mode `probe_retry`: exit 0, all assertions passed.

- Native Warp signed-zero test uses the production canonical factory/binding/descriptors and actual `wp.Launch` packed storage. Six initial/post-seal rejection controls passed; no recorded render launch executed.
- Full CUDA validator and evidence builder were replaced with rejecting instrumentation only after complete preparation. Actual timed acquire/enqueue/finish succeeded without invoking either; resolve, release, reclaim and close completed normally.
- After acquire, changed each of max_distance, sensor_role_mask, num_planes, num_bvh_nodes, num_lights, actual same-shape tensor storage via set_, raygen positional tuple and shade positional tuple. All eight rejected through actual enqueue with zero terminal event. Since enqueue authority had already set its conservative possible-submission latch, observed arena POISONED / slot RETAINED_POISONED with transaction/device holds retained; no false assertion of an unchanged pre-acquire state or ordinary reusable slot is made. These rejected arenas were retained until the disposable probe exited.
- Built a wrong-focal camera arena and forcibly changed the same frozen cached evidence object to canonical scalar bits using object.__setattr__, without editing any baseline digest. A descriptive adapter binding could be constructed, but actual `arena.bind_c3_provenance` rejected with `CUDA arena runtime stage evidence changed`. This distinguishes descriptive/registered adapter evidence from usable arena provenance. The immutable plan digest remains a separate admission check. Arbitrary in-process mutation of all private baselines is outside the explicitly closed trusted-process contract; frozen dataclasses are not claimed to sandbox hostile Python.

The first `probe` attempt exited 1 because the reviewer scratch assertion used slot name `POISONED`, while the correct existing slot enum is `RETAINED_POISONED` (arena state is `POISONED`). F1/F2 had passed and the first drift had safely rejected. Only the scratch literal was corrected; candidate code was untouched. The first log is retained separately, not overwritten or characterized as a product failure. Exact first-run processes and GPU records were verified absent before retry.

### Native core and historical regression

Mode `core` ran:

```text
python -B -m pytest -q -x -p no:cacheprovider tests/gpu/test_cuda_native_arena.py tests/gpu/test_warp_render_recording.py tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_resolved_camera_contract.py tests/unit/optics/test_sacrificial_sources.py tests/unit/benchmarks
```

Result: **372 passed in 44.78s**, exit 0, no skips. This includes the full profile/shadow/diagnostic recording matrix, initial-bit attacks, canonical wrong-scalar and ABI permutation checks, exact authority/rebinding/storage rejection, real-device restoration, fixed-cost acquire, all former F4 assertions and source-admission controls.

### Sacrificial regression

Mode `sacrificial` ran:

```text
python -B -m pytest -q -x -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py
```

Result: **135 passed in 841.85s**, exit 0, no skips, no whole-round deadline. This is a fresh complete execution, not the v1 incomplete run or the author's run. Positive cases require actual independent CPU-oracle image/geometry results and launch counters; negative cases require the intended recorded leaf audits, partial-submit state, guarded wait rejection, setup/drain failures, parent-loss transport hold and retained ownership. Real target1/ambient0 restoration and empty/resident-ray variants are included. Passing early import failure cannot satisfy those assertions. F3's mandatory regression gap is closed for this slice, without claiming future integrated C2d/e/f acceptance.

### Legacy CUDA and control contracts

Mode `legacy` ran:

```text
python -B -m pytest -q -x -p no:cacheprovider tests/gpu/test_cuda_optical_timeline.py tests/unit/optics/test_cuda_timeline.py tests/unit/optics/test_native_optical_arena.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_render_buffer_contract.py
```

Result: **130 passed in 3.55s**, exit 0, no skips. Covers existing first-hit/device-ray and direct-light timeline use, state/ownership controls, and Warp command/buffer contracts affected by shared binding machinery. These tests do not expand the approved scope.

## Process hygiene and final state

Completed remote rounds, with full command and `/proc` start identities retained in their metadata/owned-process files:

| Round | Supervisor / parent | Child/PGID | Start epoch | Exit epoch | Result |
| --- | --- | --- | --- | --- | --- |
| probe, scratch enum mismatch | 1971280 / 1971269 | 1971301 | 1790976176.4790893 | 1790976180.6757932 | 1, scratch-only assertion |
| probe_retry | 1972700 / 1972613 | 1972721 | 1790976229.2418516 | 1790976233.7381434 | 0 |
| core | 1973253 / 1973169 | 1973274 | 1790976252.883392 | 1790976298.8619883 | 0 |
| sacrificial | 1975519 / 1975434 | 1975543 | 1790976333.644442 | 1790977175.9308858 | 0 |
| legacy | 2004571 / 2004556 | 2004592 | 1790977199.7473948 | 1790977204.2924151 | 0 |

Every round's post-child GPU query was empty and its scoped process output contained only the currently printing round supervisor. No deadline escalation, broad kill, GPU reset, dirty-worktree operation or candidate edit occurred. The intended sacrificial fault tests use their reviewed supervisors' normal scoped kill/reap behavior.

Final independent `/tmp/c4-c2c4-review-hygiene-v2.py` at epoch **1790977217.6961622** checked exact recorded PID/start identities across all rounds: 6 core identities, 1 legacy, 1 first probe, 1 retry, and **271 sacrificial identities** (pytest plus 135 supervisor/worker pairs). **Zero matching owned residuals**. GPU compute records empty. Scoped PS contained only the final hygiene checker PID2005065 and its current shell. An additional explicit PS query for all ten round-supervisor/test PIDs returned only its header (PS exit1 here means none of those PIDs exist, not a test failure).

Final remote HEAD remained `4b7bf4cee0d1671e5d60c16c697d37caeec6d8a3` with empty porcelain status. Native SSH sessions were reaped before the next round. Fresh log/metadata files were downloaded and verified present under `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-independent-review-v2/`. After download, final local HEAD/tree remained the exact values above, porcelain status was empty, and a scoped Win32_Process query found no reviewer SSH/scp/Python residual. No other session's processes were stopped.

These are scoped observations, not a claim that arbitrary unrelated processes cannot hang. Prior accepted real Go2 materialization proofs were not repeated in this ABI repair review. No mock substitutes for the real-device properties tested here. No benchmark timing/result authority, full closed-worker lifecycle integration, new Warp arena, host ownership or shipping capability was validated or enabled.

## Final verdict

**Approve with comments** for C2c4 at exact `4b7bf4cee0d1671e5d60c16c697d37caeec6d8a3`. F1-F4 are closed by source inspection and fresh independent counterexamples/regression execution. C1 is a non-blocking exact-path cleanup-ledger maintenance item; retain its disposition in follow-up. The coordinator acknowledged it for a separate CPU/documentation delta with fresh independent review; that proposed delta is not approved by this report.

Approval does not extend to subsequent edits, complete C2/C4, C2d/e/f integration, product PREPARE/submission, result/baseline generation, a3b, C4 result authority, host payloads, shipping, depth-two delivery, C4c3/c4 or C4d/B3. Every such gate remains as before.
