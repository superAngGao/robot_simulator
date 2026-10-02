# C2c4 render-binding migration independent code review

Initiative: q54-optical-lab-p12-4c4c2c-render-binding-migration
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design v3 sections 3/3.1/9; render-binding-migration implementation-note v1; optical_worker/provenance.py; optics/cuda_direct_light.py; optics/warp_native_render.py; migration v12/v13 fixtures
Owner Summary: Request changes for exact native-bit validation, full CUDA preflight entering timed enqueue, broken sacrificial source closure, and three historical regression assertions that were not migrated. The complete 20-row migration independently matches the actual historical implementation. No activation or complete C2/C4 approval is granted.

## Identity and verdict

Reviewer session: `/root/c4_c2c4_render_binding_review_v1`, fresh independent code-review session. The reviewer did not author or modify the candidate. The implementation coordinator supplied additional areas to inspect during review; the findings below were independently traced and reproduced, not adopted from its test summaries.

Verdict: **Request changes**, limited to C2c4 at the exact revision below.

- Base: `9c49722d5defef6acf7add78aee730ee0fd65e5b`.
- Candidate HEAD: `8b1d13c88499a8b2ddac4279e8d41008dd860869`.
- Candidate tree: `4a89b09adfc45deaab584fbf064c5e45dd266aa9`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Remote detached checkout: `/data3/ang.gao/robot_sim_c2c4_8b1d13c` on authorized fallback `192.168.80.101`.
- Both working trees were clean at initial inspection. Final verification is recorded below. Scratch sources/evidence are outside the candidate. The dirty common checkout and old WIP were not used or touched.
- Accepted design: `collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md` at `674af14`; predecessor CPU-only sequence approval at `c8b86e7` remains bounded to its original scope.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` fully. Inspected the actual source/test/data delta, predecessor report, accepted sections 3/3.1/9 and relevant execution/authority call chains. No author-generated pass count is used as independent approval evidence.

## Findings

### F1 — P2: new Warp canonical validation can accept initially incorrect signed-zero bits

Locations: `optics/warp_native_render.py:3094`, especially descriptor-derived encoding at `3142`; underlying initial packed scalar comparison at `3002`.

The new `validate_resolved_parameters()` calls `validate_bound()`, then constructs binary32 bits from `descriptor.arguments`. The physical scalar check in `_render_launch_snapshot()` compares `parameter.value != expected`, so `-0.0` equals `+0.0`. The sealed signature includes `bytes(parameter)` and correctly rejects subsequent byte drift, but it only snapshots whatever bytes existed during initial construction. It does not establish initial byte equality to the frozen descriptor.

Real-device reproduction used the exact production factory/binding/descriptors and native `wp.Launch` objects on H200:

1. Seal a canonical asymmetric bundle and validate the frozen record: passes.
2. Change native `background_r` parameter bytes from `00000000` to `00000080`; validation of the already-sealed original bundle rejects (positive control).
3. Construct a new `WarpDirectLightRenderCommandBundle` with the same exact binding/descriptors/commands while those wrong-sign bytes are present. Initial sealing succeeds.
4. Call `replacement.validate_resolved_parameters(record)`: succeeds although actual native bytes differ from the record. No private baseline fields were overwritten and no recorded render launch was executed.

Observed record SHA-256: `3fda8f841cb62695bc207053928a30993648b1b0266f57aa46196df689e6e678`. Repro output: `argument=background_r, expected=00000000, actual=00000080, new_bundle_validation=accepted`.

Consequence: this C2c4 validator does not fulfill the exact little-endian binary32 physical-parameter contract. This is not a claim of product execution authority or a demonstrated image error; those gates are still closed. It is a reproducible acceptance of a nonconforming native descriptor within the validator's claimed responsibility.

Why existing tests miss it: they mutate scalars by `+1.0` after sealing. Numeric inequality catches those changes, and the signature catches any post-seal byte change. They do not exercise wrong physical bits at initial sealing.

Required direction/proof: compare the actual packed scalar bytes with the exact expected representation at initial seal and canonical validation, preserving integer ABI checks. Add pre-seal positive/negative signed-zero controls and retained post-seal drift tests across relevant command variants. Do not simply trust the initial snapshot or re-encode only the Python descriptor.

### F2 — P2: full CUDA preflight and scalar-record allocation now execute on the timed enqueue path

Location: `optics/cuda_direct_light.py:645-663`, particularly `_validate_cuda_transaction_stage()` at line 651. Caller: `optics/cuda_timeline.py:723-733`.

The comment says evidence creation is untimed construction/preflight. The actual path is:

`CudaActiveArenaSample.enqueue -> CudaDirectLightRenderStage.enqueue -> transaction.enqueue_stage -> _stage_evidence(stage) -> _cuda_transaction_stage_evidence -> _validate_cuda_transaction_stage`.

The new method therefore invokes complete tensor/type/shape/disjointness validation and rebuilds scalar tuples/hex strings on every render enqueue. `_validate_render_storage_disjointness()` allocates input/output/span lists and traverses all storage pairs; `_bind_shade_arguments()` also builds a positional tuple. This is extra full preparation work inside the measurement boundary, not only existing lightweight stage-stability checking.

Independent real-device repro prepared a canonical R0 CUDA arena normally, installed a counting wrapper around the production stage validator only after preparation, then ran `active.enqueue()` and `active.finish()` inside `native_timed_submission(...)`. The wrapper observed one call with `active_timed_boundary() is not None`: `calls timed=[True]`. Actual enqueue/finish/resolve/release/reclaim/close completed normally.

Consequence: the candidate violates its stated untimed-binding/preflight boundary and adds per-sample application containers and storage traversal. No numeric performance conclusion is inferred; result gates are still closed.

Why existing checks miss it: the fixed-cost test patches arena graph-walk methods and tests `acquire`, while allocation tests forbid selected Torch buffer/Event factories. Neither catches the new render-stage full preflight and Python scalar serialization during enqueue.

Required direction/proof: prepare/freeze descriptive ABI evidence outside timing and keep only the required bounded live-state checks on enqueue. Preserve rejection of actual inline first-hit scalar drift (`max_distance`, `sensor_role_mask`, and all other consumed scalar/count fields), not merely object identity; blindly caching evidence would weaken that check. Add instrumentation forbidding full render validation/evidence materialization during actual acquire/enqueue, and regress correct execution plus pre-leaf scalar/storage drift rejection and lifecycle side effects.

### F3 — P2: new eager dependency is outside the sacrificial frozen source closure

Locations: new import in `benchmarks/optical/cuda_arena.py:16` and `benchmarks/optical/schema.py:13`; unchanged `tests/support/sacrificial_sources.py::REPOSITORY_SOURCES`.

The actual cold import chain is `benchmarks.optical.__init__ -> benchmarks.optical.cuda_arena -> optical_worker.render_parameters`. The test ledger rejects `optical_worker/render_parameters.py`, which is absent from its frozen repository closure. Thus the existing sacrificial render driver fails before reaching its intended guarded setup/render work. This is a code/source-admission regression, not missing CUDA, a flaky GPU comparison, or valid fault-injection coverage.

Independent repro installed `RepositoryImportLedger(frozen_source_manifest(Path.cwd()))` in a fresh interpreter and imported `benchmarks.optical.schema`. It exited 1 with exactly `RuntimeError: repository import outside frozen closure: optical_worker/render_parameters.py`. This verifies the actual entry chain and guard, not a hand-maintained import list inference. The broader sacrificial run also produced widespread failures and was bounded/reaped as described below.

Consequence: required C1/source/sacrificial regressions cannot reach their exercised behavior. Passing standalone CUDA/Warp recording tests does not replace that validation.

Required direction/proof: reconcile only the test harness's exact transitive source closure with the new imports, retaining before-exec byte verification and negative import controls. Do not open or expand the product worker allowlist as a workaround. Rerun the sacrificial positive CPU-oracle cases and fault/hold cases after repair; explicitly verify they reached the intended states and real leaves, rather than merely exiting through early quarantine.

### F4 — P2: historical regression assertions were left incompatible with the approved schema evolution

Locations:

- `tests/unit/optics/test_native_contract_extraction.py:41`: exact all-definition AST equality for receipts/provenance conflicts with the newly added/changed provenance logic.
- Same file at `319`: full gate bytes must still equal old `a1f6c8c`, conflicting with the intentional v13 migration.
- `tests/unit/optics/test_resolved_camera_contract.py:283`: strips only `sha256` from the case catalog then expects workload payload equality, incorrectly treating the two newly added sequence catalog fields as changed workload content.

These tests need versioned migration-aware assertions. The migration itself is authorized and independently preserves old canonical identities; reverting v13 is not the requested fix. Leaving baseline regressions red obscures future genuine changes and prevents a clean required regression verdict.

Required direction/proof: narrowly account for explicitly approved evolving definitions and catalog/gate fields while keeping unrelated moved definitions, re-export identity, canonical case/content data and closed product files protected. Retain historical assertions against the appropriate historical revision. Rerun all three tests with the migration suite and source-admission regressions. Do not remove entire invariant tests or broadly exempt the modules.

## Independently verified strengths and scope controls

1. The v12 fixture gate exactly equals `git show 9c49722:benchmarks/results/q54_p12_4/gate.json`. The independent probe extracted the actual old `canonical_row_semantics_sha256` AST, executed it, and separately captured its complete preimage assignments. Every one of the 20 historical preimages/hashes matched. Every current complete preimage matched the v13 fixture and current implementation; every semantic digest changed; all old preimage keys remain. This was not just a fixture-to-fixture comparison.
2. All 20 mandatory row objects, old case/hash/scene/replay/fixture fields, unchanged top-level gates and unchanged measurement fields survive. Only the intended schema/catalog/binding/measurement/contract changes occur. The historical same-process declaration becomes false with the new independent-worker declaration. There is no live v12 fallback.
3. Stage v3, C3 v6 and receipt records form a downstream parameter chain; the render-stage record/hash must match the aggregate binding, which must match the receipt. Non-render stages reject attached render parameters. Replay frame and bound sample metadata are checked. The runner checks request frame/pass identity before its unconditional C3 timed-result rejection.
4. The CUDA adapter derives the role mask from the exact prepared scene and checks runtime stage ABI evidence before issuing registered C3 binding identity. Existing exact-resource a2 authority checks and copied-binding rejection remain in the call chain. Parsers and copied hashes are not promoted to execution authority.
5. CUDA stage sealing now blocks ordinary `_sealed=False`, field deletion and baseline rebuilding. Actual raygen/shade positional preflight is present. Warp still offers only the untimed descriptive validator, not a new C4 arena or callable result authority.
6. Product `PREPARE_FROZEN_ROW`, product source admission, a3b/results/baselines, C4 result binding, host/shipping/depth-two/B3 remain closed. The changed files do not alter the product worker, runtime allowlist or protocol. C2/C4 completion is not claimed or approved.

## Verification environment and commands

Local CPU/static environment: `C:/Python314/python.exe`, Python 3.14, available NumPy/pytest/Ruff, no Torch/Warp/trimesh. Remote real-device environment: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, with that prefix as `CUDA_HOME`, prepended `bin` in `PATH` and `lib` in `LD_LIBRARY_PATH`, exact candidate `PYTHONPATH`, `MAX_JOBS=4`, `PYTHONDONTWRITEBYTECODE=1`.

CUDA visibility, in order:

`GPU-22a6a96e-63d9-b348-e2db-ff3e4813a989,GPU-aa67d96f-c377-7912-8efb-2ac8f0ddcfe4,GPU-3cf5c216-4235-4739-5ed0-818737fae8d7`.

Observed Warp 1.12.0, H200 NVL logical 0 and two A100 logical 1/2. Normal existing Warp/JIT caches were used (probe module cache ID `73d56b1`); no old execution outputs were used as new proof. No packages/assets were installed. The packet reported primary183 transport timeout; the reviewer used the authorized fallback and independently verified exact HEAD/tree state. No new GPU round overlapped the author: clearance was received before the reviewer's first GPU run.

SSH prefix for all remote calls:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 <command>
```

Scratch source files, all outside the candidate:

- `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-static-v1.py`.
- `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-probe-v1.py`.
- `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-runner-v1.py`.
- `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-hygiene-v1.py`.

The last three were copied to `/tmp/` with `scp`. Remote fresh logs/metadata are under `/tmp/c4-c2c4-independent-review-v1`.

### Completed local checks

```text
python -B C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-review-static-v1.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_render_parameters.py tests/unit/benchmarks/test_optical_render_migration.py
git diff --check 9c49722 HEAD
python -m ruff check benchmarks/optical/cuda_arena.py benchmarks/optical/runner.py benchmarks/optical/schema.py optical_worker/provenance.py optical_worker/render_parameters.py optics/cuda_arena.py optics/cuda_direct_light.py optics/warp_native_render.py tests/unit/benchmarks/test_optical_render_migration.py tests/gpu/test_cuda_native_arena.py tests/gpu/test_warp_render_recording.py
```

Results: actual historical 20-row proof passed; **163 passed in 1.69s**; diff check clean; Ruff all checks passed. Initial PowerShell unquoted `HEAD^{tree}` was parsed incorrectly; retry with single-quoted `'HEAD^{tree}'` produced the exact tree above. This was a diagnostic command quoting error, not candidate mutation.

### Bounded broad remote regression (incomplete, not a pass)

```text
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c4-review-runner-v1.py tests
```

The runner executed:

```text
python -B -m pytest -q -p no:cacheprovider tests/unit/benchmarks tests/unit/optics/test_render_parameters.py tests/unit/optics/test_native_contract_extraction.py tests/gpu/test_cuda_native_arena.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_sacrificial_render.py
```

It reached over 93% with many sacrificial failures before its 300-second whole-round deadline. The supervisor sent SIGTERM to its owned session/process group, reaped pytest, and reported `EXIT 124`. No completed pytest summary/count is claimed. This run is retained as regression/timeout evidence only, not real fault-matrix acceptance. No inference of a GPU hang is made. A later smaller completed suite is recorded below.

### Decisive independent real-device/cold-ledger probe

```text
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c4-review-runner-v1.py probe
```

It ran `/tmp/c4-c2c4-review-probe-v1.py` with the exact environment above. Exit 0 means all assertions of the three reproduced failures and their controls held: cold ledger rejects, initial wrong-sign native Warp zero accepted while post-seal mutation rejects, and full CUDA validation invoked inside actual timed enqueue. CUDA work completed and was resolved/reclaimed/closed; no Warp recorded render command was executed. The probe's output is not an approval assertion.

### Completed smaller native/regression round

```text
/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -u /tmp/c4-c2c4-review-runner-v1.py core

python -B -m pytest -q -p no:cacheprovider tests/gpu/test_cuda_native_arena.py tests/gpu/test_warp_render_recording.py tests/unit/optics/test_native_contract_extraction.py tests/unit/optics/test_resolved_camera_contract.py
```

Result: **169 passed, 3 failed in 36.79s**, exit 1, no skips. The three failures are precisely F4's historical assertions. Both real-device native suites passed, including their new actual ABI mutation checks and existing authority/lifecycle tests. Their passes do not negate the independent F1/F2 counterexamples or repair F3's separate sacrificial driver.

## Process hygiene, final state and limitations

The bounded runner uses a new owned session for its child, records `/proc/<pid>/stat` start identity before waiting, and only terminates that owned process group on the configured deadline. No broad name kill, GPU reset, user-file cleanup or candidate edit occurred. This report and scratch scripts are retained.

- Broad round: supervisor PID `1900770`, parent `1900764`, pytest PID/PGID `1900793`, launch epoch `1790973834.5245438`, deadline/reap epoch `1790974134.78902`. After its SSH tool session completed, independent scoped process inspection showed none of these identities or candidate test/supervisor/worker processes; GPU compute records were empty. Hygiene checker PID `1916609` was the only scoped process in that query and then exited.
- Probe round: supervisor `1916967`, parent `1916880`, child/PGID `1916991`, start epoch `1790974212.5795937`, exit epoch `1790974217.149202`, exit 0. Round-end scoped PS contained only the currently printing supervisor; GPU compute query was empty. Subsequent final disappearance is checked below.
- Smaller core round: supervisor `1918382`, parent `1918374`, child/PGID `1918403`, start epoch `1790974264.6877246`, exit epoch `1790974302.6145015`, exit 1. Round-end scoped PS contained only its printing supervisor and GPU records were empty.
- Final independent scoped OS/GPU query at host-local `04:55:26` showed only the hygiene checker (`1923904`), no candidate test/worker/supervisor or earlier reviewer runner. Exact `ps -o pid,ppid,lstart,args -p 1900770,1900793,1916967,1916991,1918382,1918403` returned only its header: every launched test/probe supervisor and child was absent. GPU compute records were empty. No unresolved owned residual remained.

These are scoped observations, not proof that no unrelated process can ever hang. The initial broad runner printed an unnecessarily broad read-only OS snapshot; only owned/candidate identities were used for cleanup decisions, and later scripts filter the reported snapshot.

Final remote/local `git status --porcelain` was empty, remote HEAD remained `8b1d13c88499a8b2ddac4279e8d41008dd860869`, and local tree remained `4a89b09adfc45deaab584fbf064c5e45dd266aa9`. All long-running SSH tool sessions were reaped. A final local `Get-CimInstance Win32_Process` query found no `ssh.exe` or review-script Python process. Fresh remote log/metadata files were copied to `C:/Users/AG/新建文件夹/c4_assets/c4-c2c4-independent-review-v1/`; they remain alongside the scratch repros and this report.

Outstanding coverage is not waived: repaired source closure must undergo the complete real sacrificial oracle/fault regression; the incomplete broad run does not prove it. This review does not rerun all Go2 source-materialization proofs already accepted in the predecessor CPU slice, nor claim C2d/e/f runtime/guardian acceptance, Warp arena behavior, complete two-lane worker execution, host ownership or backend performance.

No approval of C2c4 is granted until F1-F4 are repaired and independently re-reviewed at a new frozen revision. C2/C4 and every closed activation gate remain incomplete/closed.
