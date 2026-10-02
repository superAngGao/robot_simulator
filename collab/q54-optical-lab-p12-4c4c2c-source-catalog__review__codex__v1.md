# C2c6 source catalog independent code review v1

Initiative: q54-optical-lab-p12-4c4c2c-source-catalog
Stage: review
Author: codex
Version: v1
Date: 2026-10-03
Status: in_review
Related Files: runtime-integration design codex v3 section 4; native_sources.py; native_source_policy.py; source catalog v1; candidate tests; cleanup ledger; implementation note v1
Owner Summary: Request changes. The source catalog and CPU import ledger passed the executed checks, including an independent real-Go2 scratch probe, but both committed mandatory Go2 tests fail before materialization because they omit actual asset-preflight evidence.

Reviewer session: `/root/c4_c2c6_source_catalog_review_v1` (fresh independent session; did not author or edit candidate).

Verdict: **Request changes**.

Base: `73b4172caeb155158acdba2a3e4f3d98799fdcd6`.
Reviewed HEAD: `f8e9c4b015fcb78e464f9a62d32e97a97e39a31c`.
Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
Remote repository: `/data3/ang.gao/robot_sim_c2c6_v1` on `ang.gao@192.168.80.101`.
Local and remote tracked/untracked Git status were clean before and after this review. Both remained at the exact reviewed HEAD. The seven-file diff matches the review packet. Scratch material and this report are outside the candidate, in `C:/Users/AG/新建文件夹/c4_assets/c2c6_review_v1`.

## Blocking finding

### F1 — P2: committed real-Go2 tests omit required preflight evidence

Location: `tests/unit/optics/test_native_sources.py:138`, inside `test_real_go2_cpu_materialization_has_no_unlisted_imports`.

Trigger: configure the required real asset path through `ROBOT_SIM_TEST_GO2_MODEL_DIR`, then run the new source-catalog tests on the POSIX release checkout. The generated cold-process script calls:

```python
materialize_render_source(case, go2_model_dir=Path(asset))
```

The unchanged production materializer requires both the directory and actual `asset_evidence`. `optical_worker/workloads.py:65` raises:

```text
ValueError: canonical Go2 scene requires preflighted model_dir and asset evidence
```

Consequence: both lane variants fail before importing `optics.mjcf_visual`, constructing either Go2 scene, or evaluating their actual import ledger. The candidate's mandatory remote Go2 acceptance is not executable as submitted. This is a test-driver defect, not an observed defect in the source inventory or a missing-asset/environment failure.

Why existing checks miss it: without the environment variable, both tests skip; the synthetic materializers require no external-asset evidence. A passing local run with these skips cannot exercise the erroneous call. The reviewer independently reproduced both failures, without relying on the author's later report of the same issue.

Repair direction: inside the cold script, call the production `preflight_case_asset` for each case against the actual asset checkout (for example, `repo_root=asset.parent, model_dir=asset`), then pass its returned evidence as `asset_evidence` to `materialize_render_source`. Keep the strict asset preflight and CPU-only/import-ledger assertions; do not bypass the guard, fabricate evidence, or convert failure to a skip.

Required proof after repair: freeze the revised candidate and independently rerun the committed tests in the real POSIX checkout with the real Go2 path, both lanes and both resolutions, no Go2/POSIX skips. Preserve the rest of the source-policy and C0 protocol regression checks. The scratch success below is diagnostic evidence, not a repair of the committed tests or approval of this frozen candidate.

No additional substantive production-code finding was established in this bounded review.

## Verification performed

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` fully. Read the accepted design's source-admission section 4 and surrounding scope/gates; inspected the candidate implementation note as claims rather than approval evidence. Inspected all seven changed files, the canonical codec/caller boundaries, CPU materializer/preflight call chain, eager physics/sensing imports, and selected native component imports statically.

### Independent exact remote focused suite

Remote Python: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, version `3.12.13`.

```sh
cd /data3/ang.gao/robot_sim_c2c6_v1
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 \
  /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -q \
  tests/unit/optics/test_native_sources.py \
  tests/unit/optics/test_native_source_policy.py \
  tests/unit/benchmarks/test_closed_worker_protocol.py
```

Observed: **216 passed, 2 failed, 0 skipped in 22.73 seconds**, exit 1. Both failures are F1. The actual-POSIX-checkout tests passed, as did source capture in both lanes, synthetic actual CPU import closure, catalog/revision/path/hash mutation tests, snapshot revalidation, actual unlisted/foreign-root import rejection, loaded-module metadata drift checks, policy tests and C0 protocol tests.

The prior C2c5 C1 write/escape mutations now match the four-space function line and parse successfully before rejection; those executed tests pass. Existing protocol/lifecycle regex source tests pass, while rebindings, changed source, alternate path and builtin compile substitutions reject. The implementation only permits the exact original two whole-source-pinned releases; the old C0 source files and v1 policy were not changed by this diff.

### Independent real-Go2 diagnostic outside the candidate

Scratch source: `c4_assets/c2c6_review_v1/probe.py`. Its bytes were sent to the remote Python interpreter through SSH stdin; no script or modified test was installed in the candidate. The probe runs isolated `-B -I -c` children, one per lane, against the actual frozen POSIX repository rather than a normalized packaging fixture.

Reproduction transport from PowerShell:

```powershell
$probeText = Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4_assets/c2c6_review_v1/probe.py
$probeBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($probeText))
ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes -o ConnectTimeout=10 ang.gao@192.168.80.101 "echo $probeBase64 | base64 -d | /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -I -"
```

Observed, exit 0:

- Independently read each source from `git show f8e9c4b015fcb78e464f9a62d32e97a97e39a31c:<path>` and matched SHA-256 to every one of the **71 distinct catalog entries**.
- Real production asset preflight followed by materialization succeeded for both Go2 resolutions in each lane.
- `go2_static_960x640_v1` packed-scene digest: `745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f`.
- `go2_static_1920x1080_v1` packed-scene digest: `57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923`.
- Both actual `sys.modules` audits passed with **51 repository source files** per lane, including `optics/mjcf_visual.py` and the eager physics/sensing initializers and dependencies.
- Neither cold child had `torch`, `warp`, `benchmarks`, `tests`, or `examples` in `sys.modules`.

This independently supports the current CPU materializer inventory and narrows F1 to the committed test call; it is not guarded native-import or GPU-execution validation.

### Static and format checks

Executed locally:

```sh
git diff --check 73b4172 f8e9c4b
python -m ruff check optical_worker/native_sources.py optical_worker/native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_policy.py
```

Both passed. No local pytest run or author test count is being presented as reviewer runtime proof.

Checked high-risk properties: literal ordered 65/70 lane inventories and 71-file union; package initializer coverage; exact lowercase full revision; Git-object versus working-catalog comparison; canonical schema and codec limits unchanged; source bytes retained and revalidated; symlinked path components rejected; snapshot full equality rather than digest alone on replay; actual module dictionary enumeration and expected name/file/spec-origin/root/lane identity; selected-lane optical separation; descriptive dataclasses grant no execution authority. The source-policy hash and catalog are release data, not a claim that an arbitrary new commit is owner-approved. No pre-exec/earlier-bytecode-consumption guarantee was inferred.

## Process and round hygiene

The direct pytest launch was wrapped in a parent which printed PID/start identity and waited for its actual exit:

- Remote review parent PID `2098384`.
- Pytest PID `2098385`, PPID `2098384`, `/proc` starttime `83752962`; observed exit **1** after the completed suite.
- Its SSH tool session `32888` completed with exit **1**. The local SSH PID for that short launch was not captured before it exited; this is a provenance limitation, not a claim of a full local launch ledger.
- Diagnostic remote Python PID `2101753`, PPID `2101727`, starttime `83765313`.
- Diagnostic Warp child PID `2101826`, PPID `2101753`, starttime `83765347`, observed exit **0**.
- Diagnostic CUDA child PID `2102796`, PPID `2101753`, starttime `83767931`, observed exit **0**.
- Diagnostic local SSH PID `23104`, parent `21324`, creation time reported by Windows as `2026-10-03 06:58:33`; tool session `16028` completed exit **0**.

Final remote `ps -p 2098384,2098385,2101727,2101753,2101826,2102796 -o pid,ppid,lstart,args` showed no remaining processes. A scoped full process-list check for the review checkout and its pytest fixture `/tmp/pytest-of-ang.gao/pytest-102/native-source-release0` returned only the diagnostic shell/grep executing that check, not a worker/test residual. Local PID `23104` was absent. The focused suite's internal short-lived subprocess PIDs were not individually logged; their completed subprocess calls and the scoped residual scan are the available evidence.

`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no compute-process rows. No GPU work, GPU imports, worker/supervisor launch tests, kills, resets, installs, fetches or candidate edits were performed by this reviewer. The primary host was not retried by this reviewer; the review packet's already-verified primary transport timeout and authorized fallback were used. No unresolved owned residual was observed. This is scoped hygiene evidence, not a universal no-hang guarantee.

## Scope and remaining gates

The verdict binds only the seven-file C2c6 component-source catalog/snapshot/post-import-ledger candidate at the exact HEAD above. It does not approve the whole integration design's runtime implementation or any later gate. F1 prevents approval of this candidate until repaired and independently re-reviewed.

There is no detected change to the old C0 allowlist, v1 protocol, preparation gate, dispatcher or production activation. Current component source snapshots and a post-import audit are not a pre-execution loader, not proof of earlier bytecode consumption, and not a hostile-Python sandbox. Future C2d/e component additions require reviewed catalog updates and actual source-consumption/import-boundary integration. GPU-capable modules were inspected as source only; no guarded native-import/launch, GPU target isolation, guardian custody or final full-worker closure was tested or claimed here.

C2d/e/f, full C4, B3/host delivery, shipping, GPU preparation, activation, a3b, results and baseline generation remain closed. Cleanup still requires P12.4e collapse to the selected production closure or removal if unused. No new performance threshold, alternative architecture, or execution authority is approved.
