# C2c6 source catalog independent code review v2

Initiative: q54-optical-lab-p12-4c4c2c-source-catalog
Stage: review
Author: codex
Version: v2
Date: 2026-10-03
Status: accepted
Related Files: runtime-integration design codex v3 section 4; source-catalog implementation note v1; source-catalog review v1 and followup v1; optical_worker/native_sources.py; optical_worker/native_source_policy.py; source catalog v1; source tests; cleanup ledger
Owner Summary: Approve the exact bounded C2c6 candidate. The previous P2 test-driver finding is closed by the committed repair and independent real-asset execution. Required regression observed 218 passed and zero skips. Runtime activation and all later gates remain closed.

Reviewer session: `/root/c4_c2c6_source_catalog_review_v2`, a fresh independent code-review session that did not author or edit the candidate.

Verdict: **Approve**.

Original base: `73b4172caeb155158acdba2a3e4f3d98799fdcd6`.
Repair base: `f8e9c4b015fcb78e464f9a62d32e97a97e39a31c`.
Reviewed HEAD: `3a0b5e5e5a770ef7902f314279f77743f4375d2d`.

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
Remote repository: `/data3/ang.gao/robot_sim_c2c6_v2` on `ang.gao@192.168.80.101`, detached HEAD.
Both repositories were clean at the exact reviewed HEAD before and after review. No candidate edits, commits, resets, installs or runtime activation were performed. The review packet reported the primary host's transport timeout; this reviewer used the owner-authorized fallback and did not independently retry the primary host.

## Findings and prior finding closure

No unresolved substantive finding or mandatory evidence gap was established for this bounded candidate.

Prior F1, P2, is closed. At `tests/unit/optics/test_native_sources.py:136`, the cold Go2 test now imports production `preflight_case_asset`, obtains evidence separately for each of the two real cases, and passes the returned evidence to `materialize_render_source`. Both lane parameters and both resolutions remain present. No fabricated evidence, weakened production guard, substituted asset or new skip was introduced.

The repair-base-to-HEAD diff contains only this test-driver change plus the verbatim previous review and followup document. There is no production/source-catalog/policy/gate delta from the previous candidate. The archived previous report's SHA-256 independently matches `79b7624e528770f761602a3c07c988fdd1dad45c3198e613a37cd9c828af56e2`.

Independent proof closes the actual counterexample, not merely the call signature: the committed mandatory tests passed with real assets; a separate actual-checkout probe also exercised both cases in both lanes. That probe confirmed the original missing-evidence invocation still raises `ValueError: canonical Go2 scene requires preflighted model_dir and asset evidence`, then confirmed the repaired production preflight/materializer chain succeeds.

## Inspected code and invariants

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully. Read accepted design section 4 and surrounding scope/gate material, implementation note v1, the full independent review v1 and review followup v1. Notes and author test counts were treated as claims, not approval evidence.

Inspected the original nine-file base-to-HEAD change set and the three-file repair delta; source capture, catalog validation, snapshot replay, actual module audit, source-policy exception and negative tests; the real Go2 preflight/capture/materializer call chain; and unchanged C0 gate assertions. High-risk properties checked were:

- Literal ordered lane inventories, package initializer coverage and optical lane separation: 65 Warp entries, 70 CUDA entries and 71 distinct source files. Independent probe compared all 71 Git-object source bytes to the catalog SHA-256 and actual remote disk bytes.
- Full lowercase revision identity, actual HEAD agreement, Git-object versus working-catalog equality, strict schema/ordered inventory/policy binding, unchanged canonical-codec limits, exact source bytes without production newline normalization, and rejection of symlinked source/catalog components.
- Snapshot retention of verified bytes and full equality during replay, not acceptance based only on a copied digest. Descriptive dataclasses do not grant execution authority.
- Actual `sys.modules` enumeration, repository name/file/spec/root/lane checks, unlisted real imports, foreign checkout imports and metadata mutation rejection. The trust boundary is an audit of a closed trusted process, not hostile-Python enforcement.
- Exact whole-source-pinned regex exception in only `optical_worker/protocol.py` and `optical_worker/lifecycle.py`; source mutation, rebind, builtin compile substitution and relocation remain rejected. Prior C1 write/escape mutations parse before rejection, avoiding syntax-only false positives.
- No old C0 allowlist, protocol source, worker dispatcher or GPU preparation gate change. Cleanup ledger retains selected-closure collapse/removal at P12.4e rather than creating permanent unused dual-lane infrastructure.

## Independent required POSIX regression

Environment: Linux, `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13, pytest 9.1.1. The interpreter path resolved to the supplied robot_sim environment; no packages were installed.

Exact test command, launched from the frozen remote repository:

```sh
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2 \
  /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -m pytest -v -rA \
  -p no:cacheprovider --basetemp /tmp/c2c6-review-v2-a27fqezx/pytest \
  tests/unit/optics/test_native_sources.py \
  tests/unit/optics/test_native_source_policy.py \
  tests/unit/benchmarks/test_closed_worker_protocol.py
```

Observed: **218 passed, 0 failed, 0 skipped in 74.31 seconds**, pytest actual exit **0**. The two Go2 lane tests and both actual-POSIX-checkout tests are explicitly recorded as PASSED. Each Go2 lane test executes both `go2_static_960x640_v1` and `go2_static_1920x1080_v1` through production preflight/materialization and actual module auditing. No Go2 or POSIX skip was used as acceptance evidence.

The suite also passed source capture in both lanes, cold CPU-only imports, synthetic materialization import closure, source/catalog/revision mutation rejection, symlink rejection, snapshot replay, actual unlisted/foreign-root import rejection, loaded metadata drift, source policy negative controls and unchanged C0 protocol/gate regressions.

The committed Go2 tests use their explicit LF release fixture. The separate POSIX tests verify the actual checkout. To avoid conflating those properties, the additional probe below also performs real Go2 materialization and module auditing directly against the actual checkout.

## Independent actual checkout probe

Scratch source: `C:/Users/AG/新建文件夹/c4_assets/c2c6_review_v2/probe.py`. Its bytes were sent through SSH stdin to remote `python -B -I -`; it launched separate isolated `-B -I -c` children for Warp and CUDA. No probe was installed in the candidate.

Observed, probe exit **0**:

- All 71 catalog source digests matched `git show 3a0b5e5e5a770ef7902f314279f77743f4375d2d:<path>` and the actual disk bytes.
- Both real Go2 cases passed production asset preflight and materialization in each lane, without fixture packaging or normalization.
- Asset source revision was `da76818e269b82289eba39808e2fb91d679d6994`; aggregate SHA-256 was `479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`.
- The 960x640 packed-scene digest was `745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f`; the 1920x1080 digest was `57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923`, in both lanes.
- Each actual repository module audit passed with **51 source files**, including `optics/mjcf_visual.py`, physics and sensing initializers/dependencies. Neither process had `torch`, `warp`, `benchmarks`, `tests` or `examples` loaded.
- All four original missing-evidence calls remained rejected before the repaired call was executed.

Both remote scripts (`run_review.py` and `probe.py`) and their stdout/stderr logs are retained under the reviewer scratch directory. Transport used the following pattern for each script:

```powershell
$reviewSource = Get-Content -Raw -Encoding UTF8 -LiteralPath <scratch-script-path>
$reviewEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($reviewSource))
ssh -i C:/Users/AG/.ssh/tileops_ed25519 -o BatchMode=yes -o ConnectTimeout=10 ang.gao@192.168.80.101 "echo $reviewEncoded | base64 -d | /data3/ang.gao/anaconda3/envs/robot_sim/bin/python -B -I -"
```

## Static checks

Executed locally with `C:/Python314/python.exe`:

```sh
git diff --check 73b4172caeb155158acdba2a3e4f3d98799fdcd6 HEAD
C:/Python314/python.exe -m ruff check optical_worker/native_sources.py optical_worker/native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_policy.py
git diff --name-only f8e9c4b015fcb78e464f9a62d32e97a97e39a31c HEAD -- optical_worker optics physics sensing tools benchmarks/results
```

Diff/ruff checks passed; the repair comparison over production and gate paths returned no changed path. No local skipped pytest run or author's broader test count is substituted for the independent remote execution above.

## Process and round hygiene

The required-test wrapper printed process identities and waited for actual pytest exit. Recorded remote `/proc` starttime values:

- Test wrapper PID `2117014`, parent `2116991`, start `83824969`.
- Pytest PID `2117017`, parent `2117014`, start `83824973`, actual exit **0**.
- Local test SSH PID `8040`, start `2026-10-03 07:08:30` (Windows local time). Tool session `13237` completed with exit 0. The PowerShell process object's ExitCode printed blank after completion, and the formatted CIM output did not retain its local parent PID; those two local transport fields are unavailable. Remote pytest exit was explicitly observed, not inferred from that local output.
- Probe PID `2120236`, parent `2120213`, start `83837140`.
- Warp probe child PID `2120308`, parent `2120236`, start `83837171`, actual exit **0**.
- CUDA probe child PID `2121262`, parent `2120236`, start `83839778`, actual exit **0**.
- Local probe SSH PID `19004`, parent `34800`, creation/start `2026-10-03 07:10:31`; SSH actual exit **0**, tool session `53796` exit **0**.

After the required suite, the wrapper's scoped repository/scratch process scan found no residual; pytest PID was absent; GPU compute query was empty. A subsequent remote check found wrapper, parent and pytest PIDs absent; local SSH PID 8040 was absent before starting the extra probe.

After the probe, `ps -p 2116991,2117014,2117017,2120213,2120236,2120308,2121262 -o pid,ppid,lstart,args` returned only the header. The corrected scoped scan `ps -eo pid,ppid,lstart,args | grep -e robot_sim_c2c6_v2 -e c2c6-review-v2-a27fqezx` returned only its own checking shell/grep. Local PIDs 8040 and 19004 were absent. `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader` returned no rows after each round. Both candidate worktrees remained clean at the exact HEAD.

One first final-scan command had a shell-quoting error in its grep alternation; that scan was not counted and was immediately rerun successfully with separate `-e` patterns. The final PowerShell check's exit 1 came from querying absent local PIDs, not a test failure. No kill or GPU reset was used. Short-lived internal pytest child PIDs were not individually retained; completed subprocess calls plus scoped scans are the available evidence. Read-only SSH inspection sessions were not all individually ledgered. No owned residual or observed stall remained; this is scoped hygiene evidence, not proof that no future hang is possible.

## Approval scope and remaining limits

Approval binds only the exact C2c6 component source catalog, verified-byte snapshot, actual post-import repository module audit, bounded source-policy compatibility exception, tests and cleanup obligations at the reviewed HEAD. The previous P2 mandatory test-driver defect is resolved without changing production behavior.

The snapshot and audit are **not a pre-execution loader**, are **not proof of which earlier `.pyc` bytes were consumed**, and are **not a hostile-Python sandbox**. `-B` in these tests does not establish a general no-bytecode-consumption property. Trusted configured Git revision is not a signature of owner approval. Source capture alone grants no execution authority.

Future C2d/e source consumption, cache policy, startup/HELLO/request integration and complete executable-worker source closure remain unimplemented and require their own reviewed catalog updates and runtime evidence. GPU-capable source was inspected as bytes only; no native GPU module import/launch, guardian custody, device isolation, CUDA/Warp lifecycle or runtime activation was tested or approved here.

C2d/e/f activation, GPU preparation, full C4 completion, B3/host delivery, shipping, a3b, results and baseline generation remain closed. Old C0 allowlist/protocol/gates remain unchanged. The report does not approve later phases or introduce a new policy/performance threshold. No additional non-blocking product-code comment requires disposition beyond these already explicit scope limits.
