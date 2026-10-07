# C2d2 captured-source custody — independent code re-review v2

Initiative: q54-optical-lab-p12-4c4c2d-source-custody
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v2
Date: 2026-10-07
Status: accepted
Related Files: source-custody design-request/design-review v1; code-review v1; implementation-note v1; native_bootstrap.py; guarded_custody_process.py; test_native_source_custody_process.py
Owner Summary: Approve the exact frozen revision for the bounded CPU captured-source custody checkpoint. The previous P2 no-ACK test-oracle finding is closed by independently observed supervisor-killed and worker-self-exit outcomes in both lanes. Fresh runs passed 61 process cases, 725 further regressions including both real Go2 asset cases, and eight independent identity probes. This does not complete C4 or open runtime/GPU/product gates.

## Identity and verdict

Reviewer session: `/root/c4_c2d2_source_custody_code_review_v2`, fresh independent code re-review. No candidate edits or delegation. The author process lane was explicitly released before this review launched tests; author outputs and prior reviewer outputs were not credited as this review's execution evidence.

- Overall comparison base: `af509cb5e3ed4103a1035a1298cd20bb652f45e4`.
- Prior reviewed candidate / immediate repair base: `f7692b4b7bfd50521956c5c83fcb2bc94049906a`.
- Frozen reviewed HEAD: `0dd684867a8d60a69978e878b26d98169d2b9b1c`.
- Tree: `8502f98dcd0da5927d1d67f41031c4218952292a`.
- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Independent remote candidate: `/data2/shared/ang.gao/robot_sim_c2d2_code_v2`, primary SSH alias `tileops-dev`.
- Both candidates matched HEAD/tree and were clean at initial and final checks.
- Approved design SHA-256: `088f245b305de66666d0d54912f517d9148f4dfd446a7c6b57eb84383a7671fa`.
- Tracked prior report SHA-256: `9d58f848fd9e9ced7ec9462da418c31b8e523a69827c9a14ed82f8197952558f`; matches the implementation note's verbatim-report claim.
- Verdict: **Approve**, solely for the scope at the end of this report.

No new substantive finding or unresolved blocking evidence gap was established. Prior F1 is closed. The equal-valued ModuleSpec observation remains explicitly descriptive, not a claim of spec-object identity enforcement or a new admission requirement.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` fully, with explicit UTF-8 reread of the Chinese principles. Read the approved design/review, prior full code review, implementation note, actual overall and repair diffs, and the relevant loader/controller/client/supervisor/validator/test call and state flows. Historical reports were requirements and counterexample context, not approval evidence.

## Previous finding F1: independently closed

Prior F1 was P2 at the old process test's unconditional returncode74/final-marker assertion. The real supervisor can win the equal 1000ms client/supervisor ACK-deadline race and kill/reap before any worker exception-tail write.

Inspected repair locations: `tests/unit/optics/test_native_source_custody_process.py:81–111` and `tests/support/guarded_custody_process.py:390–398`.

The repaired oracle does not treat any arbitrary negative exit as success. For the two no-ACK fault names it additionally requires validated final supervisor evidence, deadline expiry, a kill request, deadline-before-kill-before-process-exited ordering, matching actual observed returncode, one ARM, zero normal completion, no ACK write or COMPLETE request, no import-body entry, correct parent liveness, and no destructor/atexit marker. The killed branch permits absent worker evidence/final marker without calling that absence completion. The self-exit branch still requires code74, the no-runtime marker, terminal worker evidence, and full quarantine; the common worker checks require zero entered device leaves/emergency sync and no normal completion or terminal ARM attempt.

The additional `source_worker_exit_no_ack` case is a fixed test-only scheduling injector. It sleeps400ms once when the initial transport is readable, before `transport.service` receives/commits ARM. The actual client's absolute deadline begins in `GuardWaitClient._exchange` before sending ARM; the supervisor's pending deadline begins in `_accept` at commitment. Thus the new injector makes the unchanged client deadline precede the unchanged supervisor deadline. It does not extend/restart a committed deadline, alter production timers/protocol, substitute a fake client/core, send an ACK, or admit a body. The case explicitly requires its injection and self-exit74.

No unrelated fault is admitted into the no-ACK alternative. Existing self-exit faults retain their exact exit requirement and final marker; hang/parent-loss branches gained stronger event ordering. The unchanged20-second harness bound raises `RuntimeError`; the unchanged25-second test `communicate` timeout propagates failure through cleanup. Neither is an allowed endpoint. Actual reap occurs through `process.poll`/`process.wait(timeout=1)` in the supervisor and `communicate` in pytest; the observer adds no alternative acceptance rule.

Fresh independently observed no-ACK endpoints:

| Lane / case | Supervisor / worker PID | Actual worker exit | Worker record / no-runtime marker | Actual supervisor operations |
| --- | --- | --- | --- | --- |
| warp / no_ack | 2945597 / 2945605 | -9 | absent / absent | initialized, arm_committed, deadline_expired, kill_requested, process_exited |
| warp / worker_exit_no_ack | 2945735 / 2945737 | 74 | present / present | initialized, arm_committed, terminal_entered, process_exited |
| cuda / no_ack | 2947300 / 2947301 | -9 | absent / absent | initialized, arm_committed, deadline_expired, kill_requested, process_exited |
| cuda / worker_exit_no_ack | 2947404 / 2947405 | 74 | present / present | initialized, arm_committed, terminal_entered, process_exited |

All four supervisors exited0 and had runner2944933 as parent; each worker's recorded PPID matched its actual supervisor. All four had one ARM, no ACK/COMPLETE/body entry, parent_alive=true, and no destructor/atexit. Killed outcomes had terminal_reason=deadline_expired, deadline_expired=true, kill_requested=true, and observed_returncode=-9. Self-exit outcomes had input_eof, no deadline expiry/kill, actual74, and validated full worker quarantine. Both legal endings and the original absent-tail counterexample were therefore actually exercised, not inferred from test names or author logs.

## High-risk implementation assessment

Repair delta is exactly four files: two test files (31 additions/one deletion), the174-line unchanged prior report, and the89-line implementation note. No production code changed after the prior candidate. Overall delta is14 files,983 insertions/33 deletions. Actual protected-path checks found no changes to C0 lifecycle/native_setup/native_contract/worker, guarded controller/evidence/wait protocol/core, or shared collaboration rules. No gate/schema/arena/production-launcher file appears in the overall changed-path inventory.

The same `_CapturedSourceLoader`, finder and session execute six core, ten prelude and twelve definition modules; the directly executed bootstrap is a separate retained owner. The fixed operation returns no product module/callable and accepts no arbitrary module selector. Source-only code generation retains bytes and does not fall back to repository sys.path or pyc. The actual captured controller/configuration classes construct the single guardian/client. Launch configuration and the client's equal-but-distinct revalidated configuration are retained and audited separately by identity.

The session is registered/adopted before guarded imports. Each post-bind module enters the strong session registry before explicit guardian adoption and compilation/body. A failed module adoption remains held by the enclosing session. Audit verifies the literal closure, source snapshot, module/loader/source identities and guardian ownership before normal COMPLETE/DISARMED. The setup segment arms before importing; guarded failure reaches existing quarantine before driver unwind. No second release authority or lifecycle reducer was added.

Design-review cautions C1/C2 remain satisfied in the actual flow and fresh tests: first/later pre-guard adoption rejection calls explicit failure intake; swallowed nested-loader errors retain the original error identity; no retry/new ARM/sync occurs after failure; equal client-configuration/snapshot replacements reject. The26-case fresh source-custody suite checks these paths and failure stickiness after resetting the ordinary flag. Existing direct-entry, stale-bytecode, metadata, reload, source-policy and repacked-mutant tests were rerun.

Additional independent probes against the unchanged actual POSIX checkout (not a packaged mutation) confirmed, in both lanes: a distinct equal byte string cannot replace a loader's retained source; catalogued `optical_worker.worker` remains execution-ineligible; source audit rejects after terminal CPU cancellation; and a shallow equal-valued ModuleSpec copy remains accepted. The latter is consistent with the implementation note: exact module/loader/source identity plus spec-field checks, not retention of the original spec object. These probes used real production loader/controller and joined socket/core peers, not duplicate implementations. Runtime roots remained absent.

Catalog v3 retains both literal lane inventories and all source paths from v2. Only the three intended source digests differ: native_bootstrap, native_source_policy and native_sources. Bootstrap pin is `7dfb6074c7b5dc639a0763d21217c4e294afc340a03ee89b3ba058e8d088e02c`; policy isv4. Policy/catalog byte checks passed. The cleanup ledger adds the exact fixed test/catalog paths and retains its P12.4e removal obligations; neither backend is removed.

## Independently executed verification

Remote interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, Python3.12.13. Local Windows PowerShell and `C:/Python314/python.exe` were used for inspection/Ruff only; no local Torch/Warp claim. SSH prefix: `ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev`. All scratch scripts are outside the candidate, created with apply_patch and copied to `/data2/shared/ang.gao/`; tests write only their normal disposable temp fixtures. No install, push/reset, broad kill, gate edit or GPU workload.

Identity/static commands included:

```text
git status --porcelain=v1
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git log -2 --format='%H %s'
git diff --stat af509cb5 HEAD
git diff --name-only af509cb5 HEAD
git diff --numstat af509cb5 HEAD
git diff --stat f7692b4 HEAD
git diff --numstat f7692b4 HEAD
git diff --check af509cb5 HEAD
git diff --exit-code f7692b4 HEAD -- optical_worker
git diff --exit-code af509cb5 HEAD -- optical_worker/lifecycle.py optical_worker/native_setup.py optical_worker/native_contract.py optical_worker/worker.py optical_worker/guarded_lifecycle.py optical_worker/guarded_evidence.py optical_worker/wait_protocol.py optical_worker/wait_supervision.py AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md
C:/Python314/python.exe -m ruff check optical_worker/native_bootstrap.py optical_worker/native_source_policy.py optical_worker/native_sources.py tests/support/guarded_custody_process.py tests/support/guarded_peer.py tests/support/source_custody_process.py tests/unit/optics/test_native_source_custody.py tests/unit/optics/test_native_source_custody_process.py
```

Diff/checks exited0; Ruff: `All checks passed!`. Get-FileHash checked the design/prior report/scratch digests; PowerShell ConvertFrom-Json/Compare-Object checked v2/v3 lane and path equality and the three digest differences. `rg` and bounded UTF-8 reads inspected the actual code. Initial attempt to read root `REVIEW_PRINCIPLES.md` failed because the authoritative file is under `collab/`; it was then fully read there. Initial encoding/truncated display issues were handled by UTF-8 and bounded rereads. These were tooling/read issues, not code failures.

The outside-candidate observer calls the existing pytest tests unchanged. It subclasses only the parent runner's Popen and, after real `communicate` completes, prints the returned actual artifact, identity, markers and endpoint. It neither alters child/supervisor code nor changes test assertions/timers. Source fixtures explicitly package LF bytes; additional actual-checkout direct-entry tests and identity probes independently cover the unmodified POSIX release.

Process command, cwd the remote candidate:

```text
timeout --signal=TERM --kill-after=15s 300s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B /data2/shared/ang.gao/c2d2-v2-review-runner.py /data2/shared/ang.gao/robot_sim_c2d2_code_v2 -q -s -p no:cacheprovider tests/unit/optics/test_native_source_custody_process.py tests/unit/optics/test_guarded_custody_process.py
```

Result: **61 passed in132.80s**, actual command/runner exit0, tool session39282. All32 mandatory source-process cases plus29 affected old-harness cases. Observer recorded61 supervisor completions. Temp evidence root: `/tmp/pytest-of-ang.gao/pytest-110`. No outer timeout fired.

Broader regression command, same cwd:

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 timeout --signal=TERM --kill-after=15s 420s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B /data2/shared/ang.gao/c2d2-v2-review-runner.py /data2/shared/ang.gao/robot_sim_c2d2_code_v2 -q -s -p no:cacheprovider tests/unit/optics/test_native_source_custody.py tests/unit/optics/test_native_bootstrap.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_guarded_lifecycle.py tests/unit/optics/test_guarded_evidence_fixtures.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
```

Result: **725 passed in194.48s**, no skips, actual command/runner exit0, session6272. Includes both real Go2 source/materializer cases with the formal asset environment, not surrogate probes. No outer timeout fired. Historical752/2-skipped and author runs were not reused as evidence.

Independent identity command:

```text
timeout --signal=TERM --kill-after=15s 120s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B /data2/shared/ang.gao/c2d2-v2-identity-probe.py /data2/shared/ang.gao/robot_sim_c2d2_code_v2
```

Result: **eight fresh isolated -I -S -B children passed**, each actual returncode0 and reaped, joined peers, runner/command exit0, session43437. No timeout. Runner script SHA-256 `1033d27328bf9a9121e964f43511b50b383cc349f38b19460f1d88f969678a99`; identity probe SHA-256 `0ed252137b4215f4ea6ab25e63c2de81c8c2c6500ea1d20c0ebfb8e70b39f421`; local/remote hashes matched.

## Actual process ledger and hygiene

Process runner2944933(parent2944932, the scoped timeout) reaped all61 supervisors. All supervisors below returned0, their recorded parent was2944933, and each actual worker PPID matched the supervisor. Supervisors observed and reaped the worker before returning. Notation: `case: supervisor / worker / worker returncode`.

```text
warp source cases:
success: 2944944 / 2944945 / 0
prelude_raise: 2945083 / 2945084 / 74
module_adopt: 2945106 / 2945110 / 74
compile_raise: 2945134 / 2945135 / 74
body_raise: 2945190 / 2945195 / 74
caught_nested: 2945223 / 2945224 / 74
audit_raise: 2945241 / 2945242 / 74
import_hang: 2945276 / 2945277 / -9
parent_loss: 2945419 / 2945420 / -9
no_ack: 2945597 / 2945605 / -9
worker_exit_no_ack: 2945735 / 2945737 / 74
wrong_ack: 2945880 / 2945882 / 74
ack_eof: 2945941 / 2945942 / 74
stalled_disarm: 2946006 / 2946007 / 74
cleanup_hang: 2946063 / 2946068 / -9
exit_hang: 2946276 / 2946277 / -9

cuda source cases:
success: 2946577 / 2946641 / 0
prelude_raise: 2946722 / 2946732 / 74
module_adopt: 2946773 / 2946774 / 74
compile_raise: 2946833 / 2946834 / 74
body_raise: 2946911 / 2946912 / 74
caught_nested: 2946940 / 2946941 / 74
audit_raise: 2946974 / 2946976 / 74
import_hang: 2947004 / 2947005 / -9
parent_loss: 2947181 / 2947182 / -9
no_ack: 2947300 / 2947301 / -9
worker_exit_no_ack: 2947404 / 2947405 / 74
wrong_ack: 2947490 / 2947491 / 74
ack_eof: 2947572 / 2947573 / 74
stalled_disarm: 2947666 / 2947667 / 74
cleanup_hang: 2947705 / 2947707 / -9
exit_hang: 2947845 / 2947846 / -9

old affected process harness:
cpu_success: 2947989 / 2947994 / 0
cpu_cleanup_hang: 2948002 / 2948007 / -9
cpu_exit_hang: 2948017 / 2948018 / -9
normal_success: 2948039 / 2948044 / 0
normal_leaf_raise: 2948072 / 2948073 / 74
inner_abort: 2948075 / 2948076 / 74
foreign_inner_abort: 2948077 / 2948087 / 74
unknown_inner_abort: 2948088 / 2948090 / 74
invalid_permit_inner_abort: 2948091 / 2948095 / 74
cuda_import_failure: 2948096 / 2948097 / 74
warp_import_failure: 2948098 / 2948106 / 74
cuda_import_thread_failure: 2948114 / 2948130 / 74
warp_import_thread_failure: 2948144 / 2948146 / 74
pre_leaf_raise: 2948148 / 2948149 / 74
foreign: 2948150 / 2948151 / 74
unknown: 2948152 / 2948155 / 74
sync_raise: 2948156 / 2948157 / 74
sync_hang: 2948159 / 2948160 / -9
cleanup_raise_before: 2948193 / 2948195 / 74
cleanup_raise_after: 2948196 / 2948197 / 74
normal_no_ack: 2948199 / 2948200 / 74
normal_wrong_ack: 2948251 / 2948318 / 74
normal_ack_eof: 2948367 / 2948385 / 74
normal_stalled_disarm: 2948406 / 2948407 / 74
terminal_no_ack: 2948408 / 2948409 / -9
terminal_wrong_ack: 2948476 / 2948478 / 74
terminal_ack_eof: 2948483 / 2948488 / 74
target_cleanup_hang: 2948495 / 2948498 / -9
target_exit_hang: 2948580 / 2948581 / -9
```

Broad runner2948886(parent2948885, timeout parent2948884) started19:41:54 remote and exited0. Eight inherited-import probe children2953841,2953843,2953845,2953848,2953851,2953853,2953855,2953862 each recorded parent2948886 and exit0. Actual-source direct-entry child2950825 was separately sampled under2948886. The wrapper does not claim to record every other broad-suite child.

Identity runner2954767(parent2954766) launched/reaped, in lane/fault order above: warp2954796,2954860,2955146,2955224; cuda2955318,2955425,2955663,2955817. Each printed its actual PPID2954767 and returned0.

Final exact `ps -p <all listed supervisor/worker/runner/timeout/probe PIDs> -o pid,ppid,lstart,args` returned only the header. Earlier targeted no-ACK lookup likewise found none (standalone ps exit1 means no matching PID, not test failure). Final `ps -C python,python3.12,timeout -o pid,ppid,lstart,args` contained only pre-existing unrelated Python2587008(parent2586183, started14:25:56). Before, during and after inventories showed the sole GPU compute process2238410, `nvidia-cuda-mps-server`, untouched. Eight GPU UUIDs were enumerated. No owned residual or unresolved review hang was observed; this is scoped hygiene, not global idleness or GPU completion evidence.

Local final Win32_Process Python enumeration was empty. Sampled command chain: PowerShell4796 -> Codex10808(parent34516, started19:15:47). The persistent Codex/IDE chain is expected shared tooling, not an independent OS isolation claim. Final local GPU inventory contained desktop applications only; no review GPU work was run. Short-lived git/read/SSH/Ruff utility PIDs and every broad-suite descendant were not individually sampled; no complete OS process ledger is claimed.

Final local/remote candidate remained clean at the exact reviewed revision. Original remote `/data2/shared/ang.gao/robot_simulator` was only read at final verification: clean `main` at `f771e612beaeea126e26926738097512cd2d809c`, untouched. No unrelated process was signalled; no material data was deleted. Scratch scripts and disposable evidence remain outside the candidate.

## Exact approval scope and remaining limits

Approval covers only this frozen revision's CPU captured-source phase extension, same-loader source/session/module binding to the same guarded guardian, fixed native_setup/native_contract definition import under actual supervision, policy-v4/catalog-v3 pins, and the bounded CPU validation support including repaired no-ACK outcomes. It does not authorize additional callable/module admission or change the staged cleanup obligations.

No substantive blocking finding remains in this scope. The ModuleSpec metadata-copy observation is retained with its explicit current semantics; it is not presented as a rejected attack or authority-bearing object. As with any real-time process test, extreme external scheduling disruption can make a bounded test fail; that is not permission to accept outer timeouts or weaken deadlines.

Not executed or claimed: new actual-runtime/GPU matrices, performance gates, product preparation, or a whole-repository full suite. The separate optical benchmark-contract suite was not rerun in this repair review; the affected C0/closed-worker suites were. The Go2 executions are CPU asset/materialization checks, not runtime readiness. Existing tests' deliberate compile/body faults are test instrumentation, not successful frozen-source releases. Normal test caches were allowed where their contracts permit them; pytest cacheprovider was disabled and all results above are fresh executions.

Still CLOSED: external third-party/native-create/partial-import ownership and site admission; actual Torch/Warp import/init/upload and target1/ambient0, foreign pre-return, unknown-scope and upload/ready-event matrices; complete resource graph/seal/barrier/warmup/readiness; full C2d, C2e/f, C4c3/c4 and C4d/B3; arena/product GPU preparation, a3b, results/baseline, R2/host/S, shipping and activation. **C4 is not complete.** CPU object retention, frozen bytes and supervisor records do not prove real device allocation/release safety, GPU completion or hostile-Python isolation.

An unchanged verbatim archive of this report may be recorded separately under the documented evidence-only exception, referencing this approved revision. It cannot modify the verdict, implementation, operative policy, scope or gates.
