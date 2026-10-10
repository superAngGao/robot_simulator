```text
Initiative: Q54 / P12.4 / C4
Stage: Independent CODE review — M1 setup orchestration consolidation
Author: Codex, fresh independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files: systematic-refactor gap-review v1 M1; runtime-integration
  design-request/review v3; systematic-refactor code-review v4;
  implementation-note v6
Owner Summary: Approve with comments for the exact M1 candidate.
  No blocking production defect established. One pre-existing test-fixture
  defect remains a nonblocking follow-up. Later product gates remain closed.
```

**Verdict: Approve with comments.**

Approval binds **`c3472f22f1d52bfb6c853bf95cef4040cf666316`**, solely for M1 consolidation over the approved bounded R0 preparation. The implementation removes duplicated orchestration and Warp-specific failure routing while preserving the relevant lifecycle semantics. No blocking production finding or required M1 evidence gap remains.

**Independence and revision**

This was a fresh independent review session, thread **`01a12422-6645-7760-929a-0aacefd5bce4`**, agent `/root`. I received the review packet, without implementation conversation or authorship. I did not spawn agents, modify the candidate, implement fixes, or manually write `report.md`.

I read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely, correcting the initial console encoding. I inspected the requirements, accepted design/review, prior bounded approval, author claims, complete delta and surrounding control/resource paths.

| Item | Observed value |
|---|---|
| Candidate | `C:/Users/AG/新建文件夹/c4_setup_orchestration` |
| Branch | `implementation/c4-setup-orchestration` |
| Base and merge-base | `6db106caaeeac66d3bfc939b2bee44c15ce104d9` |
| Frozen HEAD | `c3472f22f1d52bfb6c853bf95cef4040cf666316` |
| HEAD tree | `0bd195f84f427f3c6ca411d7b025553556e2ff25` |
| Delta | 10 files; 677 insertions, 271 deletions |
| Working tree | Initial, intermediate and final checks clean |

The prior approved revision `640cf9f` to this base contains documentation/evidence changes only. The primary-host candidate was also clean and at exact `c3472f2`; no fallback host was used.

**M1 disposition and high-risk findings**

M1 is closed for this revision. No substantive blocking production defect was established.

- The six [SourceSession entrypoints](C:/Users/AG/新建文件夹/c4_setup_orchestration/optical_worker/native_bootstrap.py:3237) delegate to explicit functions in the existing setup module. Source admission, exact upload registry and runtime-binding validation remain bootstrap responsibilities. No callback registry, workflow engine, additional guardian or parallel slot controller was introduced.
- [Shared input validation](C:/Users/AG/新建文件夹/c4_setup_orchestration/optical_worker/native_setup.py:173) preserves original prepared-input identity, validation-report presence, retained runtime binding and guardian adoption. Backend-specific construction and existing guard segmentation remain explicit.
- [Publication](C:/Users/AG/新建文件夹/c4_setup_orchestration/optical_worker/native_setup.py:192) checks the preparing phase, performs the observation checkpoint, rechecks liveness and phase, and only then publishes ready. Checkpoint exceptions, poison and phase drift cannot publish successful readiness.
- The final graph retains its distinct existing publication protocol: graph construction, coverage seal, exact-target barrier, successful DISARMED and post-barrier validation precede admission. It does not use the ordinary checkpoint helper. I did not treat the author’s shorthand about shared publication as proof that all six entries use an identical mechanism.
- [Common failure routing](C:/Users/AG/新建文件夹/c4_setup_orchestration/optical_worker/native_setup.py:100) retains the original exception and distinguishes entry integrity rejection from ordinary setup failure. Source-integrity rejection remains terminal.
- Ordinary healthy-IDLE errors preserve completed guard history. Proven outstanding target work selects the existing single emergency synchronization; unknown scope or unverifiable custody selects quarantine.
- Active, pending or failed guards acquire no replacement normal guard or synchronization. Existing segment failure and terminal-custody machinery remains authoritative.
- CPU-only and current final-quiescent failures use the existing cancellation predicate without requiring runtime lookup or GPU synchronization. Final-quiescence validity is still checked by the lifecycle.
- Secondary retention/disposal failures do not replace the original raised exception. Relevant original and secondary exceptions remain retained, with conservative quarantine.
- Warmup authority retirement, distinct provisional/final readiness, warmed-storage reuse, final barrier ordering and closed measured-submission entry remain intact.

The native scope evidence supports target-device execution and restoration of the recorded ambient state. CUDA’s ambient observation is **logical device 0, raw device 1**; it is not evidence of raw-device-0 restoration.

**N1 — P3, nonblocking: existing graph-observer fixture omits `fixed_row`**

Locations: [fixture construction](C:/Users/AG/新建文件夹/c4_setup_orchestration/tests/unit/optics/test_prepared_fixed_render_inputs.py:123), [observer invocation](C:/Users/AG/新建文件夹/c4_setup_orchestration/tests/unit/optics/test_native_render_graph_support.py:55), and [field access](C:/Users/AG/新建文件夹/c4_setup_orchestration/tests/support/native_runtime_process.py:1095).

Trigger: run `test_native_render_graph_support.py`, covering both lanes and diagnostics modes.

Consequence: all four cases raise:

```text
AttributeError: '_SourceSession' object has no attribute 'fixed_row'
```

The fixture constructs `_SourceSession` using `object.__new__` and omits this field. The observer fails before reaching the production graph entry, so these cases currently provide no graph-observation or mutation-rejection coverage.

This is **pre-existing**: the fixture/test files are unchanged, the field access exists in base, and an isolated scratch snapshot of base reproduced all four failures at the corresponding base location.

Requested follow-up: initialize the fixture’s intended row state consistently, then rerun all four cases and retain their existing mutation assertions. No candidate fix was made during review.

This does not block M1: the affected production graph path passed separate CPU preparation tests and real native acceptance, whose graph observations were independently replayed.

**Executed verification**

Local environment: Windows PowerShell; `C:/Python314/python.exe`, Python 3.14.0, pytest 9.1.1, NumPy 2.5.2. Warp and trimesh are unavailable locally.

Tests used `-B`, `PYTHONDONTWRITEBYTECODE=1`, `-p no:cacheprovider` and scratch-only `--basetemp`.

| Execution | Actual result |
|---|---|
| Main 11-suite selection | **618 passed, 13 skipped, 4 failed**, 481.49 seconds |
| Independent probes plus provisional preparation suite | **102 passed**, 18.61 seconds |
| Isolated HEAD graph-observer reproduction, `-x` | One expected fixture failure |
| Isolated base graph-observer reproduction | Same **4 failures**, 4.75 seconds |
| Committed source/catalog/AST audit | All **90 sources passed** |
| Full-delta `git diff --check` | Passed |
| Native evidence replay on primary Linux host | All six raw records passed validation, including the historical failed-driver record |

The main command was `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`, with scratch basetemp and these files under `tests/unit/optics`:

```text
test_optical_setup_orchestration.py
test_warp_module_preparation.py
test_native_final_graph.py
test_native_render_graph_support.py
test_prepared_fixed_render_inputs.py
test_native_render_warmup.py
test_native_sources.py
test_native_source_policy.py
test_native_bootstrap.py
test_guarded_lifecycle.py
test_runtime_custody.py
```

The second execution selected scratch `test_independent.py` and `test_native_render_preparation.py`. Its 12 independent cases cover both-lane pending/failed-client classification, real active-guard entry, secondary disposal failure, and final-quiescent out-of-order setup without runtime lookup. Pending-client states were explicitly injected CPU states, not real transport-loss experiments.

The 13 local skips comprise two real-Go2 cases, four POSIX checkout/entry cases and seven pinned CPython-3.12/POSIX generated-stdlib cases. They are not local passes.

The coordinator’s Linux source/bootstrap/policy execution at `624a71f` records **293 passed, 1 platform skip**, including the configured real Go2 and POSIX checks. I inspected its command, environment setup, log and revision. This is supplied execution evidence, not a fresh reviewer rerun. Its applicable source and test bytes are unchanged at `c3472f2`.

Evidence files:

- [Main CPU log](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/cpu-core.log)
- [Independent probes](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/test_independent.py) and [results](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/probes.log)
- [Isolated base reproduction](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/base-isolated-graph-support.log)
- [Source audit](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/audit_sources.py)
- [Native audit](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/audit_native.py) and [Linux replay results](C:/Users/AG/新建文件夹/c4_assets/m1_setup_orchestration_review_v1/native-replay-linux.log)

Inspection-tool limitations were corrected and retained: Windows rejected Linux absolute paths during the first native-environment parse, so replay moved to Linux; an initial PowerShell pipe mangled a Unicode path; a remote `rg` command was unavailable/incorrectly quoted; and the first base-reproduction invocation still resolved candidate imports, so it was repeated from the isolated base directory. These are separate from candidate test results.

**Source identity and explicit catalog policy**

I validated exact committed Git blobs against the catalog and production AST policy, without normalizing production source bytes.

- Catalog SHA256: `d2c1d277c263a70ed02422a60b40e3c0453086385b6158a3b8f185a510cd35fd`
- AST policy SHA256: `6a3b153857c298712f045f0cab5b5e48c847ee61849208597a80872f71da165e`
- All 90 source entries passed.
- Explicit lane inventories are unchanged.
- Changed catalog source bodies are only `native_bootstrap.py`, `native_setup.py` and `native_source_policy.py`.
- Bootstrap body/class pins match the reviewed implementation; no broad new policy exemption was introduced.
- `git diff 624a71f c3472f2 -- optics optical_worker` is empty. The production bytes are identical; that intervening revision changes only the implementation note and acceptance-driver assertion.

**Native evidence**

I did not kill, restart or duplicate the coordinator’s matrix. I observed it running, then inspected its completed archive and ready notification.

Archive: [c4-setup-orchestration-native-evidence-v1.tar.gz](C:/Users/AG/新建文件夹/c4_assets/c4-setup-orchestration-native-evidence-v1.tar.gz)

- Size: **16,432,007 bytes**
- Members: **191**, unique, with no unsafe paths or link members
- SHA256: **`e41a65b368720b1f083c47896a3a51b08a5a28b51ffe4809984cc3da88fee90e`**
- Local and remote archive hashes match.

Using `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -` over read-only SSH, I replayed the records through the exact candidate’s lifecycle/combined-custody and runtime-custody validators. I checked request revision/catalog/environment bindings, selected custody prefixes, attachment, worker/supervisor identities and observed exits. Final cases additionally passed storage, provisional graph and warmup validators and final graph/quiescence checks.

| Revision / case | Normal completions | Total ARMs | Emergency syncs | Worker exit |
|---|---:|---:|---:|---:|
| Historical 624 CUDA IDLE entry — driver failed | 6 | 7 | 1 | 74 |
| c347 CUDA IDLE entry | 6 | 7 | 1 | 74 |
| c347 Warp IDLE entry | 6 | 7 | 1 | 74 |
| c347 Warp final graph success | 27 | 28 | 0 | 0 |
| c347 CUDA final graph success | 23 | 24 | 0 | 0 |
| c347 Warp module issuance failure | 10 | 11 | 1 | 74 |

All five new cases have acceptance results. Their supervisors observed exit without deadline expiry or kill request.

The historical 624 run remains a **failed driver execution**, with its raw artifact and missing `result.json` preserved. Its production path disposed correctly. The incorrect expectation counted a factory phase before factory preparation had started; c347 uses the existing six-element `RUNTIME_PHASES`. The new reruns separately establish acceptance. Successful replay of the historical custody record does not convert that failed driver into a pass.

Both final cases establish ten observed warmups, final readiness, three dormant plans, acknowledged target barrier, acquire/cancel, zero sample submissions and cancellation without another synchronization. The issuance failure retains the three compiled records and original error without inventing a failed normal guard.

**Scope, limitations and retained follow-ups**

Approval closes **M1 orchestration consolidation only**. It does not reopen or enlarge the preceding R0 approval.

Still unproved or closed by this report:

- Final-sample rendering, final-output correctness and native final-arena resolve/reclaim.
- R1 composition, C2f token/close integration and full cross-lane/fault/diagnostic acceptance.
- Actual parent-death/transport-loss experiments at these final boundaries.
- Product PREPARE, measured submission, activation and full C4 acceptance.
- R2/S/host delivery, external holds/depth two, C4c3/c4, C4d/B3, performance, backend selection, results/baselines/a3b and shipping.

Retain N1 and the earlier whole-C4 structural follow-ups. The existing whole-prefix replay cost remains an untimed-setup watch item; no performance claim or new optimization requirement is made here.

**Process hygiene**

Reviewer CLI PID **32416**, parent **47116**, started at `2026-10-10 12:46:36`; it remains active only to return this report. Main pytest PID **47356**, parent **35500**, started at `12:47:27`; both were absent at final scoped inspection. All test and SSH tool sessions reached terminal results. Short-lived probe/replay PIDs were not individually sampled.

Independent remote checks found all **20 recorded coordinator/CPU/driver/supervisor/worker PIDs absent**, including outer matrix PID `821476`. The recorded hygiene-check process also exited. GPU-process inspection showed unrelated MPS activity, which was untouched.

No reviewer GPU campaign, dependency installation, process signal, GPU reset, candidate edit or unresolved owned hang occurred. Final local and remote revision/status checks remained exact and clean.