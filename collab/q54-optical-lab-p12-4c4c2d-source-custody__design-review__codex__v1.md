# C2d2 captured-source custody — independent design review v1

Initiative: q54-optical-lab-p12-4c4c2d-source-custody
Stage: design-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-03
Status: accepted
Related Files: source-custody design-request v1; source-consumption design v1; guarded-custody design v2 and repair-design v2; guarded-custody code-review v5; runtime-integration design v3
Owner Summary: Approve with comments the exact bounded CPU source-custody design. The existing dependencies and failure APIs can support the proposed composition without another loader, guardian or lifecycle grammar. This is not code approval, completed C2d, external-runtime admission or a GPU/product gate opening.

## Identity, frozen candidate and verdict

Reviewer session: `/root/c4_c2d2_source_custody_design_review_v1`, fresh independent design review. No delegation, candidate edits, implementation assistance to the author, or inherited author reasoning was used as approval evidence.

- Local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Remote isolated candidate: `/data3/ang.gao/robot_sim_c2d2_design_v1`, authorized fallback `ang.gao@192.168.80.101`.
- Immediate base: `eef7a5052bed6c7699bd98d847fa7ed27147620e`.
- Reviewed frozen HEAD: `0eb29f53d4688dbc384e95a244c9938d534a37a3`.
- Candidate: `collab/q54-optical-lab-p12-4c4c2d-source-custody__design-request__codex__v1.md`.
- Candidate SHA-256: `088f245b305de66666d0d54912f517d9148f4dfd446a7c6b57eb84383a7671fa`.
- Actual delta: exactly one added design file, 235 lines. No implementation, operative workflow, gate, schema or protected C0 changes.
- Local and remote HEAD/digest matched the packet; initial/final working trees were clean. The original dirty remote checkout was not accessed.
- Verdict: **Approve with comments**, for this exact design snapshot and bounded CPU implementation authorization only.

No substantive blocking finding or missing prerequisite was identified. The comments below are non-blocking implementation/test cautions under requirements already present in this design and its accepted parents. They do not authorize new policy or waive fresh code review.

## Findings / non-blocking comments

### C1 — Explicitly carry pre-guard binding failures and swallowed loader failures into the existing failure path

Severity: non-blocking implementation caution. Locations: candidate lines 88–103, 120–142 and 202–205; `guarded_lifecycle.py::_observe_public_operation`, `adopt_host_owner`, `_fail_guard`, `_segment_context`; `native_bootstrap.py::_CapturedSourceLoader.exec_module`.

Trigger: a host adoption fails during binding, before a setup segment is active, or a nested module body catches an import error that has already invalidated the source session.

Actual dependency fact: `_observe_public_operation` does not automatically poison healthy pre-guard argument/adoption rejection. The independent probe demonstrated duplicate host adoption failing while the controller still reports healthy IDLE and `guarded_segment_failed=false`. In contrast, the existing internal `_fail_guard("operation", original)` followed by `terminate(...)` commits full terminal quarantine, retains the original exception and guardian owners, and makes zero ARM/COMPLETE/sync attempts. This mechanism exists at the reviewed base; no new grammar or public injection API is necessary.

Consequence of an incorrect implementation: relying only on a pre-guard `adopt_host_owner` exception could permit the ordinary healthy-IDLE `cpu_cancel` branch; retaining only a later source-audit exception could lose the original swallowed import exception/traceback. Neither behavior satisfies the candidate's existing partial-bind custody and original-exception requirements.

Disposition: retain as a targeted implementation/code-review obligation, not a requested design revision. The design already requires sticky failure, terminal graph custody before unwind, original exception retention, and no premature destructor/retry. Implement the new binding/loader integration so these requirements are true even when the first explicit guardian adoption fails. Tests should check first-adoption and later-adoption failure, original swallowed nested-import error identity, no successful normal completion, no new ARM/sync, and exact retained session/module/traceback owners before returning to the fixed driver. A later audit rejection alone is insufficient proof.

### C2 — Distinguish the launch configuration from the client's actual retained configuration object

Severity: non-blocking identity/test caution. Locations: candidate lines 81–94 and 195–201; `wait_protocol.py::GuardWaitClient.__init__`.

Actual dependency fact: the constructor validates the exact captured `GuardConfiguration` type, then constructs and retains a new validated configuration via `GuardConfiguration.from_payload(configuration.to_payload())`. The original launcher configuration and client configuration are equal but are not the same object. Both-lane probes confirmed this.

Consequence of an incorrect implementation: a naive assertion that the client's configuration `is` the constructor argument would reject every valid binding; replacing identity checks with payload/hash equality would fail the proposed exact-object binding requirement.

Disposition: retain this fact in implementation tests. Keep the trusted construction input and actual controller/client configuration identities distinct and verify the exact objects the binding retains; a copied equal-valued configuration is not a replacement authority. This is compatible with the proposed one-shot self-constructed controller and requires no change to the existing client, protocol, or lifecycle grammar. No caller-created controller/client should be accepted to avoid the distinction.

## High-risk design assessment

### Feasibility and one implementation of each contract

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully. Read the source-consumption design v1, custody design v2, repair-design v2, custody code-review v5, and parent runtime-integration v3 scope/source/setup/failure/validation/cleanup requirements. Checked actual bootstrap/source-policy/catalog, guardian/controller/reducer/client/supervisor code, native_setup/native_contract imports and deferred upload paths, plus existing fixed test support. Prior verdicts and the reported 769-test run were historical scope evidence, not a substitute for this review.

The current finder/loader really is limited by `_CORE`, not by catalog membership. `_verify` runs the sole `capture_native_source_snapshot` and compares captured bytes, actual loaded-module ledger and loader/module records. The new phase transition can extend that same session and same loader without introducing a second policy algorithm. The fixed literal import statements needed by the proposed operations do not require `import_module`, machinery, arbitrary module parameters, or a new compilation facility.

The actual eager repository dependency closure matches the design:

| Phase | Newly executed repository modules | Cumulative loader modules |
| --- | --- | --- |
| Verification core | optical_worker, lifecycle, protocol, manifest, native_source_policy, native_sources | 6 |
| CPU custody prelude | wait_protocol, wait_supervision, guarded_evidence, guarded_lifecycle | 10 |
| Fixed guarded definition import | native_setup, native_contract | 12 |

The directly executed bootstrap is an additional initial script owner, not a thirteenth module executed by the captured loader. Its trust/ownership remains separately identified. The prelude's guardian class cannot exist before its definitions execute; allowing this exact CPU-only prelude is explicit, not an unnoticed violation of before-CUDA guardian creation. Ordinary-import dependency probes confirmed the ten/twelve module sets and absence of NumPy/Torch/Warp, but do not prove captured-source execution.

`native_setup` imports only lifecycle/native_contract and stdlib at module execution. Its Torch/Warp/NumPy imports and upload helpers are deferred inside functions that this slice prohibits calling. `native_contract` module execution is likewise CPU/stdlib-only. No missing production module needs to be added to the catalog to obtain this closure.

### Partial ownership and failure semantics

The existing guardian supports CPU host adoption before ARM and inside an armed setup segment. Its exact-owner registry and coverage events can represent session, snapshot and module owners without changing lifecycle grammar. Existing append-only controller custody, original-exception holds, guardian freeze/quarantine and normal/emergency ticket checks provide the necessary failure destination.

The proposed register-started-before-adopt-before-compile/body order is implementable for these pure Python repository modules with the existing SourceLoader path. The adopted source session is an enclosing strong owner of a started partial module if explicit per-module adoption fails. That does not pretend the new guardian record was successfully emitted. Loader/session and guardian records serve different purposes; they are not two independent release authorities.

Normal import cycles may retrieve an existing started module without another execution. Explicit re-execution, loader replacement, unexpected dependency, drift and caught failure remain terminal. Exact registry references, not continued membership in sys.modules, retain failed partial modules. Source records stay descriptive and cannot manufacture the controller's live ticket/guardian identity.

The normal import operation can use the current `segment("setup")`: actual ARMED precedes compilation/import and audit, while normal COMPLETE/DISARMED occurs only after all succeed. Operation/ACK/audit failure must select the existing quarantine path, with no extra normal completion, emergency retry or synchronization. Healthy completion then terminal cancellation can use the already-defined non-completable emergency `cpu_cancel` envelope and whole-graph guardian release. The deadline remains armed through actual exit; C2d2 must still prove this composition with real supervised children in code review.

No seal/barrier/readiness operation is needed. Later sealed-graph mutation remains prohibited by the future integrated contract, not made safe by these CPU imports. No external-runtime create_module ownership is claimed: the proposal explicitly leaves that gap for the next checkpoint before any CUDA-capable role.

### Source policy, cached code, identities and gates

The full-bootstrap/loader/finder physical-byte pin must change when the existing loader is extended. The explicit native AST v4/catalog v3 amendment therefore addresses a real prerequisite rather than silently weakening v3. Current regex and ctypes exceptions can remain unchanged; bootstrap bytes -> policy digest -> catalog -> trusted configured release remains acyclic. Only the current catalog is executable; old artifacts are history, not fallback admission.

The design preserves the no-site/no-repository-sys.path captured-source boundary, source-only compilation, no bytecode fallback, original six-module direct-entry report and object/spec/package audits. Code review must rerun existing stale-pyc, reload and metadata attacks against the changed real loader and add new phase/owner attacks. Merely adding catalog entries or presenting a rehashed mutant must not admit another body. A deliberately mutated test release may reach a fault location, but is not successful execution of the frozen normal release.

No protected C0 edit, wire/grammar change, dispatcher, arbitrary callback, real runtime import, upload, final graph, product preparation, result schema or activation is authorized. The new internal fixed operation is test-accessed production logic, not a module/callable returned for product execution. The cleanup section assigns removal of transitional probes to P12.4e only after equivalent retained coverage, while retaining the selected source loader and supervision.

## Independent commands and observed results

Environment: Windows PowerShell; local `C:/Python314/python.exe` using `-I -S -B`; no Torch/Warp installed in this local interpreter per the packet. Remote read-only inspection used the authorized fallback. Primary183 timeout was supplied in the packet and was not independently retested in this design review. No install, push/reset, GPU test, candidate mutation, broad kill or original dirty-checkout access occurred.

Executed read/identity checks:

```text
git status --porcelain=v1
git rev-parse HEAD
git diff --stat eef7a5052bed6c7699bd98d847fa7ed27147620e HEAD
git diff --name-status eef7a5052bed6c7699bd98d847fa7ed27147620e HEAD
git diff --check eef7a5052bed6c7699bd98d847fa7ed27147620e HEAD
git log -3 --format='%H %s'
Get-FileHash collab/q54-optical-lab-p12-4c4c2d-source-custody__design-request__codex__v1.md -Algorithm SHA256
```

All command sessions completed exit0. `diff --check` was clean; status was empty; HEAD/base ancestry, one-file diff and digest matched the packet. `rg`/`Get-Content -Encoding UTF8` were used for the actual dependency and contract inspection described above. Initial default-encoding Chinese output was reread with explicit UTF-8; truncated relevant reads were split/retrieved. Those are read/display limitations, not candidate failures.

External scratch dependency probe:

```text
C:/Python314/python.exe -I -S -B C:/Users/AG/新建文件夹/c4_assets/c2d2-design-v1-dependency-probe.py
```

Probe SHA-256: `a75038dc089d585f12ae016e0205dc52a2c86f1ad4e55a5c01c9cf2f42c18a79`.

Actual result: runner exit0; four fresh isolated child cases passed, each child actual returncode0, empty stderr and peer thread joined. Each child imports the actual production dependencies, controller/client and existing real socket/core/transport peer. Normal cases demonstrate ten-module prelude closure followed by twelve-module closure inside a setup segment, one actual normal completion and one terminal cpu_cancel ARM, zero runtime sync, guardian RELEASED. Failure cases demonstrate the pre-guard rejection distinction in C1 and then the existing explicit failure-intake path: full quarantine, original error/controller/owner retained, zero normal completion/terminal ARM/sync. No NumPy/Torch/Warp module was loaded in any case.

This scratch deliberately inserts the candidate root into sys.path for ordinary dependency/API inspection. It does not use or claim the proposed C2d2 loader, before-body custody, source-frozen execution, process-watchdog kill behavior or GPU safety. The peer is a joined in-process supervisor thread, not a claimed supervisor subprocess. Scratch code was created with apply_patch outside the candidate and no implementation was changed.

Remote SSH prefix:

```text
ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Remote read-only commands, initial/final as applicable:

```text
cd /data3/ang.gao/robot_sim_c2d2_design_v1
git rev-parse HEAD
git status --porcelain=v1
sha256sum collab/q54-optical-lab-p12-4c4c2d-source-custody__design-request__codex__v1.md
ps -C python,python3.12,timeout -o pid,ppid,lstart,args
ps -C python,python3.12,timeout -o pid,ppid,lstart,comm
ps -p 2483327,2483430,2483521 -o pid,ppid,lstart,args
ps -p 2532716 -o user,pid,ppid,lstart,args
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```

No remote test process was launched. These sessions completed exit0; final remote HEAD/tree remained exact/clean.

## Actual exit/reap and round-end hygiene

Local runner27224(parent37708) launched and reaped:

| Case | Child PID | Recorded parent | Actual returncode |
| --- | --- | --- | --- |
| warp normal | 37132 | 27224 | 0 |
| warp pre_guard_failure | 2280 | 27224 | 0 |
| cuda normal | 23272 | 27224 | 0 |
| cuda pre_guard_failure | 16152 | 27224 | 0 |

Each child had a20-second failing outer bound; no timeout fired and no kill was needed. `communicate` collected actual exits and reaped each child; the tool observed runner exit0. Final `Get-Process -Id 27224,37132,2280,23272,16152 -ErrorAction SilentlyContinue` returned no process. Final Win32_Process Python enumeration was empty. The dependency peer threads joined and their sockets closed before child hard exit.

The final reviewer command process chain was PowerShell35400(parent31768), Codex31768(parent4408, started2026-10-03 03:01:32 local), Code4408(parent23388), Code23388(parent17688), Explorer17688. This is the shared host CLI chain, not a claim of a distinct OS security boundary per reviewer session. The tool command completed exit0; the continuing Codex/IDE processes are expected and were not terminated. Short-lived read/git/SSH utility PIDs were not individually sampled; no claim of a complete OS process ledger is made.

Before local probes, the remote inventory contained pre-existing Python626988 and unrelated training2483327 -> 2483430 -> 2483521; the sole GPU compute entry was2483521. At the first final query the training chain and GPU entry had ended, leaving only626988 in the queried CPU list. A follow-up query then showed a new sole GPU process2532716, owned by lin.li, parent2532626, started11:57:13 remote, running `scripts/eval_t5_holdout.py` against the ChainLoRA holdout data/output. Exact ps lookup confirmed the old training PIDs remained absent. All were external to this review's launch ledger. No signal, kill, cleanup or GPU action was applied to any of them; their transition is an unrelated external-state observation, not reviewer cleanup.

Local before/after GPU inventories contained desktop applications only and none of the five review probe PIDs. Remote nvidia-smi GPU enumeration succeeded for three devices. These are scoped hygiene observations, not proof of global idleness, no historical hang, or GPU completion. No owned residual or unresolved review hang was observed.

## Limits and exact approval scope

Approval permits implementation only of the candidate's CPU captured-source phase extension, exact controller/session/module custody binding, fixed guarded native_setup-definition import, source-policy v4/catalog v3 repin and bounded fixed CPU validation support. Comments C1/C2 remain attached as implementation/code-review cautions; they do not require a broader API or a new state-machine policy.

There is no C2d2 implementation yet to approve. This review did not rerun the historical769-test suite, execute new captured-source integration, prove partial-bind/module-body injectors, or exercise real subprocess watchdog faults for the proposed path. Those are code-checkpoint obligations, not evidence supplied by the small dependency probes. No prior test count is credited as proof of this design or future code.

Still closed: external third-party partial-import/native-create ownership; site admission; real Torch/Warp import/init/upload and target1/ambient0, foreign pre-return, unknown-scope and upload/ready-event faults; full final graph/seal/barrier/warmup/readiness; complete C2d; C2e/f; C4c3/c4 and C4d/B3; product GPU preparation, a3b, results/baseline, R2/host/S, shipping and activation. The next C2d checkpoint must explicitly design and prove external import ownership plus mandatory real-device runtime behavior before admitting any GPU-capable role. Source-frozen bytes, object identity and CPU custody do not by themselves prove those properties.
