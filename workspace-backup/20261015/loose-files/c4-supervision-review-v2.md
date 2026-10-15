# C4c1c steps 1–2 independent code re-review

Initiative: q54-optical-lab-p12-4c4c1c-supervision
Stage: review
Author: codex (independent reviewer)
Version: v2
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v3/v2; supervision review v1; tests/support/sacrificial_lifecycle.py; tests/unit/optics/test_sacrificial_lifecycle.py; test-only S/V/E supervisor
Owner Summary: Request changes for two new Ruff F821 diagnostics in the regression test. The prior ownership-retention defect is independently closed, all 78 bounded tests pass, and no further runtime safety finding was identified in this scope.

## Identity, revision, and scope

- Fresh reviewer session/task: `/root/c4_supervision_review_v2`. No implementation conversation was inherited and no subagents were used. Author notes and prior verdicts were treated as claims to check.
- Base: `c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0`.
- Previously rejected revision: `a597e8a3d5346fd0a7c18491267274b7428cbfc4`.
- Reviewed HEAD: `1ac90a2a4a067464845de3db5bfac2fa0f7e0d25`.
- Reviewed tree: `17eaee9fe3002a58e749f68984a231b411b37aae`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial`, branch `implementation/c4-sacrificial`.
- Clean remote checkout: `/data3/ang.gao/robot_sim_c4_supervision_review2`, branch `review/c4-supervision-v2`, on the authorized `ang.gao@192.168.80.101` fallback.
- Local and remote HEAD/tree identities match. Both tracked working trees were checked clean before and after verification; candidate files were not edited.
- The complete AGENTS.md, collab/README.md, REVIEW_PRINCIPLES.md, accepted v3/v2 design requests, prior review, and implementation notes were read. Initial PowerShell decoding garbled the Chinese principles; the entire file was reread with explicit UTF-8.
- V3 design SHA-256: `291d4248633a9398eb4466b5e3a0ec514e04b810c5f6575886a4fab2804e1626`.

Relative to the base, the candidate adds ten files / 1,384 lines: three collaboration records, tests/support/__init__.py, three test-support modules, two CPU test files, and the standalone test supervisor. The repair delta from the rejected revision changes only lifecycle support, its regression tests, and two evidence records. No existing C0 production source, A/K protocol, worker allowlist, manifest, product integration, or activation gate changes. Searches found no reference to these test-only modules/supervisor in optical_worker, optics, or benchmarks.

This review covers only C4c1c steps 1–2: CPU lifecycle/guardian authority and the real fixed test-only S/V/E supervisor. Snapshot payloads remain diagnostic, not serialized capabilities. The trust boundary is a fixed single-threaded CPU child with trusted in-process calls, not malicious Python code or arbitrary IPC callbacks.

## Verdict: Request changes

The prior medium-severity F1 is closed by independently reproduced evidence. One low-severity, narrow static-validation regression remains. This verdict does not allege a new runtime ownership defect; it requests a small test-only correction so the frozen candidate passes its configured review checks.

### F2 — Low — New ownership regression test fails configured Ruff F821 checks

Location: `tests/unit/optics/test_sacrificial_lifecycle.py:311` and `:313`, within `test_failed_transfer_retains_entire_input_capsule_through_quarantine`; the same enclosing test later executes `del guardian` at line 329.

Trigger: Run the focused Ruff check under robot_sim / Ruff 0.16.1 using the repository's existing pyproject.toml configuration (which selects F rules).

Observed result: Ruff exits 1 with two `F821 Undefined name guardian` diagnostics at the nested helper's finish/fail calls. The nested helper closes over a local binding that the enclosing function explicitly deletes later. This is a static-analysis limitation of this closure/deletion shape; the test does invoke the helper before deleting the binding, and the runtime suite passes. It must not be reported as an observed NameError or failed retention assertion.

Consequence: The newly added regression file fails the configured static check, so this checkpoint is not fully validation-clean. Existing pytest runs miss it because they execute the valid call ordering, not Ruff's name-binding analysis.

Repair direction: Pass the guardian explicitly as a helper argument and use that parameter inside the helper, while retaining `del guardian` and the post-deletion GC/weakref assertions in the enclosing test. This avoids retaining an extra guardian reference after helper return and preserves the actual regression being proved. Do not weaken or remove the retention assertions or globally disable F821.

Required proof after repair: clean focused Ruff output, the six invalid-position/transfer cases and full four-file CPU suite passing on the newly frozen clean revision, and a fresh independent review of the delta. No production or scope expansion is needed.

## Prior F1: independently closed

The repaired finish/fail paths adopt the entire exact incoming tuple as a strong HOST_OWNER capsule before registering its members. A null member therefore cannot strand later owners. Member-registration failure sets ownership unknown and poison, consumes the active permit, and prevents emergency recovery or ordinary release. The production guardian carries the capsule into its process-quarantine registry.

The original external repro was rerun unchanged against the actual prior checkout, whose HEAD was independently confirmed as `a597e8a3d5346fd0a7c18491267274b7428cbfc4`. Both variants reproduced the original failure:

```text
finish before_quarantine POISONED unknown_owner True resource_count 1 finalizers ['last']
finish after_quarantine QUARANTINED resource_count 1 finalizers ['last']
fail before_quarantine POISONED unknown_owner True resource_count 1 finalizers ['last']
fail after_quarantine QUARANTINED resource_count 1 finalizers ['last']
```

The same unchanged script against this candidate produced:

```text
finish exception WorkerSetupProtocolError guardian cannot adopt a null owner
finish before_quarantine POISONED unknown_owner True resource_count 2 finalizers []
finish after_quarantine QUARANTINED resource_count 2 finalizers []
fail exception WorkerSetupProtocolError guardian cannot adopt a null owner
fail before_quarantine POISONED unknown_owner True resource_count 2 finalizers []
fail after_quarantine QUARANTINED resource_count 2 finalizers []
```

The six committed regression cases independently passed as part of the clean suite. They drop caller and traceback references, force GC, cover invalid first/middle/last members for both transfer methods, require unknown ownership and poison, reject E recovery, delete the outer guardian after quarantine, and prove both owners remain alive with no finalizer.

An additional reviewer-authored scratch script, outside the candidate, tested the six cases separately and asserted active-permit clearing, rejected stale completion/new setup/setup barrier/ordinary release, rejected emergency recovery, idempotent quarantine, and retained weakrefs after outer guardian deletion. It also exercised successful duplicate-owner tuples on both normal finish and known-ownership failure paths: one capsule plus one deduplicated owner, no premature release, and zero remaining resources/collected weakrefs after authorized release. All eight cases passed.

## Other inspected high-risk behavior

- Followed tuple retention through production `NativeWorkerResourceGuardian.adopt`, whole-graph release, and `_PROCESS_QUARANTINE`; no new product cleanup authority is introduced.
- Reviewed setup permit identity and state transitions, foreign/unknown scope, irreversible ownership poison, launch-latch versus returned counters, quiescence-before-seal, and release preconditions.
- Checked normal success and clean cancellation still release the added retention capsules; duplicate individual owners remain deduplicated.
- Reviewed S/V/E canonical frame handling, exact integer/session/role/sequence ACKs, deadline commitment before ACK, S-only disarm, and V/E remaining armed through terminal processing.
- Reviewed the fixed child entry, `-I -S`, inherited descriptors, parent-liveness handling, hard-exit exception path, bounded supervisor parser, and lack of arbitrary executable/module/callback IPC.
- The real supervisor suite exercised all three roles against normal, hang, raise, no ACK, wrong role, wrong sequence, and ACK EOF. Hung normal V follows two launch-latch entries and produces no inspection/release/finalizer marker.
- Repeated the original real parent-loss repro against the new candidate: after ACKed V entry, closing parent liveness killed the worker with return code -9, `reason=parent_lost`, terminal deadline armed, no restart, and no inspection/release/finalizer markers. New artifacts: `/tmp/c4-review-parent-loss-pbircwzp`.

## Executed commands and results

Transport: `ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`. The task packet already reported the primary transport unreachable and authorized this fallback; this review did not change repository histories or switch an implementation worktree. Remote interpreter: `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python`, Python 3.12.13. Pytest and Ruff used `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim` from the clean review2 checkout.

1. `git status --porcelain`, `git rev-parse HEAD`, `git rev-parse 'HEAD^{tree}'`, `git branch --show-current`, and base diff `--stat/--numstat/--name-status`: identities and bounded file surface confirmed, final tracked trees clean.
2. `git diff --check c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0 HEAD`: passed locally and remotely.
3. `python -m pytest -q tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py`: **78 passed in 11.36s**, including clean-worker integration. This was a fresh reviewer run, not an author result or reused output.
4. `python -m ruff check tests/support/sacrificial_lifecycle.py tests/support/sacrificial_watchdog.py tests/support/sacrificial_protocol_child.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tools/optical_sacrificial_test_supervisor.py`: **failed with exactly two F821 diagnostics**, lines 311 and 313 of the new lifecycle regression test. Ruff version independently obtained: **0.16.1**. The initial shell chain consequently did not execute its later diff/old-HEAD checks; those were explicitly rerun successfully in a separate command.
5. Original repro streamed without editing the candidate:

   `Get-Content -Raw -Encoding UTF8 'C:/Users/AG/新建文件夹/c4-supervision-review-v1-repro.py' | ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S - /data3/ang.gao/robot_sim_c4_supervision_review1'`

   Repeated identically with repository argument `.../robot_sim_c4_supervision_review2`. Original failure reproduced on rejected HEAD; retained owners observed on current HEAD. The script's exit 0 denotes diagnostic completion; the substantive evidence is the state/finalizer output above.
6. Used the same streaming command pattern with `c4-supervision-review-v1-parent-loss.py` and review2: **PASS**, real supervisor parent-loss behavior as above.
7. Used the same streaming command pattern with new external `C:/Users/AG/新建文件夹/c4-supervision-review-v2-invariants.py` and review2: **8 PASS lines**, six invalid-transfer retention/terminal cases and two valid duplicate-owner release cases, exit 0.
8. Read-only `rg` searches for test-support/supervisor references under optical_worker, optics, benchmarks: no matches. Reviewed repository Ruff configuration and all changed executable files plus the production guardian composition boundary.

## Limits, residual risk, and closed gates

No GPU driver, CUDA/Warp initialization, device probe, kernel, recorded bundle execution, product preparation, or benchmark result was run. OS supervision was verified on Linux/POSIX, not Windows. CPU descriptor fields and snapshots are not proof of real device/resource identity; full import-ledger coverage, actual setup-owner adoption, storage freeze/provenance, CPU numerical oracle, real submit-then-raise faults, and the complete GPU setup/launch/drain matrix remain later C4c1c work. This review does not claim hostile in-process Python capability isolation or robustness against arbitrary OOM/process-wide corruption.

Only scratch evidence outside the candidate was authored. Standard ignored test/tool caches may have been generated; no tracked candidate files were changed. No old outputs were counted as new execution evidence. The Ruff failure is distinct from an environment failure or runtime lifecycle failure.

No approval for this exact checkpoint is granted until F2 is corrected and independently re-reviewed. Even a later bounded approval may cover only steps 1–2 at its exact reviewed revision. GPU execution and full C4c1 acceptance, C4c2–C4c4, C4d/B3, C0/C4 joint shutdown, host payload, product preparation/GPU activation, submission/result/performance evidence, backend selection, a3b, baseline access, and shipping remain closed.
