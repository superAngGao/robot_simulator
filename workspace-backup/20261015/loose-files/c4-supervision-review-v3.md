# C4c1c steps 1–2 independent final code re-review

Initiative: q54-optical-lab-p12-4c4c1c-supervision
Stage: review
Author: codex (independent reviewer)
Version: v3
Date: 2026-09-28
Status: accepted
Related Files: C4c1 ABI design v3/v2; supervision reviews v1/v2; tests/support/sacrificial_lifecycle.py; tests/unit/optics/test_sacrificial_lifecycle.py; test-only S/V/E supervisor
Owner Summary: Approve only C4c1c steps 1–2 at the exact revision below. F2 is independently closed without weakening the GC regression; F1 remains closed. Focused lint/format, six explicit GC cases, and the 78-test CPU/fault/C0 suite pass. No GPU or product gate is opened.

## Identity and exact candidate

- Fresh independent reviewer session/task: `/root/c4_supervision_review_v3`. No implementation conversation was inherited, no candidate code was authored, and no subagents were used.
- Base: `c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0`.
- Previous reviewed revision: `1ac90a2a4a067464845de3db5bfac2fa0f7e0d25`.
- Reviewed HEAD: `3c81fa14a71ee69f7a112170a2f436c5549e6725`.
- Reviewed tree: `bb83df3b3ee6e77adb9b0e39d0c899895bfc29c6`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial`, branch `implementation/c4-sacrificial`.
- Remote checkout: `/data3/ang.gao/robot_sim_c4_supervision_review3`, branch `review/c4-supervision-v3`, host `ang.gao@192.168.80.101`.
- Both local and remote HEAD/tree identities were independently checked and matched. `git status --porcelain` was empty before and after verification in both checkouts. No tracked or untracked candidate changes were made by this reviewer; ignored test/tool caches may have been generated remotely.
- Read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, accepted v3/v2 design requests, and v1/v2 review reports. Initial PowerShell decoding garbled the Chinese principles; they were reread completely with explicit UTF-8.
- Accepted v3 design SHA-256: `291d4248633a9398eb4466b5e3a0ec514e04b810c5f6575886a4fab2804e1626`.

## Verdict: Approve

No substantive findings remain in this bounded re-review. Approval binds only the HEAD/tree above and only C4c1c steps 1–2: pure-CPU guardian/lifecycle authority and the separate, real fixed test-only S/V/E supervisor. This is not full C4c1c or C4c1 acceptance.

### Delta and F2 closure

The actual delta from the previous revision has exactly two files: the 108-line v2 review record and four added/four removed lines in `tests/unit/optics/test_sacrificial_lifecycle.py`. The helper now takes `lifetime` explicitly, uses that parameter for both transfer calls, and is called with `guardian`. No implementation, production source, protocol, gate, lint configuration, or assertion changed.

I reproduced the previous revision's two F821 errors by streaming its test-file blob to the current repository-configured Ruff checker. Current focused Ruff passes. This is a closed static-validation regression, not a claim that the old runtime test raised NameError.

The helper argument is a normal temporary call reference, not a persistent closure/default reference. It returns only weakrefs; the caught exception is not retained. The enclosing `del guardian`, subsequent `gc.collect()`, owner-liveness checks, and empty-finalizer checks remain intact. All six finish/fail × first/middle/last-invalid-position cases pass individually. Thus the lint fix preserves the safety counterexample rather than bypassing it.

The committed v2 review and original external `C:/Users/AG/新建文件夹/c4-supervision-review-v2.md` have the same SHA-256, `eafc9ddfa6f52796961bd685269b58ccebedbd3ff13147d01545328491f6b33b`, confirming the record is verbatim.

### F1 remains closed

I followed whole-tuple adoption before member validation through the production guardian's strong owner list and `_PROCESS_QUARANTINE`. Invalid member registration preserves the entire input capsule, poisons unknown ownership, consumes the permit, and forbids emergency recovery to ordinary release. Normal success and clean cancellation retain release coverage in the executed suite.

The original unchanged external F1 repro was executed against this exact candidate with isolated, no-site Python. Both variants produced:

```text
finish exception WorkerSetupProtocolError guardian cannot adopt a null owner
finish before_quarantine POISONED unknown_owner True resource_count 2 finalizers []
finish after_quarantine QUARANTINED resource_count 2 finalizers []
fail exception WorkerSetupProtocolError guardian cannot adopt a null owner
fail before_quarantine POISONED unknown_owner True resource_count 2 finalizers []
fail after_quarantine QUARANTINED resource_count 2 finalizers []
```

Its exit 0 means diagnostic completion; the retained-owner/finalizer output is the substantive evidence. The six committed regression cases additionally verify retention after deleting the outer guardian. I did not rerun F1 on the original rejected revision in this round; the original failure and prior repair history were read as historical evidence, not counted as new execution.

## Inspected high-risk boundaries

- Inspected all six executable support/test/supervisor files, the full base file surface, and production guardian composition. Base-to-HEAD adds 11 files / 1,492 lines, limited to four collaboration records, test support, two unit-test files, and the standalone test supervisor. No existing production C0/A/K/allowlist/manifest or activation source changes.
- Checked permit identity, incremental strong adoption, irreversible unknown ownership, attempted-versus-returned launch counts, quiescence before seal, release conditions, and process quarantine retention.
- Checked exact session/role/integer-sequence ACK validation, supervisor-clock deadline commitment before ACK, S-only disarm, and V/E remaining armed through the terminal child path.
- Checked fixed `-I -S` child invocation, bounded canonical frames, enumerated fault scenarios, parent-liveness handling, and hard exit on failure. No caller-supplied executable/module/callback crosses IPC.
- The executed real supervisor matrix covers S/V/E normal, hang, raise, no ACK, wrong role, wrong sequence, and ACK EOF. Normal V hang follows two launch latches and asserts no inspection, release, or finalizer marker. Wrong/missing ACK cases assert the blocking callable was not entered.
- Clean C0 worker integration freshly passed, including disabled GPU preparation and forbidden raw-launch requests. Read-only searches found no references to the new sacrificial support/supervisor modules under optical_worker, optics, or benchmarks.

## Executed commands and results

Remote transport: `ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`. The packet reports the primary host unreachable and explicitly authorizes this fallback; I did not switch or reconcile an implementation worktree. Commands ran from the clean review3 checkout through `/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim`. Observed Python **3.12.13**, pytest **9.1.1**, Ruff **0.16.1**. Pytest reports the interpreter as `/home/ang.gao/anaconda3/envs/robot_sim/bin/python`; the external repro used the supplied `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python` path.

1. `git status --porcelain`, `git branch --show-current`, `git rev-parse HEAD 'HEAD^{tree}'`, base `git diff --stat/--name-status`, and prior-to-current `git diff/--numstat`: exact identity, clean trees, and bounded delta confirmed. The initial local unquoted tree expression was misparsed by PowerShell; it was rerun successfully with quotes. This was a command quoting error, not a repository failure.
2. `git diff --check c8c675accb18ecb5fe91c0761bb3aafb73a5d2b0 HEAD`: passed locally and remotely.
3. `python -m ruff check tests/support/sacrificial_lifecycle.py tests/support/sacrificial_watchdog.py tests/support/sacrificial_protocol_child.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tools/optical_sacrificial_test_supervisor.py`: **All checks passed**, exit 0.
4. `git show 1ac90a2a4a067464845de3db5bfac2fa0f7e0d25:tests/unit/optics/test_sacrificial_lifecycle.py | /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -m ruff check --stdin-filename tests/unit/optics/test_sacrificial_lifecycle.py -`: expected exit 1, exactly **two F821** diagnostics at lines 311 and 313. No candidate files were replaced for this negative control.
5. `python -m ruff format --check` with the same six file paths as item 3: **6 files already formatted**, exit 0; no formatting writes.
6. `python -m pytest -q tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py`: **78 passed in 10.97s**, exit 0, with no skips. Fresh reviewer execution, not reused author output.
7. `python -m pytest -v tests/unit/optics/test_sacrificial_lifecycle.py -k failed_transfer_retains_entire_input_capsule_through_quarantine`: **6 passed, 22 deselected in 0.10s**, explicitly observing each parameter case.
8. Original repro streamed without modification:

   `Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4-supervision-review-v1-repro.py | ssh -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S - /data3/ang.gao/robot_sim_c4_supervision_review3'`

   Exit 0 with the retained-capsule observations above.
9. `Get-FileHash -Algorithm SHA256` for the design and both copies of the v2 report: identities as recorded above. `rg` for sacrificial module/supervisor references in production directories returned no matches.

## Limits, residual risk, and closed gates

No GPU driver, CUDA/Warp initialization, device probe, kernel, recorded bundle execution, product preparation, or benchmark result was run. This verifies CPU state/ownership and actual Linux/POSIX process supervision, not real device/resource identity or Windows inherited-descriptor behavior. The trusted, single-threaded fixed child boundary is not protection against hostile in-process Python or arbitrary OOM/process-wide corruption.

The previous standalone parent-loss and eight-case scratch scripts were not rerun in this narrow test-helper-only round; their old outputs are not claimed as new evidence. The unchanged supervisor source and full existing fault suite were freshly inspected/executed. No additional required evidence gap was identified for steps 1–2.

Full GPU setup/import-ledger proof, actual setup-owner adoption, storage freeze/provenance, independent CPU numerical oracle, real submit-then-raise faults, and the complete setup/launch/drain matrix remain later C4c1c work. GPU execution and full C4c1 acceptance, C4c2–C4c4, C4d/B3, C0/C4 joint shutdown, host payload, product preparation/GPU activation, submission/result/performance evidence, backend selection, a3b, baseline access, and shipping remain closed.

This report was authored only at `C:/Users/AG/新建文件夹/c4-supervision-review-v3.md`, outside the frozen candidate. Recording it verbatim in a separate evidence-only commit does not extend this approval to any executable or operative-rule changes.
