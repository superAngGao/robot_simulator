# C4c1c recorded acceptance closure — independent code review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v5
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI design v1/v2/v3; supervision review v3; recording review v2; GPU-validation review v4; acceptance-closure design/review/implementation-note v1
Owner Summary: Request changes. F3 same-bundle repeated sentinels and F5 exact device/current-stream observation are implemented and freshly tested, and F1/F2 remain intact. Independent suites passed 423 tests with no skips. F4 is only partially repaired: a real supported Warp event-completion query runs inside the new audit interval and the audit reports no event operation or contamination. The retained contract explicitly forbids event queries. No checkpoint or product gate is approved.

## Identity, requirements, and scope

- Fresh independent session: `/root/c4_sacrificial_gpu_review_v5`. No implementation conversation inherited, no candidate authorship, no delegated agents.
- Local frozen candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`, branch `implementation/c4-sacrificial-gpu`.
- Original checkpoint base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Repair base: `95e1c9ffae819bd7add7993a9dea7f145c62ff7b`.
- Reviewed HEAD: `0c9e72b4cbb9a31dd5f6da1ae368794aa875547e`.
- Reviewed tree: `c91005f5c5710393b9c2f0af099b73c396f7662b`.
- Independent Linux execution: `/data3/ang.gao/robot_sim_c4_verify_0c9e72b`, detached matching HEAD/tree, authorized fallback `ang.gao@192.168.80.101:22`. Both candidate trees were clean on entry. Other dirty worktrees were not used or changed.
- Original base-to-HEAD: 26 files, 3,719 insertions / 89 deletions. Repair base-to-HEAD: 10 files, 789 insertions / 74 deletions. Repair executable changes are confined to test support, tests and the fixed test-only supervisor. Progress corrections do not open a gate or change operative review policy.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` completely with explicit UTF-8. Read the ABI v1/v2/v3 requirements, supervision v3, recording v2, full preceding GPU-validation v4 report, and acceptance-closure design/review/implementation note v1. Initial oversized reads were narrowed to recover omitted portions. The accepted supplement's independently computed SHA-256 is `9dec80201a66c96b47e8a03b8ccab8e74e968efb526371477c782f000c2c6f59`.

This review applies the supplement's one-seal/two-pass snapshot-and-fill schedule, one terminal V drain, nondefault target-current fixture and exact CPU-only observation of a foreign device's actual uninitialized `_stream`. It does not demand a foreign nondefault fixture or claim a hostile-Python/native-memory sandbox. Author notes, old approvals and the separate author regression log are not independent evidence.

## Verdict: Request changes

### F4 residual — P2 / blocking: event-completion query escapes the actual recorded-leaf audit

Locations: `tests/support/sacrificial_leaf_audit.py:17-29`, `:61`, `:116-118`, `:151-182`; negative-control matrix `tests/gpu/test_warp_sacrificial_render.py:169-197`.

Requirements: retained ABI v1 section 10 explicitly prohibits event construction/record/**query** inside the recorded-launch leaf; section 14.4 prohibits any observed event operation. The acceptance-closure supplement section 3 retains the event-work prohibition and requires observation of actual wrapped runtime entries. Its mention of creation/record/wait additions does not waive the predecessor's query prohibition.

The audit installs hooks for `Event.__init__`, stream event record/wait, and native `wp_cuda_event_record`, but neither `Event.is_complete` nor its native `wp_cuda_event_query` entry is monitored. Installed Warp 1.12 `context.py:3276` implements the public `Event.is_complete` property by directly calling `runtime.core.wp_cuda_event_query(self.cuda_event)`. The call bypasses all registered hooks.

**Independent real-device reproduction:** the outside-candidate script `C:/Users/AG/新建文件夹/c4-sacrificial-gpu-review-v5-event-probe.py` was copied to `/data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py` and executed against this exact candidate. SHA-256: `efc704d6fb8e3e11aba5c9a82c479b489d2f71315fa185bd7356f09deb0deb98`.

The probe creates a real owning stream, pre-created/recorded event and a real recorded empty-kernel `wp.Launch` on target 0. It does **not** construct or execute an optical render bundle. A fixed launch wrapper reads the supported public `event.is_complete` property and then invokes the original recorded launch. An observation wrapper forwards the native query to the installed native implementation and counts it. The unchanged candidate `RecordedLeafAudit.run()` surrounds the real recorded launch. A separate disposable child has a 90-second outer timeout; retained probe resources are synchronized before normal cleanup. No candidate file is changed and no callback is transported through the candidate's IPC.

Observed exit 0:

```text
CANDIDATE_AUDIT /data3/ang.gao/robot_sim_c4_verify_0c9e72b/tests/support/sacrificial_leaf_audit.py
PUBLIC_EVENT_QUERY_ESCAPED 1
AUDIT_REPORTED_CLEAN {
  'entry_calls': 1,
  'native_launches': 1,
  'native_events': 0,
  'rejected': [],
  'restoration_checks': 1,
  ...
}
```

This is a supported public API coverage counterexample, not arbitrary code disabling an installed audit hook or corrupting native memory. It proves that adding this event query to a launch wrapper can preserve every current audit success assertion. It does not claim the current production Warp `Launch.launch` actually performs that query, numerical corruption, or unsafe owner release.

Why tests miss it: the only event negative control calls `stream.record_event(event)`, which is covered. No negative control reads event completion. `native_events` observes only native event recording, so zero is not evidence of no event query.

Repair direction: extend the fixed test-only event-operation hooks to cover the supported query path (for example, the native `wp_cuda_event_query` reached by the property), including installation/restoration identity checks and observed rejection before the original body. Review the declared event-family surface for analogous gaps without claiming a universal native sandbox. Add a fixed supervised negative control that actually reads a retained event's completion property inside the audited launch wrapper. Require the specific hook observation, zero native render submissions for the rejected entry, no query-original-body execution, restoration on exception, retained graph and proper post-latch terminal disposition. Rerun the real positive/negative audit and affected lifecycle/context tests; the full clean matrix must not substitute for the missing assertion.

No additional substantive blocking finding is raised.

## Previous findings and inspected invariants

### F1 — verified source rather than stale bytecode: remains closed

`VerifiedRepositorySourceLoader.get_code()` reads the source byte buffer, checks that same buffer's digest and compiles it without invoking cache-reading parent `get_code()`. The supervisor and child create exclusive fresh bytecode-cache namespaces before the first repository helper import and disable writes. The new audit file is included in the fixed source closure; no product allowlist expansion occurs.

The original external cache repro was rerun unchanged with Windows isolated/no-site Python. It again printed current source and `EXECUTED_VALUE: new`, then exited 1 at its intentionally stale assertion `VALUE == 'old'`. This is expected repair evidence, not a failing candidate test. Fresh Linux tests cover timestamp and unchecked-hash collisions for modules/package initializers, changed source before compilation, wrong loader identity and bootstrap order.

### F2 — retained explicit readback: remains closed

`readback_input()` adopts the NumPy destination and its CPU Warp view, checks their exact guardian identities and only then enters the explicit target-stream copy. All such owners remain retained across subsequent leaves. The S-ACK-supervised stream wait precedes the shared CPU host-observation comparison. The ordinary validator still uses the same comparison policy through its lazy materializer; the full-base production diff adds no second dtype/shape/content/hash implementation. All five readback failure boundaries passed freshly in the real-GPU matrix.

### F3 — exact same bundle twice/full sentinels: closed on current code and fresh runtime evidence

Both camera executions retain the exact original bundle, command tuple, command objects and parameter/storage binding. Sentinel A includes origins/directions for camera bindings. Independently allocated first-pass GPU destinations and both sets of host destinations/views are adopted before setup quiescence and seal. Original storage is copied and then filled with sentinel B under the exact execution stream using `sync_enter=False, sync_exit=False`; no intervening host inspection or drain is introduced. Both snapshots are checked against each other and the independent CPU oracle after one terminal V drain. Zero-ray repetitions still execute both scalar-reset commands.

The guardian's schedule records render and auxiliary attempts before entry, advances only after return, rejects skipped/duplicate/wrong-role/active operations and never resets its latch or reseals. The new request schema explicitly disallows nonempty resident-ray two-pass requests; therefore the second-pass fill cannot corrupt nonempty resident ray inputs through the accepted fixed request surface. The separate independent admission probe confirms rejection before child creation, rather than treating an unreachable combination as an implementation bug.

Fresh full GPU execution includes all eight four-variant camera/zero-ray repeated-sentinel cases and both auxiliary/second-pass partial-submit failures; all passed.

### F5 — exact device/current-stream observation: closed on current code and fresh runtime evidence

Allowed target setup creates and adopts a distinct owning nondefault current-stream fixture and installs it with `sync=False`. The frozen context retains exact device/current-stream/explicit-stream objects and native handles. The observer reads `_stream` CPU metadata directly, preserving an untouched foreign device's actual `None` state; it does not call its lazy stream getter. Checks run before and in `finally` after audited launches and auxiliary work. Deliberate target-current or foreign metadata drift quarantines and hard-exits without target-only emergency synchronization. There is no claim of a nondefault foreign-device fixture.

Both real target-0/ambient-0 and target-1/ambient-0 rows executed successfully. Both injected-exception target rows and both deliberate drift negatives passed; the two-device test did not skip.

### Additional call-flow/lifetime and scope checks

- Guardian creation precedes CUDA-capable imports. Incremental owner-producing setup transfers the whole result capsule; unknown ownership cannot become ordinary release after a later sync.
- Freeze and full bundle revalidation precede execution; second-pass validation checks the unchanged binding. Snapshot objects are distinct and nonempty pointers differ from original arrays.
- Audit positive native launch arguments/count are derived from actual commands. Existing nine negative operations enter real APIs and reject before those original bodies. Hook teardown is in `finally` before legitimate auxiliary/drain work. These facts narrow F4 to the remaining coverage hole; the audit is not absent as in v4.
- Pre-call attempts and successful-return diagnostics remain separate. Auxiliary and second-pass failures are post-submit poison, retaining the full graph until exact-target E or quarantine.
- The supervisor commits deadlines before exact session/role/sequence ACK; only S disarms. V/E remain terminal. Missing/wrong ACK cannot enter the wait. Setup or validation wait failure does not retry E; parent loss kills the disposable child and journal failure hard-exits.
- CPU oracle construction is independent of migrated Warp math. Integer channels are exact; fixed float tolerances, distinct IDs/materials and explicit counter invariants remain in place.
- Product C0 A/K protocol, source policy, dispatcher and activation files are unchanged. `gpu_preparation_enabled` and `EXPECTED_NATIVE_RESULT_ACTIVATION.result_capable` remain false. Production searches show no imports of sacrificial helpers and no consumer of the recorded factory/bundle outside its defining module.

## Fresh independent verification

SSH transport:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

The packet identifies the primary 183 transport as timed out; this reviewer used the explicitly authorized 101 fallback, did not independently retry the primary and did not reconcile or alter an implementation worktree.

Observed environment: Python 3.12.13; Torch 2.9.1+cu128; Warp 1.12.0; conda NVCC 12.8.93. The actual Warp banner identifies target 0 as H200 NVL and targets 1/2 as A100-SXM4-40GB, CUDA Toolkit 12.9 / driver 13.3. The banner's toolkit is not conflated with NVCC's version. Local static/cache-probe interpreter is `C:/Python314/python.exe`.

Remote pytest commands ran from the clean candidate with:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python
```

1. Local/remote `git status --porcelain=v1`, `git rev-parse HEAD HEAD^{tree}`, branch checks, full-base and repair-base diffs. Exact identities and clean entry trees confirmed. `git diff --check 7de02e89c3e868e7554e5b1c385cfbbcf06570e1 HEAD` passed.
2. Local `python -m ruff check` and `python -m ruff format --check` over all 15 changed Python files returned by `git diff --name-only BASE HEAD -- '*.py'`: all checks passed; 15 files already formatted. No formatting writes.
3. Fresh CPU/POSIX command:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
   ```

   **192 passed in 11.75s**, exit 0, no skips.

4. Independent outside-candidate CPU state/admission probe:

   ```text
   Get-Content -Raw -Encoding UTF8 C:/Users/AG/新建文件夹/c4-sacrificial-gpu-review-v5-cpu-probe.py | ssh <above-options> ang.gao@192.168.80.101 '/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S - /data3/ang.gao/robot_sim_c4_verify_0c9e72b'
   ```

   Exit 0. All six command-count 2/4/5 × one/two-pass schedules succeeded while wrong/duplicate/active/early-release/early-drain/extra/reopen transitions were rejected. Ten malformed fixed requests (bad repetition values/types, nonempty resident repetitions, reuse-only faults with one pass, Boolean resident value and arbitrary callback field) returned 65 before a child journal existed. Source SHA-256: `34a7de70726e527fceecf2d6b9ce57de84e62012c4db6fdf960cb452bbca260a`. These checks are separate from pytest totals.

5. `C:/Python314/python.exe -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py`: expected exit 1 at the unchanged old-value assertion, with current verified source executed as detailed under F1.

6. Real-device event-query repro:

   ```text
   python /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py /data3/ang.gao/robot_sim_c4_verify_0c9e72b
   ```

   Exit 0; exact supported public API escape and misleading clean audit evidence reproduced as above. The scratch empty kernel was freshly compiled; no optical render bundle was executed outside the fixed optical supervisor.

7. Fresh independent full GPU/regression command:

   ```text
   python -m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_gpu_runtime.py
   ```

   **231 passed in 652.94s (0:10:52)**, exit 0, no skips. This includes the complete 101-case sacrificial matrix, not just the new tests: all three profiles/four shade variants, eight repeated-sentinel cases, nine real forbidden-API negative controls, real dual-device context/restoration and drift cases, resident/empty bindings, every ordinary command partial-submit position, setup/record/warmup faults, five input-readback faults, S/V/E ACK and hang/raise cases, parent-loss locations and journal failures. The remaining 130 checks cover hand-derived shade/first-hit microfixtures and recording/buffers/compatibility-runtime regressions. No test was interrupted. These tests passing do not close the separate event-query counterexample.

8. Final local and remote `git status --porcelain=v1`, HEAD and tree checks remained clean and exactly matched the identities above. Fresh independent pytest total is **423 passed** (192 CPU/POSIX plus 231 GPU/regression), no skips. Scratch state/admission checks and the event-query counterexample are recorded separately, not added to the pytest count.

## Limitations and precise disposition

The event probe establishes a contamination-observer false negative, not a production numerical, lifetime or actual stream-restoration failure. The trusted single-threaded fixed-process model is not an arbitrary Python/C++ security boundary. Installed runtime dependencies and ordinary permitted JIT caches remain trusted. No cold-cache performance, benchmark eligibility, foreign-nondefault-stream restoration or product lifecycle claim is made. No package was installed; no candidate implementation or tracked evidence was edited; no push, merge, result/baseline artifact or gate change was performed.

The author's separately running full-regression log is not counted as independent evidence. Concurrent functional-only GPU use is disclosed; no performance comparisons are derived. This report and both scratch probes are outside the frozen candidate. Temporary admission-test directories were automatically cleaned; preserved probe sources remain at the paths above. Existing workspace/user data was not deleted.

Reviewer tooling errors were limited to an initial SSH shell-quoting failure in a combined installed-source/version read, and two guessed gate-file paths that did not exist. Both were replaced with successful direct reads/searches. These were not candidate failures.

Request changes only for the substantiated F4 residual. Repair the monitored event-query path and its real supervised negative control, freeze a new candidate and obtain fresh independent re-review. This report does not approve complete C4c1c/C4c1/C4 or expand earlier narrowly bound checkpoints. C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping and backend selection remain closed.
