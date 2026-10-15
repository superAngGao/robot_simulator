# C4c1c event-family audit repair — independent code re-review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v6
Date: 2026-09-28
Status: in_review
Related Files: C4c1 ABI designs v1/v2/v3; supervision review v3; recording review v2; GPU-validation review v5; acceptance-closure design/review v1 and review-followup v2
Owner Summary: Request changes. The original Warp Event.is_complete counterexample is independently closed, including zero original getter/query/launch-body calls and exact hook restoration. A different supported event operation, torch.cuda.Event.wait, still enters and returns from its native implementation inside the audited recorded launch while the audit reports clean. This remains a blocking F4 coverage gap, not a product numerical or ownership failure.

## Identity and scope

- Fresh independent reviewer session: `/root/c4_sacrificial_gpu_review_v6`. No implementation conversation inherited, no candidate authorship, no subagents or implementation delegation.
- Frozen local candidate: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`, branch `implementation/c4-sacrificial-gpu`.
- Original checkpoint base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Last reviewed executable revision: `0c9e72b4cbb9a31dd5f6da1ae368794aa875547e`; its report-only archive is `5f6f511`.
- Reviewed HEAD: `ac4c6ad17c7cd542ab4ab1840d976d6837023801`.
- Reviewed tree: `c87136c15770d375ac110435ef2110738ddf75c1`.
- Independent runtime checkout: `/data3/ang.gao/robot_sim_c4_verify_ac4c6ad`, detached matching HEAD/tree, on authorized fallback `ang.gao@192.168.80.101:22`.
- Local and remote candidate trees were clean on entry. Final identity/tree checks are recorded below. Candidate files were not edited by this reviewer; scratch probes and this report are outside the candidate.
- Whole checkpoint: 28 changed files, 3,988 insertions / 89 deletions. Since last reviewed code: six files, 286 insertions / 17 deletions; two are the preserved report and follow-up note. Executable changes affect only the test audit, fixed child success assertions, fixed supervisor fault enumeration, and GPU assertions. No production, kernel, guardian schedule, operative collaboration rule, gate or source-policy change in this repair.

Read `AGENTS.md`, `collab/README.md`, and `collab/REVIEW_PRINCIPLES.md` fully with explicit UTF-8. Read ABI v1/v2/v3, supervision review v3, recording review v2, the full v5 report, and acceptance-closure design/review v1 plus follow-up v2. Oversized combined reads were narrowed and the omitted sections recovered. The v5 report's independently checked SHA-256 is `9dda6d5a9b193e7bc7745d6f738ad4b0c6e4abcc0547ec466d48d5f6de7f410c`.

Scope remains fixed, supervised C4c1c validation only. The accepted supplement retains v1 sections 10/14.3/14.4: no event work in the recorded leaf, same-bundle two-pass sentinels, and exact actual context observation. This review does not demand an arbitrary-Python/native security sandbox, foreign nondefault-stream setup, or product C4 lifecycle.

## Verdict: Request changes

### F4 residual — P2 / blocking: public Torch event wait bypasses the expanded event-family audit

Locations: `tests/support/sacrificial_leaf_audit.py:35-42` (Torch event/stream inventory), `:186-237` (run/evidence); negative controls `tests/gpu/test_warp_sacrificial_render.py:170-198`.

The new inventory hooks `torch.cuda.Event.query`, `elapsed_time`, `record`, `ipc_handle`, and `torch.cuda.Stream.wait_event/wait_stream`, but omits `torch.cuda.Event.wait`. Installed Torch 2.9.1 `torch/cuda/streams.py` implements `Event.wait(stream)` by directly calling its native superclass's `wait(stream)`. With an explicit stream it does not call any of the registered stream/event hooks. The native hooks in this audit cover Warp event recording and three Warp query/elapsed entries, not that Torch native wait.

This violates the retained v1 section 14.4 prohibition on event operations and the supplement section 3 requirement to reject actual forbidden runtime entries before their original bodies. It is an ordinary supported public API path; no installed hook is removed, overwritten or circumvented during the audited interval.

**Independent real-device counterexample:** outside-candidate probe `C:/Users/AG/新建文件夹/c4-sacrificial-gpu-review-v6-event-probe.py`, copied to `/data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py`, mode `torch_wait`. SHA-256 of both copies: `f9cd070c1bae24a22a1f430903e505775a6c642dfbd76f94572c047f18c3449d`.

The probe creates a real owning target-0 Warp stream and real recorded empty-kernel `wp.Launch`. Before the audit it creates a Torch `ExternalStream` view of that retained stream, creates/records a real Torch CUDA event, and synchronizes that event. A fixed launch wrapper calls `torch_event.wait(torch_stream)` and then the original recorded launch. `RecordedLeafAudit.run()` from the exact candidate encloses that wrapper. An independent Python profiler observes both the public method and the native built-in superclass wait entering and returning. All resources remain retained; synchronization occurs after the audit and in cleanup. The probe runs in a disposable subprocess with a 90-second outer timeout. It executes no optical render bundle and transports no callback/native handle through the candidate IPC.

Fresh observed exit 0:

```text
CANDIDATE_AUDIT /data3/ang.gao/robot_sim_c4_verify_ac4c6ad/tests/support/sacrificial_leaf_audit.py
TORCH_PUBLIC_AND_NATIVE_EVENT_WAIT_ESCAPED {
  'public_wait': 1, 'native_wait_enter': 1, 'native_wait_return': 1
}
AUDIT_REPORTED_CLEAN {
  'entry_calls': 1, 'native_launches': 1,
  'native_events': 0, 'native_queries': 0,
  'rejected': [], 'restoration_checks': 1, ...
}
```

Thus a real forbidden event operation can be added to the recorded wrapper without failing the current positive audit assertions. An already completed event remains an event operation; its readiness does not authorize a wait inside the leaf. This does not establish that the current production Warp launch performs this wait, corrupts render output, or releases owners early.

Why existing tests miss it: all twelve fixed negatives target other APIs; the three additions cover Warp event completion, Warp stream completion and Warp elapsed time. Hooking `Stream.wait_event` does not intercept `Event.wait`'s direct superclass call. Zero `native_events/native_queries` only reflects the enumerated Warp entries, not all event work.

Repair direction: cover the actual supported event-family entries coherently, including `torch.cuda.Event.wait`, while retaining callable/descriptor identity restoration and the fixed trusted-process boundary. Add a fixed supervised negative using a pre-created, guardian-retained real Torch event and explicit target stream/view, created outside the leaf and adopted before seal. Require exact rejection before its original/native wait body, zero native render submissions, restoration on exception, retained owners and the existing post-latch terminal E/quarantine disposition. Audit adjacent public family paths rather than asserting a universal native interceptor. Re-run the unchanged counterexample plus focused real positive/negative, lifecycle and context regressions after a new freeze.

No other substantive blocking finding is raised in this review.

## Original v5 repro and repaired paths

### Original Warp query counterexample — specifically closed

The original script is unchanged at both `C:/Users/AG/新建文件夹/c4-sacrificial-gpu-review-v5-event-probe.py` and `/data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py`; independently verified SHA-256 `efc704d6fb8e3e11aba5c9a82c479b489d2f71315fa185bd7356f09deb0deb98`.

Executed directly against this candidate, it exits 1 at `RecordedLeafContamination: warp.Event.is_complete`, at the original public-property read. It does not reach its old clean-audit assertion. That exception alone is not the full closure proof.

The new independent probe's `original_closure` mode executes the unchanged original source via `runpy`, with a call tracer only during `RecordedLeafAudit.run`. It verifies:

```text
ORIGINAL_UNCHANGED_BODY_FAILS_CLOSED {
  'property_body': 0, 'launch_body': 0, 'query_observer': 0
}
```

The actual original Warp property getter, original Warp `Launch.launch` body and original probe's forwarding native-query observer are never called. Audit evidence has `entry_calls=restoration_checks=1`, `native_launches=native_events=native_queries=0`, and exactly `['warp.Event.is_complete']` rejected. Every resolved hook restores to the exact object captured by the audit. Additionally, both property descriptors, Warp launch and all three native query/elapsed entries restore to their pre-probe identities after the original script's own `finally`. Exit 0. This is fresh body-and-state evidence, not replacement of the old assertion with a favorable assertion alone.

The `native_queries` mode separately exercises the three installed native hooks (`wp_cuda_event_query`, `wp_cuda_stream_query`, `wp_cuda_event_elapsed_time`) from real recorded wrappers. Each increments the native attempted-query count, rejects the exact entry, does not call its preinstalled original-body sentinel, performs zero native launches, and restores original callable identities. These are wrapper interception tests; they deliberately do not execute those original native query bodies. Exit 0 for all three.

The new property implementation reads descriptors on the class, wraps only the getter, preserves setter/deleter/doc metadata, installs with `ExitStack`, and restores in `finally`. Supported properties and native query identities are frozen and checked before installation. The new Torch wait gap does not invalidate the specific original Warp property repair.

## Retained findings/invariants

- **F1 remains closed:** verified loader compiles the same source bytes whose digest it checks; it never calls cache-reading parent `get_code`. Supervisor and child create exclusive fresh bootstrap cache namespaces before repository helper imports and disable bytecode writes. The audit module remains in the fixed repository source closure; the product allowlist is unchanged. Fresh source/cache/loader/bootstrap CPU regressions pass.
- **F2 remains closed:** readback adopts the exact NumPy destination and CPU Warp view before the explicit-stream copy and asserts guardian custody before entry. The S-ACK wait precedes shared CPU comparison. Whole-checkpoint production diff factors comparison into one common dtype/shape/content/hash policy; it does not duplicate numerical validation. Fresh readback fault validation is included below.
- **F3 remains closed:** same frozen bundle/command tuple and binding execute twice, with distinct GPU snapshots, both retained host inspection graphs, full camera-ray/output sentinel coverage and zero-ray resets. Copies/fills use the exact stream without implicit cross-stream waits; only the final V path inspects both snapshots. The one-seal schedule tracks attempts before calls and returns afterward, rejects skipped/active/duplicate/extra leaves and never clears latch/poison. Fresh all-eight repeated-sentinel and auxiliary/second-pass failure tests are included below.
- **F5 remains closed:** exact target nondefault-current fixture is distinct from the explicit stream. The observer reads actual `_stream` CPU metadata, retaining `None` for the untouched foreign device, and compares identity/handles before and in `finally` after execution. Drift quarantines without a target-only emergency sync. Fresh real target-1/ambient-0, both target exception paths and both drift negatives are included below; no foreign nondefault fixture is claimed.
- Guardian creation still precedes CUDA-capable imports; whole-result capsules precede per-member ownership validation. Unknown scope cannot be upgraded into ordinary release by a later sync. S/V/E deadline/ACK ordering, S-only disarm, terminal V/E, source admission and hard exit remain unchanged by the current repair.
- Searches show no sacrificial-helper imports in production and no production consumer of the recorded factory/bundle outside its defining module. Product GPU preparation and `native_result_activation.result_capable` remain false. Current repair changes no product dispatcher, manifest, allowlist, source policy, A/K protocol, backend default, kernel, result or baseline artifact.

## Fresh independent verification

Transport:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

The packet identifies primary `tileops-dev`/183 as timed out and explicitly authorizes this fallback. I did not retry primary, alter another worktree or certify primary-host ancestry.

Observed Python 3.12.13; installed Torch 2.9.1+cu128; Warp 1.12.0; NVCC 12.8.93. Fresh Warp banners identify target 0 H200 NVL and targets 1/2 A100-SXM4-40GB, toolkit 12.9 / driver 13.3. The Warp banner is not conflated with the separately checked NVCC version. Runtime commands run from the clean remote candidate with this prefix:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python
```

1. Local/remote status, branch and `git rev-parse HEAD 'HEAD^{tree}'`; full-base and last-code diffs; `git diff --check 7de02e89 HEAD`: matching exact identities, clean trees and diff checks passed.
2. Local `python -m ruff check` and `python -m ruff format --check` on all 15 changed Python files returned by `git diff --name-only 7de02e89 HEAD -- '*.py'`: all checks passed; 15 files already formatted. No formatting writes.
3. Fresh CPU/POSIX suite:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
   ```

   **192 passed in 12.05s**, exit 0, no skips.
4. Unchanged outside-candidate v5 CPU probe, streamed through SSH to `python -I -S - /data3/ang.gao/robot_sim_c4_verify_ac4c6ad`: exit 0, all six 2/4/5-command × one/two-pass schedules and ten malformed-request-before-child rejections. Wrong/order/active/early-drain/release/reopen transitions are exercised. SHA-256 `34a7de70726e527fceecf2d6b9ce57de84e62012c4db6fdf960cb452bbca260a`. Separate from pytest totals.
5. Original probe command: `python /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py /data3/ang.gao/robot_sim_c4_verify_ac4c6ad`: expected exit 1 at the newly intercepted property, as detailed above.
6. Independent probe commands, all fresh exit 0:

   ```text
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/robot_sim_c4_verify_ac4c6ad original_closure /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/robot_sim_c4_verify_ac4c6ad torch_wait
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/robot_sim_c4_verify_ac4c6ad native_queries
   ```

   `torch_wait` exit 0 means its audit-escape assertions were reproduced, not candidate acceptance. The other modes establish the bounded repair facts above.
7. Fresh independent GPU/regression command:

   ```text
   python -m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py::test_real_recorded_leaf_rejects_forbidden_runtime_before_native_submit tests/gpu/test_warp_sacrificial_render.py::test_exact_recorded_bundle_reused_with_two_complete_sentinels tests/gpu/test_warp_sacrificial_render.py::test_recorded_render_targets_second_real_gpu_and_restores_ambient tests/gpu/test_warp_sacrificial_render.py::test_exception_restores_exact_current_stream_on_real_devices tests/gpu/test_warp_sacrificial_render.py::test_stream_drift_quarantines_without_target_only_sync tests/gpu/test_warp_sacrificial_render.py::test_reuse_partial_failure_retains_both_snapshots tests/gpu/test_warp_sacrificial_render.py::test_input_readback_resources_remain_in_guardian_across_failures tests/gpu/test_warp_sacrificial_render.py::test_setup_and_drain_failures_have_process_terminal_disposition tests/gpu/test_warp_shade_microfixtures.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_gpu_runtime.py
   ```

   **172 passed in 273.67s (0:04:33)**, exit 0, no skips. This includes 42 sacrificial cases: all twelve forbidden-API controls (including the three repaired query/elapsed paths), all eight four-variant camera/zero-ray repeated-sentinel cases, real target-1/ambient-0 execution, both exception-context rows, both deliberate context-drift cases, auxiliary and second-pass partial failures, all five readback failure boundaries, and ten setup/drain/emergency raise/hang/ownership cases. The remaining 130 tests cover hand-derived microfixtures, recording/buffer contracts and compatibility runtime. A passing matrix does not close the independently reproduced Torch event-wait gap.
8. Final local/remote `git status --porcelain=v1` remained empty; both HEAD/tree still exactly match the identities above. Final remote diff check passed. Fresh independent pytest total is **364 passed** (192 CPU/POSIX + 172 GPU/regression), no skips. Scratch probe results are separate and not added to pytest counts.

## Limits and disposition

This is a bounded re-review of the repair and critical invariants, not a fresh full repository or all-104-case sacrificial run. The author's three functional-only groups completed 116/33/44 passes (193 aggregate); those logs were read for scheduling and are explicitly **author evidence**, not included in independent totals. Independent CPU checks overlapped those groups. Independent GPU probes/suite began after all three groups finished, to avoid their watchdog contention. No performance measurements or comparisons are derived from any run.

Ordinary installed runtime/stdlib dependencies and permitted Warp JIT caches remain trusted. The unchanged original kernel probe used cache; its `runpy` module variant freshly compiled. No cold-cache performance or compiler-proof claim is made. The profiler observes the actual Python-to-native wait binding enter/return; no driver profiler or arbitrary native-call security boundary is claimed. The new failure is an instrumentation false negative, not demonstrated output corruption, unsafe cleanup or a normal production call to Torch wait.

No package installation, candidate edit, destructive cleanup, push, merge, gate change, benchmark result/baseline or shipping artifact was performed. Scratch scripts are preserved outside the candidate. Temporary CPU admission directories are cleaned by that probe; ordinary ignored test/JIT caches may be generated. Existing user worktrees were preserved.

Reviewer tooling issues: an initial overly broad workspace listing was truncated; a guessed root `REVIEW_PRINCIPLES.md` path was corrected to `collab/REVIEW_PRINCIPLES.md`; an SSH process-list command lost shell quoting around a regex alternation and was replaced by `pgrep`/direct log reads. These were resolved inspection issues, not candidate failures.

Request changes for the substantiated F4 Torch event-wait gap. The exact original Warp property finding is closed, but the retained no-event-work audit requirement is not. Freeze the bounded repair and obtain fresh independent re-review. This report approves no complete C4c1c/C4c1/C4 checkpoint and does not expand prior narrow approvals. C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping, backend selection and default changes remain closed.
