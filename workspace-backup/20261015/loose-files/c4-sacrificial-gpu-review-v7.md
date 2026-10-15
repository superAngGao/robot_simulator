# C4c1c public event-family audit — independent code re-review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v7
Date: 2026-09-29
Status: in_review
Related Files: C4c1 ABI designs v1/v2/v3; supervision review v3; recording review v2; GPU-validation reviews v5/v6; acceptance-closure design/review v1 and review-followup v3
Owner Summary: Request changes. The original torch.cuda.Event.wait failure is independently closed, including zero original Python/native wait bodies and exact descriptor restoration. Distinct, documented public torch.Event and torch.Stream objects still execute real native event/query operations inside the audited leaf while it reports clean. This is a bounded F4 instrumentation gap, not proof of numerical corruption or unsafe resource release.

## Identity and requirements

- Fresh independent session: `/root/c4_sacrificial_gpu_review_v7`. No candidate authorship, implementation-conversation inheritance, or delegated agents.
- Frozen local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`, branch `implementation/c4-sacrificial-gpu`.
- Original checkpoint base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Last reviewed executable revision: `ac4c6ad17c7cd542ab4ab1840d976d6837023801`; v6 report-only archive `2e882ef`.
- Reviewed HEAD: `1c8b5f4d1c87be4c27ed3420cac65050a3dda7c4`.
- Reviewed tree: `c15d87f6366484917d5de699d43aaac54a1d7e70`.
- Independent runtime checkout: `/data3/ang.gao/robot_sim_c4_verify_1c8b5f4`, detached matching HEAD/tree, on authorized fallback `ang.gao@192.168.80.101:22`.
- Both local and remote trees were independently checked clean. All scratch work and this report are outside the candidate. Other worktrees were not changed.
- Whole-checkpoint diff: 31 files, 4,521 insertions / 89 deletions. Last-reviewed-code delta: seven files, 539 insertions / six deletions, including two collaboration records. Executable repair changes only the test audit, fixed child, GPU tests, new CPU descriptor tests, and fixed test-supervisor fault enumeration. No renderer math, guardian schedule, product protocol/allowlist, activation gate or operative collaboration policy changes in this repair.

Read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md` completely with explicit UTF-8. Read ABI design requests `q54-optical-lab-p12-4c4c1-warp-native-render-abi__design-request__codex__v1/v2/v3.md`, supervision review v3, recording review v2, preceding full GPU reviews v5/v6 and acceptance-closure design/review v1 plus follow-up v3. Oversized reads were narrowed to recover omitted text. Accepted supplement SHA-256: `9dec80201a66c96b47e8a03b8ccab8e74e968efb526371477c782f000c2c6f59`.

Retained contracts include no event work inside the actual recorded leaf (ABI v1 sections 10/14.4), actual forbidden-entry observations before original bodies (supplement section 3), same-bundle/two-pass/full-output sentinels, exact actual context restoration, and the supervised one-shot lifecycle. The supported trusted-process boundary is not a hostile-Python/native-memory sandbox. Author pass totals and prior verdicts are not independent evidence.

## Verdict: Request changes

### F4 residual — P2 / blocking: public device-generic Torch event/stream APIs bypass the audit

Locations: `tests/support/sacrificial_leaf_audit.py:43-55` (CUDA-only Torch targets), `:66-87` and `:133-142` (installed-family inventory), `:275-326` (audited execution/evidence); fixed negatives `tests/gpu/test_warp_sacrificial_render.py:169-232`.

The repaired `torch.cuda.Event`/`torch.cuda.Stream` wrappers are not Torch's only public CUDA-capable event/stream API. Installed Torch 2.9.1 documents separate `torch.Event(device='cuda')` and `torch.Stream(device='cuda')` types in `torch/_torch_docs.py:13672-13805`. Their `query`/`wait` methods dispatch directly through native built-in descriptors, not through the instrumented `torch.cuda` classes. These exact public types are absent from both the hook list and the five-family inventory. Consequently the inventory passes while these supported operations remain unobserved.

Independent real-device probe: `C:/Users/AG/新建文件夹/c4-sacrificial-gpu-review-v7-event-probe.py`, copied unchanged to `/data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py`. SHA-256 of both copies: `face20d85faf5635d43c354161b20ce323589f11b45b3ab48020ac2c41c8c0e1`.

The probe creates a real owning Warp stream, recorded empty-kernel command, and retained real `torch.Stream(device='cuda:0')` / `torch.Event(device='cuda:0')`. The generic event is recorded and synchronized before the audit. A fixed launch wrapper performs one ordinary public operation, then invokes the original Warp recorded launch. The unchanged candidate `RecordedLeafAudit.run()` surrounds it. A profiler observes the exact receiver's built-in method enter and return. Each mode runs in a disposable subprocess with a 90-second outer timeout. Resources remain retained and are synchronized during cleanup. No optical render bundle, candidate mutation, transported callback or arbitrary private/native API is involved.

Fresh results, all exit 0:

```text
PUBLIC_GENERIC_OPERATION_ESCAPED generic_query {'native_enter': 1, 'native_return': 1}
PUBLIC_GENERIC_OPERATION_ESCAPED generic_wait {'native_enter': 1, 'native_return': 1}
PUBLIC_GENERIC_OPERATION_ESCAPED generic_stream_query {'native_enter': 1, 'native_return': 1}
```

Each audit reports:

```text
entry_calls = native_launches = restoration_checks = 1
native_events = native_queries = 0
rejected = []
```

Exact-type assertions prove the receivers are `torch.Event` and `torch.Stream`, distinct from `torch.cuda.Event` and `torch.cuda.Stream`. Independent same-value `setattr` attempts fail with `TypeError: cannot set 'query'/'wait' attribute of immutable type 'torch.Event'/'torch.Stream'`; their original descriptor identities remain unchanged. The probe does not disable or replace any installed candidate hook. A successfully completed event is still an event operation and does not authorize a wait/query inside this leaf.

Consequence: a supported forbidden event/query operation can be inserted into the recorded wrapper without falsifying the current audit success assertions. This violates the retained no-event-work observation requirement. It does not establish that today's production Warp launch invokes these methods, that pixels are corrupted, or that owners are released early.

Why current checks miss it: all 25 fixed forbidden-operation controls use the enumerated Warp/CUDA-wrapper APIs. The setup inventory checks only the explicitly supplied classes, and `native_events`/`native_queries` count selected Warp entries. Neither is evidence about the distinct public generic Torch native methods.

Repair direction: cover the genuine supported public generic event/stream family, including already-created objects, using an enforceable pre-dispatch mechanism for their immutable C-method descriptors. Merely replacing a module constructor will not intercept existing instances, and ordinary `patch.object` on these immutable types cannot be the implementation. Include constructor/import surfaces coherently without claiming arbitrary native interception. Add fixed supervised negative requests over pre-created/adopted exact-target resources, require the specific operation observation before its original/native body, zero native render submissions, accurate restoration of instrumentation and context, full retained custody and the existing post-latch E/quarantine disposition. Rerun the unchanged counterexamples, original closed probes and affected positive/lifecycle/context matrix. Any proposed contract narrowing requires explicit reviewed authority, not an inventory definition that silently waives event work.

No other substantive blocking finding is raised.

## Original findings specifically closed

### v6 torch.cuda.Event.wait

Original probe is unchanged locally/remotely, SHA-256 `f9cd070c1bae24a22a1f430903e505775a6c642dfbd76f94572c047f18c3449d`. Its direct `torch_wait` mode now exits 1 at `RecordedLeafContamination: torch.cuda.Event.wait`, before its old clean-audit assertions. This exception alone is not counted as full closure proof.

New `original_wait_closure` mode executes that original file unchanged through `runpy`, retaining its original operation body and native profiler. An independent tracer captures the actual original public method and Warp launch code objects before hooks install. Fresh result:

```text
UNCHANGED_V6_OPERATION_CLOSED
  original_public_wait = 0
  original_launch = 0
  native_wait_enter = native_wait_return = 0
```

Audit: exactly `['torch.cuda.Event.wait']` rejected, `entry_calls=restoration_checks=1`, all native counters zero. Every resolved raw descriptor restores by `inspect.getattr_static` identity; the original public wait and Warp launch restore after the original script's own `finally` as well. The original profiler's `public_wait=1` diagnostic dynamically compares against the currently installed wrapper code, so it observes the denying wrapper, not entry into the original wait body; the captured original-code tracer above disambiguates this explicitly.

### v5 Warp Event.is_complete and native query hooks

Original v5 file is unchanged locally/remotely, SHA-256 `efc704d6fb8e3e11aba5c9a82c479b489d2f71315fa185bd7356f09deb0deb98`. Direct execution exits 1 at `RecordedLeafContamination: warp.Event.is_complete`. The new `original_query_closure` mode runs the previous v6 closure observer with exactly one diagnostic adaptation: `getattr(owner, name)` becomes `inspect.getattr_static(owner, name)` when checking restoration. This is necessary because the candidate now records raw classmethod/staticmethod descriptors. The v5 operation file/body remains unchanged and is executed through `runpy`.

Fresh result: `property_body=launch_body=query_observer=0`; exactly `['warp.Event.is_complete']` rejected; entry/restoration counts one and native counts zero; descriptors, native entries and original launch restored exactly. The unchanged v6 `native_queries` mode also passes all three native Warp query/elapsed entries, denying before their installed original-body sentinels and restoring identities. These are interception tests, not executions of those forbidden original native bodies.

## Retained invariants and inspected high-risk areas

- F1 remains closed: the verified source loader hashes and compiles the same byte buffer and never consults cached bytecode through parent `get_code()`. Fresh bootstrap cache namespaces precede the first repository helper import in supervisor and child. Original stale-cache probe still executes `VALUE='new'` and fails its deliberately old-value assertion; CPU source/loader/bootstrap tests pass. Audit remains in the fixed test-only source closure, not the product allowlist.
- F2 remains closed: exact NumPy destinations and CPU Warp views are adopted and custody-checked before explicit-stream input readback copies. Supervised S wait precedes the shared host comparison. Full-base production diff preserves a single dtype/shape/content/hash comparison policy; no second implementation or callback IPC is added.
- F3 remains intact: pre-seal snapshot/output graphs, both retained host graphs, full camera-ray sentinels and exact original bundle/command/parameter identities; same-stream copy/fill uses `sync_enter=False, sync_exit=False`; one final V drain precedes inspection of both passes. Schedule/latch is monotonic, tracks auxiliary attempts before entry and returns afterward, and rejects skipped/active/reordered/extra leaves and reseal/reopen. Nonempty resident-ray repetition is rejected by the fixed request schema.
- F5 remains intact: nondefault target-current fixture is distinct from the explicit execution stream; actual existing `_stream` metadata is compared by identity/handle before and in `finally`, without a lazy foreign-stream query. Target-1/ambient-0 retains the actual untouched `None` state. Deliberate context drift quarantines without target-only E synchronization. No nondefault foreign-stream fixture is claimed.
- New CUDA-wrapper controls create/adopt the genuine owning Warp stream, Torch external view, event, and IPC bytes before seal. Explicit stream handle/device checks precede execution and mismatch hard-exits through quarantine. Torch's event is recorded on that exact stream before S setup quiescence.
- Descriptor handling freezes/checks/restores raw descriptors; replacement preserves property/classmethod/staticmethod binding. CPU tests exercise normal and exceptional restoration plus original-body non-entry. Setup inventory is correctly labeled setup validation, not execution evidence; its incompleteness is the specific finding above.
- Guardian creation remains before CUDA-capable imports; whole-owner transfer capsules precede member validation; unknown ownership cannot become ordinary release after a later sync. S/V/E ACK precedes blocking entry, only S disarms, V/E remain terminal, and V failure never retries E. Fixed canonical requests admit no callback/module/native handle, and journal/parent-loss failure remains hard-exit/process-terminal.
- Product searches find no sacrificial imports and no recorded factory/bundle consumer outside its defining module. `optical_worker/manifest.py` GPU preparation and gate native result activation remain false. No product A/K/source policy/dispatcher, kernel, backend default, result or baseline artifact was changed.

## Fresh independent verification

Transport:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101
```

Primary 183 transport is recorded unavailable and the fallback is expressly owner-authorized. This reviewer did not retry primary or alter/reconcile another implementation tree. Runtime commands run from `/data3/ang.gao/robot_sim_c4_verify_1c8b5f4` with:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python
```

Observed Python 3.12.13, Torch 2.9.1+cu128, Warp 1.12.0. Direct conda `nvcc --version` reports 12.8.93; Warp's banner reports toolkit 12.9 / driver 13.3, which is not conflated with NVCC. Devices: ordinal 0 H200 NVL, ordinals 1/2 A100-SXM4-40GB. Local lint/cache-probe Python is 3.14.

1. Local/remote Git status/HEAD/tree/branch, whole-base and last-code diff inspection, and `git diff --check 7de02e89 HEAD`: exact identities and clean trees confirmed, checks pass.
2. Local `python -m ruff check` and `python -m ruff format --check` on all 16 Python files returned by `git diff --name-only 7de02e89 HEAD -- '*.py'`: all checks pass, 16 already formatted. No formatting writes.
3. Fresh CPU/POSIX selection:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_native_input_observations.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/benchmarks/test_native_readiness_authority.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
   ```

   **198 passed in 12.16s**, exit 0, no skips.
4. Original outside-candidate CPU state/admission probe streamed unchanged over SSH to `/data3/ang.gao/anaconda3/envs/robot_sim/bin/python -I -S - /data3/ang.gao/robot_sim_c4_verify_1c8b5f4`: exit 0, all six command-count 2/4/5 × one/two-pass schedules and ten malformed-request-before-child rejections. Wrong/order/active/early-drain/release/reopen cases are included. Source SHA-256 `34a7de70726e527fceecf2d6b9ce57de84e62012c4db6fdf960cb452bbca260a`. Not added to pytest counts.
5. `C:/Python314/python.exe -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py`: expected exit 1 at unchanged stale-value assertion; prints verified `EXECUTED_VALUE: new`. Separate source-loader repair evidence.
6. Original direct runtime commands:

   ```text
   python /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 torch_wait
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 native_queries
   ```

   First two expected exit 1 at their actual forbidden entries; third exit 0 for all three native hooks.
7. New independent probe commands, all exit 0:

   ```text
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 original_wait_closure /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 original_query_closure /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 generic_query
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 generic_wait
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py /data3/ang.gao/robot_sim_c4_verify_1c8b5f4 generic_stream_query
   ```

   Last three exit 0 means the escape assertions were confirmed, not acceptance.
8. Fresh GPU/regression selection:

   ```text
   python -m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py::test_real_recorded_leaf_rejects_forbidden_runtime_before_native_submit tests/gpu/test_warp_sacrificial_render.py::test_torch_wait_negative_uses_explicit_second_device_stream tests/gpu/test_warp_sacrificial_render.py::test_exact_recorded_bundle_reused_with_two_complete_sentinels tests/gpu/test_warp_sacrificial_render.py::test_recorded_render_targets_second_real_gpu_and_restores_ambient tests/gpu/test_warp_sacrificial_render.py::test_exception_restores_exact_current_stream_on_real_devices tests/gpu/test_warp_sacrificial_render.py::test_stream_drift_quarantines_without_target_only_sync tests/gpu/test_warp_sacrificial_render.py::test_reuse_partial_failure_retains_both_snapshots tests/gpu/test_warp_sacrificial_render.py::test_input_readback_resources_remain_in_guardian_across_failures tests/gpu/test_warp_sacrificial_render.py::test_setup_and_drain_failures_have_process_terminal_disposition tests/gpu/test_warp_shade_microfixtures.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_gpu_runtime.py
   ```

   **186 passed in 360.65s (0:06:00)**, exit 0, no skips. This includes 56 sacrificial cases: all 25 real forbidden-API negatives, explicit target-1 Torch wait rejection, all eight camera/zero-ray repeated-sentinel variants, real target-1/ambient-0 positive execution, both exception-context rows, both deliberate context drifts, auxiliary/second-pass partial failures, all five input-readback fault boundaries, and ten setup/drain/emergency/unknown-ownership failures. Remaining 130 tests cover independently derived microfixtures, recording/buffers and compatibility runtime. The real dual-device cases did not skip. Passing this suite does not close the separate generic-event counterexamples.
9. Final local/remote status remained empty, HEAD/tree still exactly match the identities above, and final diff checks passed. Fresh independent pytest total: **384 passed** (198 CPU/POSIX + 186 GPU/regression), no skips. All original/scratch probe outcomes are recorded separately, not counted as pytest passes.

## Limits and disposition

This is a bounded repair review, not a complete repository suite or fresh independent all-118-case sacrificial run. Author groups were used only for GPU scheduling, never as independent acceptance. CPU review overlapped author functional tests; independent GPU work began after all author groups finished. The short independent empty-kernel probes overlapped portions of the independent functional suite. No performance result is derived.

Ordinary installed runtime dependencies and permitted Warp JIT caches remain trusted. Probe kernels normally hit the shared cache; the original-wait runpy module variant freshly compiled. No cold-cache/compiler-proof claim is made. Native entry observations are actual CPython-to-native method enter/return events for the exact receiver, not a CUDA driver profiler or universal native sandbox. Local POSIX/GPU limitations were avoided by using the authorized Linux real-device environment.

No package installation, candidate edit, destructive cleanup, push/merge, gate change, benchmark result/baseline or shipping artifact occurred. Scratch probes are preserved outside the candidate. The CPU admission probe cleans its own temporary directories; normal ignored pytest/Python/JIT caches may be generated. Existing user worktrees were preserved. Tooling issues were limited to truncated combined reads (narrowed) and unavailable remote `rg` (replaced by grep); neither is a candidate failure.

Request changes for the demonstrated public generic Torch event/stream F4 residual. The original Warp property and torch.cuda.Event.wait findings are specifically closed; that does not close the retained event-family requirement. Freeze the bounded repair and obtain a fresh independent re-review. No complete C4c1c/C4c1/C4 checkpoint is approved. C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping, backend selection and default changes remain closed.
