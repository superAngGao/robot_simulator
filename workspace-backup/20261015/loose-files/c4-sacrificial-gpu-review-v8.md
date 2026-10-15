# C4c1c generic native event audit — independent code re-review

Initiative: q54-optical-lab-p12-4c4c1c-gpu-validation
Stage: review
Author: codex (independent reviewer)
Version: v8
Date: 2026-09-29
Status: accepted
Related Files: C4c1 ABI v1/v2/v3; supervision review v3; recording review v2; GPU-validation reviews v5/v6/v7; acceptance-closure design/review v1 and review-followup v4
Owner Summary: Approve the exact bounded C4c1c supervised validation checkpoint at 6d57fa789cb50e4191239a4d00f628f25824bb46. Native debugger evidence independently closes the original generic Torch counterexamples before their C bodies, including calibrated restored calls and cached aliases. Fresh independent suites pass 472 tests with no skips. No later-stage or product gate is opened.

## Identity and scope

- Fresh independent session: `/root/c4_sacrificial_gpu_review_v8`. No implementation conversation inherited, no candidate authorship, and no delegated agents.
- Original checkpoint base: `7de02e89c3e868e7554e5b1c385cfbbcf06570e1`.
- Last reviewed executable revision: `1c8b5f4d1c87be4c27ed3420cac65050a3dda7c4`; `ea6845d` archives the unchanged v7 report only.
- Frozen HEAD: `6d57fa789cb50e4191239a4d00f628f25824bb46`.
- Tree: `5b22a629b5cf0787743b79cf8f522a9fe2a20bb7`.
- Local checkout: `C:/Users/AG/新建文件夹/p12_4_c4_sacrificial_gpu`, branch `implementation/c4-sacrificial-gpu`.
- Runtime checkout: `/data3/ang.gao/robot_sim_c4_verify_6d57fa7`, detached matching HEAD/tree, owner-authorized fallback `ang.gao@192.168.80.101:22`.
- Local and remote status were clean before and after verification, with the same exact HEAD/tree. No candidate file was edited. Scratch probes, debugger commands and this report are outside the candidate.
- Whole-base delta: 33 files, 5,127 insertions / 89 deletions. Last-reviewed-code delta: seven files, 622 insertions / 16 deletions, including v7's unchanged report and the current follow-up. The current repair's executable changes are confined to the audit, fixed child, fixed supervisor fault enumeration, and CPU/GPU tests. No renderer math, product dispatcher/protocol/source allowlist, gate, kernel, or shared workflow rule changes in this repair.

Read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely; Chinese text was reread with explicit UTF-8 after the first default-encoding read. Read the supplied accepted requirements, the acceptance supplement/design review, current follow-up, and v5/v6/v7 reports. Checked the relevant retained ABI v1/v2 requirements, especially actual leaf event-work prohibition, independent CPU truth, reuse and context restoration. Oversized combined reads were narrowed to recover omitted portions. Author notes and historical pass totals are not independent evidence.

Accepted supplement SHA-256: `9dec80201a66c96b47e8a03b8ccab8e74e968efb526371477c782f000c2c6f59`. Preserved v7 report SHA-256: `15ea2b8a299f3439224e020fbfadf2af62f463234d146589d3dfc76d4b17b0ed`.

The boundary is fixed-source, single-threaded, supervised test instrumentation. It is not a hostile-Python/native-memory sandbox, arbitrary plugin/callback execution, product execution authority or a universal interceptor of calls originating inside native code.

## Verdict: Approve

No substantive blocking finding, new non-blocking finding, or required evidence gap remains for this bounded checkpoint. Approval binds only the exact HEAD/tree and C4c1c scope above. F4's original Warp-property, CUDA-wrapper wait, and generic Torch method counterexamples are independently closed; F1/F2 and the accepted F3/F5 closure remain intact. This is not automatic approval of C4c1 as a whole, C4, a later stage, benchmark eligibility or product activation.

## Original F4 counterexamples: native body proof

### v7 generic Torch query/wait/stream-query

The repair installs a CPython `sys.monitoring` CALL callback while the actual recorded leaf is active. Its constructor checks and native descriptor/receiver checks cover the installed distinct `torch.Event` and `torch.Stream` types without replacing their immutable native types. The exact three mutable CUDA wrapper constructors are delegated only after setup verifies their own frozen `__new__` hooks exist. Generic cached type aliases, bound native methods, unbound descriptors and class methods use the same observed dispatch mechanism.

The initial outside-candidate v8 observer deliberately reran unchanged original v7 code through `runpy`. Its first attempt expected zero legacy `sys.setprofile` `c_call` events and failed that diagnostic: the observed count was one `c_call`, zero `c_return`, with `RecordedLeafContamination`. This is **not** evidence that the native body ran. CPython sends the legacy pre-call notification before the separate CALL monitor refuses dispatch. I corrected only the new observer's interpretation, left the original v7 file unchanged, and independently verified actual C bodies with native debugger breakpoints.

Installed `libtorch_python.so` retains these native entry symbols, independently located with `nm -C`:

```text
THPEvent_query(_object*, _object*)
THPEvent_wait(_object*, _object*, _object*)
THPStream_query(_object*, _object*)
THPEvent_pynew(_typeobject*, _object*, _object*)
THPStream_pynew(_typeobject*, _object*, _object*)
```

GDB launched each owned disposable Python process under a 90-second outer timeout, with pending breakpoints resolved when Torch loaded. Breakpoints counted actual native function entries, not profiler notifications. In every original scenario the observer asserted exact rejection, zero original Warp launch-body entries, zero native Warp launch/event/query counts, one audit entry/restoration, one generic attempted CALL/restoration, exact raw descriptor restoration, and unchanged monitoring slot ownership/event masks. It then executed the **same method on the same live native receiver outside the restored guard** to calibrate the native breakpoint. The restored operation returned normally. All inferiors exited normally.

| Unchanged original operation | Guarded native method body entries | Restored calibration body entries | Generic event/stream constructor entries |
| --- | ---: | ---: | --- |
| `generic_query` | 0 | 1 at `THPEvent_query` | 1 / 1, setup only |
| `generic_wait` | 0 | 1 at `THPEvent_wait` | 1 / 1, setup only |
| `generic_stream_query` | 0 | 1 at `THPStream_query` | 1 / 1, setup only |

The zero guarded count follows from the total native count of exactly one and the independently successful calibrated call after restoration; the other two monitored method totals were zero in each process. This establishes pre-body rejection rather than merely observing an exception or relabeling a profiler notification.

### Cached/native aliases and restoration

A separate real-device v8 mode pre-created and retained generic Torch resources, then exercised eight fixed operations in successive fresh `RecordedLeafAudit.run()` calls: cached and unbound event query, cached and unbound event wait, cached and unbound stream query, and cached event/stream constructor aliases. Every operation was rejected at its expected generic target; no native Warp launch occurred, descriptors and monitoring state restored after every exception. A final normal recorded empty-kernel launch succeeded with exactly one actual native Warp launch.

GDB totals for that entire process were event query = event wait = stream query = **0**, and event constructor = stream constructor = **1** each, solely for the successfully constructed setup owners. Thus the two cached constructors did not enter their C bodies either. Constructor breakpoints are positively calibrated by those same setup calls; method breakpoint calibration is recorded above. No optical render bundle was run by these scratch probes.

### v5/v6 remain closed

The unchanged v7 `original_wait_closure` observer executes original v6 `torch_wait` code and observes `original_public_wait=0`, `original_launch=0`, `native_wait_enter=native_wait_return=0`, exact rejection `torch.cuda.Event.wait`, zero native Warp work and exact restoration. Its dynamic `public_wait=1` counter refers to the installed rejecting Python wrapper, not the original method; the original-code tracer disambiguates it.

The unchanged v7 `original_query_closure` path executes the original v5 property operation, using the already documented one-line adaptation of the v6 **restoration diagnostic** from `getattr` to `inspect.getattr_static`. Original v5 source is unmodified. Fresh result: `property_body=launch_body=query_observer=0`, rejection `warp.Event.is_complete`, zero native work and exact property/callable restoration.

The unchanged v6 `native_queries` mode additionally rejects all three Warp native query/elapsed entries before their original-body sentinels and restores each callable. These three interception controls do not execute the forbidden native bodies.

## Retained high-risk invariants inspected

- **F1, source admission:** the loader hashes and compiles the same byte buffer, not inherited timestamp/hash bytecode. Fresh exclusive cache namespaces precede bootstrap repository imports in both fixed supervisor and child. The audit is in the test source closure, not the product allowlist. Source/cache/bootstrap tests and original stale-cache probe were freshly run.
- **F2, input readback:** exact host destinations and CPU Warp views are adopted/custody-checked before explicit-stream copies; the supervised S wait precedes CPU comparison. The full-base production diff factors out one shared ordered dtype/shape/stride/content/hash comparison policy, not a second oracle or IPC callback. The only production lifecycle enum addition is the explicit device-to-host-copy leaf kind; product activation remains disabled.
- **F3, two-pass reuse:** snapshot destinations and both host graphs precede setup quiescence/seal; physical camera origins/directions receive both sentinels. The exact bundle/command tuple, parameter/storage bindings and stream remain unchanged. Same-stream snapshot/fill uses `sync_enter=False, sync_exit=False`. One terminal V drain precedes both inspections. Zero rays retain both scalar resets; nonempty resident two-pass requests are rejected before child creation. The frozen auxiliary schedule commits attempts before entry and returns afterward, rejects skipped/duplicate/active/extra leaves and never resets the submission latch or reseals.
- **F5, execution context:** target-current is a real owned nondefault stream distinct from execution; actual ambient/target/explicit object identities and handles are checked before and in `finally`. Foreign untouched `_stream=None` is observed as real CPU metadata without lazy stream initialization. Deliberate drift quarantines without pretending a target-only emergency sync covers unknown scope. No foreign-nondefault-stream claim is made.
- **Numerical authority:** real recorded output comparison uses `CpuDirectLightOpticalExecutor` and independent host pinhole rays, not migrated Warp math. Integer channels are exact, fixed float tolerances remain unchanged, all physical channels and counter invariants are inspected after quiescence. The separate hand-derived fixtures retain exact small-topology shadow counters, lighting results and repeated empty first-hit initialization.
- **Ownership/lifecycle:** guardian precedes CUDA-capable imports; complete transfer capsules enter strong custody before member validation; unknown ownership is irreversible. Recorded and auxiliary pre-call attempts, normal returns, quiescence and release are distinct facts. No active/missing leaf can authorize V. S/V/E deadlines commit before exact one-time ACK, only S disarms, V/E remain terminal, V failure cannot retry E, parent loss and journal failures do not unwind GPU owners into pytest.
- **Instrumentation:** descriptor-aware mutable hooks use raw identity, preserve binding and restore via `ExitStack` on exceptional and normal exits. The new native monitor reserves a free tool slot, preserves/restores the prior callback, and cleans up partial installation; no slot means refusal before command entry. Existing profile/other-tool ownership is left intact. CPU side-effect tests prove pre-body denial using real immutable builtins, not only exception text; coexistence, exhaustion and partial install failure are exercised.
- **New controls:** setup creates/adopts explicit target generic stream/event owners and records the event before S. CUDA-wrapper controls use the retained owning Warp stream view. The generic IPC-import control uses a real target Warp IPC handle because installed Torch's generic export is a stub; independent `objdump` inspection of `THPEvent_ipc_handle` shows the unconditional `c10::NotImplementedError` construction and the corresponding unsupported-IPC message. The control proves rejection, not generic IPC functionality. Neither constructor delegation nor foreign-device recovery authority is broadened.
- **Scope:** production searches found no sacrificial-helper imports and no recorded factory/bundle consumer outside its defining renderer module. `optical_worker/manifest.py` GPU preparation and `EXPECTED_NATIVE_RESULT_ACTIVATION.result_capable` remain false. No gate/default/backend/result/baseline/shipping artifact or operative collaboration rule changed.

## Fresh independent commands and results

Transport: `ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101`.

Runtime prefix, from the clean detached candidate:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python
```

Observed Python **3.12.13**, Torch **2.9.1+cu128**, Warp **1.12.0**. Real Warp banner: target 0 H200 NVL, targets 1/2 A100-SXM4-40GB; toolkit 12.9 / driver 13.3. Direct conda `nvcc --version` independently reports **12.8.93**, not conflated with Warp's toolkit banner. Local Ruff/cache-check Python is 3.14. The primary SSH transport was supplied unreachable; I used the expressly authorized fallback and did not certify primary-host ancestry or reconcile another worktree.

1. Local/remote `git status --porcelain`, HEAD/tree/branch checks; original-base, last-code and report-only-archive diffs; `git diff --check 7de02e89 HEAD`: exact identity and clean entry trees confirmed; diff check passes.
2. `python -m ruff check` and `python -m ruff format --check` on all 16 changed Python files returned by `git diff --name-only 7de02e89 HEAD -- '*.py'`: all checks pass, 16 already formatted. No formatting writes.
3. CPU/POSIX run:

   ```text
   python -m pytest -q tests/unit/optics/test_sacrificial_lifecycle.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/optics/test_sacrificial_sources.py tests/unit/optics/test_sacrificial_leaf_descriptors.py tests/unit/benchmarks/test_native_input_observations.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py
   ```

   **123 passed in 11.56s**, no skips, exit 0.
4. Affected CPU regression run:

   ```text
   python -m pytest -q -p no:cacheprovider tests/unit/optics/test_warp_render_command_contract.py tests/unit/optics/test_warp_fixed_shade_abi.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/benchmarks/test_native_readiness_authority.py
   ```

   **84 passed in 2.43s**, no skips, exit 0. CPU total **207**.
5. Original v5 CPU probe streamed unchanged to remote `python -I -S - /data3/ang.gao/robot_sim_c4_verify_6d57fa7`: six 2/4/5-command × one/two-pass schedules pass; ten malformed requests rejected before child creation. Wrong/order/active/early-drain/release/reopen behavior included. Exit 0; separate from pytest totals.
6. Original `C:/Python314/python.exe -I -S C:/Users/AG/AppData/Local/Temp/c4-review-v2-cache-fixture-20260928/check.py`: verified source `VALUE='new'`, `EXECUTED_VALUE: new`, then expected exit 1 at its unchanged intentionally stale `VALUE == 'old'` assertion. It imports this exact local candidate's loader. This is closure evidence, not a candidate test failure.
7. Old GPU closure commands, each fresh exit 0:

   ```text
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py <candidate> original_wait_closure /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py
   python /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py <candidate> original_query_closure /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py /data3/ang.gao/c4-sacrificial-gpu-review-v5-event-probe.py
   python /data3/ang.gao/c4-sacrificial-gpu-review-v6-event-probe.py <candidate> native_queries
   ```

   Bounded output filtering retained exit status using `set -o pipefail`; assertions inside the probes still ran. No giant journals were dumped.
8. Native-body checks, with `mode` equal to `original_generic_query`, `original_generic_wait`, `original_generic_stream_query`, and `aliases`:

   ```text
   timeout 90 env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim gdb --batch -x /data3/ang.gao/c4-sacrificial-gpu-review-v8-native.gdb --args /data3/ang.gao/anaconda3/envs/robot_sim/bin/python /data3/ang.gao/c4-sacrificial-gpu-review-v8-probe.py <candidate> <mode> /data3/ang.gao/c4-sacrificial-gpu-review-v7-event-probe.py --child
   ```

   All four inferiors exited normally with the exact observed native counts above. Scratch kernels and resources only; no optical bundle outside its fixed supervisor.
9. Full scoped GPU/regression command:

   ```text
   python -m pytest -q -p no:cacheprovider tests/gpu/test_warp_sacrificial_render.py tests/gpu/test_warp_shade_microfixtures.py tests/gpu/test_warp_render_recording.py tests/gpu/test_warp_render_buffers.py tests/gpu/test_optical_gpu_runtime.py
   ```

   **265 passed in 849.72s (0:14:09)**, no skips, exit 0. This freshly executes all **135** sacrificial cases: all profiles/four shade variants, real target-1/ambient-0 execution, all eight repeated-sentinel cases, all 41 forbidden-operation controls (including all 16 generic controls), both second-device Torch wait controls, success/exception context checks and deliberate drift, resident/empty rays, every ordinary command partial-submit position, auxiliary/second-pass faults, all setup recording/warmup/import/allocation/readback failures, supervised S/V/E hang/raise/ACK failures, parent-loss and post-submit journal faults. The remaining **130** tests cover hand-derived microfixtures, recorded binding/buffer contracts and compatibility runtime. Real dual-device cases ran; no skip substitutes for evidence. No test was interrupted.
10. Final local and remote `git status --porcelain=v1`, HEAD/tree/branch and `git diff --check 7de02e89 HEAD`: clean, unchanged matching frozen identity, diff checks passed. Fresh independent pytest total: **472 passed** (207 CPU/POSIX + 265 GPU/regression), no skips. Original/scratch probes and native debugger observations are separate and are not added to pytest counts.

## Preserved probe identities

Local workspace copies and `/data3/ang.gao/` copies of the original event probes have matching SHA-256:

```text
v5-event-probe.py  efc704d6fb8e3e11aba5c9a82c479b489d2f71315fa185bd7356f09deb0deb98
v6-event-probe.py  f9cd070c1bae24a22a1f430903e505775a6c642dfbd76f94572c047f18c3449d
v7-event-probe.py  face20d85faf5635d43c354161b20ce323589f11b45b3ab48020ac2c41c8c0e1
v5-cpu-probe.py   34a7de70726e527fceecf2d6b9ce57de84e62012c4db6fdf960cb452bbca260a
v8-probe.py       5f3c34c921300314ec31c4b9444f57b6adb8c40ff4b4aefa9e03c0bdd3bde7a8
v8-native.gdb     6e275b6390c704935e2aa786f4cdcb375fa7135d1e7199668fb1622730ec334e
```

Full filenames begin `c4-sacrificial-gpu-review-`. New v8 artifacts are preserved outside the candidate. Original probes were not edited; only the new v8 diagnostic was corrected as explained above.

## Limitations and scope disposition

The complete scoped matrix and original counterexamples have been independently exercised on this frozen candidate. There is no unresolved required-runtime-evidence gap for the approved C4c1c validation boundary.

This is not a full repository suite, Windows POSIX-descriptor validation, cold-cache performance measurement, compiler correctness proof, hostile-process sandbox, arbitrary native-call interception or nondefault-foreign-stream proof. Installed Python/runtime dependencies and ordinary permitted Warp JIT caches remain trusted. The first runpy kernel and new alias kernel compiled; later probe loads hit ordinary cache. Native GDB counts establish the monitored actual function entries on this pinned runtime, not a universal CUDA driver trace.

CPU review overlapped author functional tests. Independent GPU work began only after all author groups completed. The short owned GDB probe processes overlapped one another, but no fault/watchdog suite was running during those probes; the full independent suite began after all probes ended. No timing/performance conclusion is drawn. Author log totals are not included in independent results.

Tooling issues were resolved without candidate edits: initial decoding/truncated reads, one guessed non-existent gate file path corrected by `rg`, one SSH regex quoting error corrected with simple literal filters, and the new observer's incorrect legacy-profiler zero-entry assumption resolved with calibrated native breakpoints. These are not code findings or waived failures.

No package installation, destructive cleanup, candidate edit, push/merge, product gate change, benchmark result/baseline or shipping artifact occurred. Ordinary ignored test/Python/JIT caches may be generated. The admission probe cleans only its own temporary directories; existing user worktrees are preserved.

Approve only C4c1c's fixed, supervised, one-shot Warp recorded-render validation and the accepted bounded acceptance closure at the exact reviewed revision. This includes real CPU-oracle correctness, counter/sentinel reuse, actual leaf contamination observation, exact declared context preservation and the required ownership/terminal-fault matrix. It grants neither reusable product execution authority nor a performance/result outlet. Earlier narrow approvals retain their original scopes; this child-stage approval does not itself approve all of C4c1 or C4.

C4c2–C4c4, C4d/B3, product GPU preparation, a3b, result/baseline access, host payload/shipping, backend selection and default changes remain closed. Any next stage requires its own applicable authorization and independent review. Archiving this unchanged report in an evidence-only commit does not broaden approval or require recursive review of the unchanged record; any executable, requirement or operative-rule delta does.
