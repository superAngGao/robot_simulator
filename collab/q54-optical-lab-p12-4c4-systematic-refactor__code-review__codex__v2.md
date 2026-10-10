Initiative: Q54 / P12.4 / C4  
Stage: independent follow-up code review, execution-consolidation repair  
Author: Codex, fresh reviewer session `/root`  
Version: v2  
Date: 2026-10-10, Asia/Shanghai  
Status: accepted for the bounded code-review scope below  
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request v3, sections 5–8; systematic-refactor gap-review v1; code-review v1; implementation-note v2/v3  
Owner Summary: Approve the bounded consolidation/repair code at the exact revision below. Original F1 and C1 counterexamples now pass their rejection/behavior assertions. Additional independent CPU probes confirm binding-failure retention and successful standalone/guarded retirement. Native F1 closure, final graph admission and whole-C4 acceptance are not established.

**Verdict: Approve — bounded code and CPU coordination scope only.**

No unresolved blocking code finding was substantiated in this review. Original F1 is closed at the code/CPU coordination level; C1 is closed. This verdict does not approve native guarded final-arena resolution or open any later gate.

This was a fresh independent review invocation. No implementation conversation was supplied, no candidate changes were authored, and no agents were spawned. Available session identity is `/root`, with local evidence record `execution_consolidation_review_v2`. A provider/CLI session UUID was not exposed to this reviewer; none is invented.

**Candidate identity and integrity**

| Item | Independently observed value |
|---|---|
| Candidate | `C:/Users/AG/新建文件夹/c4_final_graph` |
| Branch | `implementation/c4-final-graph` |
| Overall base | `c9d537778659db8ac79ba509bdf2f73d971a8c23` |
| Prior reviewed candidate | `1934ac7673348c12ccb76a6e0b54ef883b17324b` |
| Evidence-only archive commit | `6f8759e645c3a1ef20a115050f4e05003089ee66` |
| Reviewed HEAD | `99eb1f3aff0436134346d08b2be4f235a910b19d` |
| HEAD tree | `c93a644da7bc2692736f733b2a1ea79aa6fcb0a7` |
| Working tree | Initial, intermediate and final `git status --porcelain=v1` empty |

HEAD, tree and branch remained unchanged. The archive commit changes only the prior review document. Overall base-to-candidate scope is 22 files, 1,311 insertions and 495 deletions; prior-candidate-to-HEAD scope is nine files, including archived evidence and author documentation.

I read the required collaboration documents, design sections 5–8, complete gap review, archived full code-review v1, implementation notes v2/v3, original repros and their failing logs. Author notes, historical positive findings and reported counts were treated as claims to verify.

**Original F1 — P2 blocking: closed for code/CPU coordination; native closure remains unproved**

Location: [optics/cuda_arena.py:726](C:/Users/AG/新建文件夹/c4_final_graph/optics/cuda_arena.py:726), especially `_validate_resolution_binding` at line 768.

The prior defect allowed mutable execution-context drift to escape the active sample’s identity-only checks. Resolution could succeed and drop the transaction hold despite a changed device or stream.

The candidate now invokes the same validation before waiting and again after successful guard completion, before `lease.resolve_terminal_completion()`:

- Existing active-sample configuration validation.
- Original executable/template validation, including context device, stream object identity, sealed stream handle and current native stream handle.
- Exact minted authority registration, recorded phase and live lease binding.
- Identity of the sample’s completion versus the transaction’s completion.
- Exact template terminal event, completion device and completion stream handle.
- Actual event type, initialized device and target-device agreement.

This reuses existing validators and authority registries. It does not introduce a parallel lease state machine or caller-supplied completion assertion.

The unchanged original controlled repro now produces:

| Case | Result | Resolutions | Transaction hold | Device-result hold |
|---|---|---:|---:|---:|
| Successful control | `PAYLOAD_READY` | 1 | 0 | 1 |
| Device drift during wait | `POISONED` | 0 | 1 | 1 |
| Stream drift during wait | `POISONED` | 0 | 1 | 1 |
| Stream replacement before wait | `POISONED` | 0 | 1 | 1 |

Pre-wait drift rejects without waiting. During-wait drift may reach normal DISARMED, but the post-handshake validation prevents lease commitment. Rejected leases remain `RETAINED_POISONED`.

The original smaller device/stream probes also pass unchanged.

I added 65 independent binding cases across guarded resolution before wait, during wait and during ACK processing, plus standalone resolution before/during wait. They cover:

- Different stream object with the same device and handle.
- Foreign stream and stream-device mutation.
- Context handle, native stream handle and coherent handle mutation.
- Coherent context/stream/event device mutation.
- Uninitialized or foreign-device terminal event.
- Transaction event replacement and copied transaction completion.
- Explicit frozen-completion device/stream tampering.

The frozen-record probes deliberately exceed ordinary writable-field mutation; they are not a claim that hostile arbitrary Python reflection is contained.

Every rejection preserved both holds, recorded zero resolutions/reclaims, and rejected subsequent resolve, result release and reclaim without another wait. Two successful controls completed resolve, result release, reclaim and authority retirement.

Wait-failure and injected ACK-failure probes preserved the original exception and both holds. Guarded failures transferred custody to quarantine with zero emergency synchronization attempts.

These tests use production context/template/prepared capability/minted authority/transaction/lease/guard code and actual enqueue/finish transitions. Native operations and C3 inputs remain CPU doubles. They establish coordination behavior, not actual CUDA completion semantics.

**Original C1 — P3 nonblocking: closed**

Location: [optics/cuda_lbvh.py:527](C:/Users/AG/新建文件夹/c4_final_graph/optics/cuda_lbvh.py:527).

`before_native_launch()` is now conditional on `_extension_entry is not None`. The unchanged original empty/populated LBVH probe passes. Candidate tests also cover both cases with diagnostics enabled and disabled, preserving diagnostic phase order.

An empty LBVH stage no longer declares a nonexistent native launch. This closes the reported bookkeeping discrepancy without establishing R1 native acceptance.

**Review of the broader consolidation**

No additional substantive finding was established in the inspected areas.

- **Shared events:** Guarded setup, standalone arena and compatibility wrappers use `_prepare_native_events`. Returned events are retained before subsequent fallible operations. Diagnostic initialization precedes the terminal record/wait; diagnostics-off creates no diagnostic events. The unchanged reviewer matrix exercises every create/adopt/record boundary and wait for zero, three and five diagnostic pairs in both lanes: 120 cases pass. These check custody, stopping after failure and CPU-double scope restoration.
- **Launch boundaries:** Render entries and separate triangle/plane entries explicitly invoke the launch hook. Setup submission latching is independent of diagnostic recording. Empty LBVH behavior now matches that rule.
- **Frame binder:** The extracted binder preserves canonical scalar decoding, frame metadata, retained R0 storage, Warp module/recorded-command validation and CUDA render-ABI validation. Regression tests exercise profile, shadow and diagnostic combinations and factory/context/module/returned-stage rebinding. The helper remains groundwork; no final graph consumer is established.
- **Warmup record:** Named fields preserve exact identity registries, one-shot reservation, retained transactions, active-field clearing and permanent retirement. Completion credit follows the exact completed-ticket check. The final audit now uses `SETUP_WARMUP_FRAME_COUNT - 1`; the constant remains ten.
- **C3 relocation:** Independent AST comparison found all ten definitions retained. Nine bodies are unchanged; the intended camera-selector expression changes the remaining body. All ten compatibility aliases are identical to their production objects.
- **Selectors:** `render_sample_index` participates in plan schema v2 and plan hashing. Selection/sealing use `(transform_frame_index, render_sample_index)`. Legacy builders retain their `None` camera-selector behavior. Tests cover selector independence and invalid bool/float/negative indices. They do not prove a completed final multi-camera graph.
- **Acquire:** Five inspected acquire/activation/authority-construction AST bodies are unchanged from the overall base. The repair adds checks in untimed resolution. No performance or 1/120/600-frame operation-count result is claimed.
- **Source closure:** All 88 catalog entries match exact committed LF source bytes and pass the existing AST policy. Candidate catalog SHA256 is `03b0e419b32e535758617db36b1ddfacb6b4faf2042d46f51c1417eece2f8951`. The relocated adapter is CUDA-only.
- **Gates:** Product PREPARE remains fail-closed. Final barrier admission, measured submission and clean-close entries remain explicitly closed in the guarded lifecycle. The repair does not implement worker token dispatch or final readiness/arena admission.

**Commands and fresh results**

Environment: Windows PowerShell; `C:/Python314/python.exe`, Python 3.14.0, pytest 9.1.1, NumPy 2.5.2.

Tests used `-B`, `PYTHONDONTWRITEBYTECODE=1`, disabled pytest’s cache provider, and placed temporary files under:

`C:/Users/AG/新建文件夹/c4_assets/execution_consolidation_review_v2/`

From the candidate directory, the common invocation was:

```powershell
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider --basetemp=<scratch>
```

| Execution | Files/command | Fresh result |
|---|---|---|
| Original repros and repair tests | Unchanged `reviewer_controlled_resolution.py`, unchanged `reviewer_probe_tests.py`, candidate `test_cuda_arena_resolution.py`, `test_cuda_timeline.py`; additionally `-s`; scratch `pytest-original` | **162 passed in 10.92s** |
| Independent additional probes | Scratch `reviewer_additional_tests.py`; scratch `pytest-additional` | **70 passed in 9.32s** |
| Structural regressions | `test_native_render_preparation.py`, `test_native_render_warmup.py`, `test_cuda_arena_contract.py`, `test_native_source_policy.py`, `test_native_contract_extraction.py`; scratch `pytest-structural` | **361 passed in 95.50s** |
| Static audit | `python.exe -B ../c4_assets/execution_consolidation_review_v2/reviewer_static_audit.py` | Catalog, source policy, relocation aliases and five acquire AST comparisons passed |
| Archive audit | `python.exe -B ../c4_assets/execution_consolidation_review_v2/reviewer_archive_audit.py` | Archive metadata and all ten native-channel hashes per lane verified |
| Whitespace validation | `git diff --check c9d5377 HEAD` | Passed |

Candidate test filenames above are under `tests/unit/optics/`; unchanged original repros are under `../c4_assets/execution_consolidation_review_v1/`.

Logs are `original-and-repair-tests.txt`, `additional-tests.txt`, `structural-tests.txt`, `static-audit.txt` and `archive-audit.txt` in v2 scratch. These are fresh executions, not reused author outputs. Counts are not evidence of broader acceptance.

The original repro files were not edited. Their SHA256 values remained:

- `reviewer_controlled_resolution.py`: `53c7f38754027b8cac299ad97bb4f77219d91e261a00180986c37678508fa955`
- `reviewer_probe_tests.py`: `bba3db5d0bca037d44c277f08dadf227454a4e832a1daee14a09fa31d40938db`

One initial inspection command passed unquoted `HEAD^{tree}` through PowerShell incorrectly; the quoted command returned the tree recorded above. Initial non-UTF-8 document rendering and oversized outputs were corrected with UTF-8 and smaller reads. These were inspection-command issues, not candidate failures.

**Archived native evidence: checked provenance, limited applicability**

I verified the supplied archive:

- Size: 50,722,689 bytes.
- Members: 100.
- SHA256: `cec14bada1cc7d165d5d6cfc3cb3a401a6e91facb2fdd48d55a8591748aa2a02`.

Both lane requests identify `1934ac7673348c12ccb76a6e0b54ef883b17324b`. Records show Go2 960×640, target device 1, full profile, shadows enabled and primary unprofiled mode. Warmup records contain ten distinct transactions/completions, indices 0–9 and ten completed guard sequences. Frame-10 readback and worker return code zero are recorded.

I independently hashed all ten archived native channel payloads per lane against their comparison records. Both comparison records report zero failed pixels over 614,400 pixels, with no omitted CPU channel. I also inspected the archived comparison script and its original CPU-revision, channel, shape and chunk-coverage checks.

I did **not** rerun the original CPU oracle or the native campaign. This is verification of archived evidence and its internal bindings. It is not fresh numerical execution.

The archive supports the documented R0 regression at **1934ac7**. It does not exercise guarded final-arena resolution, does not test the repaired HEAD, and does not prove native F1 closure. The records explicitly retain `final_execution_readiness_issued=false` and zero sample submissions.

**Limitations, process hygiene and remaining scope**

No GPU execution/query, SSH operation, installation, candidate source edit, candidate Git mutation, reset, process cleanup or agent spawning occurred. Reviewer-created files remained under v2 scratch. Repository paths in generated Python were obtained from `Path.cwd()`.

The structural pytest process was observed as PID 13824, parent PID 44532, with the exact review command. Its tool session returned exit zero. A final scoped query found neither that PID nor direct children, and no Python process matching the reviewer’s v2 pytest temporary paths. Other review tool sessions completed. The guard test peer closes and joins its thread. No reviewer-owned unresolved hang was observed; this is not an exhaustive OS process attestation.

The following remain unproved or incomplete:

- Real-device resolver behavior, native event lifetime and target1/ambient0 restoration for the repair.
- Actual transport-loss, watchdog and observed-process-exit behavior; ACK failures here were injected at the client completion method.
- Complete final readiness/origin, final factories/C3/arena graph, immutable coverage and acknowledged target barrier.
- Warp final arena, R1 composition, integrated worker tokens, post-reclaim close and ACK-before-cleanup.
- Broader native correctness/failure matrices and performance evidence.

M1/M4 and whole-C4 structural completion remain open. This bounded approval must not be recorded as completing those obligations.

The candidate remained frozen and clean through the final check. Approval binds **99eb1f3aff0436134346d08b2be4f235a910b19d** and the bounded consolidation/repair code reviewed here. Product dispatch, final graph admission, whole C4/C2d/C2e/C2f acceptance, activation, later R2/S/host delivery, results/baselines/a3b, backend selection and shipping remain closed.