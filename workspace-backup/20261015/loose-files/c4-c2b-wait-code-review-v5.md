# C4c2b guarded waits — independent code review v5

Initiative: q54-optical-lab-p12-4c4c2b-supervised-waits
Stage: code-review
Author: codex (fresh independent reviewer)
Version: v5
Date: 2026-09-29
Status: accepted
Related Files: design-request v3; design review v3; code-review v1/v3; implementation-note v1–v4; wait_protocol.py; wait_supervision.py; guarded-wait harness/tests; cleanup_ledger.json
Owner Summary: Approve the exact frozen CPU-only implementation. Independent Linux tests and targeted original-counterexample probes pass. Terminal publication now follows the completed OS write fence; client and supervisor failure remain monotonic in the examined interleavings. No product adoption, GPU preparation, parent-stage completion or result gate opens.

## Identity and verdict

- Reviewer session: `/root/c4_c2b_wait_code_review_v5`, fresh independent session without candidate authorship, inherited author conversation or delegated agents.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`.
- Whole implementation base: `fcb9608a916d20d1c43f64a3401b3c78d823404f`.
- Exact reviewed HEAD: `54f3e1a3363664414e434f091e5c9ed8bc7d7c3b`.
- Tree: `e82d3ce8d019c9795c1e6c0110368fb6c90d32c9`.
- Latest repair delta: `8592902240e99474aa7223f6470e6d9e64ddfcf8` to that HEAD. The review also inspected the whole implementation, not only this delta.
- Independent Linux execution: owner-authorized fallback `ang.gao@192.168.80.101`, clean detached `/data3/ang.gao/robot_sim_c4_c2b_code_54f3e1a`, matching HEAD/tree, conda `robot_sim`, Python 3.12.13.
- Local execution: Windows PowerShell, `C:/Python314/python.exe`, Python 3.14.0.
- Local and remote porcelain status were empty before and after verification. No candidate source/test/doc edits were made. This report and reviewer scratch probe are outside the candidate. `-B` and disabled pytest cache provider were used for direct test executions; ordinary ignored dependency/harness/Ruff caches are not asserted absent.
- Approved design source SHA-256 independently checked: `a3619ead4e22ad47f3133faca7788cdb85d0816f2bad8cffd8a46998fc80c872`, approved at `ff3b4db0d4c3812c7149c97cb9cf9bb095598461`.

**Verdict: Approve.** No substantive blocking or non-blocking finding remains for this exact revision and bounded delivery. Approval is not inferred from test counts or the author report.

The interrupted v2/v4 sessions did not produce final reports/verdicts; none is invented. V4's concrete interim counterexample is treated as a counterexample to re-run, not an approval or completed review.

## Scope and inspected implementation

Read complete `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, approved design v3 and its review, completed code reviews v1/v3, implementation notes v1–v4, both actual modules, the fixed process harness and both test files. Initial encoding-garbled Chinese instructions were reread as UTF-8. Truncated combined code output was supplemented with targeted reads. Inspected the shared canonical codec, complete base-to-HEAD path delta, latest repair diff, cleanup ledger and production import search.

The base delta contains 12 paths: five new executable/test files, the cleanup-ledger addition, two archived reviews and four implementation notes. Existing product modules, C0 gate/runtime JSON, worker/session/launcher/supervisor, native setup/lifecycle, C1 S/V/E driver, source/request-policy surfaces and shared collaboration rules are unchanged. Scoped import search found only the new core and fixed test harness adopting the new modules, not a product consumer.

High-risk flow examined:

- Configuration is exact, frozen and reconstructed. Wire records use the existing canonical codec with guard-specific size, identity, role, operation and integer domains. Parser storage is one bounded frame, and extra/full pending input faults rather than becoming a request queue. EOF at partial-frame prefixes is tested.
- Client ARM/COMPLETE use one absolute send-plus-receive ACK deadline. Readable ACK loses at deadline equality. ARM pending precedes output; tickets are created only after exact ACK and require the outstanding object's identity. Copy/reconstruction/retirement/emergency misuse fails before a normal COMPLETE write. The client executes no blocking callable.
- Client state/ticket/sequence are published together under an integer-keyed builtin dictionary insertion. A nested committed failure cannot be overwritten by stale success. Failure after successful publication retires the ticket; it cannot authorize subsequent protocol operations. A later asynchronous failure need not erase an immutable metadata object already committed for return. Future fixed drivers still own fail-closed leaf entry/custody policy; this is not GPU authority.
- Supervisor state/event pairs share a single append-only publication point. Nested termination wins against stale normal publication, preserving replay and deadlines. Cross-thread entry is rejected before normal mutation/output; same-thread reentry is terminal rather than serialized.
- ARM establishes operation and ACK deadlines before output. COMPLETE only creates DISARM_ACK_PENDING. The original operation guard remains until fully written DISARMED is successfully recorded. Fault folding retains the earliest deadline; partial progress cannot restart it. Clock faults retain immediate expiry/kill obligations without an endless retry budget.
- The bound socket's write side is now shut down before a TERMINATING record is inserted. On ordinary shutdown OSError the socket is closed. Terminal publication therefore cannot precede an unfinished Python cleanup fence. Every nested terminal append must satisfy the same prerequisite. Already accepted bytes remain possible and are treated distinctly from later writes; the fence does not release process or resource custody.
- Replay shares the transition grammar, exact event fields and derived summary. Expiry, kill request, kill failure and observed process exit remain separate. Only process-exited evidence is final; kill request/unknown status cannot fabricate reaping. Evidence is canonical consistency data, not authenticity or a live ticket/process handle.
- Encoding capacity remains `MAX_GUARDS=(I64_MAX-8)//4`, `MAX_SEQUENCE=MAX_GUARDS-1`; initialization, four events per guard and six terminal-tail events fit. Increment and timestamp arithmetic are checked. Full histories begin at sequence zero; no high-sequence short trace is accepted as genuine history.
- The real-process harness uses actual child poll/wait status, keeps custody after faults, records a kill attempt before calling kill, and distinguishes its outer failing safeguard from successful watchdog evidence. Hard-exit/finalizer sentinels test CPU process behavior, not CUDA ownership.
- Cleanup classification is shared safety infrastructure pending separate selected-runtime adoption, with removal if no selected production/test consumer remains. The fixed harness is a regression consumer, not a second product supervisor.

## Independent closure of prior counterexamples

### Original supervisor publication and swallowed clock reentry — closed

The external reviewer probe schedules a swallowed public `core.fault("parent_lost")` immediately before `_append` inserts the normal DISARMED record. The outer call rejects; state remains TERMINATING, effective deadline is 5,000,100 ns, completion count is zero, and evidence replays. The original constructor-clock reentry that catches its inner error also rejects the outer ARM, produces no arm commitment and leaves kill required.

Both probes ran on Windows 3.14.0 and Linux 3.12.13. The installed tests independently exercise the nine transition-commit boundaries with propagated/swallowed errors, and normally reached DISARMED/finalizer/clock/state lines. Linux tests use actual SIGUSR1. Finite line scheduling is not an exhaustive asynchronous proof.

### V3 client FAILED resurrection — closed

The original v3 executable source was inspected. Its old `_active` assignment markers no longer exist. The independent v5 probe retains the same real socketpair, real client, owner-thread core/transport and swallowed nested ARM trigger, moving injection to the equivalent pre-success publication for ARM and COMPLETE.

Observed for both phases on both interpreters: outer `GuardProtocolError`, persistent FAILED, and subsequent ARM rejected. No success ticket returns when failure wins before publication. Installed tests additionally cover nested ARM, explicit fail and close at publication and finalizer boundaries.

### V3 terminal-before-send output — closed; opposite ordering checked

The original v3 transport reproduction was inspected. A new independent probe uses its actual `_service` send boundary with real sockets and nested public service. For both ARMED and DISARMED, terminal-before-send produces **zero bytes followed by EOF**, an OS write error, preserved 5,000,100 ns deadline, and no stale ack_written event.

An adjacent independent probe injects parent loss immediately after full OS send but before offset/ACK commitment. The peer receives the already-written exact frame (ARMED 172 bytes; DISARMED 175 bytes), then EOF. Recording that ACK fails; terminal state and the original deadline remain, normal-complete count stays zero, and evidence includes no acknowledgment for the interrupted response. This checks the allowed write-before-terminal ordering without pretending already-enqueued bytes can be recalled.

### V4 double-signal fence interruption — closed

Independently read and SHA-256 checked the supplied original external script:

`C:/Users/AG/新建文件夹/c4-c2b-fence-interruption-repro-v4.py`

SHA-256: `59f700363c907b45bd3f39b9264861ece3db627783e0c939f9b0cb0705d4cded`.

Executed its original scheduling/control flow on Windows and Linux with only acceptance-observation adaptations: catch the now-expected OSError as well as GuardProtocolError; inspect raw peer bytes instead of decoding an expected full frame; replace the old bug assertion `wire == response` with `wire == b""`. The source on disk was not edited.

For both ARMED and DISARMED, the second handler now observes the still-pending state before the fence (rather than an already-published TERMINATING claim), nested handling completes the fence and terminal publication, and resumed send fails. Observed zero peer bytes/EOF, TERMINATING, kill required, effective deadline 5,000,100 ns, zero normal completions, one expiry record and replay-valid evidence. Linux executes real SIGUSR1/SIGUSR2. First terminal reason is request_out_of_order because nested entry now commits before the interrupted expiry; this is consistent with the first-committed-reason rule.

The eight new installed cases also exercise before-fence-condition, fence invocation, before insertion and after insertion, for both ACK types. They passed independently in the full selection.

## Executions and provenance

All commands ran from the local frozen repository unless the SSH command specifies its detached remote tree.

1. Local tests:

```powershell
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py
```

Observed **120 passed in 2.12s**, exit 0.

2. Independently initiated Linux regression command:

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_54f3e1a && git rev-parse HEAD && git status --porcelain && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/optics/test_sacrificial_watchdog.py tests/unit/benchmarks/test_optical_benchmark_contracts.py::test_inventory_and_cleanup_ledger_match_current_repository"
```

Observed exact HEAD, empty initial status and **181 passed in 17.02s**, exit 0, no skips. This includes real Linux process fault/backpressure/parent-liveness tests, actual signal scheduling, unchanged C0 protocol/worker regressions, C1 watchdog and inventory/cleanup consistency. The author later supplied its separate 181-pass/16.95s artifact; it was not used as proof for this review.

3. Independent scratch probes were created with apply_patch outside the candidate:

`C:/Users/AG/新建文件夹/c4-c2b-independent-probes-v5.py`

Final SHA-256: `544d107efaf3357ff522f7d625583a494ad008de13f813c0a2af1e93d9b3dae6`.

```powershell
Get-Content ../c4-c2b-independent-probes-v5.py -Raw -Encoding UTF8 | C:/Python314/python.exe -B -
Get-Content ../c4-c2b-independent-probes-v5.py -Raw -Encoding UTF8 | ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_54f3e1a && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -"
```

Both final executions passed, exit 0. Common output:

```text
client pre-success publication arm GuardProtocolError FAILED
client pre-success publication complete GuardProtocolError FAILED
wire ordering ARMED before_send bytes 0 error BrokenPipeError deadline 5000100
wire ordering ARMED after_full_send bytes 172 error GuardProtocolError deadline 5000100
wire ordering DISARMED before_send bytes 0 error BrokenPipeError deadline 5000100
wire ordering DISARMED after_full_send bytes 175 error GuardProtocolError deadline 5000100
old supervisor precommit TERMINATING 5000100 replay valid
old swallowed clock TERMINATING kill_required True arms 0
```

Final line identifies Python 3.14.0 with same-thread callbacks locally and Python 3.12.13 with actual signals remotely. No candidate method was replaced by these probes.

4. Original v4 repro execution used this in-memory adaptation, then the same local/remote stdin invocation as above:

```powershell
$repro = (Get-Content ../c4-c2b-fence-interruption-repro-v4.py -Raw -Encoding UTF8).Replace('except GuardProtocolError as exc:', 'except (GuardProtocolError, OSError) as exc:').Replace('wire = GuardFrameDecoder().feed(left.recv(1028))', 'wire = left.recv(1028)').Replace('assert wire == response, wire', 'assert wire == b"", wire')
$repro | C:/Python314/python.exe -B -
$repro | ssh -o BatchMode=yes -o ConnectTimeout=10 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "cd /data3/ang.gao/robot_sim_c4_c2b_code_54f3e1a && /data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B -"
```

Both phases passed on both platforms, exit 0. Windows OS error text was console-encoding garbled, but type/EOF/state/trace assertions and Linux Broken pipe output were unambiguous.

5. Static/integrity checks:

```powershell
C:/Python314/python.exe -B -m ruff check optical_worker/wait_protocol.py optical_worker/wait_supervision.py tests/support/guarded_wait_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py
C:/Python314/python.exe -B -m ruff format --check optical_worker/wait_protocol.py optical_worker/wait_supervision.py tests/support/guarded_wait_process.py tests/unit/optics/test_guarded_wait_core.py tests/unit/optics/test_guarded_wait_process.py
git diff --check fcb9608a916d20d1c43f64a3401b3c78d823404f HEAD
git diff --name-status fcb9608a916d20d1c43f64a3401b3c78d823404f HEAD
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git branch --show-current
git status --porcelain=v1
```

Ruff: all checks passed; five files already formatted. Diff check: no errors. Final local/remote HEAD/tree/status/Python checks match the identity section; remote branch output is empty (detached). Read-only Get-Content, rg and Get-FileHash checks supplied source/consumer/digest evidence.

The primary host timeout was supplied in the packet, not independently reexecuted in this session. The already-authorized fallback was used and its exact clean revision independently verified. No environment/toolchain failure prevented mandatory CPU verification.

## Limitations and remaining gates

Finite tests do not prove all possible asynchronous schedules, unsupported interpreters or hostile Python/native mutation safe. Single-commit reasoning is bounded to the supported CPython/process model, not a memory-security boundary. The supervisor must remain functioning/scheduled. Full-capacity histories are mathematically bounded and checked through actual primitives, not allocated at infeasible size. Actual OS kill permission failure was not induced; kill-error custody is covered at core level. The one-byte-write installed test uses a delegate around a real socket; natural socket backpressure is separately exercised in the Linux process harness. Test field mutations are substantial but not an exhaustive Cartesian product of all malformed evidence.

No GPU/native timeline, guardian completeness, allocation safety or R0/R1 lease correctness was tested or approved. No install, push, merge, product activation, result/baseline change or existing module edit occurred.

Approved only: the exact HEAD's two inert stdlib-only CPU protocol/client/supervisor modules, fixed standalone CPU harness/tests, cleanup classification and accompanying historical/evidence records. A verbatim evidence-only archival commit may record this report under the repository's bounded exception; executable or operative changes require review of their delta.

C0 A/K and C1 S/V/E remain unchanged and unadopted by this core. C4c2 runtime integration still requires its separate complete design and same-slice real-device validation, including materialization/numeric camera resolution, setup/warmup/readiness/guardian order, native timeline and exact lease/resource custody. Product GPU preparation/submission, worker/runtime-manifest/source-policy adoption, C4c3/R2, C4c4, C4d/B3, host payload/shipping, a3b, results/baselines, backend selection and product defaults remain closed. This dependency-slice approval does not complete C4c/C4 or approve any parent/later phase.
