# Native worker reuses its captured source audit

Initiative: Q54 / P12.4 / C4c2
Stage: implementation-note and native evidence
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: native-state-reader code-preflight-review v3; runtime-integration
design-request/review v3; worker-intake gap-review v1
Owner Summary: The approved reader ran on both real backends. Warp completed
two framed samples. CUDA completed input validation and final preparation but
the legacy C0 module scan rejected Torch's namespace label before the PREPARE
reply. Candidate22b1db0 removes that redundant scan from the native path and
uses the existing captured-source audit. No speed or final numerical claim.

## Exact native run and preserved evidence

Reader preflight session `01a12529-1730-7533-a977-8fccfc754ade` approved
`df1a46ae5039fe8a25718c7ec60f52fa7c60c549` for bounded native validation.
The complete unchanged report is archived, SHA-256
`21e38ffd0130ea8b486599c2fc480710b7d77cf24ed9bb7dc3ec07968c066ffd`.
Its verdict was issued before the native launch; full report archival completed
during the run. No native acceptance was granted in advance.

Actual campaign on primary `tileops-dev`:
`/data2/shared/ang.gao/c4-worker-df1a46ae50-probe`.
Code checkout: `/data2/shared/ang.gao/robot_sim_worker_df1a46a`, exact clean
approved HEAD. Existing driver `run_worker_matrix_v1.py`, SHA-256
`89ef276136c74efd830d8c4b476d62893e8d67148317f598d808dd2d93375ca3`,
was copied without modification and called with that revision and `probe`.
Existing Python3.12 robot_sim, original environment manifests, all8 visible
GPUs and target1/ambient0 were retained. No dependency install/change occurred.

| Lane | Actual result | Input scope rows | Process result |
| --- | --- | --- | --- |
| Warp | Two actual EXECUTE/RESOLVE/RELEASE samples, normal close ACK, both RECLAIMED;27 completed normal guards,0 emergency sync,guardian RELEASED | 283 | worker2554177 exit0; supervisor2554174 observed/reaped exit |
| CUDA | Input upload/readback, all6 actual extension create/exec pairs, warmups and final setup/barrier completed; PREPARE reply rejected before sample execution | 291, including12 artifact before/after rows | worker2560882 exit64; supervisor2560871 observed/reaped exit |

CUDA's actual error reply is ClosedWorkerProtocolError:
`worker loaded repository module outside source allowlist: '_ops.py'`.
It ended `cancelled_before_submission_exit`, both tokens still PREPARED,
historical submission false,22 normal guards, one CPU-cancel terminal ARM,
zero emergency synchronization and completed release. This is not CUDA worker
success. The earlier artifact-observation failure was passed on this run.

Separate CPU replay on the same clean df1a46a server checkout reconstructed
trusted launch inputs in a fresh cold child and matched the retained inputs.
For both lanes it validated custody artifact, durable selection, full
RuntimeCustodyRecord, environment/source attachment and actual input occurrence
coverage with `assert_native_input_scope_success`. Both passed. Record SHA-256:
Warp `566a11a41db7c0710a489fb9049c37cabf8039b41f7427408974418af8eb6df1`;
CUDA `0ddea169abe24863d71472dcf6a02a8d446f12d353e5d10608c0ee05f353eaf7`.
This is input-only replay, not final-sample physical/numerical acceptance.

All76 raw files (18,965,974 bytes), launch/exit records and before/after GPU
process inventories are archived in
`../c4_assets/c4-worker-df1a46ae50-probe-evidence-v1.tar.gz`:
898,686 bytes, SHA-256
`553bff46c12ec27b2f4fe819375b7ba1a80bd0bfaa6404f1dadfce43d6e83893`.
The copied archive and each member size/hash were verified after extraction
to `../c4_assets/native_reader_df1a46a_probe`. Inventory SHA-256:
`d6e088e138140a315935cd4496f632031ddf1a05b900cc72dc2715d007f97910`.
Input replay result: `../c4_assets/c4-worker-df1a46ae50-probe-replay-v1.json`.
Driver durations include imports, verification and preparation and are not
renderer performance samples or an inter-backend speed comparison.

Wrapper2553209, driver2553391 and both worker/supervisor pairs were checked
absent. GPU inventory contained none of those owned PIDs; unrelated processes
were left untouched. Reviewer37324/launcher47388 exited0 and were checked
absent. No second GPU campaign overlapped this one.

## Cause and bounded implementation

Installed Torch source `torch/_ops.py` has `_Ops(types.ModuleType)` with class
attribute `__file__ = '_ops.py'`; it constructs the `torch.ops` namespace. The
actual custody journal contains its already-admitted module_alias with
creator_attribute provenance. The legacy scanner uses getattr and resolves
that synthetic relative label against cwd, incorrectly classifying it as a
repository file. The existing native snapshot audit instead uses raw instance
metadata, exact repository names/paths/specs and verified captured bytes.
No new external namespace admission is needed.

Pinned source SHA-256:
`3b58079d356d15cc73082c20b1279468ee9cfe7f5e1340d329a93b6e877a30d5`.
Source excerpt and actual publication/artifact rows are retained in
`../c4_assets/native_reader_df1a46a_namespace_evidence_v1.json`, SHA-256
`f55a72d87c6d7918823a761efed7f9c9cf99831cf30503b857a44d627821abb2`.

Candidate: `22b1db0ee619f05e30f03bd5e339f03e883e97a5`.
`_SourceSession.audit` now returns the observed tuple it already cross-checks
against executed sources, after all custody checks. The worker uses this same
audit at startup, HELLO and the post-request/pre-response boundary. Native
HELLO derives its ledger digest from that result. C0 retains its existing
scanner. No backend-name exception, new registry, loader, source admission rule,
native query or lifecycle authority was introduced; the duplicate native scan
is removed rather than maintained alongside the captured ledger.

Bootstrap LF SHA-256:
`3bf4f6ec30562e033a865c31122e4abee4c8dfa7d128085d8574614db3e2d9d6`.
Catalogv18 LF SHA-256:
`192574b4037f28880ed1f9a204ed289eaba148bf1336f42756e505cb7cfc2695`.
All91 prospective LF entries pass the production AST validator.

The prior non-blocking bytecode comment is also addressed: two fixed isolated
source-policy subprocesses add `-B` and assert `sys.dont_write_bytecode`.
Old reviewer cache files remain preserved; no previously rejected cleanup was
retried or bypassed. Fresh-review cache absence still needs verification.

Four cold tests passed using both actual captured SourceSession audit returns
and real native snapshot checks with a modeled Torch class label; a genuine
unlisted repository file still rejects. Two HELLO cases and two isolated
import cases passed after correcting the new HELLO test's list-vs-tuple
expectation. Production returns a tuple before wire serialization. Broader
committed CPU/source/worker regressions and Linux C0/Go2 checks are running;
their outcomes are recorded below. Ruff passed worker/integration tests.

Author regression completed at exact candidate22b1db0:

- Local Python3.14: native source policy/sources/custody, integrated worker
  lifecycle, closed worker protocol and optical closed worker:308 passed,
  13 skipped in272.26s, exit0 (tool session91300). Skips are not acceptance
  evidence for asset/POSIX-dependent cases.
- Primary server Python3.12: actual frozen Go2 CPU materialization, POSIX
  checkout catalog and optical closed worker file:15 passed without skips
  in141.98s, exit0 (tool session72158, driver2627533/pytest2627572).
  The wrapper observed pytest absent and the exact server checkout clean.
- These are CPU/source/worker regressions, not a native rerun, full hardware
  fault matrix, numerical acceptance or speed comparison.

Fresh independent preflight session `01a1253f-d23b-7d22-b40b-b66701cc189e`,
launcher/reviewer7056/5648, started2026-10-10T09:58:21Z, is reviewing frozen
`../c4_native_worker_source_audit_review` at22b1db0. Request SHA-256:
`82c1d0ae3cc4d03a1ceb587473ded8d1340de360d766e22121100a1e554e2bba`.
It receives full original raw evidence and must decide whether the delta
preserves existing admission policy and may resume bounded native validation.
The complete returned preflight report is now archived verbatim as
`q54-optical-lab-p12-4c4c2-native-worker-source-audit__code-preflight-review__codex__v1.md`,
SHA-256 `ce80f7d68b7e9532495cef0454acf9aede3a9b8284e97152991d31518578fe31`.
Verdict Approve: exact22b1db0 may resume bounded native setup/framed validation.
Independent tests:238 targeted and7 additional boundary probes passed, with
no skips;91/91 source entries verified. The reviewer independently checked
the preserved base campaign and replayed both input records. This approves
the source-audit delta/preflight, not hardware or whole C2f/C4 acceptance.
Its clean isolated server checkout is
`/data2/shared/ang.gao/robot_sim_worker_22b1db0`.

After receiving and archiving that verdict, a new two-lane/two-sample probe
started2026-10-10T10:08:51Z, wrapper2670393/driver2670408, using the unchanged
campaign driver. Output prefix `/data2/shared/ang.gao/c4-worker-22b1db0ee6-probe`.
The execution ledger preserves command, code revision, process start identity,
stdout/stderr and GPU inventories. It is still running; no outcome is claimed.

Complete reader failure/Go2 native acceptance, final-sample physical and
numerical proof, bounded preallocated traces and fixed work, full worker fault
matrix/close and whole C2f/C4 review remain open. C4c3/c4, C4d/B3, activation,
formal performance, result/baseline, backend selection, R2/S/host delivery,
a3b and shipping remain closed. The Warp smoke is not the parent goal.
