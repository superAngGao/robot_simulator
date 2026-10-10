Initiative: Q54 / P12.4 / C4c2 bounded native-worker source audit
Stage: Fresh independent CODE/PREFLIGHT review only
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Session: 01a1253f-d23b-7d22-b40b-b66701cc189e
Related Files: native-state-reader design-request v1 and code-preflight-review v3; runtime-integration design/review v3; source-custody design v1; worker-intake gap-review v1; optical_worker/worker.py; optical_worker/native_bootstrap.py; native_source_policy.py; native_sources/v18.json
Owner Summary: Approve exact candidate 22b1db0ee619f05e30f03bd5e339f03e883e97a5 to resume bounded native setup/framed-worker validation on the pinned deployment. No substantive finding remains in this source-audit delta. Independent CPU checks pass 245 cases; preserved base campaign evidence was independently replayed. This is neither candidate hardware acceptance nor whole-reader/C2f/C4 approval.

**Verdict: Approve — bounded CODE/PREFLIGHT scope.**

This exact candidate may resume bounded native setup/framed-worker validation under the approved preparation-reader design and pinned deployment. There is no unresolved blocking finding or required preflight evidence gap in this delta. No new native/GPU run was performed on this candidate. Its actual native success and affected failure behavior must still be observed and independently reviewed; this verdict does not grant that acceptance in advance.

**Identity, independence and preservation**

| Item | Independently verified value |
| --- | --- |
| Candidate | `C:\Users\AG\新建文件夹\c4_native_worker_source_audit_review` |
| HEAD | `22b1db0ee619f05e30f03bd5e339f03e883e97a5` |
| Source base and merge-base | `df1a46ae5039fe8a25718c7ec60f52fa7c60c549` |
| Branch | Detached HEAD |
| Initial and final porcelain, including ignored/untracked files | Empty |
| Base-to-HEAD delta | 11 files, 354 insertions, 11 deletions |
| Whitespace | `git diff --check df1a46ae5039fe8a25718c7ec60f52fa7c60c549 HEAD` passed |
| Reader approved design, committed LF SHA256 | `0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5` |
| Reader predecessor report v3, committed LF SHA256 | `21e38ffd0130ea8b486599c2fc480710b7d77cf24ed9bb7dc3ec07968c066ffd` |
| Predecessor reviewer session | `01a12529-1730-7533-a977-8fccfc754ade` |
| Candidate bootstrap LF SHA256 | `3bf4f6ec30562e033a865c31122e4abee4c8dfa7d128085d8574614db3e2d9d6` |
| Candidate AST policy SHA256 | `a219791cda442b3f79da5b72becd6d2bfe85f17452b3e096377036107c80da8b` |
| Candidate catalog v18 LF SHA256 | `192574b4037f28880ed1f9a204ed289eaba148bf1336f42756e505cb7cfc2695` |
| Unchanged reader LF SHA256 | `3a67a897834c7c118d9cac850ea1f3f68a8f0686daf318b0a0715d0ddd3b40f7` |

I received the review packet without an implementation conversation and did not author the candidate. No agents were spawned. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely in UTF-8, followed by the specified reader design/predecessor report, runtime-integration design and review v3, source-custody requirements and worker-intake gap report. Oversized outputs were followed by focused reads of the relevant complete documents/code. Prior verdicts supplied scope and gates, not proof of this delta.

All reviewer-created scripts, tests, logs, packaged test releases and this report are confined to `C:\Users\AG\新建文件夹\c4_assets\native_worker_source_audit_scratch_v1`. No candidate, old report, raw campaign, old cache, private agent context or other implementation file was edited. No installs, resets, signals, native-library acquisition, GPU query or GPU execution occurred. SSH used only primary `tileops-dev`; no host switch occurred. The frozen remote base checkout remained clean/detached at df1a46ae5039fe8a25718c7ec60f52fa7c60c549. This candidate was not deployed.

**Findings and policy classification**

No substantive blocking or non-blocking production finding was identified in this bounded delta.

The removed C0 scan was an inappropriate second interpretation of native source identity, not an additional required native admission rule. Runtime-integration v3 section 4 requires rejection of unexpected repository imports at startup, HELLO and request boundaries. Source-custody requires captured byte consumption, exact module/loader/source ownership, phase closure and sticky failure. Those obligations remain enforced at the same boundaries by the already approved native audit. Neither design mandates the generic C0 `getattr(__file__)` algorithm for the native path.

This is a bounded implementation correction within the accepted designs. It changes the native worker's audit routing and returned descriptive ledger, without expanding admitted module names, paths, external publications, runtime phases, source-policy exceptions, IPC operations, backend authority or native query capability. The policy digest and bootstrap/class pins necessarily change to bind the reviewed source bytes. C0 still uses its original scanner. I did not infer this classification merely from the author's description.

The two scans are not identical predicates over arbitrary hostile Python objects: the native snapshot intentionally treats relative labels on external namespaces differently and does not invoke their lazy metadata. That is consistent with the declared closed trusted-process boundary, the captured loader and separately admitted external publication machinery. This approval does not claim a sandbox against arbitrary in-process reflection, forged objects or replaced audit methods.

**Checked call, data and state paths**

- `native_bootstrap.py:3651`, `_SourceSession.audit`: the existing topology/liveness, generated-stdlib/interpreter-alias, phase/module closure, exact registry identity, loader/source-cache, spec/package/path, and started/completed trace checks remain. Snapshot verification still precedes comparing the observed set with executed source paths plus the bootstrap. Guardian coverage of the original session, snapshot, bootstrap and every captured module still follows. Only after all these checks does the method return the snapshot's actual observed tuple. It does not return the catalog or a caller-provided ledger. Independent AST comparison, after removing the new empty-tuple initialization and return, proves the complete `_SourceSession` class otherwise matches base.
- `native_sources.py:232`, `NativeRepositorySourceSnapshot.check_loaded_modules`: unchanged. It first recaptures/verifies the configured Git release, literal catalog, policy and source bytes. It reads module instance metadata, validates repository names, absolute contained paths, exact name/path/spec correspondence and redirected-path rejection, then returns sorted observed paths. A real absolute unlisted repository file remains rejected in both lanes. A repository-root name cannot bypass checks merely by omitting its file or using a non-module/relative/foreign object.
- `worker.py:43`, `_audit_repository_imports`: the sole routing helper selects the unchanged C0 scanner only when source is None; native selects the source audit. It contains no Torch/Warp name exception and no alternate inventory. All three changed call sites use it: startup at line 610, HELLO at line 270, and post-handler/pre-response at line 635.
- Native source/controller/manifest binding remains mandatory. `open_worker` requires the live captured session at phase imported. `ClosedWorkerSession.from_source_session` checks the exact integrated manifest type, rebuilds it from actual source/process/frozen contract, checks the original controller, enforces one worker per source, retains the worker and binds it to the controller. HELLO still calls `_require_native_binding` and checks its manifest digest/lane/device before audit. The helper itself is not an IPC admission API. The earlier bootstrap-only state where audit can return an empty tuple cannot open the native worker.
- HELLO now derives `repository_module_paths` and `canonical_sha256(list(paths))` from the returned observed tuple. The new HELLO regression exercises the production dispatcher with modeled source intake; the cold SourceSession tests independently exercise the real captured audit return. Neither test is misrepresented as a full native integration run.
- Request loop order remains handle -> audit -> encode -> complete response write -> `commit_written_response`. CLOSE cleanup/normal ACK commitment still follows successful complete writing. Audit failure cannot export the already constructed success response or commit CLOSE. Startup audit failure occurs before `open_worker`; it propagates to the existing outer supervised driver rather than creating a usable session.
- Failed native audit remains within the existing `except BaseException: self.fail(error); raise` boundary. `fail` commits retained failure, calls the original bound controller's failure path and terminates conservatively. Real captured two-lane probes show original exception retention in source/controller, full quarantine, no emergency synchronization or replacement terminal ARM, and rejection after removing the offending module and resetting the ordinary failed flag. No restored metadata or successful later audit revives admission.
- External `module_alias` admission remains separate: the unchanged publication reconciler follows raw creator attributes, requires the exact published object and original creator/publisher identities, retains it, and records provenance. The new worker route does not replace that contract with a backend-name allowlist.
- C0's scanner implementation, manifest implementation, native snapshot, reader/native ABI/query code, native setup, runtime custody parser and integrated runtime contract are byte-identical to base. The two C0 paths in the shared dispatcher retain their original scanner calls and ordering through the helper. Production/test diff contains no second reader, new wait/guard, runtime query or resource-release authority.

**Independent check of the motivating native evidence**

The preserved base campaign is evidence about df1a46ae5039fe8a25718c7ec60f52fa7c60c549, not this candidate's runtime result.

All 76 inventory entries independently match local raw size and SHA256, totaling 18,965,974 bytes. The archive SHA256 is exactly `553bff46c12ec27b2f4fe819375b7ba1a80bd0bfaa6404f1dadfce43d6e83893`. All archive members match their local copies; its 77 files/18,980,883 uncompressed bytes include the inventory itself. The namespace evidence SHA256 is exactly `f55a72d87c6d7918823a761efed7f9c9cf99831cf30503b857a44d627821abb2`. These checks were repeated at closure and remained unchanged.

Read-only primary SSH freshly hashed the installed `torch/_ops.py` to `3b58079d356d15cc73082c20b1279468ee9cfe7f5e1340d329a93b6e877a30d5` and read its `_Ops(types.ModuleType)` definition: `__file__ = "_ops.py"` is a class attribute, while initialization supplies the `torch.ops` module name. No backend was imported. The actual CUDA raw custody journal has the matching `torch.ops` module_alias creator_publication event with creator_attribute provenance; its alias row and all artifact rows exactly equal the supplied namespace evidence rows.

The old scanner calls `getattr(module, "__file__", None)` and resolves that inherited relative class label under repository cwd. Cold CPU reproduction reaches its exact `_ops.py` error. Native snapshot uses instance metadata and does not invoke lazy namespace lookup. This explains the observed rejection without treating a class label as a real repository import. Both the committed negative test and new real-bound-source probe still reject an actual unlisted absolute repository path.

Fresh cold `-I -S -B` subprocesses on the pinned Python reconstructed both lanes' trusted source/configuration/interpreter-alias inputs from the clean base release. They matched the preserved inputs exactly. I then replayed each raw record, durable selection and custody attachment using the production parsers, and ran `assert_native_input_scope_success`. The parser/checker files used for replay are unchanged in this delta. Replay imported no Torch, Warp, NumPy or ctypes and acquired no native library.

| Raw evidence | Independently observed |
| --- | --- |
| Warp | 283 required input scope rows; HELLO/OPEN/PREPARE, two sample execute/resolve/release sequences and CLOSE successful; both samples RECLAIMED; historical submission latch retained; normal close ACK committed; release completed; clean_reclaimed_exit; zero emergency sync |
| Warp process chain | Worker 2554177, supervisor 2554174, parent 2553391; supervisor evidence EXITED, observed worker returncode 0, 27 normal completions, no kill/deadline expiry |
| CUDA | 291 required input scope rows; 12 artifact rows: three create before/after pairs and three exec before/after pairs; upload/readback completed; warmup work and final setup graph/barrier present |
| CUDA rejection | PREPARE response is error; raw journal fault is `worker loaded repository module outside source allowlist: '_ops.py'`; no successful sample submissions; both sample states remain PREPARED; cancelled_before_submission_exit; zero emergency sync |
| CUDA process chain | Worker 2560882, supervisor 2560871, parent 2553391; supervisor evidence EXITED, observed worker returncode 64, 22 normal completions; terminal cpu_cancel ARM once; no kill/deadline expiry |

The input record digests are Warp `566a11a41db7c0710a489fb9049c37cabf8039b41f7427408974418af8eb6df1` and CUDA `0ddea169abe24863d71472dcf6a02a8d446f12d353e5d10608c0ee05f353eaf7`. Their persisted record summaries stop at input_validation; the later final setup/barrier facts come from the attached terminal worker trace. I did not relabel the input cut as final sample coverage. Both attachments validate as exact worker prefixes. Fresh `/proc` checks find all four worker/supervisor PIDs, parent 2553391 and wrapper 2553209 absent.

This verifies the stated campaign diagnosis and bounded input evidence. It does not prove GPU arithmetic of final samples, every physical scope after input validation, or this candidate's native success.

**Independent commands, results and limits**

Local environment: PowerShell and `C:/Python314/python.exe`, Python 3.14. Pytest used explicit repository PYTHONPATH, `-B`, `PYTHONDONTWRITEBYTECODE=1`, disabled plugin autoload/cacheprovider, and TEMP/TMP/basetemp under the allowed scratch. Exact expanded argv/cwd/PIDs/times/exit codes are in the `*_execution.json` records. Fixed test releases are LF packages created only in scratch; they are not an attempt to execute the Windows CRLF checkout as a native deployment.

Commands from the frozen candidate directory, where `<scratch>` is the full scratch path above:

```text
C:/Python314/python.exe -B <scratch>/audit_evidence.py
C:/Python314/python.exe -B <scratch>/run_checks.py
C:/Python314/python.exe -B <scratch>/check_remote.py
C:/Python314/python.exe -B <scratch>/run_replay_remote.py
C:/Python314/python.exe -B <scratch>/run_checks.py boundaries
C:/Python314/python.exe -B <scratch>/audit_evidence.py
C:/Python314/python.exe -B <scratch>/final_remote.py
```

| Fresh independent check | Result |
| --- | --- |
| Targeted pytest selection | **238 passed in 224.10s**, exit 0; no skips |
| New reviewer boundary probes | **7 passed in 23.95s**, exit 0; no skips |
| Catalog/policy/source checks | **91/91 committed LF sources** pass digest and actual AST validator; both literal lane inventories match; current worktree LF-normalized bytes equal HEAD; exact bootstrap/reader pins validate |
| SourceSession AST comparison | Only the new observed-tuple initialization and return differ in the class |
| Raw inventory/archive/namespace checks | All supplied sizes/digests and archive copies match, before and after tests |
| Primary cold input reconstruction and CPU replay | Both lanes pass record/selection/attachment and input-scope coverage; no backend/FFI imports |
| Candidate hygiene | Initial/final porcelain including ignored files empty; exact HEAD unchanged |

The targeted selection, fully enumerated in `run_checks.py`, covers the new audit-return and namespace cases; captured module/spec/package replacement and no-reexecution cases; the full CPU source-custody and source-policy files; native HELLO, two-sample ACK-before-release and manifest mutation cases; C0 scanner rejection, closed preparation gates and loop read/write failures; and missing C0 transitive bootstrap sources. It deliberately excludes GPU cases and destructive timeout/kill campaigns. No competing full-suite rerun was started, and no author's ongoing broad-suite result is claimed here.

The new reviewer probes add two real captured-source/controller poison cases, four production `run_worker_loop` ordering cases (success, startup audit failure, post-handler audit failure, write failure), and unchanged C0 helper delegation. Loop I/O/session leaves are modeled; actual codec and loop control flow execute. These tests do not claim real socket loss, OS worker reaping or native resource behavior. The real captured poison probes use the existing CPU peer and exact production loader/session/controller.

Prior nonblocking C2 is closed for this candidate: both isolated source-policy children now explicitly include `-B` and assert `sys.dont_write_bytecode`. Both tests ran successfully from a checkout with no ignored caches; final ignored status remains empty. The old checkout's six ignored pyc files were not touched. Its earlier automatic cleanup rejection was neither bypassed nor retried.

One reviewer tooling limitation occurred: initial local Linux-evidence parsing on Windows failed the production normalized-absolute-path check. No validator was weakened. Replay was moved to the authorized pinned Linux interpreter and passed. This was a host/path mismatch in the review harness, not a candidate failure. One guessed support filename was absent; the actual runtime_custody_launch/preflight helpers were then located and read. No candidate assertion failed and no implementation was changed to obtain a pass.

**Process closure**

Reviewer CLI PID 5648, parent 7056, creation Unix milliseconds 1791626301817, corresponds to this independent session; it is still serving this review, not an abandoned child. Targeted pytest PID 31392 (driver 44220) and boundary pytest PID 47848 (driver 44116) were waited/reaped with exit 0. Initial SSH PID 31796 (driver 43708) and replay SSH PID 46172 (driver 15956) returned exit 0. `process_closure.json` records a scoped Windows process query at 2026-10-10T18:05:37+08:00 with no recorded test/SSH/driver or scratch-scoped Python/SSH/SCP residual.

Primary inspection PID 2642847/parent 2642840 and cold preflight PIDs 2642854/2642885 were absent at the final remote check, together with the six campaign PIDs above. Final closure probe PID 2657305/parent 2657301 returned synchronously through SSH PID 40800 (driver 41260), exit 0. Remote replay was a synchronous stdin-script command with SSH exit 0; its remote PID was not separately sampled. Routine Git/inspection subprocess identities and every short-lived pytest child PID were not individually inventoried; the runners observed completion, and final scoped process scans found no residual. These are bounded observations, not exhaustive process attestation.

No worker/supervisor/GPU job was launched by this review. No process was signaled. No GPU-process query was made because the packet prohibits GPU queries. Existing campaign GPU inventory files were preserved, not represented as a fresh query. There is no unresolved owned residual or hang observed in this round.

**Exact permission and remaining gates**

Approval binds only **22b1db0ee619f05e30f03bd5e339f03e883e97a5** and this native source-audit/test-hygiene delta on top of the predecessor's approved preparation-reader preflight. It authorizes resuming bounded native setup and same-production-framed-worker validation on the pinned deployment: both lanes, setup/input observations, CUDA's six real create/exec pairs, affected audit/query/scope failures, retained custody, response/ACK ordering and actual observed/reaped closure. Changes to code, pins or scope require the existing independent review workflow.

Still required are this candidate's fresh native run and affected failure evidence, the reader's outstanding real failure/Go2 matrix, and complete C2f final-sample physical/numerical, workload/profile/diagnostic/generation, post-submit/disconnect/ACK/cleanup/hang acceptance. No CPU test or preserved base run substitutes for them. The predecessor F3 compact-reply/journal-recovery implementation is outside this approval; inspecting its unchanged use in fault paths does not approve it.

**Full reader hardware acceptance, final-sample physical/numerical proof, full C2f/C4, F3 predecessor approval, C4c3/c4/C4d/B3, formal performance, product activation, result/baseline generation, backend comparison/selection, R2/S/host delivery, a3b and shipping remain closed.** No speed or final numerical claim is made.

The review stops here. This complete unchanged report may be archived verbatim under collab in a separate evidence-only recording under the shared workflow exception. Such recording cannot change the reviewed revision, verdict, requirements, scope or gates. No candidate edit or evidence commit was made by this reviewer.
