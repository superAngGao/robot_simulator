Initiative: Q54 / P12.4 / C4c2 native-state-reader repair
Stage: Fresh independent CODE / NATIVE-EVIDENCE acceptance, review only
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files: native-state-reader design-request/design-review v1 and complete code-preflight reviews v1/v2/v3; native-scope-aggregate design v2 and cache supplement v1; native-worker-source-audit, terminal-target and stream-adoption-order code-preflight reviews v1; Warp-exit-fault test-code-preflight review v1; runtime-integration design/review v3; stream-adoption-order implementation-note v1
Owner Summary: Approve the complete bounded preparation-reader repair at frozen HEAD 1286cbb94040b83e44cffbfcec98497f9af77138, whose production/test source matches cf7f5a6d9ff9b3971144410a7c3ad156a806646c. No substantive unresolved finding or mandatory native-evidence gap remains in this predecessor scope. Independently executed 612 CPU checks and verified/replayed retained native evidence. No new native execution occurred. Full C2f/C4, final observations/output correctness, remaining integrated refactor/fault work and all product gates remain unapproved.

**Verdict: Approve — bounded CODE / NATIVE-EVIDENCE acceptance.**

This accepts the promised preparation reader as a prerequisite repair: its sole production query implementation, source policy and captured custody; runtime initialization and CUDA artifact observations; input upload/readback observations and eligible input-stage terminal observations; identity, cache, retention, failure and adoption ordering; consolidated test facade and CPU coverage checker; and the affected successful framed-worker integration repaired by source-audit routing and terminal-target binding. It does not accept the complete integrated worker or grant future final-observation permission. No additional native campaign is required to decide this bounded predecessor.

**Session, revisions and preservation.**

Independent session: **01a125a2-0c60-7aa3-8736-a5b6c3a977f7**. Actual reviewer CLI PID **40760**, parent **46608**, creation Unix milliseconds **1791632739233**. The launch packet and live Win32_Process identity agree. I received the acceptance packet without the implementation conversation, did not author the candidate, and spawned no agents or equivalent reviewers. This session itself is the fresh independent review.

| Item | Independently checked |
| --- | --- |
| Candidate | C:/Users/AG/新建文件夹/c4_native_reader_acceptance_review |
| Actual HEAD | 1286cbb94040b83e44cffbfcec98497f9af77138 |
| HEAD tree | 10ee84ac76c466f00fc5ff5b5504e0d466fbb868 |
| Source/native-evidence revision | cf7f5a6d9ff9b3971144410a7c3ad156a806646c |
| Source revision tree | 2ab19d4d0e9a221516584fa302e87d57bdcd7888 |
| Branch | Detached HEAD |
| Initial/final status | No tracked, untracked or ignored entries; porcelain-v2 contains only detached branch headers |
| Complete reader repair base | 868225db0ac5794bd7089cd73357b5c920d0c0e9 |
| Reader preflight source | df1a46ae5039fe8a25718c7ec60f52fa7c60c549 |
| Source-audit fix | 22b1db0ee619f05e30f03bd5e339f03e883e97a5 |
| Terminal-target fix | 0be9876b9778cf4826a62f45c373a38d60ff33f1 |
| Immediate stream-order repair base | 91e11eab275f6434754b9449200bf5cb7bf45655 |
| Accepted reader design, committed LF SHA256 | 0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5 |
| Current reader LF SHA256 | 3a67a897834c7c118d9cac850ea1f3f68a8f0686daf318b0a0715d0ddd3b40f7 |
| Current bootstrap LF SHA256 | 7822eba259d80919cb5791e884d1084b4c099ba3fa5bf78fe3c6e38cc962546c |
| Catalog v18 LF SHA256 | 0433890aa9f5720001e91426c8bd37bb9061370338f56861a22f852dce5297cc |

I read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md fully; corrected the initial PowerShell Chinese decoding with UTF-8. The required reports and original reader findings were read, including v1 R1/R2, v2 R2-followup/R3 and v3 dispositions. I also inspected the original worker-intake findings, cache/ordered-diagnostic findings, accepted parent design and actual repair history. Oversized outputs were followed by focused reads. Prior verdicts supplied requirements and scope, not proof of current correctness.

The actual full base-to-HEAD diff has 46 files, 5,059 insertions and 643 deletions, including accumulated review/design records. The cf7f5a6-to-HEAD diff contains only PROGRESS.md and seven collab documents. `git diff cf7f5a6 HEAD -- optical_worker optics tests` is empty. `git diff --check 868225d HEAD` passes. Thus neither the proposed final-observation v1/v2 designs nor their review adds implementation to this frozen source. They were not used as accepted predecessor requirements. No moving main worktree was inspected.

All review-created files, extracted archive copies, copied unchanged repros, logs, temporary directories and this report are under **C:/Users/AG/新建文件夹/c4_assets/native_reader_acceptance_scratch_v1**. No candidate file, private operating context, ignored cache or original evidence was edited or deleted. No installs, deployment, GPU/native calls, process signals or host switch occurred. Read-only SSH used primary tileops-dev only, for pinned source/environment/assets, existing evidence CPU replay and scoped process checks. No remote files were written by those scripts; subprocess preflight used -I -S -B.

**Findings and original-finding disposition.**

No substantive blocking or non-blocking implementation finding remains in the reviewed scope.

| Original issue | Current disposition and evidence |
| --- | --- |
| Worker-intake F1: production CUDA lacks required artifact pairs | Closed for preparation intake. Production bootstrap directly uses the reader around original create/exec calls. Current actual framed CUDA evidence parses with all three create and three exec pairs, PREPARE succeeds, and both samples resolve/reclaim/close. No successful test fact producer is installed on runtime_worker_success. |
| Worker-intake F2: missing Warp physical evidence | Preparation portion closed: complete occurrence-based input observations and physical ambient/interior restoration exist on current source in both standalone and framed runs. Final-sample/resolve/close physical and numerical portions remain outside this repair and unaccepted. Their absence is not a preparation-reader regression. |
| Worker-intake F3: compact terminal reply/journal recovery | Outside this approval. Its unchanged use is not a separate code or parent acceptance. |
| Reader v1 R1: queries after emergency failure or release | Closed. native_scope.py:41 admits only healthy original setup or original in-flight target sync with retained custody, PARTIAL_FROZEN guardian and no release authorization; live emergency ticket alone is insufficient. Preserved post-release/failure probes and current active-sync/post-sync/failed-sync/cleanup cases pass before any modeled getter. |
| Reader v1 R2: lost source, copied/substituted reader or binding, caught rejection | Closed. Guardian-held original source, exact reader/binding custody and seven-slot record are checked; mandatory captured input rejects phase drift. Preserved original identity probes pass; caught failures cannot COMPLETE or revive admission. |
| Reader v2 R2-followup: mutable upload-context source rejection not sticky | Closed at native_setup.py:1412. Registration is consumed and the retained original controller supplies failure authority before checking replaceable fields. Both-lane field-drift/replacement probes reject before native work and poison the original controller, including substituted lifecycle fields. |
| Reader v2 R3: caught raw getter failure allows later queries/release | Closed at native_scope.py:356. The shared raw query body itself retains/fails/rethrows. Actual diagnostic-facade probes cover CUresult and Python exceptions in setup and active sync, preserve ordered prefixes/original exceptions, reject later getters and prohibit ordinary release. |
| Design C1: input/later-helper/C0 distinction | Satisfied. Captured input requires the reader; later shared helpers do not emit input receipts; diagnostic facade delegates to the one reader; C0 remains reader-free. Zero-size occurrence and normalization regressions pass. |
| Reader preflight v3 C2: isolated children write bytecode | Closed in current tests: isolated source-policy children include -B. Both ran; final ignored status is empty. No old cache was removed. |
| Earlier native-scope cache F1 and ordered-diagnostic F2 | Retained fixes verified: exact pinned Device model rejects stale/error/NULL/unproved cache before getter; first stream must match. CUDA getter-position tests preserve successful prefixes and exact failed calls without retries. |
| Source-audit integration failure on torch.ops relative label | Closed within the accepted native source authority. Native startup/HELLO/post-handler audit uses captured source audit and its observed ledger; C0 keeps its scanner. Current framed CUDA path reaches all responses successfully. CPU namespace/unlisted-real-file controls pass. |
| CUDA terminal-target identity failure | Closed. Exact factory/transaction context, context device argument and exact stream remain required; indexed device value agrees with retained target. The retained target supplies scope checks. Distinct-device two-sample and substitution/wait-failure CPU tests pass; current real CUDA RESOLVE/RELEASE/CLOSE succeeds. |
| Stream-adoption-order regression | Closed against the actual failed 22b1db0 case. Current returned foreign0 yields dirty [0,1] before rejection; target streams are adopted before reader failure. Original-method CPU probes reproduce the old order; current probes and all four native foreign/unknown cases pass. |

**Code, ABI and resource assessment.**

The full extraction and subsequent corrections were inspected, not just the last four-line reorder. NativeStateReader is the sole query/cache/state implementation. The SourceSession retains it and its mutable owner/raw containers before initialization, and the existing guardian adopts it. Exact source/controller/module/runtime/target/stream/binding and owner-thread/ticket checks precede queries. Source-bound upload and readback do not recover a missing reader through metadata or a test hook. No second guardian, release mechanism, wait, deadline extension or execution capability was introduced.

Library initialization still checks the fixed environment descriptor, resolved regular path, bytes and already-mapped path before RTLD_NOLOAD|RTLD_LOCAL acquisition. Handle/function retention precedes configuration/calls. The literal ABI remains cudaGetDevice(int*) -> int; Warp's five checked CUresult calls use c_int device/result, c_uint flags and c_void_p context. Actual CUdevice enumeration is retained instead of assuming handle equals ordinal. Nonzero CUresult is rejected before consuming output; successful null context is distinct. Warp getter declarations are validated, not rewritten: [] -> c_char_p error string and [c_void_p] -> c_void_p cached stream.

Cuda cold count/primary queries, three exact queued callbacks and empty lazy-seed tracker precede real stream creation. Logical0 is derived only from raw0/uninitialized/all-inactive facts; warming logical/current-stream getters follow the retained real return. Postbinding Triton nonexecution remains checked. CUDA expected restoration is logical0/raw1 after initialization, with only primary1 active and unchanged captured current-stream baseline; it is not an invented raw0 requirement.

Warp cache proof requires absent raw context/stream and exact empty error bytes before the original get_stream, then exact returned/raw-stream identity, positive context/handle and exact empty error bytes before enabling native stream observation. The first native stream equals the held return. I read the retained hash-matched Device/ContextGuard source and freshly re-read pinned ScopedDevice, native cache getter/setter and error paths. The getter's cache miss can select context and allocate; it remains forbidden as a fallback. The persistent native error buffer excludes the original void-setter failure branch. No cache initialization, error clearing, setter or retry was added to the reader. Existing held-graph lifetime and terminal checks bound cache use.

Initialization keeps immediate session retention and returned-owner journaling ahead of returned metadata/adoption. Existing device adoption now precedes reader.stream_returned in both lanes. Foreign adoption expands known scope before rejecting; a pre-return or metadata failure does not invent a known foreign owner. CUDA artifact before/after pairs surround the original scoped delegate once, with created-module retention inside its original scopes before unwind. Helpers retain returned arrays/events before after-query, append, metadata or adoption can fail. Readback report/host owners remain held before outer exit.

Warp outer observations occur before real scope entry and after successful original exit. Body failures unwind once and skip successful after observations. Complete contradictory snapshots are retained/appended before scope rejection; failed getters keep only their truthful ordered prefix. No normal continuation, successful COMPLETE, emergency retry or ordinary release follows the failed original guard. The shared raw diagnostic route has the same sticky boundary.

The CPU custody collector and runtime_custody.py retain their original roles and meanings; the complete diff does not weaken the v3 six-pair/artifact/file/returned-owner joins or historical v1/v2 grammar. The shared success checker derives exact eligible occurrences from completed input guards, not counts or all later helpers. Sacrificial success requires one original terminal sync; integrated clean close requires zero emergency sync and actual reclaim/ACK/exit evidence. Setup facts do not certify final samples.

All **91** catalog source entries match committed LF bytes, match LF-normalized local checkout content and pass the actual AST validator. This normalization is an audit comparison, not permission to deploy CRLF bytes. Remote deployed source bytes match those exact LF pins. Full-source reader/bootstrap exceptions and the v18 runtime-contract release remain coherent. The source-policy suite passes changed-symbol/path/mode/recomputed-hash and exception-transfer negatives. The packed-render exception is unchanged. The six relevant GPU/test-support files were separately hashed on primary against this candidate's Git bytes. No backend or ctypes module was imported by source/evidence audit processes.

**Native evidence independently verified — archive replay, not new native execution.**

All four supplied archive SHA256 values match. Every raw inventory entry was independently checked for exact size/hash; member-set equality was checked, including no omitted or additional raw file. All retained primary files were then compared individually to those local inventory bytes before replay. Original archives remained byte-identical at final rehash.

| Archive | Raw files / bytes, excluding inventory | SHA256 |
| --- | --- | --- |
| c4-reader-cf7f5a6-native-scope-v1-evidence-v1.tar.gz | 378 / 185829259 | 5e54d56e64de3e65c09c73df0f47a9ecfbfd6e9e37a41fc3ae41f003336fd652 |
| c4-reader-cf7f5a6-foreign-scope-v1-evidence-v1.tar.gz | 96 / 29000102 | 47f16a1092d84f4f0e9aae8b84d0020e0dbaca92993c4c2083dbc1d696a28af6 |
| c4-worker-cf7f5a6d9f-probe-evidence-v1.tar.gz | 93 / 13829929 | 700412e637e00bda645431ba6bd20067b15f8f8095ccf600efdc318c6c936a7c |
| c4-reader-22b1db0-foreign-scope-v2-evidence-v1.tar.gz | 33 / 2226370 | f57b304eaee6af6f73a04a3038bd9e832754112cfc758eccae884da50f9b7df7 |

The two supplied reader runners match their exact SHA256 values ab65b9a… and de6f3cc…. Their actual-item.module hook changes only pinned profile selection and validates collected parameter sets/function globals; it does not replace test assertions or produce successful state facts. The remote worker runner matches 89ef276…; its revision argument selects cf7f5a6 despite the older filename. The frozen test source independently confirms runtime_worker_success does not install _NativeScopeProbe. Fault/diagnostic facade use in the other tests is distinguished from the uninstrumented successful framed path.

Read-only primary CPU replay used the exact frozen cf7f5a6 source, pinned Python 3.12.13, production record/selection/attachment/lifecycle validators and existing occurrence-based acceptance helpers. It reconstructed trusted launch inputs with the original cold source preflight and compared them exactly to archived inputs, rather than merely trusting copied source hashes. Both environment contracts were parsed and all their current files verified against the requested digests: Warp 02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb; CUDA 000fe3bece3850327e3930630804fd5d5f5a9585172415dc7e0f34ec18ceb0e2. No native libraries were acquired. The 16 retained source/header/library identities were freshly rehashed and match. The archived eight-device mapping includes exact target1/ambient0 UUIDs and driver595.71.05.

Actual Go2 source/asset preflight independently passed at frozen revision da76818e269b82289eba39808e2fb91d679d6994, canonical origin and 25-file complete model inventory. Aggregate SHA256 is 479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0; manifest SHA256 is 5bdac372dd79897e203a1295e5c3e7cb524b1eb018e08f371db603bbfbcee5d5. Current source and archived input validation bind the actual canonical manifests, storage and content. This is input validation evidence, not a new final-output oracle run.

| Current successful input case | Warp native rows | CUDA native rows |
| --- | --- | --- |
| Asymmetric 7x3 | 285 | 293 |
| Dynamic rigid 960x640 | 195 | 203 |
| Real Go2 960x640 | 285 | 293 |
| Real Go2 1920x1080 | 285 | 293 |

These counts are reported only after exact occurrence/identity/state joins pass. Every successful standalone record is current, matches the full terminal worker trace, has actual validated input and all required normal and terminal pairs, one emergency synchronization, completed release and observed exit0. Every successful CUDA case includes three create and three exec before/after pairs. Warp outer state restores default0/null native context; interior helpers observe exact target context/ordinal1. CUDA exhibits the accepted cold and postbinding states. Each record's worker PID/TID agrees with its independently observed live worker identity.

All **nine** scope-matrix negative cases independently replay to worker exit74, UNKNOWN, no input-validation report, no emergency/terminal ARM or sync, no authorized ordinary release, and original failed-guard quarantine. The six named getter/after-return/after-append cases are both-lane controlled failures around real preparation. The selector named runtime_scope_getter_error raises through the reader failure boundary before the allocation snapshot; it does **not** induce a real driver/C10 error return. Actual per-getter errors and prefix behavior are established separately by CPU models. No stronger hardware-error claim is made.

The physical negative controls are distinct:

- Omitted Warp readback scope: the first interior copy-before fact records default0/null current context where target1 is required; it fails before that copy. This is not an exit-restoration test.
- CUDA stream-only copy: the contradictory copy-after fact records raw device0 with primaries0 and1 active; the complete bad state is preserved before failure.
- Actual Warp omitted native pop: the original scope enters and the readback body returns. The test omits its context-pop while restoring Python default0. Exactly one readback_scope/after observation records **current_context = target_context = 98784103096784**, current ordinal1, only primary1 active. Marker guard is setup5, boundary readback_exit_omitted_context_pop. No later normal ARM/COMPLETE, terminal ARM/sync or ordinary cleanup appears after the marker. This is an actual native-context contradiction caused by controlled omission, not a modeled-only test or a spontaneous driver malfunction. The older fc23b4e outcome is not substituted for this current-revision result.

All **four** current foreign/unknown cases independently validate their records, selected cuts, custody and live identity chains. Returned foreign0 in both lanes records dirty **[0,1]**. Unknown pre-return failures retain unknown scope and do not manufacture returned-owner adoption. Every case exits74 with full quarantine, zero sync and no ordinary release. The original 22b1db0 Warp foreign-return case remains a **failed** test: dirty[1], unknown, exit74, safe quarantine but missing known0. Its assertion failure is preserved; the other three original selections did not execute under -x. It is a regression comparison, not a historical four-pass result.

Both actual framed-worker archives contain the full ten request/response exchanges: HELLO, OPEN, PREPARE, two EXECUTE/RESOLVE/RELEASE sequences, CLOSE; all responses are successful with matching sequence numbers. Production parsers verify custody and full terminal lifecycle. Warp has **283**, CUDA **291** input rows; CUDA has all six artifact pairs. Both have exactly two terminal records, resolutions and reclaims; samples are RECLAIMED, historical submission stays latched, normal-close ACK is committed before completed release, guardian is RELEASED, disposition clean_reclaimed_exit, and worker exit is0. Warp has27 and CUDA24 successful normal segments; both have zero emergency sync. Their journal files are empty because no transport fault occurred. This closes affected successful integration only, not final numeric, complete worker-fault or fixed-work acceptance.

The 17+4 reader cases use external live /proc stat/command/executable/start/ancestry chains, cross-checked against requests, worker/native rows, supervisor outputs and closure inventories. The worker probe uses the supervisor's separately captured live /proc PID/PPID/start triples; it does not retain the same full raw-stat/executable record as the reader runner. I inspected that producer and cross-checked its actual child PID, parent/launch identities, framed evidence and closure. These are evidence within the frozen trusted-process model, not cryptographic attestation or arbitrary-hostile-process resistance.

**Fresh commands, results and limits.**

Local interpreter: C:/Python314/python.exe, Python3.14. Tests use -B, PYTHONDONTWRITEBYTECODE=1, disabled plugin autoload and cacheprovider, explicit candidate PYTHONPATH, TEMP/TMP and unique basetemp under scratch. Exact argv, cwd, environment, PIDs, elapsed times and exit codes are retained in execution JSON. No performance comparison is inferred from test durations.

| Fresh independent execution | Result |
| --- | --- |
| Eight committed reader/upload/readback/lifecycle/source-policy/binding/artifact CPU suites plus unchanged copied predecessor probes | **595 passed**, no skips, 42.01s |
| Focused native source-audit and integrated target/identity/wait/reclaim/ACK CPU cases | **17 passed**, no skips, 38.85s |
| Catalog/policy/LF source and archive audit | 91 sources and all four inventories pass |
| Existing native archive CPU replay | 24 distinct cases: 23 current outcomes plus original failed comparison; all asserted record/selection/attachment and scope/lifecycle checks pass |
| Source/environment/assets and process verification | Exact frozen source, environment files, pinned dependencies, Go2 inventory and scoped closure checks pass |

Total newly executed pytest evidence: **612 passed, no skips**. This comprises 506 committed cases and89 preserved predecessor probes in the main run, plus17 focused cases. The copied probes are byte-identical to their originals, with provenance hashes; their own output paths consequently resolve inside current scratch. They include original-method ordering reproductions, identity/caught-failure cases, raw diagnostic boundaries, AST normalization mutations and pair/adoption parser attacks. They are new executions of retained tests, not newly invented independent assertions or reuse of their old outputs.

Principal reproducible commands, with S denoting the exact scratch directory above:

```text
git status --porcelain=v2 --branch --untracked-files=all --ignored
git rev-parse HEAD 'HEAD^{tree}' 'cf7f5a6^{tree}'
git diff --stat 868225db0ac5794bd7089cd73357b5c920d0c0e9 HEAD
git diff --name-status cf7f5a6d9ff9b3971144410a7c3ad156a806646c HEAD
git diff --check 868225d HEAD
C:/Python314/python.exe -B S/inventory.py
C:/Python314/python.exe -B S/audit.py
C:/Python314/python.exe -B S/run_cpu.py
C:/Python314/python.exe -B S/run_extra.py
C:/Python314/python.exe -B S/run_remote.py
C:/Python314/python.exe -B S/run_remote.py v2
C:/Python314/python.exe -B S/run_remote.py v3 worker-
C:/Python314/python.exe -B S/run_sources.py
C:/Python314/python.exe -B S/final_evidence.py
```

Remote replay/source scripts are passed over stdin to `ssh -o BatchMode=yes -o ConnectTimeout=15 tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`. Evidence remains local in source_archive_audit.json, evidence_summary.json, remote_stdout_v2.jsonl, remote_stdout_v3.jsonl, remote_sources_v2.jsonl, final_remote.json and associated command/error records. Replay used Linux path semantics without modifying production parsers.

Review-harness failures are preserved and distinct from candidate failures. The first replay assumed every fault marker included lifecycle_event_count; the two older omission controls do not. Its corrected version uses their first actual guard_failed cut for post-failure checking. That version completed all22 reader/original cases before discovering the worker identity record lacks a raw stat string. The worker-only replay then validated its actual PID/PPID/start schema and completed both worker cases. final_evidence.py checks the combined24-case inventory and the actual physical negative controls. Initial Go2 source verification supplied the application root instead of the original materializer's model-parent root; the production validator correctly rejected it. Using the actual source call's root passed. Two retained-source JSON reads initially encountered diagnostic/non-JSON lines; filtered reads succeeded. No candidate assertion was weakened or source changed to obtain a pass.

The author's 317-pass/13-Windows-skip log, exact13 primary-Linux pass log, launch/exit/closure records and listed tests were inspected and remain **author evidence**, not part of612 fresh passes. Those Linux cases cover real Go2 materialization, actual POSIX release/entry and pinned generated-stdlib behavior. The predecessor211 CPU report/scratch remains intact; relevant original probes were rerun as stated. No blanket full-lint or full-suite result is claimed.

No new native worker, getter/library acquisition, GPU query, GPU kernel, real output recomputation or native fault execution occurred. Native conclusions derive from exact-source inspection and independent replay/verification of the terminal author archives. No binary rebuild equivalence, proprietary driver-internal proof, arbitrary deployment/thread support or exhaustive real driver-error matrix is claimed. The admitted deployment remains eight visible devices, target1/ambient0 and pinned dependencies.

**Owned-process closure and final state.**

Archived OS/GPU closure records cover37 current scope-run identities,11 current foreign-run identities,6 worker-probe identities and5 original failed-run identities. All list no remaining owned OS/GPU process. Their GPU inventories retain unrelated entries; no unrelated process was touched. Fresh read-only /proc checks independently find these59 campaign PIDs absent, plus the two author Linux-CPU PIDs and sampled replay/source-reader processes. No fresh nvidia-smi or native/GPU inventory call was made by this reviewer.

Local pytest PIDs **43712** and **44260** were waited/reaped with exit0. SSH processes, including replay attempts that returned1 for harness schema assumptions, were all synchronously waited. At **2026-10-10T19:55:49.7370892+08:00**, the Win32_Process check found none of16 recorded test/driver/SSH IDs alive and no scratch-scoped Python/SSH/SCP residual. The reviewer40760 and launcher46608 are active review infrastructure, not abandoned test processes. Final source/closure SSH39128 returned0; its remote inspection returned synchronously. Short Git/inspection/preflight children were not each separately sampled; their callers observed completion. These are bounded closure observations, not exhaustive process attestation.

Final candidate HEAD is unchanged and status remains empty including ignored files. Original archive hashes were rechecked unchanged. This report and the scratch evidence are ready for verbatim archive under the shared evidence-only exception; recording cannot change verdict or scope.

**Remaining gates.**

This approval is only for the complete bounded native-state-reader repair at1286cbb, with production/test source exactly cf7f5a6 and the verified terminal evidence above. It neither approves nor implements final-observation v1/v2. Owner work on integrated refactoring, bounded/preallocated control traces, final output/oracle correctness, full R0/R1/profile/diagnostic/generation matrices, remaining setup/post-submit/disconnect/ACK/cleanup/hang faults and whole C2f/C4 continues separately. F3 separate acceptance, C4c3/c4/C4d/B3, product activation, performance, backend comparison/selection, result/baseline generation, R2/S/host delivery, a3b and shipping remain closed. No further campaign or implementation is initiated by this review.
