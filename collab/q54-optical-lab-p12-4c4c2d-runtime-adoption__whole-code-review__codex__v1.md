# C2d3 independent whole code and acceptance review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent whole code/acceptance review
Author: codex, independent reviewer /root/c2d3_whole_review_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: runtime-integration design/review v3; runtime-adoption design/review v2; runtime-custody implementation-plan/gap-review v3; native-scope-aggregate design/review v2; native-scope-cache design/review v1; complete C2d2-to-candidate source/test/evidence delta
Owner Summary: **Request changes.** Found reproducible upload-event scope validation and aggregate completeness defects, plus twelve failing current CPU regression cases. Existing real-device results provide substantial bounded evidence but do not close those findings. This is an independent whole C2d3 review, not another bounded preflight or an approval of C2d/C4.

## Identity, independence and actual candidate

Fresh reviewer session: `/root/c2d3_whole_review_v1`. I received requirements, exact revisions, accepted designs and evidence locations, not the implementation conversation, and authored no candidate code. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Initial Chinese text decoding was corrected by explicit UTF-8 reads. The referenced integration approval is named `runtime-integration__review__codex__v3.md`, not `__design-review__`; I located and read the actual file.

Actual local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`; branch `implementation/c4-supervised-waits`; HEAD `dedd35eb24d02bb3b5a78c02f7babcf3cc0349ea`. Initial and subsequent tracked/untracked status were clean. The full review base is C2d2 evidence `ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3`, whose approved implementation was `0dd684867a8d60a69978e878b26d98169d2b9b1c`. The whole delta spans 186 files, not merely the last commit.

HEAD changes only PROGRESS and the unchanged bounded terminal preflight report. Its parent is code/test candidate `7cde8879b307090d54350ca3511529bfd8e5d7c8`. The complete e585956-to-HEAD changed-path list contains only tests, cleanup ledger and evidence/PROGRESS records. Production directories `optical_worker` and `tools` have no diff from `e585956eb29e3a33483d8cb0c7444f893b390ac4`. Read-only SSH to primary `tileops-dev` independently observed `/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip` clean at detached `7cde8879b307090d54350ca3511529bfd8e5d7c8`. I did not modify the separate binding worktree.

AGENTS.md expressly authorizes independent review agents. I used two fresh `fork_turns="none"` read-only subreviews: `/root/c2d3_whole_review_v1/bootstrap_review` for full bootstrap/environment/provenance/source-policy flows and `/root/c2d3_whole_review_v1/native_flow_review` for materialization/binding/upload/storage/readback/lifecycle. I reviewed runtime custody, driver/observer, acceptance, raw artifacts and whole scope; independently reran both the upload counterexample and all twelve failing test cases. Subreview conclusions were checked against actual source and were not standalone stage approvals.

## Verdict and blocking findings

**Request changes.** F1 and F2 are production contract defects reproducible without new GPU execution. F3 is broken required regression coverage. The findings do not claim ordinary pinned runtime behavior has produced a foreign event, nor that descriptive records confer live native authority.

### F1 — P2: both upload cores wait on a foreign ready event while retaining target-only scope

Locations: `optical_worker/native_setup.py:286–313` and `:400–427`; compare the actual-event checks in `optical_worker/native_readback.py`.

The CUDA and Warp upload cores retain/adopt the ready event with the configured target ordinal, record it, and enter a target-labelled wait. They do not validate its actual post-record device before that wait. Runtime-adoption v2 section 5 requires actual returned-device validation, expansion of dirty scope on an unexpected ordinal, and rejection before further work. The readback path already has separate creation/post-record validation for precisely this category.

Reproduction uses the sole production upload cores, actual `NativeGuardedSetupLifecycle` and actual CPU guard peer. Array doubles report target1; the event-record double sets the event's device to0. In **both** lanes, the core calls the event wait once on actual ordinal0, returns registered prepared input and completes `normal_disarmed`. The resulting facts remain `guarded_segment_failed=false`, dirty ordinals `[1]`, UNKNOWN=false. A subsequent terminal attempt calls only the CPU target1 sync double and reports `aborted_synced_exit`, guardian RELEASED and release completed.

This is a supported fault-containment counterexample. It is not evidence that actual Torch/Warp normally returns a foreign ready event or a hostile-Python security exploit. The defect is that the promised actual-resource validation does not notice a wrong returned/recorded event and therefore permits later target-only cleanup for a graph whose observed resource was foreign.

Why existing checks miss it: upload-core tests preserve the extracted loop and validate arrays/stream; the new D2H foreign-event tests exercise readback, not H2D readiness. The prepared registry retains the event's identity but does not supply this missing actual-device check. Normal native positives necessarily exercise correctly bound events.

Repair direction: preserve retain-before-inspection, validate creation and post-record event device metadata before the next leaf, and permit CUDA's legitimately unbound event only before its record. A foreign or unavailable actual device must expand/mark scope and irreversibly fail under the original ticket; no later wait, COMPLETE, new emergency sync or ordinary release. Reuse one policy/mechanic where appropriate without creating a second uploader or changing C0 abort semantics. Required proof: both lanes' foreign-create, foreign-record and unknown-record cases, retained returned owner on metadata failure, zero subsequent wait/COMPLETE/sync, existing C0/upload/readback/authority tests, and affected real positive readiness/disposal validation after the repaired candidate's preflight.

Artifacts: `c4_assets/whole_review_v1_native_upload_event_repro.py` and `whole_review_v1_native_upload_event_repro_stdout.txt`. Main reviewer independently reran the saved script with exit0 and observed the same full facts. The repro labels its backend/native behavior as CPU doubles; `Peer.close()` joins the thread, checks peer errors and closes sockets.

### F2 — P2: phase completion accepts a missing external file-admission completion

Locations: `optical_worker/runtime_custody.py:921` (`phase_completed`) and `:947` (`finish`); external transition handling from `:409`.

The parser replays individual `file_call` transitions and external file outcomes separately. `phase_completed` requires every call to have state completed, but never requires each executed file's external admission state to have reached `completed`. `finish` compares call/component histories and derives the lower completed-file count, so adjusting that descriptive count does not restore the omitted check.

Independent CPU replay loaded all eight real successful v2 records from the pinned native-scope archive. For each, I removed only NumPy's external `completed` journal occurrence, renumbered journal/external indices and object first-reference indices, and updated the two derived counts. The full original lifecycle, trusted launch inputs, environment, native observations and component facts remained unchanged. **All eight mutants were accepted** by `RuntimeCustodyRecord.from_payload`, by the original worker-prefix attachment validator, and by `assert_native_scope_success`. Their NumPy external state is still `exec_started`, while later canonical/upload/validation phases and terminal success are accepted.

The approved plan requires ordered file admission, missing-event rejection, completion cross-joins and successful phases to include required admission/publication/fallback outcomes. A delegate returning is not the external adapter's final successful audit/admission. Thus this is a concrete incomplete-evidence acceptance bug, not a demand for cryptographic process attestation and not a live-registry authority forgery.

Why tests miss it: existing mutations cover altered fields, hashes, guards, owners and phase order; they do not remove a final external completion while preserving the already completed delegate calls and recomputing descriptive indices/counts. `validate_runtime_custody_attachment` intentionally checks an exact lifecycle prefix, not a second copy of external admission semantics. Native-scope acceptance checks real helper occurrences and restoration and cannot recover the absent admission outcome.

Repair direction: at normal phase completion, require every external file execution begun in or before that phase to have a consistent successful final external outcome, including required publication/fallback facts. Preserve valid resolution-only probes such as the reviewed deferred Triton `file_resolved` record, and preserve truthful interrupted prefixes before a phase succeeds. Add deletions of completed events, omitted exec progression and related incomplete admission mutations using fixed trusted inputs; retain genuine early/failure cuts and complete real positives. Do not solve this by accepting weaker components or inventing missing completion rows.

Artifacts: `c4_assets/whole_review_v1_replay.py` and `whole_review_v1_replay.json`. The script reads original files only, mutates in memory, imports no NumPy/Torch/Warp, and records all eight accepted counterexamples explicitly. It also validates the unmodified records before mutation.

### F3 — P2: twelve current CPU regressions fail, including ten before their retention scenarios run

| Location | Trigger and actual failure | Consequence / repair proof |
| --- | --- | --- |
| `tests/unit/optics/test_torch_module_replacement.py:148,216`, also assignments in terminal doubles at 147/225 | The importer fixture now supplies a real lifecycle subclass `_Owners`; `termination_evidence` is a getter-only property. Six first-terminal-publication variants and four invalid-alias-publisher variants assign this property and raise AttributeError during setup. | These tests never exercise request/return/frame/audit/malformed/native-global failure custody or callback-free invalid-publisher retention. Adapt the test observer to the real fixture/lifecycle. Do not add a production setter. Require each intended injection, first-terminal retention, single termination, original exception, zero hostile callback and sticky poison assertions actually to run. |
| `tests/unit/optics/test_runtime_environment.py:246–262` | AST oracle expects literal `ENVIRONMENT_SHA256` in the cold test; the consumer now imports shared `PROFILES`. The literal query returns `[]`. | Packaging pin coverage is stale. Verify the actual shared profile and consumer through a CPU-safe path, with meaningful drift rejection; do not reintroduce a duplicate digest constant. |
| `tests/unit/optics/test_native_contract_extraction.py:349–361` | It demands byte equality of the whole current runtime contract with a historical base; the reviewed lane_contract/native_manifest owner additions break it. | Reconcile the test with the explicitly approved two-owner extension while preserving all unrelated closed product/gate assertions. This failure does not prove product activation. |

Main reviewer directly ran the twelve named cases: **12 failed in 2.16s**, pytest exit1. Full output is `c4_assets/whole_review_v1_regression_repro.log`. The independent broad subreview run gives 11 failed/468 passed/3 skipped for importer/environment/policy/global/replacement suites; native-flow regression gives 148 passed/1 failed for contract extraction/host observations/guarded lifecycle. These are current-candidate failures, not the historical 451-pass singleton-stride test-oracle failure.

The supplied 637-case foundation selection excludes the failing test files, and historical b0f22fb results do not establish them at this HEAD. Mandatory source/publication/C0 compatibility regressions cannot be declared closed from those subset totals. Repair the expectations/fixtures without weakening production invariants, run the complete affected suites and obtain independent re-review.

## Nonblocking comments and scope qualifications

C1: the cleanup ledger has explicit historical catalog paths through v11 but omits current `optical_worker/data/native_sources/v12.json`; several newer test-only files are also absent from its explicit paths. The ledger does retain the relevant long-term selected-backend/source/schema policy and P12.4e exit conditions. Add the current catalog and reconcile new temporary test entries during repair; do not delete history or a lane now. This is not evidence of activation or a resource leak.

C2: native-module-name design/review v1 asked for import key and observed intrinsic `__name__` separately in final custody. The newer approved runtime-custody plan v3 explicitly freezes a five-field file provenance without intrinsic name; the implementation follows that newer grammar and performs the name check live. Record this older/newer design inconsistency explicitly rather than claim intrinsic-name observations are present in aggregate. I do not classify conformity to the later exact grammar as another production bypass. The bootstrap subreview documents a CPU observation showing the factual omission.

Full-base `git diff --check` reports trailing whitespace in two unchanged archived review documents (optional-import gap-scan v1 and runtime-binding-fixture review v1). This is not a code finding; reports must remain verbatim under the evidence exception. Production-delta whitespace inspection does not expose a corresponding implementation issue.

## Whole-scope assessment

I reviewed the complete C2d3 call chain and source/test delta, with the two bounded independent subreviews, rather than treating prior local preflight approvals as whole acceptance.

The architecture retains one captured repository loader/finder, one guarded controller and the existing lifecycle/wait/evidence machinery. Runtime environment and publication outcomes are fixed deployment tables, not request-driven module discovery. Exact file/spec/delegate/module relationships, native create/exec intervals, requester contexts, creator roots, Torch original/replacement identities and interpreter alias ownership are checked live. Prefixes, type names and copied digests are not accepted as live owner provenance in the inspected supported path. Native/interpreter/driver internals remain the approved trusted dependency boundary; none of these Python checks claims hostile same-process sandboxing or the ability to undo unsafe native constructor cleanup.

Canonical materialization calls the existing production workload/materializer and uses one immutable canonical source; the actual input binding has exact object/session/controller/lane/target/stream identity, one consumption and sticky failure. Shared cores preserve C0 wrapper signatures and abort-policy separation. Storage-only registration checks are separate from readiness; stdlib-only manifest extraction reexports the same definitions. Real validation reads back into adopted host buffers using the existing copy helpers and shared CPU comparator. The Warp readback scope is an actual original `ScopedDevice`; the observer retains owners/original functions and compares native post-exit context, not merely successful Python exit. F1 is the identified missing upload event check in that otherwise shared flow.

The aggregate is an immutable descriptive codec, uses separately captured source/configuration/case/interpreter launch facts, delegates lifecycle validation, and does not register live module/binding/storage/readiness authority. Object tokens are held identity references with actual guardian joins, not reconstructed owner labels. Issued bytes, fail-cut fallback, raw diagnostics and record-before-sidecar persistence preserve earlier validated cuts without resuming execution. F2 is the identified gap in external-admission completion replay. The native_scope version keeps lifecycle versus journal indices distinct, ties observations to actual latest leaves and exact guards, permits nested Warp scope observations, and treats emergency sync separately from normal DISARMED completion.

The fixed test observer is outside captured production code. The inspected ABI/query paths use pinned mapped libraries and RTLD_NOLOAD, separate successful null from getter errors, and gate Warp's potentially initializing stream-cache lookup on the approved original-path/cache/error preconditions. CUDA cold logical state remains explicitly source-derived while postinit state is actually queried. Contradictory full rows are retained and fail acceptance; failed getters do not fabricate successful rows. I did not run native getters myself.

Product PREPARE remains rejected. The only C0 exact allowlist additions are the reviewed CPU `lane_contract` and `native_manifest` owners in both literals. No renderer, warmup, final graph/readiness, sample token, optical sample submission, benchmark result, baseline or dispatcher activation was added by this candidate. The selected-backend custody/schema infrastructure and losing-lane/test cleanup retain the P12.4e decision boundary.

## Requirements to raw evidence and coverage limits

Archive hashes and sizes below were independently recomputed from local files; the corresponding raw summaries/logs/residual records were read, rather than relying only on collaboration summaries.

| Required coverage | Inspected original evidence | Independent review result / remaining limit |
| --- | --- | --- |
| Both lanes, asymmetric/replay/Go2 960/1920, actual v2 native state and readback | `c2d3-e585956-native-scope-v1.tar.gz`, 7,821,994 bytes, SHA256 `c84af059c9e32a8048b1b6d9bb5efa7af09e62a8a7e38d54fa3b5d0b444dc230` | Raw 16-pass run, eight positives/eight negatives. Independently parsed all 16 exact records/selections/attachments and same-run external identity joins; all eight positives pass native occurrence acceptance, totaling 1,988 native rows. Eight negative records are current terminal cuts with UNKNOWN and no authorized release. This is input/native scope proof, not rendering. |
| Import/init/allocation/copy/event/readback faults; ACK/EOF; real foreign/UNKNOWN; cleanup/missing-exit hangs | `c2d3-e585956-runtime-fault-matrix-v4.tar.gz`, 18,390,205 bytes, SHA256 `df9f92da1bfe8dcf4095dd598381c22f4b0974f1e982b89af97ef56eed83b359` | Raw 85 passed/4488.89s, no filter/skip/outer timeout. Independently replayed all 85 records and same-run external identity joins. Selections: 51 current terminal, 28 current prefix, six prior_validated_cut terminal. Those six older cuts remain explicitly partial with diagnostics; no manufactured full-tail claim. This matrix excludes the first three terminal selectors. |
| Exact candidate C0/guard/lifecycle/arena/source/codec regression | `c2d3-e585956-foundation-cpu-v2.tar.gz`, 16,327,350 bytes, SHA256 `7396ebb1ae39d692a3d58648ce4dae29085fe4b50dac23f23d47073121d7ba9d` | Original stdout really says 637 passed/125.94s; supplied residual record has 65 observed identities gone. It is a selected fourteen-suite run, not all C2d3 CPU regressions; F3 is not covered. Historical 303-pass/5-fail missing-owner archive remains a distinct prior failure. |
| Actual-record mutation and no parsed authority | v4 SHA256 `2b6b073073c9ab82e2af4669459c70f2eda3db3c4f45b1bba79b35ca8a7a019d`; v5 SHA256 `82f462d83c4483b48d40cf20243d9af341254132dcf02eca3900c74b78f7713f` | Raw v4 is 451 passed/1 failed/36 explicitly inapplicable. The failed [1,1] CUDA [1,9] singleton stride is consistent with the approved nondegenerate-axis rule, not a demonstrated parser defect. Current repair mutates a dimension >1; raw v5 is 96 affected passed/392 deselected/49.90s. I do not present this as a new full 488-case pass. F2 adds a distinct uncovered mutation. |
| Both lanes × two cases × terminal success/sync-raise/sync-hang | Running `c2d3-7cde887-terminal-matrix-v1` | At the independent replay cut, seven completed records existed: four successful disposals and three sync-raise terminal failures. All seven parsed/attached and joined their live identity files. Full twelve-case result and final archive/residual check were not yet supplied at that cut. No whole terminal pass is inferred from seven cases or from the previous 85. |

The first two archives' raw residual checks contain respectively 34 and 172 unique owned PID references, successful GPU query return codes and empty owned OS/GPU residues. The fault archive has 85 external chains; native-scope has 16. I independently checked each actual worker/supervisor PID against its original supervisor stdout and its external PID/PPID/start-tick chain in the replay. This validates the supplied retained same-execution evidence; it is not new GPU execution or cryptographic process attestation. Historical b0f22fb 501-pass/one-platform-skip and earlier v0/v1 records remain historical, not reassigned to the current candidate.

## Executed verification and artifacts

All candidate operations were read-only. Main commands used PowerShell, `rg`, Git and `C:/Python314/python.exe -B`; pytest did not use `-I`, autoload was disabled and cacheprovider disabled. Scratch/repro files and pytest basetemp paths were outside the candidate. No package install, GPU launch/reset, private context change, broad signal, fallback host, source edit or second upload implementation was performed.

Main local six-suite selection: test_runtime_custody, test_runtime_native_scope_codec, test_native_scope_observer, test_runtime_custody_native_failure_cold, test_runtime_failure_observations and test_native_runtime_driver_process: **54 passed, 45 skipped in 59.84s**. The skips require POSIX or retained deployment evidence and are not passes. Existing real Linux archive replay was separately performed; I did not relabel all local skipped test functions as rerun.

Native-flow subreview: test_native_upload_core, test_guarded_runtime_binding, test_guarded_native_readback and test_native_manifest: **119 passed**. Its second selection (test_native_contract_extraction, native input host observations, guarded lifecycle): **148 passed, 1 failed in 12.74s**. Bootstrap subreview's final logged five-suite run: **468 passed, 11 failed, 3 skipped in 30.05s**. Counts overlap other supplied rounds and are not additive unique-case claims. The main reviewer then directly reproduced the twelve failures with a focused command (12 failed/2.16s) and reran F1's saved production-core script.

Remote CPU replay command:

```text
Get-Content -Raw ../c4_assets/whole_review_v1_replay.py |
  ssh -o BatchMode=yes -o ConnectTimeout=8 tileops-dev
  /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B -
```

It inserted only the pinned repository for CPU imports, read original artifacts in place, checked roundtrip, selection, lifecycle attachment, same-run identity and positive native occurrence coverage, then ran eight in-memory F2 mutants. Result: **108 originals validated** = 16 native-scope + 85 fault + seven then-complete terminal, eight incomplete-completion mutants incorrectly accepted, `backend_imports=[]`, exit0. Main artifacts: `whole_review_v1_replay.py/.json`, `whole_review_v1_archive_inspection.json`, `whole_review_v1_regression_repro.log` and F1 script/stdout. PowerShell redirection initially wrote the replay output as UTF-16; it was converted to UTF-8 JSON outside the repository without altering its data. An initial inline archive-reader path was mangled by PowerShell's legacy stdin encoding; the corrected script uses the actual working directory. These were review-tool errors, not candidate failures.

Bootstrap detailed report/log are `whole_review_v1_bootstrap_review.md` (SHA256 `050a30a6e72f849de4bb3b3e16869729d0e1669e520226357fa29c1109fb05f6`) and `whole_review_v1_bootstrap_cpu_tests.log` (SHA256 `cc6292284228dd68061383621ecb5fef7d9776198615aaf6e5fdd041e3b4958f`). Its separate intrinsic-name probe is diagnostic only. All original supplied archives and candidate files remain unchanged.

Not executed by this reviewer: new native/GPU acceptance, actual foreign-event injection, dependency installation or complete current CPU suite. Native fault/hang/reap evidence is inspected original execution, not a reviewer rerun. The approved native library/driver exception-safety trust boundary remains; CPU double success cannot prove native internal cleanup safety. No optical numerical rendering, performance or later-stage validation was attempted.

## Process hygiene and closed gates

Remote replay recorded Python PID3176312/parent3176311; a subsequent exact `ps -p 3176312,3176311 -o pid,ppid,lstart,args` returned only its header. Reviewer CPU peer threads closed/joined; completed main tool sessions returned exit codes. Final scoped local process search found no Python command matching this review's scratch/test names. Short-lived utility and all local pytest descendant PIDs were not exhaustively sampled, so this is a bounded residual check, not a complete OS attestation. The supplied native runner/worker residual evidence remains separate; no reviewer signal was sent to any ongoing terminal worker or unrelated process.

Approval is withheld for the whole frozen C2d3 candidate. Repair F1/F2/F3, retain original failed evidence, refresh affected checks and obtain a fresh independent whole re-review. A finished terminal matrix cannot by itself close the code/parser/test findings. A Request changes conclusion does not imply prior 16/85 native results are false; it means they do not cover all promised correctness and acceptance obligations.

Still CLOSED: C2d as a whole and C4 completion; C2e/f; C4c3/c4; C4d/B3; renderer/final graph/seal/quiescence, warmup/R_setup/R_final and prepared-sample authority; product GPU preparation/dispatch; a3b; result/baseline; host payload, R2 and shipping/activation. No backend selection or losing-lane deletion is authorized. An unchanged report/evidence-only archive commit may record this verdict but cannot change it.

Final integrity check after writing this report: local HEAD remains `dedd35eb24d02bb3b5a78c02f7babcf3cc0349ea`, parent remains `7cde8879b307090d54350ca3511529bfd8e5d7c8`, and porcelain status is empty. The coordinator confirmed the original terminal runner/pytest were still live; this report deliberately retains the seven-case evidence cut and does not wait for or predict their eventual result. It can be followed by a separate unchanged-result archive without changing this Request changes verdict.
