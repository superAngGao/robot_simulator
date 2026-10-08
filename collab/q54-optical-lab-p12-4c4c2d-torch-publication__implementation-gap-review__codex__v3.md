Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: implementation-gap-review
Author: codex, independent reviewer session /root/torch_publication_implementation_gap_v3
Version: v3
Date: 2026-10-08
Status: accepted for bounded repair scope only
Related Files: approved torch-publication design-request/design-review v1; runtime-adoption design-request v2; native-global-publication design-request v1; implementation-gap-review v1/v2; implementation-note v2/v3
Owner Summary: Approve the bounded F1-F4 repair delta at exact a3d750b. The callback-bearing publisher counterexample is closed, failure and original-error custody precede ancillary intake, and the original three defects remain closed. Independent Windows and pinned-primary183 CPU checks passed within their stated limits. This is not full Torch supplement, captured runtime, aggregate custody or whole C2d3 approval.

VERDICT: Approve — bounded F1-F4 repair delta only.

1. Exact revision, independence and scope

Fresh independent code-review session, not the implementation author; no inherited implementation conversation. Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, rereading Chinese with UTF-8 after the initial shell decoding was garbled. Read the approved Torch supplement and adjacent independent design review, governing runtime-adoption v2 and native-global-publication v1, mandatory prior review reports v1/v2, and author implementation notes v2/v3. Author claims and prior verdicts were inputs to check, not substitute evidence.

Local repository: C:\Users\AG\新建文件夹\p12_4_c4_warp_timeline_design
Local branch: implementation/c4-supervised-waits
Evidence-only base: cbfb1cec1e1a783ee5a36b2dbb70fdb8246bbaa9
Code predecessor: 68e10d2b13249dfb38d6a92f4335e344602f4854
Reviewed HEAD: a3d750bbaf79d760a6727e56146d277e1cb3b770
Reviewed tree: 9982c6d9ec44e7232537643254d3c00e8f76d445
Approved design SHA-256: 2dace48ad6002db9cb6523a6852ef2ec64314130c59563762c17dd95371f7186
Prior v2 report SHA-256: b273f1b08d265f94d22d1ae0cb813a2a26eb55a3a27be2da69cc1968c059c68c
Both supplied document digests independently matched. Base cbfb1ce contains only the prior unchanged report archive. The base-to-HEAD delta contains the implementation note, two production bootstrap changes, exact policy/catalog repins and ten regression cases in two test files. No new permission, alternate importer, upload implementation or product gate appears in this delta.

Initial and final local status --short were empty, with exact unchanged HEAD. After author synchronization, independently verified primary183 /data2/shared/ang.gao/robot_sim_c2d3_binding_wip was clean at exact a3d750b before testing; its detached HEAD was preserved and final revision/status remained unchanged. No fallback host was used.

The candidate's tracked files, private context and installed dependencies were not edited. Independent scratch tests/report/logs are outside the candidate under c4_assets and /data2/shared/ang.gao. No GPU operation, real backend initialization, package change, reset or broad process kill was requested or performed by this reviewer. Ordinary temporary release checkouts and interpreter caches were produced by CPU tests.

2. Findings and prior-finding disposition

No new substantive finding or unresolved blocking defect was identified in the bounded repair delta. This statement does not approve missing whole-stage evidence by subtraction.

F4: Closed for the demonstrated callback and non-sticky-failure defect.

At native_bootstrap.py:2597-2612, alias intake first retains every available __main__, __mp_main__ and multiprocessing value by identity. Metadata traversal now occurs only when type(publisher) is ModuleType. This uses the built-in exact type, so neither an invalid object's __class__ property nor its __getattribute__ executes. Exact ModuleType has the built-in raw dictionary path; arbitrary spec/loader values are retained without invoking their behavior. Rejected ModuleType subclasses remain held through the publisher object and cannot gain alias permission.

At :2098-2122, SourceSession.fail commits self.failed and the original exception identity before ancillary intake. It still performs finite replacement/global/alias retention before forwarding into ExternalAdmission.fail or the controller terminal branch. The external path at :1306 onward commits its own failed flag/original error, retains available module deltas, and calls the source path on its first failure only. Source reentry sees admission.failed and reaches the controller once. Existing termination_evidence/_terminating checks prevent a second terminal operation. Later failures continue holding available objects; they do not grant normal adoption, observations or retry.

The new repository regressions independently ran on both interpreters. Cold actual source sessions exercised __class__ and __getattribute__ traps via audit, direct fail and poisoned intake. Bound source/admission tests checked zero callback calls, both flags and original exception identity, publisher/replacement custody at the first terminal endpoint, late publisher retention, one terminal operation, no new success facts and rejection after bad-key removal.

I added seven external independent cases rather than relying solely on those tests. Six begin with a real cold standard multiprocessing import, successful observation and audit; they then replace its publisher with either an arbitrary object or ModuleType subclass whose __getattribute__ raises, through audit/direct fail/already-poisoned entrypoints. They verify zero calls, retention, original-error identity, sticky failure, unchanged historical alias evidence and no recovery after removal. The seventh enters through actual ExternalAdmission.fail while all three alias keys plus torch._VF contain trap objects, checking the full available graph and both flags/original error before the first terminal endpoint. It also checks late retention and no second terminal or success fact. All seven passed on Windows 3.14.0 and primary183 3.12.13.

The original v2 repro was preserved, not rewritten. Its audit branch now fails the expected 'publisher callback' assertion because the actual production alias validator raises SourceBootstrapError instead. Its direct-fail branch now reaches the old 'operation unexpectedly passed' assertion because fail returns after committing poison without calling the trap. Those two failures alone would not prove all repaired invariants; the explicit current-contract tests above establish the actual sticky state, original retention and zero callbacks. The v2 ordinary routing/late-retention test continues passing.

F1: Remains closed for the exact earliest-custody counterexamples. The unchanged simpler probe now observes [True], and the real main-drift/SourceSession.fail/first-controller-terminate probe sees ('interpreter alias main/identity drift', True, False). The exposed replacement is already held in the adopted admission graph; the False direct-controller-owner observation is expected because this is transitive custody, not a requirement to normally adopt an invalid object. Existing request/return/frame/audit/malformed/native-global earliest-terminal regressions passed. This is available Python ownership evidence, not native quarantine/reap proof for real Torch.

F2: Remains closed. The unchanged main_token=True probe raises 'interpreter alias facts drift'. _same_facts requires exact dictionary/key/value types before equality; alias issuance and live checks preserve frozen scalar facts, as do the replacement and native-global comparisons. The equal bool/int/float negative tests passed, including poisoned historical issuance.

F3: Remains closed. The unchanged resolved-then-removed publisher/alias probe raises 'interpreter alias publisher disappeared'. The final absent/None publisher branch rejects issued incomplete resolution while legal non-final resolution intervals and untouched optional absence remain supported. Normal real cold import and repeated observation/audit tests passed.

The four v1 and two v2 old-behavior failures were checked individually and were not relabeled as six ordinary passing tests. Both original files retained their supplied SHA-256 before and after execution.

3. High-risk inspection and verification

Inspected actual base-to-HEAD changes and surrounding intake, raw type/dictionary access, source/admission failure recursion, terminal idempotence, source and external audit order, common request/return boundaries, pending/admitted replacement graphs, exact class/native binding, immutable typed evidence and alias source/resolution/publisher history. Checked that early poison does not skip required available-object intake and that normal cold success remains covered. The finite intake helpers use identity retention, not validation callbacks; real first-terminal graph assertions exercise the relevant order. Catalog semantic comparison found only ast_policy_sha256 and source_digests changed, with exactly bootstrap/policy source rows repinned and row identities/lanes unchanged. Source-policy/catalog and cold production tests passed on matching POSIX checkout bytes.

Local interpreter: C:\Python314\python.exe, CPython 3.14.0, Windows.
Commands from local candidate:

  C:\Python314\python.exe -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_import_admission.py -q -p no:cacheprovider
  Result: 291 passed, 3 skipped in 89.08s; exit 0. The three skips are pinned CPython3.12/POSIX generated-stdlib variants, not acceptance of unavailable platform behavior.

  C:\Python314\python.exe -m pytest ../c4_assets/test_torch_publication_gap_v1.py ../c4_assets/test_torch_publication_gap_v2.py -q -s -p no:cacheprovider
  Result: 6 failed, 1 passed in 13.06s; exit 1. These are the individually explained old-behavior assertion failures above.

  C:\Python314\python.exe -m pytest ../c4_assets/test_torch_publication_gap_v3_independent.py -q -p no:cacheprovider
  Result: 7 passed in 19.77s; exit 0.

Remote: ssh tileops-dev; /data2/shared/ang.gao/robot_sim_c2d3_binding_wip; /home/ang.gao/miniforge3/envs/robot_sim/bin/python, CPython 3.12.13. Independently executed at verified exact clean HEAD:

  python -m pytest tests/unit/optics/test_torch_module_replacement.py tests/unit/optics/test_interpreter_alias.py tests/unit/optics/test_native_global_publication.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_runtime_import_cold.py tests/unit/optics/test_generated_stdlib.py tests/unit/optics/test_native_runtime_driver_process.py tests/unit/optics/test_runtime_environment.py -q -p no:cacheprovider
  Result: 590 passed, 2 skipped in 192.82s; exit 0. Both skips are real Go2 materialization variants because ROBOT_SIM_TEST_GO2_MODEL_DIR was not supplied. This does not waive real Go2 validation. The fixed driver-process tests reject before NumPy/backend execution and exercise actual POSIX supervisor failure wiring; they are not successful native setup tests.

  python -m pytest /data2/shared/ang.gao/test_torch_publication_gap_v1.py /data2/shared/ang.gao/test_torch_publication_gap_v2.py /data2/shared/ang.gao/test_torch_publication_gap_v3_independent.py -q -s -p no:cacheprovider
  Result: 6 failed, 8 passed in 16.14s; exit 1. Same six original rejected-bad-behavior assertions; v2 normal routing and all seven independent current-contract cases pass. Remote scratch digests matched local.

  git diff --check cbfb1cec1e1a783ee5a36b2dbb70fdb8246bbaa9 HEAD
  Result: exit 0/no output locally and remotely.

Each test invocation preserved its actual exit status while saving a fresh log. No cached test result or author log was reused. All reviewer test/SSH commands completed. Remote observed suite PID206082/parent206081 were absent in the final scoped ps check, as were matching repro, guarded_peer and guarded_custody_process commands. Final local process inspection found no matching reviewer Python test/helper process. No GPU residual query was needed or performed because this review executed no GPU work. CPU process results cannot establish real GPU residual absence.

One exploratory remote ps command failed due to Windows/SSH shell quote removal around a regex. It was corrected by filtering the remote ps output in PowerShell. This was an inspection tool error, not a candidate failure. No cleanup or process kill was needed.

4. Evidence artifacts

All local paths below are under C:\Users\AG\新建文件夹\c4_assets:

test_torch_publication_gap_v1.py
  SHA-256 57bbfead7bae23deab212907e020e3ab2b9be935421c77052b783b669b79e63a
test_torch_publication_gap_v2.py
  SHA-256 e30aedcdbaa19a56a6c4a47c6a1127d84ebd151b9354e337fdde4e6d2c78721a
test_torch_publication_gap_v3_independent.py
  SHA-256 0f03bed85b5e835ab16e0f8ccf2327dd1ffe557d2ea72f408ee25cda86bf8e34
torch_gap_v3_independent_local_suite.log
  SHA-256 2f8455aa6ad16ae3c5701e12b3a4732e39aad75bbc5d7cd6de492ca46fa98f57
torch_gap_v3_independent_original_repros.log
  SHA-256 3da972f32d60c36126e47e497e67d86523c5b56c2986147f6819946014369769
torch_gap_v3_independent_extra.log
  SHA-256 6449aa5125a2a9a3e1fe8f6708f355bf59cd47376398cda94a7aec97381bdbf2
torch_gap_v3_independent_remote_suite.log
  SHA-256 744e038133505606b6110a4c16e1a310f0446d58d8ad3a96a2dc6e9ffb34718c
torch_gap_v3_independent_remote_repros.log
  SHA-256 e5b7e4f64a917b5ebcbb96f18b4630c09714a4714753fe705e4011c725745532

5. Exact approval boundary, limitations and required continuation

Approve only the inspected F1-F4 repair delta and its affected CPU import/failure-custody behavior at a3d750bbaf79d760a6727e56146d277e1cb3b770. No separately deferred non-blocking finding was needed for this bounded disposition. This does not certify arbitrary hostile same-process mutation, memory-exhaustion failure behavior, native exception safety or unobserved dependency mutations between supported boundaries.

The complete Torch supplement and C2d3 implementation are not approved by this report. Actual captured Torch must still demonstrate all seven original/exposed relations, native _C/ops/classes publications, exact interpreter alias and custody under the supervisor with a frozen reviewed deployment/native-scope closure. Ordinary census/deferred-init diagnostics and these CPU doubles do not supply that evidence. The draft captured diagnostic was not run in this review. UNKNOWN-before-import must not be bypassed to create success; normal import/init scope remains a real proof obligation.

The final versioned runtime-custody aggregate remains required, including session/lane/target, actual file/import/guard ordering and interrupted/failed-prefix replay. Component records are descriptive and grant no import/storage/release authority. No admitted-event prefix with interrupted file completion may be promoted to completed custody. This repair approval neither completes nor defers away that obligation.

Whole C2d3 still requires both lanes, canonical asymmetric and dynamic replay cases, both real Go2 inputs, actual guarded upload/readback, complete native scope/source audit, fault/hang/ACK/EOF/terminal and foreign/UNKNOWN matrices, real dual-GPU target1/ambient0/restoration evidence, actual exit/reap/residual checks, the final aggregate and fresh whole-stage independent review. Native internal exception safety remains a trusted dependency obligation, not something loader retention or CPU tests prove.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 is not complete.
