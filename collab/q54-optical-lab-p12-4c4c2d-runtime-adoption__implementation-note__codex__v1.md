# C2d3 implementation work in progress

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-07
Status: draft
Related Files: approved runtime-adoption design v2/review v2; native_setup.py;
native_contract.py; runtime_environment.py
Owner Summary: Initial implementation toward the whole approved C2d3 checkpoint.
Not a completed CPU substitute, code approval or runtime activation. The complete
importer/binding/environment freeze and real-device acceptance still must follow.

Base:956d1fa8d51b85c25ac17825adef17640f0e69b5. Approved design:f668b5b.
This draft records ongoing work, not a request to approve a smaller checkpoint.

## Actual changes so far

- Added CPU-only RuntimeEnvironmentContract: canonical immutable configuration,
  interpreter/file hashes and distribution METADATA checks, explicit file-module,
  absent-probe, creator/alias and namespace tables, contained paths and fixed
  plugin/quiet settings. Site roots are distinguished from native-library roots;
  containment roots may overlap but the same physical path cannot be inventoried
  twice. Parsing/verification returns no import/readiness capability. No actual
  deployment inventory or permission table had been frozen at that initial
  slice. The later cold-Warp test evidence below records the first narrow author
  fixture; it is not the complete required runtime matrix or a product profile.
- Extracted each existing CUDA/Warp native_setup upload body into its single
  internal core. C0 wrappers retain existing signatures, fresh-epoch entry and
  catches/abort policy. A registered exact internal context is held before upload
  and burned once at consumption. This context is not RuntimeSetupBinding,
  guarded execution authority or readiness; the guarded importer/driver still
  needs its proper registered binding from the approved design.
- Extracted the original registered-storage validation from full prepared/
  readiness validation. The new storage-only entry does not mint readiness; full
  C3/C4 checks still require the exact registered readiness object separately.
- Added CPU negative tests and AST comparison against the committed original
  upload bodies, proving the allocation/copy bodies were moved without a second
  algorithm. These are not GPU ownership/correctness or cold-admission proofs.

## Author verification, not independent approval

Local C:/Python314/python.exe; no local Torch/Warp installation. No GPU tests
were attempted or skipped as acceptance. Completed runs:

- Initial environment contract:34 passed in0.93s.
- Environment/core/guarded lifecycle:152 passed in7.70s.
- Added storage-only tests:166 passed in7.73s. Windows Start-Process reported no
  readable ExitCode on that first invocation; its stdout summary was observed,
  but that invocation was not claimed to have a separately verified exit0.
- Fresh confirmation, process handle retained before waiting:166 passed in7.70s,
  HasExited=True, ExitCode=0. PID36820, parent PowerShell31508, start
  2026-10-07T20:39:02.0426861+08:00. Earlier PID24716/parent13704 also finished.
- Final environment root-kind amendment plus directly affected tests:58 passed
  in1.30s, tool command exit0. The guarded controller was unchanged by that last
  amendment. Ruff check and git diff --check pass on all changed/new code/tests.

An initial Ruff E501 in the new test was corrected by formatting, then the full
changed-file check passed. Initial exploratory reads used two nonexistent test
paths and one Windows rg wildcard argument; rg subsequently located the actual
files. Those read/lint issues are not represented as product acceptance failures.

Author logs outside the repository:
C:/Users/AG/新建文件夹/c4_assets/c2d3-core-cpu-author-confirm.log and .err.
Earlier no-exit-code invocation: c2d3-core-cpu-author.log/.err.

## Required next implementation work (still incomplete)

### Additional unreviewed importer work after 46618be

The existing captured-source finder now has an internal external-admission adapter,
not a second repository finder. It verifies the trusted environment, retains exact
file specs/delegates before create, retains returned modules before exec, executes
held Python source rather than existing bytecode, and separates absent probes,
creator-owned publications and namespaces from file execution. Segment audit
rechecks deployment files, exact module metadata and creator identities. A started
import error stays failed even when the dependency catches it; available partial
modules and original errors remain held. Namespace/search-path and lookup-table
drift also fail closed. The added configuration method is internal only; no product
IPC dispatch or runtime/sample activation calls it.

These mechanics are still unfinished C2d3 implementation. The 21 added tests use
real importlib and actual temporary Python files but unit-double the source-session
and guardian prerequisites. They neither execute a native extension nor prove
cold captured-source/supervisor/GPU resource safety. The actual environment tables,
runtime binding/driver, evidence replay and mandatory native fault tests below
remain missing. No new independently approved subcheckpoint is claimed.

An initial run exposed Windows mixed path-separator comparisons (13 failures,
7 passes), corrected by deriving platform-native search-root spellings from the
canonical environment paths. The importer-only rerun passed 20 cases; after adding
the retained-source/file-drift check it passed 21 in0.57s. The bytecode negative
test first proves a standard loader would accept its unchecked-hash hostile cache,
then proves this loader executes the frozen source instead.

Combined local author regression:270 passed in8.14s, Python3.14, PID8216,
parent32284, start2026-10-07T21:05:03.7594155+08:00, explicitly observed
HasExited=True/ExitCode=0. This includes import/environment/core, guarded lifecycle,
guarded evidence fixtures and readiness-authority tests. Logs outside the repo:
c4_assets/c2d3-importer-cpu-author.log and .err. Changed-file Ruff check passes.
No source-policy/catalog repin, product admission or GPU acceptance is implied.

Read-only primary183 check at21:03:42 found no ang.gao test/worker/Python process;
only the existing MPS2238410,60MiB was listed by the GPU process query. No GPU work
was launched for this local author run. Historical hang resolution is not claimed.

Remote cross-platform CPU confirmation of WIP7b7279aecb1aa24385ac3e435a06ea7bc12ba496:
primary183, robot_sim Python3.12.13, new detached worktree
/data2/shared/ang.gao/robot_sim_c2d3_importer_wip. Bundle SHA-256 matched locally
and remotely:f6aef8ba7992c68c8aeaab488a118c0df5f15296257e7f604ddfcb290a40fd21.
The same six test files passed270 in9.57s. Timeout wrapper PID3204483,
parent3204480, observed start21:06:56, shell date21:06:57; wait returned0 and
the SSH command completed0. The Python child PID was not separately sampled;
do not treat the wrapper PID as its PID. The45s timeout/5s kill grace was not
triggered. This CPU run is not the design's actual supervised GPU fault matrix.

At21:07:22, remote user-process inspection found no ang.gao Python/pytest/worker/
supervisor/timeout residual, and GPU inventory still showed only pre-existing
MPS2238410,60MiB. Local process inspection also found no Python/SSH residual.
The isolated WIP checkout and original /data2/shared/ang.gao/robot_simulator were
clean; original HEAD remained f771e612beaeea126e26926738097512cd2d809c. No101
fallback, GPU test, dependency installation, process kill or original-tree edit.

### Guarded runtime/source wiring after 2235056 (also unreviewed)

Added a literal CPU materialization module closure to the same captured finder,
including the actual physics/sensing package-initializer dependencies. The internal
session calls resolve_render_source and CanonicalNativeInput.from_scene_contract
on the same packed scene; synthetic replay and R0 fixed fixtures still come from
the sole production builders. It does not accept caller arrays or shipping cases.
Partial repository modules are held before body execution. Extended-phase audits
check the exact executed subset of this frozen closure, while the previous CPU
core/prelude phases retain their exact full-set requirement.

Added internal indexed Torch/Warp initialization, actual-object RuntimeSetupBinding,
and one-shot upload dispatch into the shared cores. The registry freezes exact
session/controller/module/runtime/target/stream/source and stream metadata.
Registration failures retain returned resources in the previously adopted session
before unwinding. Partial initialization records UNKNOWN and guarded failure;
no C0 notifier or emergency retry is used. Product dispatch/activation is unchanged.
The new bootstrap bytes still are NOT admitted by the old source policy/catalog.

The shared Torch copy helper now enters an explicit target-device scope before
StreamContext and exits them in reverse order, including copy exceptions. This
does not change the extracted allocation/upload bodies. Installed Torch2.9.1
Python sources on183 and upstream native code were inspected, not executed as
proof of cold scope: StreamContext restores streams through set_stream, while
CUDA12 device-guard restoration has a separate lazy-device mechanism. Sources:
[Torch CUDA Module.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/csrc/cuda/Module.cpp),
[CUDAFunctions.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/c10/cuda/CUDAFunctions.cpp),
[CUDAHooks.cpp](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/cuda/detail/CUDAHooks.cpp).
These source observations do not establish that all linked native dependencies
touch only the target device. That cold real-device proof remains mandatory.

25 new tests use real guard client/core state transitions with simulated CUDA
objects, plus real canonical CPU materializers. They cover exact identity,
handmade copies, drift, consumption, partial-init UNKNOWN, returned-owner custody,
the actual upload entry rejecting before its core, and nested device/stream
scope unwind. They do not prove real GPU safety. An initial owner-custody assertion
used the wrong enum spelling; it was corrected to the existing formal
TRANSFERRED_TO_QUARANTINE state (no production enum change).

Local combined regression:295 passed in10.29s, Python3.14, PID31928,
parent12880, start2026-10-07T21:25:36.4027532+08:00, HasExited=True/ExitCode=0.
Logs:c4_assets/c2d3-binding-cpu-author.log and .err. Ruff/diff checks pass.
Before remote work,183 check21:26:04 showed H200×8 and only existing MPS2238410;
no owned Python/test process. No101 fallback or dependency change.

Remote WIP50d4e05 CPU confirmation:295 passed in10.26s, robot_sim3.12.13,
Python PID3228715, wrapper3228713, parent3228711, start21:27:31. Wait/SSH exit0.
Bundle hash e328e22a6da5473425df8bc126e99c1712aae8aabcb1bce3d399c42ce80c1d8c
matched both sides; isolated worktree /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.

The subsequent two existing GPU C0 a2 positives FAILED, not skipped or hung:
2 failed in5.83s, Python3229353/wrapper3229351/parent3229349, start21:27:53,
wait/SSH exit1. Both encountered TypeError while creating weakref(lifecycle):
the real C0 NativeWorkerSetupLifecycle is slotted and non-weakrefable. The earlier
unit double and guarded subclass had not exposed this regression from46618be.
Their C0 exception handling ran; no C2d3 cold setup proof is claimed.

Repair: retain exact strong lifecycle identity in the one-shot upload context
registry until immediate consumption. Failed context adoption removes the
registration before propagating the original error; context construction itself
submits no GPU work. The weak context identity/callback and consume-before-use
checks remain. Added the real C0 lifecycle CPU test plus failed-adoption removal
test; affected core/binding set36 passed in2.38s. GPU re-verification is recorded
separately below once actually run. Do not treat the failed run as acceptance.

Repair re-verification at1fae16aa2f72d8c032befd56c2011e5068456212:

- Bundle SHA-2568577ed641de33908b3be57f9b416c264e1a7408667ee3efd71797a7a82150cce
  matched on both hosts; the clean isolated WIP checkout fast-forwarded, no reset.
- Two original GPU positives:2 passed in6.31s, Python3231674/wrapper3231672,
  parent3231637, start21:30:27, wait/SSH exit0.
- Entire tests/gpu/test_optical_closed_worker_setup.py:10 passed in10.90s,
  no skips, Python3233498/wrapper3233496/parent3233494, start21:31:13,
  wait/SSH exit0. This includes existing failure/custody/quarantine/warmup
  regressions, not the new C2d3 cold captured setup acceptance matrix.
- Local combined CPU regression:297 passed in10.14s, Python30928,
  parent29240, start21:31:23.9481449+08:00, HasExited=True/ExitCode=0.
  Logs:c4_assets/c2d3-binding-fixed-cpu-author.log and .err.
- Remote same seven CPU files:297 passed in9.72s, Python3234723,
  wrapper3234721/parent3234719, start21:32:06, wait/SSH exit0.

No timeout fired. Final183 check21:32:41 found no ang.gao Python/test/worker/
supervisor/timeout processes and only the unchanged existing MPS2238410,60MiB.
The isolated worktree is clean at1fae16a; the original checkout was unchanged
and clean at the21:31:55 check. Local process inspection at that check found no
Python/SSH residual. No101 fallback, unrelated kill, reset or package installation.
Neither these passes nor author source inspection is independent code approval.

### Explicit input readback WIP after c4bc795

Added native_readback.py, called once by the captured session after its own
registered upload. Before calling it, the session rechecks the original runtime
record, prepared object, stream identity, module/runtime roots and source. It
does not remint a consumed upload binding or issue execution readiness.

The helper adopts a custody list before returning any new host owner, then
retains each NumPy destination and Torch/Warp CPU view before the next copy.
It uses the shared explicit-stream copy/event helpers inside one already armed
setup segment. Only after the supervised event wait succeeds does the existing
_validate_native_input_host_observations compare actual host observations. This
is input verification, not B3 host delivery. Storage signatures are rechecked
before copies and after the wait. Returned event device metadata is checked;
foreign/unknown scope prevents the wait and takes the guarded quarantine path.

Also fixed duplicate guardian adoption in _hold_runtime_owner: the importer
already owns Torch._C, and the upload core already owns its prepared result.
An already held exact object now keeps its original custody record instead of
attempting a second registration under a different role. A negative/positive
unit covers the pre-adopted native root; real cold-path verification is still
required, not inferred from that unit.

Readback tests use the real guarded controller/protocol peer, a2 physical-storage
registry and shared host validator, with CPU-backed fake device/copy/event leaves.
They cover empty/integer/float arrays, completion-before-inspection, wrong content,
storage rebinding, copy/event/record/wait failures, host-view/event/report adoption
failure, foreign/unknown event device and owner retention through quarantine.
Event observations use weak references rather than retaining the event in the
test harness. Session tests cover ordering, single validation and unchanged burned
upload authority. They do NOT prove native GPU execution, cold import or hung
native-call supervision. The source catalogue remains deliberately un-repinned.

Author checks:

- Focused readback/binding tests:60 passed in4.95s; Ruff passed.
- Eight-file CPU regression:388 passed in17.93s on Python3.14. Python25960,
  parent21916, start2026-10-07T21:52:42.4224070+08:00; explicitly waited,
  HasExited=True, ExitCode=0. Logs:c4_assets/c2d3-readback-cpu-author.log/.err.
  Files: test_runtime_environment, test_runtime_import_admission,
  test_guarded_runtime_binding, test_guarded_native_readback,
  test_native_upload_core, test_guarded_lifecycle, test_guarded_wait_core,
  test_native_contract_extraction, all under tests/unit/optics/ with .py suffix.
- Primary183 read-only preflight21:50:37 found no owned test/worker processes,
  only session/SSH readers and the existing MPS2238410 (60MiB). No new GPU work
  was launched by these local CPU tests; remote results, if run, follow below.

This is still incomplete C2d3 author work, not independent approval or a new
CPU-only acceptance boundary. No shared workflow or activation gate changed.

Remote confirmation of readback WIP e82d1a041470486d976a1c4103c794957014cbf0:

- Transfer bundle SHA-256
  dfc54d23cb7dc75441a4ff03c11636d5997d9032f89c4558c0c5d098c37cd890
  matched locally and on183. The clean isolated binding_wip worktree was
  fast-forwarded; the original worktree was not changed.
- Same eight CPU files on robot_sim Python3.12.13:388 passed in16.44s,
  Python3254995, timeout wrapper3254993, parent3254991, sampled process start
  2026-10-07 21:54:32+08 (command date21:54:33). Explicit wait and SSH exit0;
  the60s safety timeout did not fire. No GPU test or cold-runtime pass is claimed.
- Final183 check21:55:35 found no owned Python/test/worker/supervisor/timeout
  process, only user-session/current SSH readers. GPU compute inventory contained
  only existing MPS2238410,60MiB, untouched. Local process query also found no
  Python/SSH residual. These are scoped observations, not a claim to repair all
  historical hangs or to prove the whole shared machine globally idle.
- Isolated worktree clean at e82d1a0; original robot_simulator clean and unchanged
  at f771e612beaeea126e26926738097512cd2d809c. git show --check passed locally.
  No101 fallback, package install, unrelated process kill or GPU reset.

### Still required before the whole C2d3 code review

Source admission WIP after021f8af: catalog v4 and AST policy v5 pin the current
candidate's exact Git LF bytes, including runtime_environment/native_readback.
The bootstrap import exception adds only standard source/extension file loaders,
explicit PathFinder.find_spec, namespace Loader and existing spec_from_loader;
every importlib use must be an exact complete attribute chain, not a namespace
alias. Full bootstrap bytes plus seven source/loader/admission/binding classes
are pinned. The public direct entry remains CPU source verification only; C0's
policy/dispatcher and all activation gates are unchanged. This supersedes the
earlier WIP statement that the source catalog has not yet been repinned, but
does NOT authorize an external dependency or a GPU setup request. The complete
runtime contract and fixed supervised driver remain unfinished. New pins identify
an author candidate, not independent approval. Regression results follow after
execution; old v1-v3 catalogs remain historical records.

At58c480d3754953a5b3e61d61288d9b5ba2dee4ad, local source policy/catalog/
bootstrap/source-custody files completed289 passed,6 skipped in212.29s.
Python34868, parent7040, start2026-10-07T22:13:49.3235689+08; explicitly waited,
HasExited=True/ExitCode=0. Skips are2 real-Go2 cases without the local asset
configuration and4 actual-POSIX checkout/entry cases. They require183 evidence.
Logs:c4_assets/c2d3-source-v4-cpu-author.log/.err. Ruff passed.

The remote first combined five-file batch did NOT pass: its outer240s budget
expired, wait/SSH exit124. Python3277682/wrapper3277680/parent3277678,
start22:17:14. It had reached the actual-supervisor source fault matrix, not
stopped at an unexplained first hang. At22:20:57 the current child3282251 and
supervisor3282246 were3s into the declared source_cleanup_hang fixture; no
component watchdog failure was observed. Do not reuse completed dots as a full
pass. At22:21:44 the terminated batch had no owned Python/timeout/worker residual;
GPU inventory was only unchanged MPS2238410. Separate bounded reruns follow;
per-operation supervision deadlines are not relaxed. Transfer bundle SHA-256
66366d3dde6edd9e64c25214aa7e19a9266609bf966a1acb06151871883be3c7 matched both
hosts; only the isolated binding_wip checkout was fast-forwarded.

Added actual -I -S -B captured-session tests with fixed CPU-only external fixture
modules. Unlike the earlier importer units, these use the real SourceSession,
guardian/controller, environment configuration, captured contract type and
external loader together. They check held modules/aliases, expected absence,
started-body failure, swallowed unexpected absence and identity drift, preserving
poison without a second ARM/sync. The fixture imports no NumPy/Torch/Warp and
does not prove cold GPU custody. First local run10 passed in56.59s; a subsequent
explicit hold assertion still requires the fresh rerun below. No checkpoint is
approved from this CPU-only composition.

Remote reruns at02be61fc20891c50cb967264cdef8d1a1e264997:

- Transfer bundle SHA-2567587872f5a3f99702780883b403ac6be9f5adea3cca052046b5566dda74ae918
  matched. Same isolated worktree, no original-worktree edit/reset.
- Five ordinary files (the four local source files plus test_runtime_import_cold):
  305 passed in211.91s, no skips, with the real Go2 path configured. Python3284363,
  wrapper3284361, parent3284330, start22:23:02. Explicit wait/SSH exit0,360s
  batch budget not reached. This fresh run includes the final explicit guardian
  hold assertion in the new ten-case cold CPU composition test.
- Separate test_native_source_custody_process.py:32 passed in121.87s. Python3287651,
  wrapper3287649, parent3287647, start22:25:27; wait/SSH exit0,180s batch budget not
  reached. Each real supervisor reported exit0 and validated/reaped its child;
  success children exited0, injected failures74 or supervised kill -9 as specified.
  Examples: import-hang Warp child3288048/CUDA3289293; cleanup-hang3288632/3289780;
  exit-hang3288815/3289935 all observed/reaped -9. Parent-alive expiry, parent loss,
  no/wrong/EOF ACK, stalled disarm, compile/body/adoption/audit failure and cleanup/
  exit hangs are covered. These are CPU source-custody branches, NOT real native
  CUDA/Warp initialization faults or a C2d3 GPU acceptance claim.
- Final183 check22:27:54 found no owned Python/test/worker/supervisor/timeout
  residual, only session/SSH readers. GPU inventory still only MPS2238410,60MiB.
  Local Python/SSH residual query empty. Both isolated checkout and original
  f771e612beaeea126e26926738097512cd2d809c checkout clean. Local show/diff checks
  passed; cleanup-ledger validator1 passed in0.47s. No101 fallback, install,
  unrelated process kill or GPU reset. The earlier interrupted batch stays failed.

### Fixed supervised runtime driver at ae5d3f4 (unreviewed)

Candidate: ae5d3f4b1ddd11d4fbed363b49f3fc70ddae5762. Added only the fixed
tests/support/native_runtime_process.py driver, its supervisor dispatch/test
coverage and exact cleanup paths. No production code/catalog, C0 protocol,
product request, activation gate or shared collaboration rule changed.

The driver sequences the captured session's setup definitions, external contract,
canonical materialization, runtime binding, shared upload, explicit validation
readback and terminal disposal. It does not copy preparation/upload loops or
issue final execution readiness. Fixed terminal-sync failure/hang selectors are
present but NOT yet exercised with a real successful native setup. Per-phase
v0 observations are debug facts only, not the required runtime-custody schema.
The complete dependency table, GPU positives and failure matrix are still missing.

Actual author verification on primary183, clean isolated
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip at ae5d3f4, robot_sim Python3.12:

- test_native_runtime_driver_process.py:4 passed in12.61s; wrapper3302020,
  Python3302022, parent3301980, start2026-10-07T22:40:34+08. Wait/SSH exit0;
  outer90s fail-bound did not fire. Both lanes reject environment-digest mismatch
  and an unlisted NumPy import via the real captured entry and real supervisor.
  Supervisors/children:3302036/3302041,3302078/3302079,3302110/3302111,
  3302181/3302183. All supervisor exits0, actual reaped child exits74. Evidence
  replay validates full quarantine, zero emergency/terminal sync attempts and
  no ordinary atexit cleanup. These are pre-native rejection tests, NOT GPU
  initialization/upload/cold-runtime successes.
- test_native_source_custody_process.py plus test_guarded_custody_process.py:
  61 passed in134.65s. Wrapper3302945, Python3302947, parent3302943,
  start22:41:32+08; actual wait/SSH exit0; outer190s fail-bound did not fire.
  All actual child exits were observed/reaped by their supervisors. Deliberate
  hang fixtures, including source import/cleanup/exit and terminal sync/cleanup/
  exit, were watchdog-killed (exit-9), not counted as successful disposal.
  Examples: source-import Warp3303268/CUDA3304352; source-cleanup3303755/
  3304810; source-exit3303883/3304911; sync-hang3305160; target-cleanup3305220;
  target-exit3305235. This remains CPU-fixture supervision proof, not a new
  real-GPU hang matrix or a claim that the historical101 hang is fixed.
- Local Ruff check for all three changed Python files passed. Cleanup validator
  in test_optical_benchmark_contracts.py:1 passed,40 deselected in0.42s. An
  initial command named a nonexistent cleanup test file and collected no tests;
  only the corrected command is counted. Local git show/diff checks passed.
- Bundle c2d3-runtime-driver-wip.bundle SHA-256
  cf8fe53d17bb2474735ada628c661d34dc31a361266724619203b89bf0d3a5d2
  matched before fast-forwarding the isolated checkout. No original checkout
  modification, install,101 fallback, GPU reset or unrelated process kill.
- At22:43:58+08, post-test read-only OS inventory found no ang.gao Python/test/
  worker/supervisor/timeout processes, only user-session/current SSH readers.
  GPU inventory showed only the pre-existing MPS2238410,60MiB. No scope-wide
  approval or GPU activation follows from these scoped hygiene observations.

### Remaining work after this driver round

Subsequent actual runtime progress at022b2ee is recorded in
q54-optical-lab-p12-4c4c2d-runtime-cold-warp__test-evidence__codex__v1.md.
The first111-module Warp/NumPy author fixture is frozen and real canonical
upload/readback plus terminal success/failure/hang now have3 non-skipped GPU
tests. The run exposed and repaired case-sensitive array-name operation IDs in
both sole upload cores/readback. C0 GPU and expanded295-test regression passed
on183. These facts supersede the prior statement that only pre-native rejection
tests exist, but do not close the rest of the whole-C2d3 obligations below.

1. Derive and freeze the actual required deployment/native closure, absent-probe
   and publication tables from observed/source-audited dependencies. The offline
   32-wheel facts are not this permission table. The narrow Warp fixture is not
   CUDA/Go2 admission or complete native scope proof. Finish runtime-custody evidence
   schema/replay; CPU fixture loader/adoption tests do not replace real dependencies.
2. Verify the now-wired captured RuntimeSetupBinding/fixed supervised driver and
   terminal disposal against the actual frozen native environment. Explicit
   readback and fixed dispatch are implemented; actual Warp asymmetric terminal
   selectors now pass. The rest of the real cold runtime/fault matrix and
   versioned runtime-custody replay remain incomplete.
3. Keep catalog v4/policy v5 pins synchronized with any subsequent reviewed-scope
   source edits, and verify the literal actual runtime closure. Current pins and
   CPU source-entry tests are WIP acceptance of bytes only, not GPU authorization
   or independent code approval. Only the explicitly configured sacrificial test
   consumes the frozen Warp author table; product runtime admission remains off.
4. Fixed supervised actual Torch/Warp cold setup and all required real assets,
   dual-device and fault/hang tests; existing C0/C2d0-2 regressions. No CPU mock or
   preloaded process substitutes for that matrix.
5. Update exact-path cleanup inventory, freeze the complete candidate and start a
   fresh independent code review. Repair/re-review until that scope is approved.

No code review was requested for this incomplete intermediate state. Existing
approvals do not cover these code edits. Product worker, gate.json, source-policy
admission, sample/result/baseline activation and shared collaboration rules are
not expanded. Full C2d/C4 and all previously closed later paths remain incomplete.

## Earlier round-end hygiene (20:43; later rounds recorded above)

Later progress at a502ef1 is recorded in cold-warp test-evidence v2: six real
asymmetric/dynamic Warp cold tests passed, including terminal raise/hang with
actual supervisor reap. Both Go2 CPU-source regressions passed independently of
new runtime admission. Fresh optional-import gap-scan v1 confirmed a design gap:
pinned trimesh has optional adapter bodies that fail and are caught by parents;
the existing absent-leaf rule cannot admit those started failures. Next work is
a bounded design amendment and independent review, not a blanket exception or
package install. Current sticky failure/activation rules remain unchanged. This
updates the dynamic-workload status above but does not close full C2d3 or C4.

Final local Win32_Process query showed no python.exe/python3.exe/ssh.exe residual.
The confirmation test process was explicitly waited and observed exited0; other
short-lived test utility/child PIDs were not all sampled, so no complete process
ledger is claimed. Each subprocess test has bounded wait and successful return.

Primary183 remained reachable. Read-only check at20:43:10 showed no ang.gao
Python/test/worker/supervisor processes, only user-session/current SSH readers;
GPU inventory showed existing MPS2238410,60MiB, untouched. Remote design checkout
remains clean f668b5b; no WIP code was installed/copied there and no GPU work was
launched in that earlier round. No101 fallback, unrelated process kill, install, reset,
original-worktree edit or historical-hang-resolution claim.
