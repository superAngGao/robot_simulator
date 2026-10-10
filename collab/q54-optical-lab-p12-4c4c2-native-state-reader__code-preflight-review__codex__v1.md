Initiative: Q54 / P12.4 / C4c2
Stage: Fresh independent exact-code/native-call PREFLIGHT review
Author: Codex, independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: in_review
Session: 01a12501-54b8-7220-a419-ead734cb7b14
Related Files: native-state-reader design-request/design-review v1; worker-intake gap-review v1; native-scope-aggregate v2; native-scope-cache v1; runtime-custody implementation-plan v3; runtime-integration v3; fixed-CUDA-artifacts v2
Owner Summary: Request changes. Production extraction and source pins are coherent, but reader admission permits queries outside the eligible terminal interval and permits captured-input observation bypass/substitution. Repair and obtain fresh independent exact-revision preflight before new native execution.

**Verdict: Request changes.** This exact candidate may **not** proceed to bounded native setup/framed-worker validation. Two blocking implementation findings follow. These are CPU-reproduced contract failures, not observed GPU failures or evidence of a broken dependency. No runtime acceptance is issued.

**Identity and independence**

| Item | Independently checked value |
| --- | --- |
| Checkout | C:\Users\AG\新建文件夹\c4_native_reader_code_review |
| HEAD | 4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd |
| Base and merge-base | 868225db0ac5794bd7089cd73357b5c920d0c0e9 |
| Branch | Detached HEAD |
| Initial/final candidate tree | Clean, including untracked files |
| Diff | 18 files; 1,223 insertions, 620 deletions |
| Whitespace check | git diff --check base HEAD: passed |
| Design LF SHA256 | 0b6dab55c496e730219dfbb253b7d9530dff40faa6c90676c71a2b90e0f41ca5 |
| Design review | 01a124e2-d69d-72d3-a86c-b59248c84055, v1; implementation-only approval |
| Reader LF SHA256 | 14cfade215dc383bbf8d66e0b49ad3e8b1d2628b46310c52ed7f9106560c17aa |
| Bootstrap LF SHA256 | 5a8f54ce74f9f991f11ad554cf0a7fa1a5f24793c48be4738a2444aa49fff277 |
| Catalog v18 LF SHA256 | c056a01fc60ad951ac60e579c5b0cd8d047d9732e4d74faa3c5e8c8ec7aa6cb8 |
| Integrated runtime contract LF SHA256 | cff830f90fca7c966916ceb43b7ad0dfebd621011df66b9aa66d313653c7ebe2 |

I received the task packet without the implementation conversation and did not author the candidate. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely in UTF-8; inspected the approved design, its C1 comment, predecessor requirements and worker-intake F1/F2/F3. No agents, candidate edits, installation, GPU/native-library acquisition or query, reset or signal occurred. CPU tests use modeled device leaves. Remote work used read-only SSH/source inspection on tileops-dev; no fallback host was used.

The complete report, logs, remote source excerpts/hashes and repro source are archived outside the candidate at `C:\Users\AG\新建文件夹\c4_assets\native_reader_code_scratch_v1`. This is the verbatim report archive. It may later be copied unchanged into collab through the evidence-only recording exception; this reviewer did not mutate the candidate to do so.

**R1 — P1: a live emergency ticket permits native queries after terminal failure or ordinary release.**

Location: `optical_worker/native_scope.py:41-50` (_ticket), `_state`, and `:399-402` (snapshot); `optical_worker/guarded_lifecycle.py:726-782` (_synchronize/terminate).

Trigger: retain a reader reference and invoke its real state method after healthy emergency synchronization has completed ordinary release; alternatively, call reader._fail during emergency work and subsequently invoke _state. The emergency ticket intentionally stays armed through process exit. _ticket checks session.failed, replay.compromised and the normal replay.failed flag, then selects emergency role whenever controller._terminating is true. It does not reject replay.terminal_failed, completed release/terminal intent, or emergency access outside the actual sync interval.

Consequence: the native getter path remains callable outside its admitted lifecycle interval. A reader failure during emergency work does not itself stop subsequent reader queries. A later append/parser rejection is too late because _state has already queried. This violates the requirements that only the existing active eligible sync may query and that queries stop after sticky failure or ordinary release. The reproduced emergency failure still ultimately quarantines; this finding does not claim it authorized release.

Reproduction: `test_independent_identity.py::test_actual_state_queries_after_ordinary_release` uses the real NativeGuardedSetupLifecycle/Peer, real NativeStateReader._state and exact pinned Device AST model. It establishes the modeled original cache, completes setup and terminates with a modeled successful sync. Actual terminal evidence reports release_status='completed'. A subsequent _state call increments the modeled native stream-getter count. `test_actual_state_queries_after_emergency_reader_failure` sets the actual terminal_failed flag through reader._fail inside synchronization; _state still increments that count before quarantine. Both probes passed as assertions of the defect; neither executes native functions.

Why existing checks miss it: observer tests primarily exercise normal setup failure, where replay.failed/UNKNOWN prevents further work. They do not distinguish a live emergency deadline from active synchronization eligibility or exercise reads after release. Keeping the deadline armed is correct; treating its whole lifetime as query permission is the defect.

Direction: use existing lifecycle state to admit normal setup queries and the eligible active emergency-sync interval only. Reject terminal_failed, sync completion/failure, release/cleanup and terminal intent before any getter. Apply the same admission to raw diagnostic access. Preserve the original guard/deadline; add no wait, synchronize, retry, ticket or supervisor.

Repair proof: invert the two original repros to require rejection before the first getter. Cover terminal failure, sync success/failure, release authorization/cleanup, ordinary release and terminal intent while preserving legitimate before/after observations inside active sync. Require quarantine and zero retry. Actual native failure evidence follows only after repaired preflight approval.

**R2 — P2: captured-input reader provenance can fail open, and identity rejection is not sticky.**

Locations: `optical_worker/native_scope.py:477-521` (require_reader/reader_for_input), `:41-42` and `:399-402` (identity checks outside failure handling); `optical_worker/native_setup.py:1415-1431` and `:1733-1741`; `optical_worker/native_readback.py:44-54` and `:115-119`.

Triggers and observed behavior:

- Create a registered upload context while a source is bound, then remove lifecycle._native_source_session before consumption. The context and its registry still retain the original source. _consume_upload_context validates that retained source against its own record but returns only lifecycle/runtime/target/stream. reader_for_input rediscovers the absent lifecycle link and returns None. The actual Warp upload function enters its scope and reaches a modeled allocation without native_scope facts. The retained context source did not enforce rejection before the call.
- Both-lane source-bound readback likewise performs its original copy/event/record/wait sequence with zero observation rows after the link is removed. A mismatched source phase selects the same optional-reader branch; this does not distinguish broken captured-input entry from legitimate later shared-helper use.
- Replacing the reader in session._native_scope_record with copy.copy(reader) passes require_reader/_ticket and input collection. The existing copy test leaves that slot naming the original. This probe tests the explicitly required copy/substitution boundary, not an IPC exploit or hostile-Python sandbox.
- require_reader validates binding field consistency but does not compare reader.binding to the source's originally issued binding identity. A copied binding with identical fields passes the CPU input model. The model is not claimed to exercise a fully captured runtime-binding registry; the absent comparison is also directly visible in the actual code.
- Catch a missing-reader rejection from actual readback, restore the record and retry in the same setup segment: replay.failed stays false, input work proceeds and normal_disarmed is recorded. A rejected cloned-reader _ticket can also be caught without poison. These identity checks precede snapshot's try/_fail, and require_reader/reader_for_input have no sticky failure handling.

Consequence: C1's requirement to reject absent/substituted readers before captured input work is not enforced at the shared boundary. Later coverage/parser rejection cannot undo input-native calls. Uncaught errors reaching the normal outer SourceSession/segment do poison correctly; that does not satisfy the explicit caught-failure and pre-call rejection obligations. These are local integrity checks, not claims that arbitrary Python mutation is reachable through worker IPC.

Reproduction: `test_independent_identity.py` contains both-lane missing-source readback, substituted-reader, copied-binding, caught-identity/caught-input-rejection probes and a Warp registered-upload-context probe: 11 parametrized cases characterize these behaviors. The upload probe preserves context.source_session and reaches an allocation sentinel through production upload. Its initial fixture attempt failed because the reused readback Scope expected an entered leaf. The corrected CPU probe explicitly enters that modeled leaf. The original fixture failure remains in pytest_boundary.log and is not candidate-failure evidence.

Why prior checks miss it: test_captured_input_cannot_omit_its_reader keeps a non-null source at the expected phase and clears only the reader record. test_copied_reader_rejects_before_query leaves the checked slot unchanged and lets the exception escape the segment. The later-BVH test intentionally expects no input facts. None proves the retained source is mandatory after context creation, original binding identity is checked, or caught rejection stays failed.

Direction: carry the registered source through upload consumption and validate it against the controller before input-native calls. Give the source-owned readback entry a corresponding mandatory source check. Preserve the explicit C0/no-source route and later diagnostic route. Bind reader/binding validation to original source-held identities rather than only replaceable, self-consistent fields. Poison through retained original authority before raising admission/identity failures, including caught errors. This needs no new general infrastructure or callback surface.

Repair proof: reject missing/replaced source, wrong captured-input phase, reader copy/substitution and binding copy before allocation/copy/event/query. Require sticky failure after caught rejection, no COMPLETE, emergency retry or ordinary release. Preserve empty-array coverage, later BVH raw diagnostics without input receipts, and reader-free C0. Invert the original repro assertions after repair.

**Native-call and resource assessment**

The ordinary preparation placement substantially follows the design; the following source findings do not override R1/R2:

- The captured setup closure contains NativeStateReader. Construction retains/adopts it before query initialization. Its held/raw containers retain library handles/functions and diagnostics. Acquisition requires the exact environment row, resolved regular path, matching bytes and that path in /proc/self/maps; CDLL uses RTLD_NOLOAD|RTLD_LOCAL. No fallback loading, setter, primary retain/release, memory query or synchronization was added to the reader.
- Installed headers match cudaGetDevice(int*) -> int and the five Warp driver signatures: CUresult/CUdevice=int, CUcontext=void*, flags=unsigned int. Return codes precede state consumption, and successful null context differs from error. The primary-state API is distinct from primary retain/release. [NVIDIA primary-context API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html).
- CUDA count/primary checks precede Stream; cold collection avoids the warming logical getter. The installed-version upstream source confirms getDevice invokes device_lazy_init, while count/primary have separate paths and explicit-target current-stream reads C10 state. [Pinned Torch Module.cpp](https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/csrc/cuda/Module.cpp).
- Fresh Torch source reads confirm the capability/cubin callbacks, later _register_triton_kernels queue entry, initial _LazySeedTracker fields and Stream(device)'s scoped path. Candidate checks those identities/empty seed state and checks Triton nonexecution at binding. Retained pinned C10 GetDevice/MaybeExchangeDevice supports the accepted logical0/raw1 distinction; it does not claim raw0 restoration.
- Warp's stream getter is only conditionally readonly. Fresh warp.cu shows get_context_info's cache miss can select context and allocate/free a byte. The candidate checks initially absent raw context/stream and exact empty error bytes; after the original get_stream return has been retained/journaled, it requires returned identity, positive raw context/handle and another exact-empty error result. It then enables the getter and requires its first result to equal the actual returned stream. The pinned void setter and persistent error buffer support this proof. Inspected cache erasure occurs in context destruction, outside the normal held setup path; R1 still leaves the terminal boundary incomplete.
- ScopedDevice restores metadata and delegates ContextGuard's void native pop. The candidate retains and checks direct post-exit physical state rather than trusting Python return. Body exceptions skip successful-after observations. The committed contradictory-context model appends the wrong state before poison.
- Bootstrap holds/journals stream before cache and metadata work. Artifact create retains/journals module inside the original target/stream scopes, before unwind and the after snapshot; file completion follows the pair. Arrays/events are held on helper return before after-query, metadata/adoption and later outer exit. Existing host staging/report retention remains. No second guardian or release route was added.
- Complete snapshots enter raw storage and journal before expected-scope comparisons. Getter failures preserve ordered prefixes without inventing complete state; normal _fail marks UNKNOWN and fails the guard. Executed models cover normal getter/cache failures. R1/R2 are the uncovered lifecycle/admission cases.

**Consolidation, coverage and PID evidence**

_RuntimeObservation and runtime_custody.py are unchanged by this diff. Historical v1/v2 meanings and v3 artifact joins were not weakened. Production now surrounds all three create/exec operations with required pairs, addressing the structural cause of predecessor F1. It has not yet established real CUDA intake success.

The old successful test-fact producer was removed. _NativeScopeProbe delegates to the production reader, retaining fault/diagnostic hooks. Later shared-helper diagnostics use that reader's _state and raw records without successful input receipts. No second native ABI/state implementation was found in the test delta. CPU models are explicitly models.

The shared coverage validator selects actual guards associated with runtime_initialization/native_upload/input_validation completion and joins each eligible site/edge/lifecycle occurrence with uniqueness and exact-set checks. The supplied standalone probe passed: later helpers excluded; missing/transplanted input facts rejected. Integrated acceptance parses records and validates durable selection/attachment first. Its success branch requires reclaimed samples, zero emergency sync, completed release and supervisor-observed exit; existing lifecycle replay checks close/ACK ordering. The sacrificial checker separately requires one emergency sync.

process_identity.json is produced from live /proc stat entries for actual worker/supervisor/parent, including start ticks. Acceptance compares worker/supervisor PIDs to launched process results. This is coherent bounded provenance, not cryptographic attestation. No new execution PID evidence was produced here. Prior campaign facts remain historical; setup rows cannot prove final-sample/resolve/close physical scope.

All 91 catalog entries independently match committed LF bytes and pass the actual AST validator. Both lane inventories, policy digest, reader/bootstrap pins and integrated contract pin match. The exception is full-source pinned; selected tests reject recomputed-hash symbol, path/mode mutations and exception transfer. C0 policy/product activation remains unchanged, and definition import did not acquire ctypes or a backend. No new guard/wait/deadline renewal appears in the production diff.

**Executed verification and evidence**

Environment: C:/Python314/python.exe -B; PYTHONDONTWRITEBYTECODE=1; PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; pytest -p no:cacheprovider; all basetemp paths under review scratch. ROBOT_SIM_WARP_CONTEXT_SOURCE was the packet's copied pinned_warp_context.py; exact bytes are hash-asserted by Device models.

- Git status/revision/branch/merge-base/diff/check: clean detached exact candidate; correct merge-base; whitespace pass. hashlib of git show HEAD:<path> produced identity_hashes.json.
- Independent LF catalog inventory/hash/AST loop: 91 passed; lane/policy/contract checks passed, no Torch/Warp/NumPy/ctypes imported by that check. See catalog.log.
- `python -B -m pytest -q -p no:cacheprovider <scratch>/test_independent_identity.py tests/unit/optics/test_native_scope_observer.py` with the initial six repro cases: 30 passed in 4.24s, comprising 24 committed observer tests and six defect probes; pytest_independent.log.
- Subsequent bounded selections of the same repro file: `-k actual_state`: 2 passed; `-k 'registered_upload or caught_input'`: two caught-input passes and one fixture failure; corrected `-k registered_upload`: 1 passed; `-k copied_runtime_binding`: 2 passed. Total: 13 distinct defect-characterization cases passed. See pytest_terminal.log, pytest_boundary.log, pytest_upload.log and pytest_binding.log. These passes confirm defects, not candidate acceptance.
- `python -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py -k 'reader or cold_import or cleanup_obligation'`: 8 passed, 167 deselected in 1.12s; pytest_source.log.
- `python -B -c "import runpy; runpy.run_path('../c4_assets/native_scope_coverage_probe.py')"`: passed; coverage.log. Synthetic coverage only.
- Read-only `ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`: installed source/header reads and hashes, no backend imports. Remote old checkout remained clean at 26542543d067d20c64670d375065535420ede203; this candidate was not deployed there. See remote_sources.jsonl and remote_native.jsonl.
- Rehashed 16 retained upstream Torch source files against their manifest at 5811a8d7da873dd699ff6687092c225caffcf1bb; inspected relevant C10/getter paths. Installed torch/version.py names that revision and 2.9.1+cu128.

Fresh dependency hash matches:

| File | SHA256 |
| --- | --- |
| warp/_src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| warp/_src/utils.py | 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24 |
| warp/native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b |
| warp/native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 |
| warp/native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f |
| warp/native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 |
| warp/native/error.cpp | 877b9f1e889d3e5bf8950c3a3deaf28f44bc4a9a76e780dcc19606d80f6229f2 |
| warp/native/error.h | 4ebf4a1dcd7ed121df72f9390888c5f9eb30547f0e42fde5e04ace50de7face0 |
| torch/cuda/__init__.py | 9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99 |
| torch/cuda/streams.py | cdad950977c5eb5a0fe744eaff8c7f9194817de40f380b40fdbc29032bad3fef |
| libcudart.so.12 | c3a75b33af334a3486d197dbd1584a2985183ba4688d237a2be5f2f679329920 |
| libcuda.so.595.71.05 | 76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6 |

Installed CUDA headers confirm the six ctypes signatures. Readonly denotes admitted state-query operations under this frozen deployment, not zero driver initialization, linker bookkeeping or CPU-cache activity. Cold CUDA and Warp cache purity depend on the inspected prerequisites and pinned trusted-dependency boundary. This does not prove proprietary driver internals, binary/source rebuild equivalence, actual error-buffer contents or actual hardware restoration.

Some oversized combined reads were truncated and replaced by focused reads. An rg used two nonexistent guessed filenames and was replaced by actual lifecycle reads. One official CUDA runtime documentation fetch failed; installed headers and retained source evidence remained available. The initial large shell report-write command was rejected by tool policy; the report was instead written directly with apply_patch outside the candidate. These are tool/inspection issues, not candidate failures or permission requests.

**Unexecuted checks, residuals and exact scope**

I did not rerun the author's 18 integration tests or 103-pass/9-platform-skip group, nor launch a competing full source/artifact suite. Their summaries remain author claims. Observer tests were rerun because extraction changed the real reader; independent probes targeted uncovered admission/terminal invariants. Source tests were limited to the changed exception/C0 checks plus full release-byte/AST verification. The suite still running at packet creation is not claimed passed. Historical raw inventory and numerical samples were not re-executed or relabeled.

No new native reader, framed worker, GPU failure/restoration, Go2, final-sample numerical or full C2f matrix ran. Resource-retention paths were inspected and committed stream/cache/getter/void-exit CPU cases executed; this was not an exhaustive fresh fault campaign. Full runtime acceptance still requires actual native evidence after repaired preflight: both production framed lanes, all six CUDA pairs, complete Warp setup coverage, contradictory physical after-state, foreign/UNKNOWN failures, retained resources and raw diagnostics, observed exit/reap and scoped residual closure.

Reviewer CLI PID was 28572, parent 42276. All test/tool sessions returned. Final scoped local descendant and scratch-process checks and exact remote inspection PID checks are recorded in process_closure.json; no owned Python/SSH/SCP residual was found. No unrelated process was signaled. These are scoped OS checks, not a GPU-process query or exhaustive attestation.

Predecessor F1 has the intended producer structurally, but real CUDA intake remains unaccepted. F2 final-sample physical/numerical proof and complete C2f/C4 remain open. F3 compact reply/journal recovery is predecessor code and receives no approval here. Design C1 is only partially met because R2 defeats its mandatory-reader boundary.

Approved scope: **none for new native execution at 4dcf73d78ef36dbfd40ac3025a228cf9be1b21cd**. These findings support one consolidated repair and fresh independent exact-revision preflight. Complete C2f/C4, activation, formal performance, result/baseline, backend selection, R2/S/host delivery, a3b and shipping remain closed. No subset is parent acceptance. This bounded review ends here; it does not continue the parent goal.
