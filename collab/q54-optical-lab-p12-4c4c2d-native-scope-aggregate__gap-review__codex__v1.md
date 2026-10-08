# Independent native-scope aggregate gap review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent bounded gap review
Author: Codex, fresh independent reviewer /root/native_scope_aggregate_gap_v1
Version: v1
Date: 2026-10-08 (Asia/Shanghai)
Status: in_review — Request changes at the bounded gap scope
Related Files: native-scope-aggregate gap-request v1; runtime-adoption design-request v2 sections 6–8; cuda-ambient design-request/review v1; captured-cuda-ambient gap-review v1; runtime-custody implementation-plan/gap-review v3; runtime-custody test-evidence v1 and implementation-note v2; optical_worker/runtime_custody.py; native_bootstrap.py; native_setup.py; native_readback.py
Owner Summary: The required native-scope observations remain outside the actual v1 aggregate. All twelve retained v1 records pass fresh CPU replay and attachment, but that does not prove ambient state. A bounded exact amendment is needed before adding observation capability. Quiet Warp initialization offers a feasible metadata-only ambient selection, but the current full readback path has a distinct native-context restoration problem that a metadata-only observation would miss.

## Verdict and exact scope

**Request changes**, scoped solely to closing the native-scope/aggregate requirement and the proposed Warp observation direction. This is not a rejection of every existing aggregate property, a whole-C2d3 code verdict, approval of the eleven new negative tests, or permission to run an unspecified new observer. No exact new schema or frozen Warp observation harness was submitted, so none is approved here.

The CUDA ambient decision explicitly requires descriptive observations in the final versioned runtime-custody aggregate, cross-bound to actual guard and import/runtime order. The current aggregate and sidecars do not fulfill that requirement. The v3 aggregate design approval approved its exact fields and failure-cut contract; it did not waive the earlier ambient requirement. Both remain relevant, so the author should draft one bounded exact supplement and obtain fresh independent design review before changing the production collector/parser or attempting a newly instrumented positive Warp run.

This was a fresh independent task packet, without candidate authorship, inherited implementation reasoning, subdelegation, or candidate edits. AGENTS.md, collab/README.md, and collab/REVIEW_PRINCIPLES.md were read in full first. The initial PowerShell Chinese decoding was garbled; REVIEW_PRINCIPLES.md was reread in explicit UTF-8. Governing documents and their approval boundaries were read, including narrower rereads where combined output was truncated.

Inspected state:

- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Frozen local HEAD: `3617ab5f0f778b7f7410c84fa44ec5762dc4e228`.
- Packet base: `bcc1bac4a6fa0ae9cfc758e85bf0a7c77fdca455`.
- Initial and checked working tree: clean. `git diff --check` passed.
- Base-to-HEAD: only the gap request and two test files, 294 insertions/11 deletions. `git diff --stat 9b5702b4016824ea7fc5b5e7c108c5adf02a1bd1..HEAD -- optical_worker` is empty.
- Request SHA256 independently verified: `0f8c0767b92627a89d8d899e9decbddadfed3a1209565b7b531caf6214481fce`.
- Governing implementation-plan v3 SHA256 independently verified: `6dc1fa68fef830b5c5f631128cedb78a35f79ee31767879424945550fa483beb`.
- Read-only primary remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip`, clean at `bcc1bac4a6fa0ae9cfc758e85bf0a7c77fdca455`.
- Original binding worktree: clean at `a3d750bbaf79d760a6727e56146d277e1cb3b770`.
- Actual retained execution revision: `9b5702b4016824ea7fc5b5e7c108c5adf02a1bd1`; archive SHA256 independently verified locally and remotely: `eb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510`.

## F1 — P2: final aggregate lacks the required native observations and occurrence joins

Locations: cuda-ambient design-request v1 section 4; runtime-custody implementation-plan v3 exact journal schema; `runtime_custody.py:31` (`_DETAIL_FIELDS`), `:560` (journal replay), `:776` (exact top-level parser), `:807` (attachment); `native_bootstrap.py:2090` (`_RuntimeObservation`), `:2754` (runtime binding emission).

Trigger: a complete actual target1 upload/readback and terminal execution emits a current v1 record and passes existing attachment. A reviewer then treats that record, its selection sidecar, or an unrelated historical ambient report as proving the earlier native ambient contract.

Observed facts:

1. The event vocabulary has import/file-call/publication facts, returned owners, a runtime binding, storage, validation, phase completion and failures. It has no raw current device/context, primary-context active state, Torch logical selection, Warp default-device observation, current target stream observation, or before/after native observation event. Exact-key parsing rejects an added top-level native observation field.
2. `runtime_binding.stream_identity` is the validated identity of the held stream. It is not a time series of the selected current stream, much less evidence of raw/logical/context restoration around native leaves. `native_bootstrap._stream_identity` explicitly reads fixed stream metadata without ambient queries.
3. Lifecycle exact-target descriptors/dirty history are important existing authority but are not native getter observations. A recorded target-only declaration cannot independently establish that an unobserved dependency never selected another context.
4. Selection sidecars contain only schema/selection/record digest/event counts/diagnostic presence. They prove durable pair consistency. `validate_runtime_custody_attachment` validates the custody artifact and exact worker prefix; it does not consume a scope sidecar.
5. Fresh independent CPU replay of all twelve archived records succeeded. Every selection was `current`; every attachment returned `worker_prefix_matched`; backend imports were empty. None of these records had native observation events. Thus parser/attachment success does not close this requirement, even on actual successful executions.
6. The archive contains 252 regular files. Its run artifacts include the record/input/selection triplets and older phase diagnostics, but no ambient/scope observation file. The historical a3d750b CUDA run separately has 267 snapshots with native values, 132 helper pairs, and actual matching old guard references. Its runtime phase diagnostics are explicitly `q54.c2d3.driver-observation.v0`.

Concrete independent non-transplant check: historical snapshot 0 (`binding_adopted`) references old lifecycle event 2419, whose actual role/operation is `runtime_setup_binding`. In the archived v1 CUDA asymmetric run, event 2419 is an `external_import_owner` adoption of a `ModuleSpec`; the new runtime-binding observation references event 2421. Both use setup guard sequence 3. Guard sequence equality or matching counts would therefore accept a false correspondence. The old request also names a3d750b, while every new record names 9b5702b. No conversion of these old observations can truthfully make them observations of the new execution.

Consequence: final native-scope acceptance remains unproved. This is an evidence/schema coverage gap, not evidence that the existing twelve runs performed foreign work. Existing storage/content tests and import/lifecycle replay miss it because the necessary native state is absent from their input.

Smallest honest direction: keep the existing lifecycle/source/binding/storage authorities and add exactly one descriptive native observation mechanism under an independently reviewed supplement. Reuse the existing journal if that produces fewer independent cut/order rules. An outer versioned aggregate attachment is also possible in principle, but only if the supplement explicitly makes it part of the final aggregate contract and defines exact same-execution/upstream/guard/leaf/binding/prefix joins and failure selection. A free-standing JSON dump, aggregate hash alone, or the existing selection sidecar is insufficient.

Proof needed: actual new same-execution positive observations, required negative/failure cases, exact parser/attachment tests, and rejection of missing/reordered/duplicate observations, wrong phase/guard/leaf, another run's module/binding/stream, and contradictory evidence being relabeled as a pass. Preserve old v1 files byte-for-byte with their old meaning.

## F2 — P2: Warp's current readback path does not restore the pristine native context

Locations: `native_readback.py:146`–`:158` (D2H helper dispatch); `native_bootstrap.py:2905` (unscoped readback call); `native_setup.py:105`–`:110` (direct Warp helpers), `:337` (upload's outer ScopedDevice). Installed Warp sources: `_src/utils.py:1267`–`:1284`, `_src/context.py:3153`–`:3167`, `:8827`–`:8829`; `native/warp.cu:894`–`:910`; `native/cuda_util.cpp:146`; `native/cuda_util.h:253`–`:270`.

Trigger: a pristine Warp worker has default metadata device cuda:0 and no current native context. The real target1 stream is created and initialization restores the prior null context. Upload runs under `wp.ScopedDevice(target)` and restores the prior context/default device at exit. The subsequent first nonempty D2H readback calls `_warp_copy_exact` without an enclosing ScopedDevice.

The inspected pinned path is direct: `_warp_copy_exact` -> `wp.copy(..., stream=held_stream)` -> contiguous D2H -> `wp_memcpy_d2h(target_context, ..., explicit_stream)` -> native `ContextGuard guard(context)`. Its default `always_restore` is false. The constructor sets the target current context, and its destructor restores only when `need_restore` was set by a true restore policy. No production call configures that policy; a read-only search of installed Warp Python sources found no policy setter use. The later event/wait path has no outer saved-context restoration either. Thus source flow predicts target1 remains native-current after this readback, even though `runtime.default_device` can still be cuda:0. This conclusion is conditional native source reasoning, not a newly executed GPU result or a claim that device0 was touched.

Warp upload has an outer scope, so a before/after snapshot taken *inside* each upload helper correctly sees the temporarily selected target1, not ambient0. Its ambient restoration boundary is the enclosing scope's exit. The proposed observer cannot simply reuse CUDA's per-helper expectation that logical0 holds at every helper boundary. Both nested helper boundaries and outer scope exits need exact meaning in the new contract.

Consequence: a test that checks only default-device metadata would report restoration while the actual native context differs. Conversely a blanket assertion that every observed native context remains null would fail on legitimate scoped upload interior work. The CUDA logical/raw clarification expressly does not decide Warp's rule. No existing v1 field or normal readback numerical test detects this distinction.

Smallest honest direction: define Warp's entry/interior/return observation boundaries in the bounded supplement. Prefer restoring the prior context with an audited existing runtime scope on the real readback path, preserving explicit target/stream, original guard/deadline and failure ownership. If the author instead wants a metadata-only restoration interpretation that permits the raw context change, that is a separate explicit Warp policy decision requiring independent design review; it cannot be inferred from CUDA's exception or silently accepted by recording the unexpected result.

Proof needed: after the exact remedy/decision is reviewed, a fresh captured Warp positive showing default metadata, actual current context, all primary active states, and actual held target stream across initialization, upload scope exit, D2H, event/wait and terminal boundaries. A nonempty D2H case must distinguish this current path from a restored implementation. No device0 context fixture or setter may manufacture the positive ambient condition. Full required fault/UNKNOWN cases remain separate.

## Warp observer feasibility and limits

The suggested narrow direct-driver route is feasible enough to design and preflight; it is not approved for execution by this report.

Quiet runtime initialization: `Runtime` registers `Device(..., is_primary=True)` metadata, with `_context=None` and no stream initialization for those devices. Device attributes/UUID/PCI data are drawn from audited native metadata enumeration. `cuda_init` calls the driver initializer and `cuDeviceGet*` metadata functions; the inspected success chain contains no primary-context retain. When native current context is null, Runtime calls `set_default_device("cuda:0")`; that method assigns the resolved metadata object. It does not invoke `Device.make_current`. `config.quiet=True` avoids the greeting's memory query, whose native implementation otherwise temporarily retains each primary context. These are actual source distinctions, not assumptions from API names.

Production `get_stream(explicit_target)` is different: Device.context retains target's primary context, initializes real streams, and restores the saved current context. That target work belongs to its already-entered indexed runtime leaf. The stream must be strongly retained immediately on return before fallible metadata, observer calls or adoption. There is no reason to create a stream on ambient device0.

`wp.set_device` calls `device.make_current`, which accesses Device.context and can retain a primary context. With the valid cuda:0 device/string it is therefore unsuitable as a positive metadata-only fixture. The request's shorthand `set_device(0)` should not be copied into a harness: this installed DeviceLike is a Device/string/None contract, and the relevant forbidden operation is making cuda:0 current by a setter.

Use the already-loaded exact `libcuda.so.595.71.05` only, with explicit RTLD_NOLOAD and no fallback load. The fresh remote file hash matched the profile: `76e0d9678d41cf6b6ae71d18549d88a963d3d434f08108baa12457eb53ca88d6`. Warp itself loads libcuda through its native loader. The new observer must verify successful real init and actual same loaded-library identity, hold its CDLL and function objects before later work, and freeze ABI types and failure behavior. RTLD_NOLOAD changes loader reference bookkeeping but avoids newly loading an absent object; it does not by itself prove which GPU actions another getter performs. [Linux dlopen reference](https://man7.org/linux/man-pages/man3/dlopen.3.html).

Direct `cuCtxGetCurrent` returns a result code and a context pointer. A successful null result means no current context on the calling thread; an error must not be mapped to absence. `cuDevicePrimaryCtxGetState` separately returns flags and active state for a CUdevice. Both documented APIs report state rather than retaining/creating a primary context, and may surface earlier asynchronous errors. Freeze typed output pointers/result checks and exact device-handle/ordinal mapping; do not assume a configured ordinal is an undocumented CUdevice encoding. If needed, audited `cuDeviceGet` resolves that mapping without creating a context. A current-context-to-device query should only run after a successful non-null context result, without setters. [NVIDIA current context](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html), [primary context state](https://docs.nvidia.com/cuda/archive/12.3.1/cuda-driver-api/group__CUDA__PRIMARY__CTX.html), [device handles](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__DEVICE.html).

Two concrete observer traps were found in the installed native code:

- `wp_cuda_context_get_current` wraps `get_current_context`, which prints/checks a driver error and returns NULL on failure. It cannot distinguish failed query from successful no-current-context by pointer alone. Prefer the proposed typed direct driver result checking for evidence.
- `wp_cuda_context_get_stream` calls `get_context_info`. For an unseen context, that helper can switch context and issue a one-byte `cudaMallocAsync`/`cudaFreeAsync` workaround. `wp_cuda_context_is_primary` can temporarily retain a primary context. Neither is automatically a harmless cold observer. A target stream getter is eligible only after exact real target initialization has established the relevant existing context/stream cache, with that call chain audited. Cached Python `_context is None` alone remains insufficient native absence proof.

An observed native-current pointer does not itself prove identity with the target's primary context, and flags/active values are distinct from pointers. The exact amendment must choose how those relations are demonstrated without acquiring a context for observation. Likewise no snapshot proves absence of transient context creation/destruction between samples. The pinned native call-chain audit and trusted dependency exception-safety obligation remain required. This scan did not rebuild Warp or certify proprietary driver internals.

## Required bounded amendment, without prescribing a candidate schema

The existing version has already been emitted as actual execution evidence even though whole-stage acceptance is pending. An additive event would be rejected by the old exact parser and would change its semantics. The supplement must explicitly choose either a new version, or a documented additive grammar whose old records remain valid only as the historical coverage they actually contain. It must never let absence of new observations in old v1 imply a full native-scope pass. A separately versioned outer attachment naturally preserves old bytes but still needs full join/cut rules. This report chooses neither unspecified schema on the author's behalf.

Before implementation, freeze:

1. Exact per-lane fields/types/enums, observed versus source-derived values, nullable/not-yet-created states, and occurrence identifiers. Torch logical/raw and Warp default metadata/native context remain separate facts; no common `ambient_restored` boolean.
2. Producer ownership: only the fixed audited driver records primitive native returns under the original live ticket. The CPU collector executes no native getter, import, callback or second storage audit. Existing guard/import/runtime authorities supply the joins; parsing never creates authority.
3. Exact observation sites: pre-first-stream pristine evidence, successful returned-owner retention before observers, before/after original helper pairing against the actual indexed leaf, enclosing scope returns, runtime binding when available, and emergency terminal observations. A binding cannot be required before it exists. Current v1 normally permits nonfailure execution facts only under a setup guard; emergency observations need an explicit narrow rule, not a generic relaxation of that rule.
4. Source/environment/library and actual module/runtime/object-token relations, same thread/controller/session, exact lifecycle event and journal-prefix joins. Include original guard/operation facts rather than relying on repeated setup sequence numbers. Target current stream selection may differ from the held upload stream handle; specify which value is being compared.
5. Structural validity versus native acceptance: contradictory completed snapshots remain evidence, while the guard's UNKNOWN/poison transition prevents continuation. Define partial getter failure with no fabricated values, unmatched before on a thrown/hung helper, observer failure after a real return, terminal observation after an actual sync, and no retry/new ARM/deadline renewal.
6. Immutable validated cuts, raw held failure tails, persisted record/attachment/selection mismatch handling, prefix-only coverage and unavailable worker traces. Extending an outer attachment beyond the selected aggregate cut must not silently claim its facts are in that cut. Missing after/terminal output is incomplete evidence, not inferred success. Existing authorized release cannot be retrospectively erased by a later evidence write failure.
7. Fresh mutation/failure tests: wrong module/runtime/binding, wrong guard/leaf or run, source-derived labeled observed, omitted/duplicate/reordered pairs, contradictory value preserved then poison, getter error versus null, return retained before failed observation/adoption, failed selection/persistence windows, and terminal guard handling. Fixed upstream launch inputs remain fixed during replay attacks.

These are design obligations and smallest completion directions, not authored candidate fields or implementation permission.

## Verification, evidence, and unexecuted work

Read-only inspection covered the actual collector/parser/attachment, runtime initialization and return retention, upload/readback dispatch and scopes, historical observer wrapping/tickets, latest eleven negative-case delta, and installed Warp metadata/context/stream/init/native copy paths.

Independent commands/results:

- Local PowerShell UTF-8 reads, `rg`/`rg --files`, Get-FileHash, Git revision/branch/status/diff/diff-check established the exact boundaries above. No candidate files changed.
- `ssh tileops-dev` ran read-only Git/status/hash checks in both named worktrees. Remote `rg` was absent, so source search used stdlib Python over SSH stdin. No fallback host or remote write was used.
- Source readers ran `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`, importing only stdlib, and copied source text/hash observations into local scratch JSON files. Python reported 3.12.13 and an empty numpy/torch/warp module set. The Warp shared-library hash independently matched the profile: `4afdc3ddd8d4c7e4f68837e9b1f4268e767ef8527769321b152dde8cc3b11acd`.
- `native_scope_archive_probe_v1.py` was piped unchanged into that interpreter on primary183, using the existing CPU production parsers from clean bcc1bac. Final execution (`native_scope_archive_probe_v3.log`) validated **12/12 actual records, 12/12 selection pairs, 12/12 worker-prefix attachments**, and rejected an added unknown native-scope field in all twelve copies. Backend imports were empty. This is fresh CPU replay of retained runtime evidence, not twelve new GPU executions.
- Two earlier scratch-probe attempts stopped because the reviewer caught ValueError instead of GuardProtocolError for the expected unknown-field rejection, then because a historical request's environment file was incorrectly assumed to be inside the repository. The corrected script uses each retained request's original environment path. These were reviewer harness errors; no candidate behavior was changed or excused. Their tool outputs are retained in the session; v1/v2 log files contain only captured stdout and can be empty because the errors were stderr.
- Local `C:/Python314/python.exe -I -S -B -` read the archive in place (no extraction), enumerated artifact names/event vocabularies, and compared the historical observation's actual old guard/index to the new trace. Old 267 snapshots all matched their old guard references. This limited historical check is not a repeat of every old acceptance assertion.
- Local source-text hashing and profile comparisons matched context.py, utils.py, warp.cpp, warp.cu and cuda_util.cpp. The first broad membership assertion stopped on cuda_util.h because that header is absent from the profile. The corrected check explicitly records the two read headers as independently hashed installed source, not profile-pinned source. Source/binary equivalence remains the existing trusted pinned-dependency obligation, not a reviewer rebuild claim.
- Official NVIDIA API references and Linux dlopen documentation above were consulted to validate query/loader semantics. No driver function or backend was called by this reviewer.

New scratch artifacts, all under `C:/Users/AG/新建文件夹/c4_assets`:

- `native_scope_warp_source_read_v1.jsonl` and `native_scope_warp_utils_read_v1.json`: actual remote source text/hash observations.
- `native_scope_archive_probe_v1.py`, `native_scope_archive_probe_v1.log`, `native_scope_archive_probe_v2.log`, `native_scope_archive_probe_v3.log`: CPU probe and attempt outputs.
- `native_scope_cross_evidence_check_v1.log`: concrete old/new event mismatch plus the header membership error described above.
- `native_scope_source_pin_check_v2.log`: corrected explicit pin coverage.
- This complete unchanged report.

Relevant actual source hashes:

| Installed source | SHA256 | In current profile |
| --- | --- | --- |
| warp/_src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee | yes |
| warp/_src/utils.py | 01bd20f279f5762881c50cf8b6d766734fe21bd5f9b1ec286b34b1f3fcb27f24 | yes |
| warp/native/warp.cpp | b9feea467eb11e0d9016ca33fceb7b1a4358fc21c65513a692378155f1d05fc7 | yes |
| warp/native/warp.cu | 7c7e6e7a02db7b285ca1800d0ec52e28d2cf8a704435d925c70a8c5ccd8f176b | yes |
| warp/native/cuda_util.cpp | 21d9dfe57b861ada6b98d0fa2ccfb3b8ed0b14972ed840eb04d0ad7e5f70a89f | yes |
| warp/native/cuda_util.h | 59207856d17907bca935612ccb3ce7cde0952d8491011e1d45e8bb2d5b9ed1b8 | no |
| warp/native/warp.h | 0bb164157668ad83ce5fe78d3a583c68258ee333ce37d97e746b5d85a2e94511 | no |

The new test-only cases were read, not executed or approved. Their foreign stream, UNKNOWN-before-return and CUDA stream-only C1 paths deliberately create/observe adverse state. Such negative evidence cannot establish positive ambient restoration or add missing aggregate observations. They must prove the original production handler's sticky disposition and actual exit/reap after execution. Collection is not execution.

During this review the author reported a separate existing cleanup-hang injection defect: assigning `release_whole_graph` on a slotted guardian instance raises AttributeError. The author reports a scratch CPU reproduction and plans a class-level exact-instance repair. I did not independently reproduce or review that repair; it remains an explicitly reported test-fixture issue outside this native-scope verdict. No running matrix result was used as acceptance. The owned 70-case bcc1bac matrix was neither modified, signaled nor inspected for a final result.

No new GPU run/query/library load, backend import, package installation, reset, broad kill, worker launch, remote file mutation, private instruction edit, or change to the frozen candidate occurred. No pytest suite was needed to demonstrate F1's absent data; the more direct actual-record CPU replay was run. F2 is a source-based call-chain finding and still requires the bounded real-native proof after an exact reviewed remedy/decision.

## Remaining gates

Required next steps are the bounded exact amendment and fresh independent design review; reviewed production/harness changes if needed; actual both-lane ambient/foreign/UNKNOWN evidence with the new aggregate relationship; required full import/init/allocation/copy/event/adoption/audit/ACK/EOF/hang/cleanup/exit matrix; original regression and evidence persistence checks; then fresh independent whole-C2d3 code review. Historical v0 observations, current valid v1 records, this CPU replay, and unfinished test matrices cannot substitute for any missing real-native property.

No full C2d3/C2d/C4 approval, final graph seal, R_setup/R_final, prepared-sample authority, warmup, measured optical work or result is issued. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain CLOSED. C4 remains active.

Final integrity check after report creation: local HEAD remained 3617ab5f0f778b7f7410c84fa44ec5762dc4e228 with a clean tree and unchanged request SHA256. Primary remote runtime-custody worktree remained clean at bcc1bac4a6fa0ae9cfc758e85bf0a7c77fdca455; original binding worktree remained clean at a3d750bbaf79d760a6727e56146d277e1cb3b770. All foreground reviewer commands finished; no reviewer background process was launched. The author may archive this complete report under the existing evidence-only exception without enlarging its verdict or scope.
