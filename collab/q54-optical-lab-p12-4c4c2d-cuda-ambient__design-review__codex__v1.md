Initiative: q54-optical-lab-p12-4c4c2d-cuda-ambient
Stage: design-review
Author: codex (fresh independent reviewer)
Version: v1
Date: 2026-10-08
Status: accepted
Related Files: CUDA ambient design-request v1; runtime-adoption design-request v2; torch-publication design-request v1; torch-native-scope gap-review v1
Owner Summary: Approve with comments the exact bounded CUDA ambient clarification. Logical Torch device/scoped stream restoration is a coherent, implementable boundary in the single-runtime worker. Raw CUDA state remains distinct evidence; foreign device work and UNKNOWN remain disqualifying. This is neither code approval nor a successful captured Stream/upload result.

IDENTITY, REVISION AND VERDICT

Reviewer session: /root/cuda_ambient_design_v1. Fresh task packet, no inherited implementation conversation, no subdelegation, no candidate edits.
Repository: C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design
Branch: implementation/c4-supervised-waits
Actual frozen HEAD: e390b20b418a81d24ef65012d704e4d1df461d51
Review base: 3932387b00b30644b9c2f29588ebcfb835125a1b
Production implementation: a3d750bbaf79d760a6727e56146d277e1cb3b770
Candidate: collab/q54-optical-lab-p12-4c4c2d-cuda-ambient__design-request__codex__v1.md
Independently verified SHA256: 9e6dee864f4575d8ae559c86038484658ee3829befa3ab4a88fe65ee4173ae9e
Initial and final candidate tree clean. HEAD unchanged. Base-to-HEAD adds only the 174-line candidate. Since a3d750b, the only changes are this design and two archived collab reports (374 inserted lines total). git diff --check passed.

Verdict: Approve with comments, solely the exact design interpretation and its bounded implementation/validation requirements. No substantive blocking design finding was established. Native-scope F2 is resolved at the contract level: restoring Torch logical0 is explicitly not restoring raw CUDA0. No runtime acceptance or full native-scope proof follows from this decision.

CONTRACT AND SCOPE ASSESSMENT

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully, rereading the Chinese principles with explicit UTF-8 after the first PowerShell decoding was garbled. Read the ambient candidate fully, runtime-adoption v2 (including the operative sections 2/3/5/6/7/8), torch-publication v1, and the native-scope report. Checked the runtime-adoption v2 approval scope rather than treating its status field as an approval.

The governing requirement already selects runtime scoped operations while independently forbidding foreign primary-context acquisition. The clarification chooses one of the alternatives explicitly left open by native-scope F2. It preserves the stricter side-effect requirement, single-runtime worker boundary, explicit indexed leaves, original watchdog, ownership and failure rules. It does not permit raw-current-state consumers to treat logical0 as raw0 or admit a mixed-runtime bridge.

This choice does not grant a performance advantage by hiding preparation work: no renderer, measured sample, baseline or result is enabled, both lanes retain their real target1/ambient0 obligation, and all upload/readiness/storage and termination requirements remain. The distinction is an API-state contract for a sealed worker, not a promise to restore a foreign caller's raw CUDA context in a reusable mixed-runtime process.

The design is testable: the positive case fixes pristine order, explicitly separates derived pre-Stream logical state from observed post-Stream state, pins observer/runtime identities, records all primary-context states, and requires explicit targets and normal-return restoration throughout the actual chain. The negative clauses reject warmed pre-observation, foreign context, incorrect logical restoration, unexpected raw changes, implicit escaping work, identity drift and failed initialization. The prior UNKNOWN/foreign real negatives remain mandatory. State authority stays in the guard; the aggregate stays descriptive and cannot mint live authority.

INDEPENDENT SOURCE CHECK AND COUNTEREXAMPLE ASSESSMENT

Read the actual locally retained exact-commit C10 CUDAFunctions.cpp, CUDAStream.cpp, Torch CUDA Module.cpp and Stream.cpp, installed streams.py and relevant torch/cuda/__init__.py sections. Independently verified all 16 files against torch-scope-upstream-5811a8d/manifest.json, and all 10 files against torch_native_scope_extra_upstream_v1/manifest.json. Hash verification of those extra files is not a claim to have repeated the previous review's complete import-callee audit.

CUDAFunctions.cpp defines thread_local targetDeviceIndex=-1. With cold raw0 and that initial state, GetDevice resolves0. ExchangeDevice(1) clears the deferred selection and calls cudaSetDevice(1). CUDA12 MaybeExchangeDevice(0), when primary context0 is absent and raw1 is current, writes targetDeviceIndex=0 instead of calling cudaSetDevice(0). Subsequent C10 GetDevice returns0 while cudaGetDevice remains1. This is actual control flow, not an inference from function names. The before logical0 is conditional on the documented cold state; the candidate correctly labels it derived.

Installed Stream.__new__(device) surrounds its native constructor with torch.cuda.device(device). Its enter path invokes lazy initialization before ExchangeDevice; the native constructor obtains the pooled stream on that selected device. The pool's default ordinal branch uses current_device and SetTargetDevice only after this exchange. CPU initialization of per-device stream-id arrays does not itself establish foreign contexts. Stream exit uses MaybeExchangeDevice. These paths substantiate implementability of the chosen logical/raw distinction without a foreign fixture.

Checked an adverse alternative, not merely the happy path: StreamContext alone can start on logical0 and end by set_stream(source_previous_stream0). THCPModule_setStream_wrap calls c10::cuda::set_device when the stream device differs from current logical selection; SetDevice can call cudaSetDevice(0), acquiring the forbidden context. Therefore the general statement 'Torch scoped APIs are safe' would be false. The candidate does not say that: it requires an audited scope selecting the same target and every device leaf to remain target-only.

The inspected production helper optical_worker/native_setup.py:82-90 already puts torch.cuda.device(stream.device) outside torch.cuda.stream(stream). The inner stream scope consequently captures/restores target1's stream, then the outer device scope restores logical0 virtually. native_readback.py imports and uses that same helper for CUDA D2H. Allocation uses an explicit target; event record receives the exact stream. SourceSession.initialize_native_runtime retains the returned stream before metadata and marks partial initialization UNKNOWN. NativeGuardedSetupLifecycle._check_running rejects compromised scope before another normal leaf. These checks show a concrete implementation direction exists; they do not certify all native allocation/event/copy callees or the entire implementation.

Consulted the pinned official C10 source online and NVIDIA CUDA12.8 primary documentation. The latter expressly identifies cudaSetDevice as initializing and making current the selected primary context. Thus adding cudaSetDevice(0) to force raw equality would violate the retained foreign-context rule, rather than repair this design.
Sources:
https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/c10/cuda/CUDAFunctions.cpp
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__DRIVER.html

NON-BLOCKING COMMENT C1 — RETAIN THE NESTED STREAM SCOPE AS A REGRESSION OBLIGATION

Location: candidate sections 2.2, 3 normal-chain checks and 4; native_setup.py:82-90; pinned StreamContext.__exit__/THCPModule_setStream_wrap.
Trigger: a future simplification removes the outer device scope and uses only torch.cuda.stream(target_stream) after pristine Stream initialization.
Consequence: returning to the ambient source stream may acquire context0 even though logical selection eventually reports0. A scalar logical-device check would miss the violation.
Disposition: non-blocking because the candidate already prohibits the effect and requires native audit plus foreign-context rejection, and the current helper already has the correct nesting. Preserve the nesting and carry this concrete counterexample into the code-review proof. Required proof before full acceptance remains the actual captured upload/readback case with native scope rationale and context observations, plus an appropriate regression that distinguishes this incorrect nesting. Do not run an unaccounted foreign-device positive fixture to produce that regression; an intentional foreign negative must retain its true dirty history and quarantine.

RAW EVIDENCE CHECKS

Read c2d3_torch_dependency_probe_child_v4.py in full and the actual ordinary_deferred_init_scope_observations.json, supervisor_stdout.json and worker.json. Independently verified:
- Native-scope archived report and scratch original: 5c20a33fbafddcf9dcb8c916f772a85a50ab4fc896367fd8c5ccefbaae11d8f3
- v4 child source: 8b5f531086cad67bd4a051b1fa687b665a8782ae382f3c8c7d4f1af1c2614f78
- v4 scope observation: 1b7def31a6c142ef239647a8e9d56d6315f2cd43dc37c98aa295159d3a365cfa
- v4 supervisor output: 573eefb387e943c7f4df1f83e2101739444f8696a214da4d8506b2e0b0c1c585
- v4 worker output: e783ba536bb7defb65d0a18dcab46a8ab503fa61fb708d2c8b9af9668d6daa71
- 16-source manifest: 32a9b6deb348201e9893f8b7c0cb90d450dd1f9017cdf4a356de6e5e45ca9f91
- 10-source manifest: dac8a89314241f7f81ab0e5241a13d2bb240895e98a7f49b37969f1437d43985
- Installed read artifact: dfdd990b390ae80f418f09f0cf9305c5e1b72fb506ecd80383b1344c4d83fbaa

Verified all five installed-read source strings independently against their SHA fields and the offline wheel inventory: streams.py, CUDAGraphsUtils.cuh, CUDAGraphsC10Utils.h, Context.h and CUDAGuard.h. Recorded environment is Python3.12.13, clean detached a3d750b. This is inspected stored provenance, not a newly queried remote deployment.

The v4 script obtains the _cuda_getDevice function object but does not call it before Stream. Its actual before getter uses RTLD_NOLOAD on the hashed libcudart, with typed arguments. Context observations follow the raw getter, and the raw _initialized flag is checked false. Actual before: raw0, all eight contexts absent, _initialized=false. Actual after: stream device1, logical0, raw1, only context1 present, _initialized=true. The returned stream is appended to the retained graph before metadata inspection. This corrects the earlier getter-order defect for this ordinary diagnostic, without making it a captured success.

Actual retained supervisor/worker records show parent216922, supervisor216928, worker216938, worker return74 and supervisor process_exited/EXITED, initial ARM1, normal COMPLETE0, emergency sync0, terminal ARM0, unknown_device_scope=true and release_status=not_authorized. No new GPU or OS residual query was performed. The report cannot independently elevate the old residual-cleanliness assertion beyond its historical evidence.

The ordinary diagnostic deliberately runs dependency operations after marking UNKNOWN and ultimately quarantines. It is not a normal production sequence and supplies no exception to _check_running. I did not use it as captured runtime binding, upload, readback or disposal authority. Snapshots cannot rule out transient native context activity; the pinned native trust/audit obligation remains explicit, and unsafe cleanup must stop admission rather than be excused by quarantine.

COMMANDS, RESULTS AND LIMITATIONS

- CPU-only PowerShell Get-Content -Raw/-Encoding UTF8 and Select-Object ranges read the documents and sources above; rg/rg --files located source and evidence paths. The initial garbled Chinese decoding was repaired by explicit UTF-8. One oversized combined read was truncated; the operative runtime-adoption sections and publication design were reread separately in full.
- git status --short, git rev-parse HEAD, git branch --show-current, git diff --stat 3932387b00b30644b9c2f29588ebcfb835125a1b HEAD, git diff --stat a3d750bbaf79d760a6727e56146d277e1cb3b770 HEAD, and git diff --check established the frozen and document-only boundaries. Final status and HEAD rechecked unchanged.
- Local python -I -S -B stdin scripts using only json/pathlib/hashlib verified candidate/report/diagnostic hashes, 16/16 and 10/10 source manifests, five installed source strings against the 18065-row inventory, and inspected raw worker summary fields. No Torch import, CUDA library load or target code execution occurred.
- web open read the exact official C10 revision and NVIDIA12.8 documentation listed above. No unpinned latest-runtime assumption is used for the verdict.
- Wrote only this scratch report outside the candidate. No production/test/design file edit, package installation, SSH execution, worker launch, GPU call, reset, kill, private context or process mutation occurred. No newly executed runtime tests are claimed. Real-device validation is intentionally outside this read-only design review.
- Did not re-review every publication relation, external profile row, import callee, allocator/copy/event native path, source policy or aggregate schema. Their existing and future bounded reviews remain required. The design's normal-chain assertion is an obligation to prove, not a result obtained here.

REMAINING GATES

Approval permits the clarified design to guide bounded C2d3 implementation and validation; it does not certify complete CUDA initialization, either full lane, upload/readback, terminal cleanup, or the product. The full captured production both-lane canonical/replay/both-real-Go2 upload/readback, failure/hang/dual-GPU/terminal matrix, versioned custody aggregate and replay negatives, and fresh whole-stage review remain mandatory. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 remains active.
