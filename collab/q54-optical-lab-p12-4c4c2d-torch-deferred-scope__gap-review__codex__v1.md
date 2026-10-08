Initiative: q54-optical-lab-p12-4c4c2d-torch-deferred-scope
Stage: independent gap review
Author: codex, fresh independent session /root/torch_deferred_scope_gap_v1
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: torch_deferred_stream_scope_request_v1.md; runtime-adoption design/review v2; cuda-ambient design/review v1; native_bootstrap.py; guarded_lifecycle.py; evidence listed below
Owner Summary: Request changes before executing the proposed captured pre-upload diagnostic. The inspected first-Stream CUDA/native chain supports the intended target1-only normal scope and logical0/raw1 restoration under the approved dependency trust model. However, the frozen request omits a real third deferred callback and its currently rejected Triton resolution, and places post-binding native observations after the runtime guard has disarmed. Neither defect requires a new native sandbox or relaxing the device-scope contract.

## Verdict and frozen identity

**Request changes** for advancing the exact frozen request to execution. This is a bounded source/evidence gap review, not a test execution or full C2d3 code approval. The positive native-scope assessment below is not permission to bypass either finding or the separately reviewed cold deployment profile.

- Request SHA256 independently verified: `670475f0aa755457a489f6a5d512c541103b09398d2eb8df8883dc08f49f863c`.
- Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Actual local HEAD: `e3e6d791cb30a9c2677b850f33c56f51971ccebf`; branch `implementation/c4-supervised-waits`; tree `4cd7d865435207c759a3045c8b88c4d280c68f4a`.
- Production base: `a3d750bbaf79d760a6727e56146d277e1cb3b770`. The base-to-HEAD delta is eight documentation/evidence/profile files, 634 added lines. `git diff --exit-code ... -- optical_worker AGENTS.md collab/README.md collab/REVIEW_PRINCIPLES.md` passed: no production or operative workflow delta.
- Initial and final local status clean. Candidate and request remained unchanged during review.
- Independently checked primary remote `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`: clean detached HEAD `a3d750bbaf79d760a6727e56146d277e1cb3b770`. Read-only CPU inspection used `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -`, Python 3.12.13. No fallback host used.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Reread the Chinese principles and README explicitly as UTF-8 after initial PowerShell decoding artifacts. Read the runtime-adoption v2 contract and approval, cuda-ambient v1 contract and approval, and native-scope gap-review v1. Prior approvals define scope and obligations; their source/evidence conclusions were independently checked where relevant here.

## Blocking findings

### F1 — P1: proposed post-return native observations have no live runtime-segment deadline

Location: frozen request, “Proposed actual diagnostic obligations,” paragraph beginning “After source runtime audit/binding returns”; `optical_worker/native_bootstrap.py:2349-2433`; `optical_worker/guarded_lifecycle.py:187-238`.

Trigger: follow the request literally, call `initialize_native_runtime()`, then perform raw CUDA/current-device/primary-context observations and deliberately mark UNKNOWN. Production `initialize_native_runtime` exits `with controller.segment("setup")` before setting `phase = "runtime_ready"` and returning the binding. Successful context-manager exit performs `normal_body_returned`, COMPLETE, DISARMED and clears the guard ticket/reference. The post-return observer calls therefore run after the original operation deadline has ceased to protect them.

Consequence: an observer that hangs can escape the promised original armed operation budget. The evidence would also contain successful runtime COMPLETE before the intended diagnostic fault, rather than a fixed failure inside the original runtime operation. UNKNOWN set afterward cannot retroactively protect an earlier blocking native call or undo the completed segment. Keeping ACK1000/op120000 numeric settings unchanged does not fix this placement.

Why existing checks do not prevent it: raw ctypes/native observation performed by the fixed diagnostic is outside the controller's guarded leaf API. Source review of the observer's normal device scope does not make its execution bounded. The approved ambient contract expressly subjects observers to native-scope review, and the runtime-adoption failure contract preserves the original deadline.

Required repair/proof: freeze a concrete fixed diagnostic hook that executes after production returned-stream retention/adoption and the intended identity/audit/binding work, while the original runtime segment is still armed and before its normal completion begins. Keep the production Stream constructor and custody/binding implementation actual. Perform the fixed post-Stream observations there, mark UNKNOWN and raise into the existing failure/quarantine path without returning normally from that segment. No new ARM, nested segment, renewed deadline, synchronization or ordinary release. Have fresh review verify the exact hook position and byte hash. A hook around the actual final binding-adoption boundary is one possible diagnostic mechanism, provided it first preserves the original production call and cannot trigger on another phase/owner.

The subsequent run must prove zero runtime-initialization COMPLETE, no post-fault normal work, no terminal sync/release or extra ARM, and actual supervisor exit/reap. Earlier successful CPU/prelude COMPLETE records are legitimate and must not be confused with runtime COMPLETE. No corrected diagnostic artifact was supplied for approval in this frozen request.

The author independently raised this same placement concern during review. I confirmed it against the actual segment exit and return code; the finding does not rest on that author message.

### F2 — P2: the claimed two-callback deferred chain omits `_register_triton_kernels` and an unadmitted present-package probe

Location: frozen request, author-inspected chain item 4; installed `torch/cuda/__init__.py:356-357,416-421,1695-1728`; `torch/__init__.py:2699-2710`; frozen `tests/data/runtime_environments/183_cuda_cold_import_author_v1.json`; `native_bootstrap.py:1361-1444`.

Trigger: the pristine first Stream enters `_lazy_init`. The installed CUDA module queues three callbacks, in order: `_check_capability`, `_check_cubins`, `_register_triton_kernels`. The third registration at line 1728 is unconditional. The seed tracker is separately appended by `_lazy_init`; absence of seed calls does not remove this third callback.

The third callback executes `importlib.util.find_spec("triton")`. CPU-only remote file inspection independently found `triton/__init__.py` present, 1661 bytes, SHA256 `4ec1df745c6bfd03257370a2aab1534e7a31eaa62110cad2ff6c3b56a67cfc50`, matching the pinned inventory. The frozen profile has no top-level `triton` module rule, absent-probe row or disabled-optional row. It has only a distinct `torch._library.triton` module row. The captured finder's rule check rejects the top-level probe as `unlisted external import`, before resolving a spec. Its failure path is sticky.

Consequence: the proposed actual captured chain cannot normally return a Stream and binding with this profile. A diagnostic assertion expecting two queued calls would fail even earlier. The omission also means the request's asserted native-call coverage is incomplete as written. This is a concrete deployment/coverage failure, not evidence that Triton kernel compilation or a foreign CUDA context actually occurs.

Why prior evidence misses it: the previous captured diagnostic stops before Stream, so it never executes queued callbacks. The ordinary v4 process uses normal import resolution and is not constrained by the captured profile. Its successful Stream cannot prove captured probe permission. A file hash/inventory entry alone does not authorize resolution or execution in this profile.

Required repair/proof: correct the frozen queue contract to the three exact retained function identities in the actual pinned `torch.cuda` module and require the seed tracker to contain no pending callbacks; do not clear/replace a callback or seed state. Freeze the necessary exact present-package resolution facts and deployment metadata in a separately reviewed profile delta. The current finder can return a pinned spec and record `file_resolved` without create/exec; evidence must honestly identify that resolution-only outcome. Do not call a present Triton package “absent” or “disabled” to bypass its real branch, and do not import it merely to make the probe succeed. Verify the returned spec's pinned origin, package locations and loader, zero Triton body/create/exec for this path, and actual callback order/registration outcome. Audit any new import or publication discovered by the real path without learning permissions during execution.

I inspected the third callback's installed Python and additional exact-commit dispatcher sources. The inner sparse-kernel imports are inside function bodies and are not invoked by registration. In the present-Triton branch `_TritonLibrary.registerOp` defines two schemas and installs held Python kernel wrappers through `Library.define/impl`; the inspected native binding constructs a `PythonKernelHolder` and dispatcher schema/kernel entries, rather than executing its call operator. Dispatcher registration and the JIT registration listener retain callable/metadata state. This supports a CPU registration rationale under the approved trusted-native boundary; it does not excuse the missing captured resolution rule. Additional source/evidence paths are recorded below for re-review.

The author later confirmed the same omitted callback and a two-element assertion in an unexecuted draft child. That draft was not supplied or reviewed here; the finding follows directly from the frozen installed source and profile.

## Native scope and custody assessment

No substantive unsafe-native-cleanup finding was established in the inspected first-Stream path. The following is a normal-path source assessment for the pinned pristine CUDA deployment, conditional on correcting F2 and satisfying the actual profile/observer prerequisites. It does not certify all exceptions inside the native dependency, general Stream calls, warmed processes, extra queued seeds or plugins.

1. **Explicit index and lazy ordering.** Installed `streams.py` uses `with torch.cuda.device(device)` around the native constructor. The actual explicit `torch.device("cuda", 1)` flows through `_get_device_index` without the optional-current-device fallback; I read the installed `_utils.py` and CUDA utility adapter. `THCPModule_exchangeDevice` calls `device_lazy_init(CUDA)` before `ExchangeDevice(1)`. The lazy helper invokes Python `_lazy_init`; `_cuda_init` reaches `globalContext().lazyInitDevice(CUDA)` and `CUDAHooks::init`. Thus initialization must be audited while raw ambient0 is still current; merely passing device1 to Stream would not prove it safe.

2. **Allocator/P2P initialization.** `CUDAHooks::init` performs device count, allocator init and P2P cache initialization. Native allocator `init(count)` creates CPU `DeviceCachingAllocator` objects. Their constructor initializes block pools, counters/config and recorder state, with no CUDA allocation. Inspected allocator configuration defaults and unset `PYTORCH_CUDA_ALLOC_CONF` select the native backend and do not enter configured driver/allocation branches. The alternate async `init(count)` also only creates CPU vectors; this does not approve later async allocation. `init_p2p_access_cache` initializes vectors/diagonal flags, not peer enablement. The required future worker configuration must be checked before import; an earlier shell's unset variables are not future-state proof.

3. **Actual MAGMA variant.** Read the installed CUDAConfig/version evidence and actual callback disassembly. `AT_MAGMA_ENABLED()==1` is not the deciding fact. Installed `libtorch_cuda.so` SHA256 `02250527966ae122ccfc89d0306736874f9c619ba04431871ef76175e7253b66` has the MagmaInitializer callback at `0x3631450` consisting of `ret`, matching `LinearAlgebraStubs.cpp`'s empty callback. The separate lazy linalg DSO's `magma_init` implementation is not the callback executed here. No basis was found to attribute that other DSO's initialization to first Stream.

4. **Default generators before target selection.** `_cuda_init` creates a default generator for each explicit ordinal slot and seeds CPU state. `CUDAGeneratorImpl` construction/seed/set_current_seed call `assertNotCapturing`; the installed `CUDAGraphsUtils.cuh` implementation first asks `hasPrimaryContext(current_device())`. With pristine raw0 and no primary contexts this returns None without `getCurrentCUDAStream`/`cudaStreamIsCapturing`. No per-generator device selection occurs in that loop. `GeneratorImpl` and CUDA seed source use CPU state and `std::random_device`.

5. **Primary-context observer build variant.** `_hasPrimaryContext` validates the index against device count and invokes `nvrtc().cuDevicePrimaryCtxGetState`; it does not call PrimaryCtxRetain. `CUDAHooks.cpp` has a `USE_DIRECT_NVRTC` alternative, so I did not assume LazyNVRTC solely from the request. A fresh CPU-only disassembly of the installed, hash-matched `at::cuda::detail::nvrtc()` at `0xc82430` shows its handle initialized to `at::cuda::detail::lazyNVRTC` at `0x3aeba040`. This independently closes the actual build-variant question. The lazy stub resolves the single called driver function from libcuda; binding the table's separate retain function is not invoking it. No requirement to reverse-engineer the closed CUDA/OS driver was introduced.

6. **Metadata callbacks and NVML.** `_check_capability` and `_check_cubins` enumerate explicitly indexed properties. Native `getDeviceProperties` initializes CPU caches and calls `cudaGetDeviceProperties` for that index, without an all-device guard/setter. Python `device_count` may load `libnvidia-ml.so.1`, initialize NVML and query count, with the known runtime-count fallback. This is a real native dependency, despite being metadata. I freshly rehashed the exact NVML file to `9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5`, size 2621336, and read its retained readelf/ldd audit. The future environment must freeze/admit and verify its actual resolved identity plus the existing dependency closure. This review does not independently approve a yet-unfrozen NVML profile row. The third callback is addressed in F2.

7. **Target selection and pool creation.** After `_lazy_init` completes, `ExchangeDevice(1)` calls `cudaSetDevice(1)`. `THCPStream_pynew` then observes selected device1 and calls `getStreamFromPool`. Global stream initialization obtains device count/priority and creates CPU stream-id slots; on this path the priority query occurs after target selection. `initDeviceStreamState(1)` creates the pool via `CUDAGuard(1)` and `cudaStreamCreateWithPriority`, only for target1. It may create many pooled streams, not merely one. The native pool deliberately retains streams for process lifetime. No all-device pool creation, target sync, tensor allocation/copy/event submission or native release of an in-use allocation was identified on this path.

8. **Logical/raw restoration.** CUDA12 `MaybeExchangeDevice(0)` with primary context0 absent writes the thread-local logical target index rather than calling `cudaSetDevice(0)`. C10 GetDevice therefore reports logical0 while raw CUDA remains1. That is precisely the accepted cuda-ambient v1 boundary. Stream creation does not change the thread's per-device current stream selection to the returned pooled stream. No current-stream getter should be added before Stream merely to construct a comparison fixture. This approval rationale does not extend to the later outer-device/inner-stream upload helper; its preserved nesting and required regression remain separate obligations.

9. **Returned ownership and partial failures.** Production appends the returned stream to the already held session's `runtime_owners` immediately, before reading `stream.device.index`; `_hold_runtime_owner` also retains before registration. `_stream_identity` validates actual indexed device and native handle, then external/source audits and binding adoption occur. Stream/device metadata or registration rejection cannot discard the already returned stream. The inner initialization exception handler marks UNKNOWN; outer session failure and the segment controller preserve errors/available graph and quarantine. Native constructor failure before return remains the approved trusted-native exception-safety obligation; Python process quarantine cannot repair an already completed native use-after-free. No such unsafe cleanup was found, but it was not proved absent for every injected native failure. The pool's retained native streams and the Python returned wrapper are distinct resource facts.

## Historical runtime evidence and its limits

Read the complete ordinary v4 child and its raw scope/worker/supervisor JSON; rechecked hashes:

- Child `c2d3_torch_dependency_probe_child_v4.py`: `8b5f531086cad67bd4a051b1fa687b665a8782ae382f3c8c7d4f1af1c2614f78`.
- Scope observations: `1b7def31a6c142ef239647a8e9d56d6315f2cd43dc37c98aa295159d3a365cfa`.
- Supervisor output: `573eefb387e943c7f4df1f83e2101739444f8696a214da4d8506b2e0b0c1c585`.
- Worker output: `e783ba536bb7defb65d0a18dcab46a8ab503fa61fb708d2c8b9af9668d6daa71`.

It uses typed cudaGetDevice through hash-pinned RTLD_NOLOAD cudart before Stream, observes contexts afterward and checks raw `_initialized` false. Obtaining the `_cuda_getDevice` function object does not invoke it; its first invocation is after Stream. Observed before: raw0, eight absent primary contexts, initialized false. After: stream device1, logical0, raw1, only primary context1, initialized true. Returned stream is held before metadata.

This is historical ordinary execution under deliberate UNKNOWN, with normal COMPLETE0, terminal ARM0, sync0, release not authorized and full quarantine. It did not use captured admission, did not record the exact queued callback identities, and is not a new review execution. Snapshot endpoints cannot exclude transient native context creation/destruction; the source chain plus the approved pinned-native trust obligation remains necessary. I did not rerun OS/GPU residual checks or adopt prior residual assertions as fresh evidence. The separately reviewed captured cold profile and pre-Stream v6 success are not assumed approved by this report.

## Verification, paths and additional evidence

All paths below are under `C:/Users/AG/新建文件夹/c4_assets` unless repository-relative. Independently verified all source-file hashes in the three supplied manifests: 16/16, 10/10 and 6/6 passed. Verified all five stored installed-read sources and all five dill/sites-audit source strings against both their own SHA fields and the 18065-row offline inventory. Fresh remote library/utility/Triton file reads also matched that inventory.

Supplied evidence inspected:

- `torch-scope-upstream-5811a8d/manifest.json`, SHA256 `32a9b6deb348201e9893f8b7c0cb90d450dd1f9017cdf4a356de6e5e45ca9f91`; operative files inspected: CUDAFunctions.cpp, CUDAStream.cpp, CUDAHooks.cpp, CUDAContext.cpp, CUDAGeneratorImpl.cpp, CUDACachingAllocator.cpp, PeerToPeerAccess.cpp, LinearAlgebraStubs.cpp, CUDA Module.cpp/Stream.cpp and device_lazy_init.cpp.
- `torch_native_scope_extra_upstream_v1/manifest.json`, `dac8a89314241f7f81ab0e5241a13d2bb240895e98a7f49b37969f1437d43985`; relevant Context and async-allocator init paths inspected. Hash verification of all ten files is not a claim to have re-reviewed their entire cold-import closure.
- `torch_deferred_sources_v1/manifest.json`, `794a007a71935fe0ab1fc74809ca2100683eeeeb9403cde2e1af728a1128f79c`; inspected LazyNVRTC, ATenNVRTC header, driver_api, CUDAAllocatorConfig.cpp/.h and GeneratorImpl.cpp. C10 DriverAPI's other allocation-related resolution branches are not silently treated as executed by the ATen LazyNVRTC observer.
- `torch_native_scope_installed_read_v1.json`, `dfdd990b390ae80f418f09f0cf9305c5e1b72fb506ecd80383b1344c4d83fbaa`.
- `c2d3-torch-dill-sites-audit-v1.json`, `80a77febaabb2408a0da25b12789342d80124719a96f1d7f9450ab41ef06c8c7`.
- `c2d3-torch-init-native-audit-v1.json`, `1b01df2dc18acde8ec7d7939a6c6daf8c9f901576eac6626ea79944965be18ec`.
- `c2d3-torch-deferred-nvml-audit-v1.json`, independently hashed `327ef49aa51d2ad75b4f7cc3ef1c0fbb7c47c3cf9f660a23215bc91cbb22e8da`.

New reviewer scratch evidence, no candidate edits:

- `torch_deferred_review_cpu_read_v1.json`, SHA256 `0999152a93220531d8136b7d222ced3a495c377e501d133ca682c40b7774ffbe`: CPU-only remote identity/status, full installed torch/library.py, torch/_utils.py, torch/cuda/_utils.py and Triton root source, plus independently rehashed NVML file. Reading the Triton source did not execute it.
- `torch_deferred_review_nvrtc_binary_v1.json`, SHA256 `2a19c660b78f7585b74cdd6545f33ae6a4e12f15d0d83e758a54b2a5b377e199`: rehashed installed libtorch_cuda and targeted nvrtc() disassembly.
- `torch_deferred_review_extra_v1/manifest.json`, SHA256 `f68a4a993b3807542d2a92a39166cf1186c89948426db1dc36f141fd4de1894e`: six additional official upstream files at exact `5811a8d7da873dd699ff6687092c225caffcf1bb`: torch/csrc/utils/python_dispatch.cpp; torch/library.h; aten/src/ATen/core/library.cpp; aten/src/ATen/core/dispatch/Dispatcher.cpp and OperatorEntry.cpp; torch/csrc/jit/runtime/register_c10_ops.cpp. Inspected relevant schema/kernel registration, holder construction and listener code. These are upstream source bytes, not a claimed wheel-binary reconstruction. [Pinned dispatcher binding source](https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/torch/csrc/utils/python_dispatch.cpp).

Commands were CPU-only PowerShell `Get-Content -Raw/-Encoding UTF8`, targeted ranges, `rg`/`rg --files`, `Get-FileHash`, Git status/revision/tree/diff checks; local `python -I -S -B -` using json/pathlib/hashlib for manifest/inventory checks and profile rows; SSH stdlib scripts for file reads and git facts; targeted `objdump -d --disassemble=_ZN2at4cuda6detail5nvrtcEv` invoked through the remote stdlib script. Exact-commit extra sources were read through official raw GitHub and saved with urllib into scratch.

Tool-only errors were corrected: initial Chinese stdin path encoding in local Python, the first manifest parser expecting `files` instead of `sources`, a Windows rg wildcard and two wrong-working-directory reads, and unquoted PowerShell `HEAD^{tree}`. Corrected checks succeeded. Oversized initial outputs were reread using targeted ranges. These are tooling issues, not runtime or product failures.

No Torch import, CUDA/NVML dlopen, GPU call, new worker/supervisor, fault injection, test suite, package change, kill/reset/private context, candidate edit or delegation was performed. Short CPU inspection commands completed; no GPU execution or new residual-cleanliness proof is claimed. This review does not duplicate the separate whole cold-profile audit, validate every profile permission row, or approve any future diagnostic/profile bytes that were not frozen here.

## Smallest next proof and remaining gates

Repair F1/F2, freeze the concrete diagnostic and exact profile delta (including actual NVML identity), and obtain fresh independent re-review before execution. Preserve the pre-Stream actual pristine checks after the observer, raw `_initialized=false`, all contexts absent, exact three callback identities/no seeds, held hash-pinned observer handles, and actual production Stream/adoption/audits/binding. Post-Stream observations and intentional UNKNOWN must remain under the original runtime segment and deadline. Any drift, missing profile permission, unexpected context, callback, import, observer return or failure is a failed proof and sticky quarantine, never normalization.

This report provides neither a captured Stream success nor complete deferred-chain execution authority for the frozen request. Allocation, copy, event/ready wait, readback, terminal target sync/release, outer-device/inner-stream regression, both-lane ambient/foreign/UNKNOWN and failure/hang matrices, canonical/replay/both real Go2 inputs, custody aggregate/replay negatives, and full C2d3 review remain required. No R_setup/R_final, final graph seal, prepared sample or product capability is granted. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload, R2 and shipping remain closed. C4 is not complete.
