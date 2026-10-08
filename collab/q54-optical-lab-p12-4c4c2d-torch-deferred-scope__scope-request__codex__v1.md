# C2d3 bounded deferred Stream scope gap review

Frozen production a3d750bbaf79d760a6727e56146d277e1cb3b770; local documentation
HEAD e3e6d791cb30a9c2677b850f33c56f51971ccebf; no production delta.
Request: independently determine whether the pinned pristine first
torch.cuda.Stream(device=torch.device('cuda',1)) call has enough target-only
normal-scope justification to execute in the existing captured controller,
followed by a fixed pre-upload UNKNOWN/quarantine. This is a review of evidence
and existing native entrypoint scope, not new policy or full upload acceptance.
No prescribed verdict. Identify exact unresolved callees/build assumptions.

Authority: runtime-adoption design v2 sections3/6, native-scope gap-review v1,
cuda-ambient approved design/review v1. Trusted pinned native internals remain an
explicit boundary; do not require an unapproved allocator interposer or pretend
that snapshots prove exception safety. The previous review approved cold import
only. Current captured cold import v6 reaches a pre-Stream checkpoint with
_initialized=false, all42publications/7replacements reconciled. A separate ongoing
review checks that cold-profile evidence; do not assume its verdict.

## Author-inspected chain (claims to verify)

1. Installed streams.py __new__ enters torch.cuda.device(explicit indexed target)
before calling its native constructor. cuda.device enters _exchange_device;
THCPModule_exchangeDevice invokes device_lazy_init before ExchangeDevice.
The latter calls Python torch.cuda._lazy_init -> _cuda_init ->
globalContext.lazyInitDevice(CUDA) -> CUDAHooks::init.
2. CUDAHooks::init calls device_count_ensure_non_zero, cachingallocator.init(n),
init_p2p_access_cache(n), and the installed MAGMA callback. The native allocator
init constructs CPU DeviceCachingAllocator objects for each ordinal; its
constructor only initializes config/counters. P2P init fills CPU vectors and
diagonal flags, not cudaDeviceEnablePeerAccess. The actual installed MAGMA
callback is ret (see binary audit), not the separate linalg DSO magma_init.
3. _cuda_init builds one default CUDAGeneratorImpl per explicit ordinal and
seeds it. CPU state creation and nondeterministic seed generation are preceded
by assertNotCapturing. CUDAGraphsUtils currentStreamCaptureStatus first checks
hasPrimaryContext(current_device); in the pristine ambient0 case this is false,
so it does not call getCurrentCUDAStream/cudaStreamIsCapturing and does not
materialize context0. The seed loop does not select each generator's ordinal.
_hasPrimaryContext uses LazyNVRTC's cuDevicePrimaryCtxGetState from libcuda;
binding that function does not call cuDevicePrimaryCtxRetain.
4. Python queued calls are _check_capability and _check_cubins. With no previous
CUDA work or user seed calls, the lazy seed tracker has no queued seeds. Both
checks enumerate device metadata. ATen getDeviceProperties calls explicit-index
cudaGetDeviceProperties, not a device guard/context acquisition. Python
device_count may use ctypes NVML loading/init/count. This is additional native
deployment closure, even though it is metadata rather than queue work.
5. Only after lazy init returns does ExchangeDevice(1) select/create target1.
THCPStream_pynew reads current_device1 and getStreamFromPool. Stream global state
checks count/priority and allocates CPU ids; target1 pool initializes with
indexed CUDAGuard and cudaStreamCreateWithPriority. No all-device stream pool is
created. Return restores logical0 with MaybeExchangeDevice; rawCUDA may stay1,
as explicitly clarified by the approved ambient design.
6. SourceSession immediately appends the returned stream to runtime_owners before
metadata access and binding validation. Its exception path marks UNKNOWN and
retains available graph; native constructor failure internals remain trusted,
not repaired by process quarantine. The proposed diagnostic ends after actual
initialize_native_runtime returns/audits, before upload. It introduces no sync,
release, final readiness or product capability.

## Exact evidence locations

All below in C:/Users/AG/新建文件夹/c4_assets. Existing raw evidence also on183.
- torch-scope-upstream-5811a8d/manifest.json SHA32a9b6deb348201e9893f8b7c0cb90d450dd1f9017cdf4a356de6e5e45ca9f91:16 pinned upstream files including CUDAHooks,CUDAFunctions,CUDAStream,CUDAGeneratorImpl,CUDAContext,PeerToPeerAccess,Module.cpp,Stream.cpp,device_lazy_init.
- torch_native_scope_extra_upstream_v1/manifest.json SHAdac8a89314241f7f81ab0e5241a13d2bb240895e98a7f49b37969f1437d43985:10 extra files; contains Context.cpp and async allocator alternative.
- torch_deferred_sources_v1/manifest.json SHA794a007a71935fe0ab1fc74809ca2100683eeeeb9403cde2e1af728a1128f79c:6 official files downloaded at exact5811a8d7da873dd699ff6687092c225caffcf1bb:driver_api.cpp,LazyNVRTC.cpp,ATenNVRTC.h,CUDAAllocatorConfig.cpp/.h,GeneratorImpl.cpp.
- torch_native_scope_installed_read_v1.json SHAdfdd990b390ae80f418f09f0cf9305c5e1b72fb506ecd80383b1344c4d83fbaa:installed streams.py,Context.h,CUDAGuard.h,CUDAGraphsUtils.cuh,CUDAGraphsC10Utils.h, inventory-matched source strings.
- c2d3-torch-dill-sites-audit-v1.json SHA80a77febaabb2408a0da25b12789342d80124719a96f1d7f9450ab41ef06c8c7:includes full pinned installed torch/cuda/__init__.py.
- c2d3-torch-init-native-audit-v1.json SHA1b01df2dc18acde8ec7d7939a6c6daf8c9f901576eac6626ea79944965be18ec:installed native libtorch_cuda hash and actual MAGMA callback disassembly.
- c2d3-a3d750b-torch-pristine-init-diagnostic-v4/ordinary_deferred_init_scope_observations.json SHA1b7def31a6c142ef239647a8e9d56d6315f2cd43dc37c98aa295159d3a365cfa:ordinary UNKNOWN diagnostic, not captured approval. Before raw0/initializedfalse/all8contextsfalse; after logical0/raw1/onlycontext1/stream1. No Torch getter before Stream. Full script c2d3_torch_dependency_probe_child_v4.py SHA8b5f531086cad67bd4a051b1fa687b665a8782ae382f3c8c7d4f1af1c2614f78.
- c2d3-torch-deferred-nvml-audit-v1.json:new CPU-only read of exact /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.595.71.05,size2621336,SHA9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5. readelf/ldd and observed unset allocator/visibility/preload config. Not yet a reviewed environment row or future environment proof.

## Proposed actual diagnostic obligations, if the scope rationale holds

Keep exact captured path/supervisor and ACK1000/op120000. No deadline extension.
Freeze the additional NVML file and existing dependencies, check actual pristine
state after metadata observer and enforce relevant unset environment assumptions
before import. In worker, use held, hash-pinned RTLD_NOLOAD cudaGetDevice and
_cuda_hasPrimaryContext; never a lazy Torch getter before Stream. Record exact
queued callback identities and absence of pending seeds before allowing first
Stream, without clearing or replacing them. Normal stream call remains actual
production implementation, no monkeypatched Torch constructor or fake binding.
After source runtime audit/binding returns, record logical/raw/context state and
exact stream device, then mark UNKNOWN and invoke existing failure/quarantine.
No upload or terminal sync. Any anomaly is a failed proof, not normalization.

This diagnostic cannot approve allocation/copy/event/readback callees; those
and the existing outer-device/inner-stream regression remain separate next
obligations. Full C2d3/C4 and every later gate remain unaccepted/closed.
