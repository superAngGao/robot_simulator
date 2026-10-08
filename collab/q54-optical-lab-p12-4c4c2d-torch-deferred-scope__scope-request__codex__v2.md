Initiative: q54-optical-lab-p12-4c4c2d-torch-deferred-scope
Stage: bounded scope / diagnostic / deployment re-review
Author: codex
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: scope-request/gap-review v1; runtime-adoption design v2; cuda-ambient design/review v1; captured-import gap-review v1
Owner Summary: Correct the armed observer boundary and omitted third callback before the first captured Stream diagnostic. This freezes concrete scratch driver bytes and a three-file deployment delta. No Stream diagnostic has yet run; no upload or runtime acceptance is claimed.

## Frozen scope and predecessor

Production remains a3d750bbaf79d760a6727e56146d277e1cb3b770, clean detached183
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip. Local evidence base is
4b3585f7fbbf83d7956c63f2a04930f56218bb1a. No production code or watchdog change.
The predecessor request670475f0aa755457a489f6a5d512c541103b09398d2eb8df8883dc08f49f863c
and unchanged Request changes report77631d10678e4dc20b3dcbb344891aaad7b63861bf93ecd77e9f9e521a030b77
are preserved. That review substantiated the pinned normal first-Stream scope
rationale, conditional on correcting F1/F2. It did not approve execution.
The cold profile/evidence at e3e6d79 separately received bounded approval;
its unchanged report is archived, SHA2e463abf9745d1ab4428fffe4c5d7e632c3d546c4656050529b8a2ce0cb0b362.

This request supersedes v1's diagnostic placement and two-callback statement.
It retains the existing native trust model, source audit, no-foreign-context
rule, logical/raw distinction and every later-stage exclusion. It introduces
no importer exemption, source-policy exception, runtime permission learning,
callback removal, fake absence or new native shim.

## F1 repair: exact hook inside the original operation

Concrete child is sibling c4_assets/c2d3_captured_torch_stream_child_v3.py,
SHAaae8ae0f30832ef0ae05a887546fbd390598dfd55e5a1cfe7f1fdf9e83111ad3.
It derives from the already inspected captured-import child v2. Its fixed
controller.begin_device_leaf hook performs only the pre-Stream observer and
then delegates to the original begin. The actual production Stream constructor,
returned-stream append/adoption, identity validation, external/source audits,
binding construction/registry and binding adoption remain unchanged.

Its second fixed hook wraps session._hold_runtime_owner. It always invokes the
original method first; only role runtime_setup_binding triggers the diagnostic.
That call is the final adoption INSIDE initialize_native_runtime's existing
with controller.segment('setup'), after actual stream/binding/audits and before
normal segment exit. The hook requires the live setup ticket and adopted exact
binding/stream, observes raw/logical/context state, checks the Triton resolution
outcome, emits component observations, marks UNKNOWN and raises. It cannot
return a successful binding or reach runtime COMPLETE, upload, readback or
ordinary terminal release. A post-Stream observation exception also marks
UNKNOWN before propagation if the controller is not already terminating.
No observer is placed after initialize_native_runtime returns.

The supervisor wrapper changes only the fixed child path and retains the actual
tests/support/guarded_custody_process.py supervisor. Original ACK1000/op120000
and session target1 remain unchanged; no nested segment, extra ARM or renewed
operation budget is introduced. The invoker's existing outer process ceiling
is not used to claim watchdog compliance. Actual custody/reap must come from
the production supervisor record, as before.

Fixed remaining scratch files:
- c2d3_captured_torch_stream_supervisor_v3.py SHA617d3526e511e940b13ce6a215b78bd14fdffe2b1a0065b84a73ca3d17e7f20c.
- run_c2d3_captured_torch_stream_v3.py SHA4a9f5e3c583937cc21c34aabba93fb816ffe4428c4b2e6c341c1c9eea4381422.

## F2 repair: actual three callbacks and present-package resolution

Before Stream, require the exact ordered functions in held torch.cuda:
_check_capability, _check_cubins, _register_triton_kernels. Require the exact
held _LazySeedTracker class and raw manual_seed_all_cb/manual_seed_cb=None,
call_order=[]; do not invoke/clear/replace pending callbacks or call a Torch
lazy getter to establish ambient state. Earlier scratch children v1/v2 remain
unexecuted; v1's erroneous length2 assertion is preserved as history.

The third callback's installed source1695-1728 queries find_spec('triton'),
constructs two _WrappedTritonKernel Python objects and registers their schemas/
dispatch holders. Sparse kernel imports are inside their uncalled bodies.
The previous reviewer independently followed Library.define/impl, native
PythonKernelHolder construction, dispatcher and JIT registration listener.
Its six exact-commit sources remain in
torch_deferred_review_extra_v1/manifest.json,
SHAf68a4a993b3807542d2a92a39166cf1186c89948426db1dc36f141fd4de1894e.
This is CPU registration under the pinned dependency trust model, not a claim
that every Triton operation is CPU-only or admitted.

The new tests/data/runtime_environments/183_cuda_stream_author_v1.json is
byte-identical to external author draftv9, rawSHA
44f0833cdf507127695052b97b56864eb7b39783122d4b511240bcb6d6ef87d7,
canonicalSHA b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47.
Compared with the approved cold profile it adds exactly:
- os_native/libnvidia-ml.so.595.71.05 native file,2621336bytes,
  SHA9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5.
- site/triton/__init__.py Python file and one exact runtime file-module rule,
  1661bytes,SHA4ec1df745c6bfd03257370a2aab1534e7a31eaa62110cad2ff6c3b56a67cfc50.
- site/triton-3.5.1.dist-info/METADATA and its exact distribution identity.

Totals860files,759namedfilemodules,19distributions;42publications,7replacements,
8absence rows unchanged. No Triton descendant or native Triton library is added.
The existing standard file admission can resolve its pinned spec without
create/exec. The expected result is one file_resolved event for triton, zero
create/exec/body/publication, no module in sys.modules/records. The diagnostic
also checks exact held wrapper/delegate binding, source origin, package location,
delegate type/name/path, the two expected ops_table entries and both wrappers'
kernel_invoked=false. A resolved spec is not labelled an executed module.
Unexpected real imports still fail closed; no trace changes the table.

CPU authoring audit c2d3-torch-triton-probe-audit-v1.json SHA
929b01634f061f6dbda1f05db2ccc83ee36b8aceed4b7999d9026bf3d4dae47a
contains installed sources/metadata matched to the prior wheel inventory and
actual standard PathFinder resolution. No Triton body ran. PID320497/parent320496
exited and scoped ps found none. NVML audit SHA
327ef49aa51d2ad75b4f7cc3ef1c0fbb7c47c3cf9f660a23215bc91cbb22e8da
records readelf/ldd, exact file, observed shell configuration and CPU process
identity298708/298696; both exited. Builders draftv8/v9 verify all unchanged
NVML ELF dependencies' existing hashes before adding these exact files. Actual
worker asserts required unset allocator/visibility/preload settings before any
external import; configure_external_environment sets autoload0 and checks it.

## Required actual result after review, not yet obtained

Before Stream: rawCUDA0,raw Torch_initialized=false,all8primarycontexts absent;
logical0 is derived from the pinned pristine C10 initial state, not observed by
a warming getter. Observer is held, hash-pinned RTLD_NOLOAD cudart cudaGetDevice
with typed arguments; _cuda_hasPrimaryContext follows it. The current native
build's LazyNVRTC variant is substantiated by independent installed disassembly
torch_deferred_review_nvrtc_binary_v1.json SHA
2a19c660b78f7585b74cdd6545f33ae6a4e12f15d0d83e758a54b2a5b377e199.

After actual Stream/binding adoption INSIDE the original segment: logicalTorch0,
rawCUDA1,onlycontext1,initializedtrue,streamdevice1. Record actual native mappings
for comparison to fixed inventory; mappings cannot create permission. Snapshot
endpoints support the audited call chain but do not exclude transient side
effects or independently prove native exception safety.

Then intentional UNKNOWN and hard quarantine:4totalARM,3earliernormalCOMPLETE,
zero runtime COMPLETE,zero terminalARM/sync/release,worker74,reaped without
deadline expansion; verify actual PID/start/parent and OS/GPU residual evidence.
Any failed assertion, permission/scope mismatch, hang or different termination
remains failed evidence. Local syntax compilation only has passed so far; no
captured Stream run and no full runtime success is claimed.

Allocation/copy/events/readback/native exception matrix, preserved outer-device/
inner-stream regression, both-lane canonical/replay/both Go2, dual-GPU/foreign/
UNKNOWN/failure/hang/terminal matrix, versioned aggregate and full independent
C2d3 review remain mandatory. C4 remains active and all later gates stay closed.
