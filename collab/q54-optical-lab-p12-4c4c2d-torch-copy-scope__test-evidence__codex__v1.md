Initiative: q54-optical-lab-p12-4c4c2d-torch-copy-scope
Stage: bounded test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: draft
Related Files: cuda-ambient design/review v1, comment C1; native_setup.py::_torch_copy_exact
Owner Summary: An ordinary supervised UNKNOWN diagnostic distinguishes the actual nested production copy helper from the explicitly faulty stream-only alternative. Both preserve logical0, but the faulty alternative creates foreign context0. Neither is captured a2/readback or normal-disposal acceptance.

Production remote clean detached a3d750bbaf79d760a6727e56146d277e1cb3b770,
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip; pinned robot_sim CPython3.12.13,
Torch2.9.1+cu128,8H200,driver595.71.05. Local frozen e0a3b7a unchanged during
these external scratch diagnostics. No production/test/profile edit occurred.

This addresses the already approved ambient review's C1: preserve outer
torch.cuda.device(stream.device) around inner torch.cuda.stream(stream).
The positive invokes the actual imported optical_worker.native_setup helper.
The negative deliberately calls the same one-copy body under stream scope alone.
It is a fault injection, not a proposed production implementation.

Both fresh children use the existing guardian and supervisor, -I -S -B,
ACK1000/op120000, target1, and mark UNKNOWN inside the single armed segment BEFORE
backend import. That UNKNOWN remains sticky throughout direct diagnostic calls;
no attempt is made to resume a normal controller leaf or complete the segment.
The actual graph list is guardian-held before all imports/allocations. Returned
target/stream/NumPy host/Torch host view/destination/current-stream wrappers are
appended before later calls. Original exceptions/available modules remain held.
No ordinary atexit/destructor cleanup is used; both hard-exit74 after quarantine.

Native observer is the same hash-pinned RTLD_NOLOAD cudart cudaGetDevice with
typed signature and native _cuda_hasPrimaryContext. Before Stream, both observe
raw0,raw_initializedfalse,8absentcontexts. No warming Torch getter is called.
Create actual indexedStream1, eight-float host/view, explicit device1 empty
tensor, then one actual H2D copy. All allocations are outside measurements.
The copied values are not read back or claimed validated by this diagnostic.
The target1 current-stream handle is observed only after target1 initialization.

| Mode | Parent/supervisor/worker | Before copy | After copy |
|---|---|---|---|
| production_nested |343132/343137/343138|logical0,raw1,onlycontext1,stream1current0|same|
| fault_stream_only |343735/343740/343742|logical0,raw1,onlycontext1,stream1current0|logical0,raw0,contexts0+1,stream1current0|

The actual supervisor artifacts show1ARM,0normalCOMPLETE,0terminalARM/sync,
UNKNOWN/full_quarantine/not_authorized release,worker74 observed EXITED/reaped,
no expiry/kill, no ordinary_atexit. Worker evidence does not enumerate these
direct diagnostic GPU leaves: UNKNOWN is the deliberate whole-process scope
authority, while before/prepared/after JSON are descriptive observations of
the actual one allocation/copy. Do not call the guard's zero leaf counter
proof that no GPU work ran. Both worker.json files happen to have the same
digest because their guarded state histories are identical, not because the
actual native effects are equal. The separate scope records distinguish them.

Scripts in sibling c4_assets and /data2/shared/ang.gao:
- c2d3_torch_copy_scope_child_v1.py SHA322267c77f5df266da8956cca3c9b749a2e86cd02239be613b0a1097849bac38.
- c2d3_torch_copy_scope_supervisor_v1.py SHA8fe5dcc15a0485d86a004bad0fdf8b6238ecaae7ec1ff3dbaf446ad075b79803.
- run_c2d3_torch_copy_scope_v1.py SHAdb1d8c8c6cd2d52b7b1032ce0bb5787f44f98b05766b2a093fd4626fd31d9574.
Each mode ran sequentially using robot_sim Python and its literal mode argument.
Logs: c2d3-torch-copy-scope-nested-v1.log and c2d3-torch-copy-scope-stream-only-v1.log.

Artifact directories (both hosts):
c2d3-a3d750b-torch-copy-scope-v1-production_nested;
c2d3-a3d750b-torch-copy-scope-v1-fault_stream_only.
Identity files and invoker logs include /proc start/parent facts. Positive after
SHA33b8373c8bf3b9716fd2bde3014ec44df4b8401f214ab8367c91e81b9f7c2316;
custodySHAbf9d6692867094b3b4624d9f62416e969dc1d6b3214ab4695e103dae016d7148.
Negative afterSHA3967cf271b0c0dc222d3c764875cd1809aa121a99d1c5c774699c87ff0b5f4c3;
custodySHA0285af4a2c06fe0d89bd3f57a32870cb26f447e07bba775eb20ed27bfa64fe49.
Both workerSHAc4d5a19c010f96c105e0d94f9e5636faa154b61953aa41bbbd30b938994ed140.

Final scoped ps343131/343132/343137/343138/343734/343735/343740/343742 returned
only a header. NVML listed only pre-existing MPS2238410,60MiB,untouched. GPU0 UUID
GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274;GPU1 UUID
GPU-3a3112ff-837c-3b87-38d1-cb203d656589. No CUDA_VISIBLE_DEVICES remapping.
No process signal/reset/package change or interaction with others' work occurred.

Snapshots alone do not exclude transient native activity. The pinned source
counterexample and this actual observation jointly substantiate C1's specific
regression; broader allocation/copy/event/readback source/fault proofs and the
captured positive remain outstanding. No whole C2d3 or C4 approval follows.
