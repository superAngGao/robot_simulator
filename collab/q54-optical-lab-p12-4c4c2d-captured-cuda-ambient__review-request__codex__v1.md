# C2d3 captured CUDA full-chain ambient observation review v1

Fresh bounded code/gap review for the already-approved CUDA ambient designv1,
not a policy revision or whole-stage approval. Read repository AGENTS.md,
collab/README.md,collab/REVIEW_PRINCIPLES.md fully first.
Repository C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design, clean frozen
HEAD68fdb5084363eb70416580fd31816a0a69871ac6, base00c1e407a9ab6afd2ef5d33ca385b51df8634f6e.
Production and tests/support remain unchanged froma3d750bbaf79d760a6727e56146d277e1cb3b770,
also clean on tileops-dev /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.

Purpose: observe actual pristine raw0/uninitialized/no-context -> first indexed
Stream/binding -> every allocation/copy/event/ready/readback helper normal return
-> sole target terminal sync, using the real shared production path. Distinguish
logical0 restoration from raw1 and require no foreign primary context. This
closes a specific remaining normal-chain observation gap; snapshot evidence
alone cannot exclude transient native work and does not replace source proof.

Frozen candidates outside repo in C:/Users/AG/新建文件夹/c4_assets, staged same
bytes at /data2/shared/ang.gao; not executed:
- c2d3_captured_cuda_ambient_child_v1.py
  SHA b160e335eaf86809ee57c000b7d6f2c899aa976051a60d26eebdf060a4ae1ee2
- c2d3_captured_cuda_ambient_supervisor_v1.py
  SHA 7db32e76570cc1768b551015ad42f8652f7c4e72010258910193157614775053
- run_c2d3_captured_cuda_ambient_author_v1.py
  SHA d7d0c3108bf9a220d77f75fe572368829c982bb151522f3ea8efe606be8957ec
- launch_c2d3_captured_cuda_ambient_fixed_v1.py
  SHA fc04c350bfaa78c023111ea26ed0102d4cc17c106eac30da6eba6a25e0894b6d

Invocation: pinned robot_sim Python3.12.13 -I-B wrapper above asym selector only.
Fresh child-I-S-B. Exacta3d750b, CUDAasymmetric_synthetic_7x3_v1,target1,
ACK1000ms/op120000ms, session9*32, profile draftv9 canonical
b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47.
Exclusive artifactc2d3-a3d750b-captured-cuda-ambient-author-v1. Invoker verifies
both diagnostic child/supervisor hashes as well as profile/revision/clean tree.
Outer launcher verifies invoker bytes and preserves12-variable absent/native
environment enforcement. No remapping/warmup/raw device setter/deadline renewal.

Child derives from reviewed c2d3_captured_torch_stream_child_v3.py
SHAaae8ae0f30832ef0ae05a887546fbd390598dfd55e5a1cfe7f1fdf9e83111ad3.
First-stream pre-observation and actual post-adoption checkpoint retained; remove
only intentional pre-upload UNKNOWN/raise so the real original sequence continues.
After actual binding adoption, in the original armed runtime segment, install
fixed wrappers around the five existing native_setup Torch helpers. Every wrapper
calls its held original exactly once, retaining any actual return value in the
already-owned session graph before post-observation; original production adoption,
one-use binding, upload/readback loops, content validation and terminal policy remain.
Readback imports the same wrapped shared helpers. No backend function is replaced,
no second upload implementation or fake tensor/stream/event is used.

The new observer is only after initialized Stream/binding, with live original
setup ticket or live emergency ticket for the two terminal samples. It records
logical/raw device,8primary-context booleans, target1 current raw stream, actual
guard sequence/role/event-index and monotonic observation time. It compares scope
with the post-binding baseline on each before/after sample. Any normal observation
exception/contradiction marks UNKNOWN then raises into original quarantine; during
terminal it raises into existing failed-terminal quarantine, without new ARM/retry.
Already-collected snapshots stay in the owned graph. Upload/validation CPU-only
copies to files occur after each segment; all CUDA observers execute inside tickets.
The final terminal snapshot write remains inside the emergency callback/deadline.
Expected positive267snapshots:binding1,132actual helper calls*2,terminal2. Counts
are expectations only. Native leaf count remains136; wrappers add no render/sample.

Added native observer: torch._C._cuda_getCurrentRawStream(1), first called only
after actual Stream already initialized global/thread stream state and target pool.
Audit exact pinned5811a8d torch/csrc/cuda/Module.cpp:186-197 ->
c10/cuda/CUDAStream.cpp:getCurrentCUDAStream(explicit1) -> CUDAStream::stream.
Explicit1 avoids -1/current_device/SetTargetDevice branch; global/thread state was
initialized by actual priorStream. Default stream returns nullptr without creating
an ambient stream; CUDA(nonROCm) pooled branch reads existing native stream.
Sources in c4_assets/torch-scope-upstream-5811a8d, manifestSHA
32a9b6deb348201e9893f8b7c0cb90d450dd1f9017cdf4a356de6e5e45ca9f91.
Review this source rationale independently, including error/initialization branches.
Other observer APIs/raw cudart RTLD_NOLOAD/pinned identity and native upload chain
have prior source/evidence listed in deferred-scopev2 and upload-scopev1 reports.
All retained source/upload/host-allocator manifests remain in c4_assets.

Requirements/reports under collab q54-optical-lab-p12-4c4c2d-:
runtime-adoption designv2; guarded-custody designv2; cuda-ambient design/reviewv1;
torch-deferred-scope gap-reviewv2; torch-upload-scope gap-reviewv1;
torch-upload-workloads gap-reviewv1; cuda-normal-workloads test-evidencev1.
Actual normal CUDAasym/replay/bothGo2 have now passed original driver, realCPU
reference comparison and existing component replay; all worker0/reaped. Their
normal runs are uninstrumented and cannot substitute this full ambient observation.
Actual original firstStream and ordinary C1 positive/stream-only fault artifacts
remain in c4_assets and remote shared. Prior asym worker start ticks absent;
new replay/Go2 runs have actual live OS identity; do not repair history by inference.

Independently inspect exact code/diff, retained native owners, ticket placement,
original helper use/closure identities and failure behavior. CPU exact-AST/mock
controlflow tests may exercise wrong-scope/observer exceptions/partial return and
no extra guard/sync. Do not mutate candidate, run GPU, install dependencies, change
remote code, broad-kill or touch others. All CPU-owned processes must exit/reap.
No new nativeGPUcall may be accepted solely because its name contains 'get'.

Approve only this positive actual attempt if substantiated. Full captured C1 fault,
foreign/UNKNOWN/dual-GPU and both-lane import/allocation/copy/event/adoption/audit/
hang/ACK/EOF/terminal matrix, final versioned runtime-custody aggregate and whole
independent C2d3 review remain. No later gates or performance claim open.
Write complete unchanged report to c4_assets/captured_cuda_ambient_gap_review_v1.md
with independent identity, exact digests, findings, commands/proof limits/verdict.
