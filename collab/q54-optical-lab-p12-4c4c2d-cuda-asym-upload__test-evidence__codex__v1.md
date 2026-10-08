Initiative: q54-optical-lab-p12-4c4c2d-cuda-asym-upload
Stage: actual author test evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: bounded evidence; C2d3 not accepted
Related Files: torch-upload-scope gap-review v1; cuda-go2-profile gap-review v1;
runtime-adoption design v2; CUDA ambient design v1; PROGRESS.md
Owner Summary: The unchanged captured CUDA path completed actual42-array a2
upload and D2H input validation, one supervised target sync, guardian table
release and worker exit0/reap. Fresh CPU reference and existing evidence
components agree. This uninstrumented run is not full ambient or whole C2d3
acceptance and lacks a worker OS start-tick observation.

## Exact execution and approval

Primary183 tileops-dev; remote isolated repository
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip at clean
a3d750bbaf79d760a6727e56146d277e1cb3b770. Local evidence/profile-only checkpoint
fd89a7ea7fec6032e9756db8d0602aa304393a0b; production code is identical.
Pinned robot_sim CPython3.12.13,Torch2.9.1+cu128,8H200,driver595.71.05.
Target1 UUID GPU-3a3112ff-837c-3b87-38d1-cb203d656589; ambient device0 UUID
GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274. No visibility remapping or warmup.

Fresh /root/torch_upload_scope_gap_v1 approved only this asym launch with
non-blocking C1/C2. Complete report SHA
d7f4eed814649296ea5776c39795cc179343dac5ecde6b54d957084ecd9362a6
is archived unchanged. C1: use the hash-pinned wrapper's exact environment
snapshot and reject12 listed native allocator/loader settings. C2: RELEASED
clears the guardian owner table; other Python/native caches can remain until
hard process exit. Neither comment is silently broadened to other workloads.

Exact command:
`ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B /data2/shared/ang.gao/launch_c2d3_cuda_upload_fixed_v1.py asym`

Wrapper SHA f5bbb7dd6fce0b788537dd623aa4666076409d19031d34a91d5373f3a958bbd6;
inner invoker SHA7ff0466fb595d87ee326c2dceeb2859605925369af895365ed4ed6c1f929b8d4.
Both outside repository in local c4_assets and remote /data2/shared/ang.gao.
Uses unchanged tests/support/native_runtime_process.py and
guarded_custody_process.py, actual -I -S -B worker. Profile external draftv9,
canonicalb4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47,
raw44f0833cdf507127695052b97b56864eb7b39783122d4b511240bcb6d6ef87d7.
Caseasymmetric_synthetic_7x3_v1; sessiona*32;ACK1000ms/op120000ms. No renewal.

Launch stdout observed all12 variables null: PYTORCH_CUDA_ALLOC_CONF,
PYTORCH_ALLOC_CONF,PYTORCH_NO_CUDA_MEMORY_CACHING,CUDA_MODULE_LOADING,
TORCH_USE_RTLD_GLOBAL,CUDA_VISIBLE_DEVICES,PYTORCH_NVML_BASED_CUDA_CHECK,
LD_PRELOAD,LD_LIBRARY_PATH,LD_AUDIT,CUDA_INJECTION64_PATH,CUDA_INJECTION32_PATH.
Wrapper execve preserves that exact environment. This emitted observation is
in the tool transcript; it was not written as a worker-issued runtime receipt.

## Actual result and independent CPU reference

Artifact basename local c4_assets and remote /data2/shared/ang.gao:
`c2d3-a3d750b-cuda-upload-author-v1`.

6normal ARM/COMPLETE pairs then emergencyARM6;7ARM total, no emergencyCOMPLETE.
Worker136 entered leaves:43allocation (42arrays plus indexed Stream),42H2D,
42D2H,2event-create,2event-record,2event-wait,3runtime-import/init.
UNKNOWNfalse, possible dirty upper bound[1]. One terminal target sync succeeded;
guardian table RELEASED, worker0 observed/reaped, no deadline expiry or kill,
no ordinary_atexit/failure record. No final optical readiness or sample submission.
Allocation/copy counters are entered leaf counts, not claims that zero-sized
arrays performed a native cudaMalloc/memcpy. Ready/validation completion precede
host content inspection; the sole final target sync is checkpoint disposal.

Fresh ordinary CPU reference script c2d3_expected_canonical_inputs_v1.py used
the sole canonical scene/fixed/replay builders; no Torch/Warp import. Its
output c2d3-a3d750b-fresh-cpu-reference-v1 is retained locally/remotely. Actual
CPU407517/parent407515 completed and scoped ps confirmed absence. This new run
generated42fixed arrays,27replay arrays and42arrays for each real Go2 size.
For this asym run all42names plus manifest/content/topology match:
- manifest f65056b303775f4d13ad1a48660467cbae06a9d9dbe09fc5596a341ba5af6728
- content 1771f0ce65a929075d542032c54a95e4c777de361e904a377c6f2a197303a32d
- topology 6a6136c2e6991470f5fd6823af9fbcf87faddf71a7d574cfa42ce198a4e81966

Fresh check_c2d3_cuda_upload_v1.py1 invoked production custody/native-call/
replacement/alias parsers and checked phase-prefix stability, actual leaf counts,
zero sample counts and the fresh CPU manifest. Passed. It is author verification,
not the required final aggregate runtime-custody schema.3841external events:
759resolved,758each create-started/created/exec-started/completed,42publications,
8absence probes. Triton is the resolution-only row. No new dependency is admitted
from this observed history. Raw runtime_initialization bytes happen to match the
earlier Stream diagnostic because this descriptive record is deterministic;
equal bytes do not prove same process/run/live objects. Actual guard/process
artifacts must carry run identity; final aggregate cross-binding remains open.

## Process identity and residual limits

SSH wrapper431188,parent/execve-launch431190,supervisor431371,worker431372.
Parent and supervisor raw /proc stat were emitted by the invoker; worker
identity.json records pid431372/ppid431371, cross-checked with supervisor output.
The unchanged worker driver did NOT capture worker OS start ticks. No such value
is reconstructed after exit. This is an explicit full-matrix evidence gap;
subsequent runs will capture the owned live /proc chain externally.
Supervisor actual stateEXITED/observed_returncode0 and completed wait establish
reap. Scoped ps for all four owned identities returned header only.

Subsequent NVML snapshot showed MPS2238410/60MiB on device0 plus unowned transient
python436331/582MiB on device1. A scoped ps436331 read returned no row by the time
it ran; its user/parent were not established. No signal, reset or modification
was issued. Do not call the snapshot empty or assert this process was a known
other user's workload. All owned worker/supervisor identities were absent.

## Raw SHA256

| File | SHA256 |
|---|---|
| request.json | 0ca1aa3a3eb94ad2daa0a3cc36e8b1eb5abbbbec77c5b79739a04d24544a1c5b |
| identity.json | d3c2a4b744c77e301ffd155183825d0b3ee075dc798c51184433fe487102ee55 |
| runtime_initialization.json | cc49030f879519ab6d69910d28d42dc6253b18cd5f6085d6212fe45549870c21 |
| native_upload.json | cc17b99ebbb5b1652bc86e8f3cf4a4825a8dfb8483013352b97f3996e3d2d39c |
| input_validation.json | 8f00917087184972f15f6aa4969155da33747987e39d8ed305c10f22d5fb8590 |
| input_validation_report.json | 334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760 |
| worker.json | 717b8ec97070bef3274c87f7afca0e732fa2210a602734c15dc08af756202315 |
| custody_artifact.json | 6f30c50fc0d5aa069cd08a174153d176532f9ade089d4236f36419d466ca4668 |
| supervisor_stdout.json | b371e095d0d2e19029fb70d213e551cb0bb8f7929ec329b4f16a6278a4f029c8 |
| author_component_verification.json | 65757bd925908a9851a543648988491441ef68e2b81d1cea591c9a1944be74ca |

Remaining: actual CUDA replay/bothGo2; full captured logical/raw/stream/context
observations and foreign/UNKNOWN cases; both-lane complete fault/hang/ACK/EOF/
terminal/dual-GPU matrix; versioned aggregate schema and fresh whole C2d3 review.
C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host,R2/shipping
remain closed. C4 remains active. No performance or selected-backend claim.
