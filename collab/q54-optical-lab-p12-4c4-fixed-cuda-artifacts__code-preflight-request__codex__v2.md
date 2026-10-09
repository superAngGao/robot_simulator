# Fixed artifacts: real manager evidence and driver-state correction

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: code repair preflight and actual manager evidence review
Author: Codex
Version: v2
Date: 2026-10-09
Status: in_review
Related Files: fixed-cuda-artifacts design-request/design-review v2;
  code-preflight-request/review v1
Owner Summary: The six real manager cases passed. The first build stopped before
  compilation because a mapped libcuda was mistaken for initialized CUDA. Check
  the small correction and unchanged bounded continuation; no new process system.

## Frozen candidate and scope

Whole replacement base remains dcfa7a7120690a4c79bfbc5baa36a116c641ab28.
Prior reviewed code: a43f0c1e292e586b2740bfb2142de142e5284639, packet6fc47d9.
Unchanged preflight report archived at5391d4c, SHA256
f88e3b161d9a39cec55136165273745b6a7ed3b471546aad2ebf4d434571f3b4.
Repair/code execution HEAD: **66240e4fedc24ad8a5109690e3b1e78b450e2f62**.
Only tools/optical_cuda_build_recipe.py and its build-status unit tests changed
since the reviewed code; intervening commits are evidence/documents. This packet
is another documentation-only descendant. Exact packet HEAD is in the handoff.
Same repository, host, isolated checkout, interpreter, profiles and accepted
design as v1. No worker source/catalog/environment ABI change is proposed.

## Actual execution at a43f0c1

The v1 approved commands were executed with the exact clean env prefix. Git
transfer installed clean detached a43f0c1 in the named isolated checkout.

1. Manager fixture at `/data2/shared/ang.gao/c4-fixed-artifact-manager-v1`:
   terminal exit0 (tool session35077), **all six cases accepted**. Each recorded
   cgroup was observed absent. Normal and released-descendant mains exited0;
   ignored-TERM timeout returned waiter1; launcher loss returned waiter-9 and
   manager timeout; main SIGTERM produced manager success/killed/TERM and waiter0
   but correctly failed the normal-exit gate; nonzero main exited7/waiter7.
   The sentinel stayed alive across every case, then was terminated/waited (-15).
   Independent scoped `ps -o pid,stat,args -p` observed all16 recorded main/child/
   waiter/sentinel PIDs absent: 2045207,2045212,2045401,2045651,2045209,2045648,
   2045218,2045213,2045402,2045221,2045211,2045400,2045650,2045208,2045647,2045217.
2. Deployment command targeted `/data2/shared/ang.gao/c4-fixed-cuda-artifacts-v1`.
   It stopped on the first scene build, tool session63546 exit1. The manager unit
   was `robot-sim-cuda-a59890bf38eb469696857fc48b0b9097-0.service`; raw job:
   `/data2/shared/ang.gao/.cuda-stage-a59890bf38eb469696857fc48b0b9097/jobs/0`.
   Footer invocation24aaf6c8592d4a46bd691a19ab49b7e3, result exit-code,
   exited/status1; service runtime1.479s. Builder PID2047742 and exact owned cgroup
   were subsequently observed absent. No artifact deployment exists; the output
   reservation/staging remain. No second build, worker or GPU campaign was run.

Failure stdout records before_dependencies: no driver maps/FDs, initialized=null;
after_dependencies: mapped `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`,
no GPU FDs, torch_initialized=false. stderr locates the checkpoint rejection
before CUDAExtension construction or compilation. This proves neither successful
driver initialization nor its historical absence. The library mapping itself is
insufficient to decide that requirement. Read-only ELF NEEDED inspection of
libtorch_cuda/libc10_cuda/libtorch_nvshmem/nvshmem_host/cufile did not establish
the precise loading call; no such attribution is claimed.

Raw manager directory and failed job copied unchanged outside the candidate:
`C:/Users/AG/新建文件夹/c4_assets/c4-fixed-artifact-manager-v1/` and
`C:/Users/AG/新建文件夹/c4_assets/fixed-artifacts-build-v1-job0/`.
Combined archive `C:/Users/AG/新建文件夹/c4_assets/fixed-artifact-execution-v1.zip`,
SHA256 **100270980dece1bb0dab48ca8c51f51879eef4999e21fb806408fc5a2c115afa**.
Remote originals remain available. All owned tool sessions are terminal.

## Correction and evidence limits

At the existing checkpoints, if exactly one libcuda is already mapped, the builder
uses RTLD_NOLOAD|RTLD_LOCAL and the fixed `cuCtxGetCurrent(CUcontext*)` getter.
It does not call cuInit, load an absent driver, create a context, enumerate devices
or invoke NVML. It admits only CUDA_ERROR_NOT_INITIALIZED=3 and a null pointer;
CUDA_SUCCESS with a null context is rejected because the driver is initialized.
Other statuses, multiple driver mappings, NVML mapping, GPU FDs and Torch CUDA
initialization still fail closed. Getter/load exceptions terminate the build under
the same manager deadline. Raw status/context join the existing stdout checkpoint.
The already pinned dependency profile includes the installed driver file.

This uses the same RTLD_NOLOAD/fixed-getter pattern already present in the native
scope observer, without importing test code or adding a generic probe framework.
The helper is deployment-only, outside the captured worker source closure.
NVIDIA's [CUDA12.8 initialization contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__INITIALIZE.html)
and [cuCtxGetCurrent contract](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html)
support querying uninitialized status without first calling cuInit. Error3 also
allows failed initialization, so it is a state observation, not proof that no
earlier call was attempted. The selected-source/environment proof and checkpoint
limits from v1 remain necessary; no syscall-history or compiler-child claim.

Fresh CPU test command at the repair (Windows, no actual native getter):
`C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`
with tests/unit/optics/test_cuda_artifact_build_status.py,
test_cuda_artifact_manifest.py, test_cuda_extension_recipes.py:
**62 passed in6.09s, exit0**. Checks include mapping-only/code3 acceptance,
initialized-driver/code0/null rejection, unexpected status rejection, no getter
when absent, and unchanged GPU-FD/NVML/Torch/injection rejection. Mock native facts
are not the real-driver observation. git diff --check passed.

## Proposed continuation after independent approval

Review the repair, the actual six manager cases and their original command, plus
affected whole-packet invariants. CPU/read-only SSH allowed; do not run the new
getter, services, builds or GPU campaign before preflight. Report outside the
candidate to c4_assets/fixed_artifacts_preflight_v2_report.md, for verbatim archive.

The manager implementation/argv/footer/fixture is unchanged; the completed six
cases need not repeat for this builder-only repair. Reuse its actual evidence.
After clean Git transfer of **66240e4fedc24ad8a5109690e3b1e78b450e2f62**, execute
v1 steps2,3,4 in order with these exact substitutions and no others:

- revision=66240e4fedc24ad8a5109690e3b1e78b450e2f62;
- deployment output `/data2/shared/ang.gao/c4-fixed-cuda-artifacts-v2`;
- environment output `/data2/shared/ang.gao/c4-fixed-cuda-environment-v2.json`;
- runtime output `/data2/shared/ang.gao/c4-fixed-artifact-runtime-v2`.

The same fixed env -i prefix, build/profile digests, independent expected-platform
file, 120s/10s build limits and ten fixed runtime cases apply. Manifest/environment
digests come only from successful new outputs. Preserve failedv1 staging and
reservation; do not overwrite/resume them. Stop at the next unexpected result.
No dependency/budget change, installation, new fault selector or product activation.
Real artifact/GPU/numerical acceptance, later parent setup, whole-C4 refactor and
all product/performance gates remain unproved, as enumerated in v1.
