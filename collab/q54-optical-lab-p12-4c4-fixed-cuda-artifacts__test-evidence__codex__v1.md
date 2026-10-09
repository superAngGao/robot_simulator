# Fixed artifacts: actual builds and ongoing Go2 acceptance

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: actual deployment evidence; native matrix still running
Author: Codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: fixed-cuda-artifacts code-preflight-request/review v1/v2
Owner Summary: All three real extensions built and the first real Go2 CUDA
  preparation case passed. The remaining native matrix is running; no whole
  artifact/C2d3/C4 acceptance or rendering/performance result is claimed.

Fresh independent reviewer /root/fixed_artifacts_preflight_v2 accepted the six
actual manager cases and approved the precise v2 continuation. Unchanged report
SHA256 ec035f93ac1c247213482bca468e8b6cb5e43e9db34e5947c428220a238cfb54,
archived at a341584eecd72dc3e7d2039d45deeb81511aa669. The named isolated primary
checkout was verified clean and moved with Git to exact reviewed execution code
66240e4fedc24ad8a5109690e3b1e78b450e2f62. No original checkout was modified.

## Three actual builds

Executed v1 step2 with v2 substitutions and the exact env -i prefix, interpreter,
profile/digest and service budgets. Deployment tool session34340 reached exit0.
Published root: `/data2/shared/ang.gao/c4-fixed-cuda-artifacts-v2`.
Its actual canonical cuda-artifacts.json SHA256 is
**9981192f6c3874777755425fbe28e257c3b85d582be8303a665f2d2209d3c71a**,
independently read with sha256sum after successful publication.

All units have prefix `robot-sim-cuda-00ee1bec4fbd4171948893f538808924-` and suffix
`<index>.service`. Each raw manager footer is success/exited/0; the publisher
verified normal waiter0, actual owned-group closure and all output/provenance
bytes before publication. Raw waiter logs:

| Index / extension | Runtime | Builder PID | Invocation |
| --- | --- | --- | --- |
| 0 / scene_v1 | 50.195s | 2074763 | 34581134aa3944d4b5ba155c49f170db |
| 1 / lbvh_native_v3 | 54.493s | 2077412 | 6b64a4ce47f449f1afca4fe66325adae |
| 2 / direct_light_v8 | 51.561s | 2079106 | a56994121a7f4f4189051b25083739cd |

All three PIDs were subsequently observed absent. Each builder's before-dependency
checkpoint has no driver mapping/FD; both later checkpoints have one libcuda
mapping, driver_status3, current_context=null, no GPU FDs and Torch initialized=false.
These are real current-state observations under the reviewed source/environment
argument, not syscall history or compiler-child tracing. The no-visible-runtime
warning and missing GCC-version-bounds warning in stderr are preserved; neither
build failed. All three actual CXX11 ABI values are true and match the selected
platform. Python extension suffix/SOABI also match. No extension was imported by
the builder. No build budget, dependency or expected ABI was changed.

Raw published bundle retains the complete manifest, generated sources/Ninja,
stdout/stderr, invocation/termination/build-details/waiter logs and artifacts.
A selective evidence archive additionally includes those exact records and the
three top-level extensions (not intermediate object files):
`/data2/shared/ang.gao/c4-fixed-artifacts-build-evidence-v2.tar.gz`, copied to
`C:/Users/AG/新建文件夹/c4_assets/c4-fixed-artifacts-build-evidence-v2.tar.gz`.
Local archive SHA256 **35caf5fe21dc14b358829a5e72e9e4e9b41985d607385a074f429c443ab49ba6**.

## Cold selection and live matrix

V1 step3 with the v2 root/manifest above completed exit0. Canonical environment:
`/data2/shared/ang.gao/c4-fixed-cuda-environment-v2.json`, SHA256
**a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce**.
sha256sum independently agrees; local copy is in c4_assets. The base profile and
independent expected platform were exactly those in the preflight packet.

V1 step4 with v2 substitutions is **still running**, tool session **94514**.
Do not restart it. Acceptance parent PID2083102; output directory
`/data2/shared/ang.gao/c4-fixed-artifact-runtime-v2`. Poll this exact session;
the original supervisor owns each worker's deadlines and terminal disposal.

The missing/changed/partial private-copy cases all rejected before a worker.
The changed copy reports `artifact file bytes changed`; missing/partial report
`artifact file missing or outside root`. Original deployment revalidation passed.

First `cuda_fresh` Go2 960x640 case accepted, worker2083344 exit0 and supervisor
2083343 exit0. The acceptance script checked worker PID absence, original terminal
disposal, all guarded phases, actual target1/ambient0 restoration, factory binding,
input validation and public runtime-custody replay. Raw cuda_factory_binding.json
records the three actual module names, target1 stream, before/after equality,
only target1 primary context active, and real LBVH sizing for376139 primitives:
node_capacity752277, alignment256, required workspace4590335 bytes. It explicitly
records no final readiness and sample_submission_count0.

The fresh worker that reuses this deployed set is currently running. Remaining
cases and final results.json have not passed merely because the first did. After
the matrix terminates, preserve raw evidence, inspect all owned terminal states
and obtain fresh independent artifact/evidence acceptance. Then continue parent
Warp custody, timeline/arena, warmup/readiness/seal, C4c3/c4, C4d/B3 and the required
whole-C4 refactor/review. Product PREPARE, benchmarks/results and selection remain
closed; C4 stays active.
