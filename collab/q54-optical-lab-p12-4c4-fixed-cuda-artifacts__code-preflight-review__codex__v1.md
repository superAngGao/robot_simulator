# Independent code preflight: fixed CUDA artifacts v1

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: whole replacement code preflight before real execution
Author: Codex independent reviewer /root/fixed_artifacts_preflight_v1
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted - bounded execution preflight only
Related Files: fixed-cuda-artifacts design-request/design-review v2; code-preflight-request v1; runtime-integration design-request/review v3; remaining-setup gap-scan v1
Owner Summary: Approve the frozen replacement and the packet's exact sequential manager/build/environment/native commands for execution. No substantive blocker was found. This is not acceptance of unexecuted services, artifacts, GPU behavior, numerical rendering, C2d3, C4, or product activation.

## Verdict and independence

**Verdict: Approve.** No substantive blocking or non-blocking code finding is raised for this bounded preflight. Required real observations remain obligations of the subsequent execution and independent evidence review, not facts established by this verdict.

I received the requirements packet in fresh session `/root/fixed_artifacts_preflight_v1`, without implementation conversation/history. I did not author or change the candidate, launch subagents, or execute a real service, compiler build, GPU import/getter, or native fault campaign. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md in full, the complete preflight request and accepted fixed-artifact design, and relevant parent integration/gap-scan requirements. Author test summaries and previous approvals were not substituted for current verification.

## Actual identities and tree state

Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.

- Branch: `implementation/c4-supervised-waits`.
- Whole replacement base: `dcfa7a7120690a4c79bfbc5baa36a116c641ab28`.
- Reviewed packet HEAD: `6fc47d9a0ddad568ca77cea60d4eda418475ae02`.
- Code/execution revision: `a43f0c1e292e586b2740bfb2142de142e5284639`.
- Both base and code revision are verified ancestors of HEAD. Code-to-packet delta contains only PROGRESS.md and the review request: two files, 232 insertions.
- Initial, intermediate and final porcelain status: empty. HEAD remained unchanged. Base-to-HEAD `git diff --check` passed.
- Whole diff inspected: 72 files, 4030 insertions, 5383 deletions, including deletion/movement rather than only the last factory delta.
- Accepted design working-byte SHA256 independently matched `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.
- Current catalog SHA256 independently matched `3a21fcc175518b0e8bdbdfe3e484278f4d97056c03d28ae91e7525fbdc028b79`.
- The three canonical profile digests independently matched the request: build `a9f42cc833088f308545749e4f239c246f787d9ec39096825815ae9557bb4829`, CUDA base `79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919`, Warp base `88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8`.

Read-only SSH independently observed `tileops-dev:/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` at `92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`, with empty porcelain output. This is the old isolated checkout; I did not transfer code, change its HEAD, or imply that it already contains the approved code revision. No other remote checkout or fallback host was used.

## Findings and inspected high-risk behavior

No substantiated correctness/requirement blocker or maintainability finding remains in this preflight. The following conclusions are based on source tracing and the executed CPU negatives, with their limits stated below.

**Deployment and publication.** The builder consumes the sole fixed recipe/source modules and uses standard CUDAExtension/BuildExtension. Kernel-body preservation and explicit direct-light wrapped bindings are checked against retained original hashes. The launcher reserves a fresh output, creates private staging, records fixed service argv, waits for each job, validates manager-provided normal main exit, separately observes the owned cgroup, and verifies output/source/toolchain bytes before renaming the complete set. A clean signal, nonzero exit, timeout, missing footer, incomplete output, or uncertain closure cannot publish a manifest. The recorder itself neither runs arbitrary commands nor publishes. `observe_group_closure` distinguishes absent owned groups from missing interfaces in a still-existing group and rejects populated groups; it checks the exact fresh unit below the selected user manager.

The immutable-set assumption is the accepted trusted deployment boundary. This review does not invent hostile publisher/root attestation. The source/table validators alone are descriptive; selection is the trusted launcher's explicit manifest digest plus verification of actual files and expected platform. No worker compiler installation or mutable JIT cache becomes runtime authority.

**Cold selection and compatibility.** Manifest verification joins fixed recipe order, source bytes, emitted inputs, footer contents, actual extension bytes, ABI, argv/job relations and invocation identity. Environment composition binds exactly the three modules plus manifest into ordinary v5 and checks the base selection and running interpreter ABI. Compatibility uses the same artifact and entry rules, explicit single-use deployment selection, fixed-name imports and identity rechecks. Missing selection cannot trigger JIT; an incomplete compatibility load is not retried as a fresh success. The runtime admission path remains separately tied to live session identity.

**Ordinary loading, ownership and failure.** In native_bootstrap/native_setup, each CUDA create/exec goes through the existing ordinary external loader after runtime binding and in original guard3. Create returns are placed in the already retained admission registry inside target/stream contexts before journal/adoption/context-exit failures. File/spec/delegate/module identity checks remain before and after execution. Named entry adoptions and the set adoption precede issuance; partially populated sets remain held and unregistered. Session/controller/binding/admission/module/entry/context identities are checked again at preparation. Failure preserves the original exception/available roots and follows poison/UNKNOWN/quarantine rather than acquiring a replacement guard or asserting readiness.

**Factories and source closure.** After the six input phases, the existing controller guards CUDA context/scene/LBVH/renderer preparation. It uses the registered uploaded input and issued extension set, retains each return before the next fallible work, and queries LBVH workspace with the same resolved module used by its factory. Direct-light now rejects bare-module construction. The factory preparation adds no output slot, rendering launch, warmup or final readiness. Factory source admission is restricted to the later CUDA preparation phases; Warp compatibility imports are lazy at their actual compatibility boundaries. The source-policy/catalog changes remove the old compiler/setuptools exceptions instead of broadening the captured loader.

**Replay and removal.** V3 joins six recipe/create-exec scope pairs to actual file frames, leaf descriptors, original guard3 and retained create returns, then joins exact module/spec/delegate and individually named entry/set adoptions. CPU parser attacks retain independent lifecycle/file facts while swapping entries, substituting owners, removing pairs, changing guard/frame/leaf, moving returns outside context pairs, or delaying adoption. They reject for the intended relations. V1/v2 interpretation is retained; the moved archived readers differ by documentation/import paths and explicit historical v2 selection, rather than acquiring artifact-v3 requirements. The historical native/archive tests skipped on Windows were not claimed as executed archive replay.

The live worker cuda_compilation, setuptools_admission, compiler_supervision, direct loader/root/hook wiring and compiler-specific transport/driver branches are removed. Searches found cpp_extension imports only in deployment tooling and inert historical/profile declarations, not a live worker JIT route. Shared recipe, artifact validation, loading resolution and factory logic are reused. The replacement is a net deletion; the remaining large parent bootstrap/lifecycle infrastructure is not falsely represented as completion of the separately required whole-C4 structural refactor.

## Installed-source build preflight

Read-only `cat` fetched the actual installed files to outside-candidate scratch. Independently computed hashes:

- `torch/utils/cpp_extension.py`: `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`.
- `torch/cuda/__init__.py`: `9ba786aee9b5aac710f49c3beade1722610a0d6d8ec406fad9a9793df1316e99`.

I inspected module initialization, `_find_cuda_home`, the selected availability/device-count/visible-device path, deferred CUDA capability checks, CUDAExtension include/link selection, BuildExtension's Unix/Ninja flow and compiler-version checks, explicit architecture generation, and the separate load_inline/PCH path. With the exact clean builder environment, empty CUDA_VISIBLE_DEVICES parses to an empty list, NVML device count returns zero before discovery, and device_count does not call its negative-result CUDA fallback. Explicit `9.0` bypasses device/capability architecture discovery. Deferred capability/cubin checks remain queued without CUDA initialization. The standard CUDAExtension/BuildExtension route does not call load_inline or its PCH create/remove branch and does not load its output.

The code validates the builder-only controls before repository/third-party imports. Its three OS checkpoints inspect driver-library mappings, GPU device FDs and Torch initialization state, and fail if those reveal driver use. Under the accepted trusted installed-dependency boundary, the inspected selected source path plus these fail-closed checkpoints is sufficient to authorize the first bounded build attempt. It is **not** a syscall history, a compiler-child trace, or proof already obtained that an actual build never initialized a driver. Later evidence review must inspect the actual checkpoints, raw logs, emitted inputs and selected provenance. The platform CXX11=true expectation is an independent expectation; this preflight does not claim to have observed the native ABI value or permit rewriting it to accommodate failure.

## Executed CPU verification

Windows interpreter: `C:/Python314/python.exe -B`; repository cwd above. All pytest invocations used `-q -p no:cacheprovider`; no build/native acceptance script was executed. Paths below are relative to `tests/unit/optics/`.

1. Command: `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider` with:
   `test_cuda_artifact_build_status.py test_cuda_extension_loading.py test_guarded_runtime_binding.py test_cuda_artifact_manifest.py test_cuda_artifact_runtime.py test_cuda_artifact_replay.py test_native_scope_observer.py test_cuda_extension_recipes.py test_cuda_toolchain_inventory.py test_runtime_native_scope_codec.py test_setuptools_replay.py test_cuda_compilation_codec.py test_compiler_job.py`.
   Result: **213 passed, 52 skipped in 20.01s**, terminal exit0 (tool session 36143).

2. Same command prefix with:
   `test_native_source_policy.py test_native_sources.py test_native_bootstrap.py test_runtime_import_cold.py test_runtime_environment.py`.
   Result: **366 passed, 13 skipped in 362.22s**, terminal exit0 (tool session 6190).

3. Same command prefix with:
   `test_runtime_custody.py test_guarded_lifecycle.py test_guarded_evidence_fixtures.py`.
   Result: **214 passed in 21.24s**, terminal exit0 (tool session 50341).

Total: **793 passed, 65 skipped**. These are fresh executions on the frozen packet. Mocks/modelled file/native facts establish CPU ordering, identity and parser behavior; they do not establish native create/exec, cgroup semantics, target restoration or numerical correctness. POSIX/archive/native-dependent skips are explicit coverage limitations, not passing native evidence.

Other read-only checks used Git status/revision/branch/ancestry/diff/diff-check, `rg`, UTF-8 source reads, and stdlib canonical JSON/hash calculations. No new repro was needed for an asserted defect because no substantive finding is raised. The existing focused negatives were independently run and their intended checks inspected.

## Command scope authorized by this approval

Approval binds the entire replacement at code revision `a43f0c1e292e586b2740bfb2142de142e5284639` and only the exact sequence in `collab/q54-optical-lab-p12-4c4-fixed-cuda-artifacts__code-preflight-request__codex__v1.md` at packet HEAD above:

1. Transfer the reviewed commits with Git into the stated isolated remote checkout, preserve history, and verify clean exact code HEAD.
2. Run the six-case installed-manager fixture at `/data2/shared/ang.gao/c4-fixed-artifact-manager-v1` using its 6s runtime/start and 1s stop bounds; inspect actual closure and sentinel evidence before proceeding.
3. Run the three fixed standard builds through optical_cuda_deploy at `/data2/shared/ang.gao/c4-fixed-cuda-artifacts-v1`, each with the reviewed 120s runtime/start and 10s stop service properties and pinned dependency profile. Inspect terminal results and checkpoints; derive the manifest digest from the actual published file, not the mixed progress stdout.
4. Compose `/data2/shared/ang.gao/c4-fixed-cuda-environment-v1.json` using the exact pinned base, manifest and expected-platform inputs; derive its actual digest.
5. Run cuda_artifact_runtime_acceptance at `/data2/shared/ang.gao/c4-fixed-artifact-runtime-v1`: the ten fixed sequential Go2/partial-return/adoption/factory cases and private-copy artifact negatives through the existing supervisor, with the unchanged 120000ms operation/1000ms ACK contract and the packet's outer fixture bounds.

Every Python invocation retains the packet's exact `/usr/bin/env -i` prefix and fixed interpreter `-I -S -B`. Builder-only empty visibility must not be applied to runtime workers. All named outputs must be absent; existing output or an unexpected result stops the sequence. This approval grants no overwrite/retry, dependency installation, budget increase, alternative host/checkout, service command, fault selector, or broad process signal. Preserve failed evidence and inspect owned processes/jobs to their actual terminal state.

## Limits, process hygiene and remaining gates

All three local pytest sessions reached terminal exit0. The longer pytest PID 30864 and the later regression PID 11648 were observed while running; final scoped OS inspection found PIDs 30864, 11648 and the owned SSH PID 45872 absent. Test child processes used their existing subprocess timeouts. The first run's individual PIDs were not sampled; its tool session nevertheless reported terminal completion. Thus this is scoped process hygiene, not exhaustive OS attestation.

One combined read-only SSH command printed remote revision/status and hashes, then stalled during its quoted sed portion. I identified and stopped only my owned local SSH PID 45872 and observed its tool session 98297 end with exit1. It launched no service/build/native job. Subsequent single-command `cat` SSH reads completed normally with bounded local subprocess timeouts. A first attempt to save one fetched file hit a local PowerShell-to-Python Unicode-path encoding error; the repeat used Path.cwd().parent and succeeded. These inspection/tooling issues did not alter the candidate or remote code. Only outside-candidate source copies and this report were written; pytest scratch used its normal temporary location. No service/GPU/worker/compiler job was launched or signalled by this review.

Real manager behavior, actual compilation within 120s, publication/native ABI, real target1/ambient0 create/exec/factory restoration, actual partial-owner failure/hang custody, both-lane Go2 upload/readback and public replay still require execution and independent evidence review. Numerical rendering/kernel equivalence is not proved by source-body preservation, workspace queries, or these input-only acceptance commands; it remains a later real-device obligation.

This preflight does not approve fixed-artifact acceptance, C2d3/C4 completion, Warp NVRTC/cache/ModuleExec custody, parent timeline/arena, one-slot provisional/final graph, ten warmups, final readiness/seal/barrier, C4c3/c4, C4d/B3, or the owner-required whole-C4 structural refactor/review. Product PREPARE/submission, benchmarks/results/baselines and backend selection remain closed.

The unchanged complete report may be archived verbatim under the repository's evidence-only recording exception. New operative code or command changes require review of their delta.
