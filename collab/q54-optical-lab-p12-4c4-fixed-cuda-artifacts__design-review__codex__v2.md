# Independent design re-review: fixed CUDA artifacts v2

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: independent bounded design re-review
Author: codex, independent reviewer /root/fixed_artifacts_design_v2
Version: v2
Date: 2026-10-09 (Asia/Shanghai)
Status: accepted — design only
Related Files: fixed-cuda-artifacts design-request v2 and original design-review v1; runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-admission design-request/design-review v3; setuptools-admission design-request/design-review v2; systematic-refactor implementation-note v1; scope-convergence proposal v1
Owner Summary: Approve this exact replacement design. The three original findings are closed at design level: publication requires normal main-process exit, the builder environment closes the identified CUDA availability path, and independently named guardian adoptions bind entries and the extension set. Implementation, installed-manager acceptance, native execution and whole-C4 acceptance remain separate gates.

## Verdict and reviewed identities

**Verdict: Approve.** No substantive blocking or non-blocking finding remains in this bounded design re-review. This conclusion concerns the specified contracts and their source-level feasibility; it is not a claim that the future implementation passes its required tests.

I received the review packet and repository access in fresh independent session `/root/fixed_artifacts_design_v2`, without the implementation conversation. I did not author or edit the candidate, launch subagents, execute tests, or run native/build/service/process experiments. AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read in full. The Chinese principles were reread explicitly as UTF-8 after the initial default-decoding display. I read the entire frozen candidate and original fixed-artifacts report, and inspected governing predecessor sections and relevant production paths independently.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base: `0b8f4f527934e71cbb780c75dff9cc4ff251a33a`.
- Reviewed HEAD: `3e31778766d7ea16ebe6b936dd00b9de0a0af5f0`.
- Sole base-to-HEAD change: `collab/q54-optical-lab-p12-4c4-fixed-cuda-artifacts__design-request__codex__v2.md`, 346 added lines.
- Independently recomputed candidate Git LF SHA256: `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`.
- Independently recomputed working-byte SHA256: the same value.
- Original report working-byte SHA256: `8b06f872d3057431b56d3f578449945c805fd788c8a205abc30540ceed558bac`, matching the packet.
- Initial and final local porcelain status: empty; HEAD and candidate digest unchanged. Base-to-HEAD `git diff --check` passed.

The packet's primary target is `/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` at code `92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`, using `/home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12.13`, Torch 2.9.1+cu128 and systemd 255.4-1ubuntu8.11. These are supplied deployment identities, not newly observed remote facts from this review. I did not use SSH or claim refreshed remote cleanliness.

## Original findings: independent dispositions

### R1 — closed at design level

Candidate lines 100–132 and 177–186 now distinguish the manager's service result, the main process's termination classification, waiter completion, and subsequent cgroup closure. A fixed `ExecStopPost` recorder writes exact manager variables and the unit/cgroup identity into an exclusive private record. Publication requires `service_result=success`, `exit_code=exited`, `exit_status='0'`, waiter zero, expected outputs, and post-wait absent/empty owned cgroup. The raw footer and invocation association are retained before collection; log text is not termination evidence.

The original output-then-SIGTERM counterexample now fails the `exited` predicate even when the service result and waiter remain successful. The pinned systemd documentation explicitly distinguishes `success/killed/TERM` from `success/exited/0` and defines the footer variables as main-process facts. Missing facts and recorder failure cannot satisfy the proposed success predicate. [systemd v255 termination variables](https://raw.githubusercontent.com/systemd/systemd/v255/man/systemd.exec.xml).

I also checked an affected implementation risk: recording `/proc/self/cgroup` from a control command must identify the service group, rather than only a control subgroup. In v255, `exec_params_get_cgroup_path` selects `.control` only when delegation is enabled; the candidate explicitly prohibits delegation. The recorded path is then validated against the fresh unit and user manager. [systemd v255 cgroup path selection](https://raw.githubusercontent.com/systemd/systemd/v255/src/core/execute.c).

The service code passes result variables to stop-post, and its cgroup liveness check uses recursive emptiness. The candidate correctly delays its own closure observation until the recorder and waiter finish, so the recorder's presence is not mistaken for a leftover builder and footer completion is not mistaken for group closure. [systemd v255 service lifecycle](https://raw.githubusercontent.com/systemd/systemd/v255/src/core/service.c).

This is source-level closure of R1. Section 6 still requires the actual installed-manager normal, separate-PGID descendant, ignored-TERM timeout, launcher-loss, output-then-SIGTERM and nonzero-exit cases, with terminal facts and an unrelated sentinel. None was run here. A permission/I/O error observing closure still fails publication, and an unreaped or uncertain task must not be reported as closed.

### R2 — closed at design level

Candidate lines 70–89 now freeze both builder-only variables before dependency import: empty `CUDA_VISIBLE_DEVICES` and `PYTORCH_NVML_BASED_CUDA_CHECK=1`; missing/different values reject. Explicit `TORCH_CUDA_ARCH_LIST=9.0` independently prevents architecture discovery. The worker keeps its original visibility and target/ambient verification.

I independently read the preserved installed `cpp_extension.py`, whose SHA256 is `52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`. Its `_find_cuda_home` still calls availability during module initialization. In the pinned Torch CUDA source, the NVML-selected availability path calls `device_count`; empty visibility produces an empty parsed list, so `_device_count_nvml` returns zero before native discovery. Zero does not select the negative-result CUDA fallback. Thus the original import-time counterexample does not survive the revised environment. [PyTorch v2.9.1 CUDA implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/torch/cuda/__init__.py).

This establishes the mechanism for the identified path, not blanket proof about every installed build dependency. The design appropriately requires complete installed-source preflight and actual absence-of-driver-initialization evidence during later build acceptance, explicitly refusing to equate `torch.cuda.is_initialized() == False` with that stronger property. No environment monkeypatch, backend import or native query was performed in this review.

### R3 — closed at design level

Candidate lines 227–273 now provide independent named adoption anchors. Each entry requires role `cuda_artifact_entry`, HOST_OWNER scope and operation `<recipe>.entry.<entry_name>`; the set requires role `cuda_artifact_set` and operation `cuda_artifacts.bind_set`. Successful binding requires the original guard3, prior event order and distinct direct guardian sequences. The verified receiving boundary assigns these names using the common recipe table. Root retention alone remains failure custody, not successful issuance.

Hold the independent lifecycle/object table fixed and swap the scene tokens for `update_triangles_world_aabb_out` and `update_planes_world_out`: the swapped tokens now have the wrong required operation anchors. A cross-recipe exchange likewise changes the recipe-qualified anchor. An unrelated adopted owner cannot meet the set's specific role/operation. Late, wrong-guard and root-only cases are explicitly excluded. These conclusions follow from the new relations, not from uniqueness, a digest or an entry-name string alone.

The existing `runtime_custody.py:329` adoption helper already supports role, scope, operation and temporal joins. The guard and strict-before conditions are additional explicit checks for the new event. Existing `file_call` grammar joins resolved spec/delegate/module identities to actual started/returned/completed calls. V2 retains that source of truth rather than readopting the same objects.

The six native scope pairs are now recipe-specific: fixed site, operation, leaf kind/source/runtime identity, active file frame, original guard and unique leaf. Create-return retention must occur within the pair before context unwind; file completion follows pair closure. Thus swapping a recipe or create/exec pair cannot pass merely because six valid pairs exist elsewhere. The current scope parser has lifecycle-operation and paired-observation machinery that can support these required joins without a second registry.

The mandated production collector/replay negatives in lines 318–323 retain the independent lifecycle/file facts while making the original swaps. They must still be implemented and run. I did not execute a schema model and do not present static counterexample analysis as a future parser test.

## Affected architecture and regression assessment

**Ordinary loading and retention.** `native_bootstrap.py:589` currently retains a delegate-create return only after the delegate call. The extracted helper in `cuda_compilation.py:261` demonstrates the needed retention inside target/stream contexts. V2 explicitly moves the ordinary registry retention inside those contexts before exit can fail. The existing `_RuntimeSetupBinding` is constructed within the runtime-init guard, so fixed imports after its creation and before that guard's completion have a concrete insertion point. Failure remains terminal with held partial roots, no extra guard and no emergency synchronization to repair the failed segment.

**Factory authority.** The current LBVH query and prepare independently call the cached loader; v2 requires one resolved module to serve both. The existing CUB query, bounds and ABI remain intact. Direct-light's constructor at `optics/cuda_direct_light.py:295` really accepts an unguarded module today; lines 281–283 now explicitly require equivalent private preparation issuance. This closes the original report's non-blocking wording note. Captured authority cannot be replaced by the compatibility route, while shared artifact/name/entry checks avoid a second kernel/factory implementation.

**Build and artifact provenance.** The sole recipe table already supplies the three names and ordered entries. The preserved Torch source confirms the old header insertion and wrapped direct-light pybind generation, the separate load_inline PCH-removal branch, standard BuildExtension behavior, and architecture selection. Explicit headers/bindings and recorded actual emitted build inputs are necessary migration work; numerical equivalence and the 120-second budget remain acceptance questions. No compiler flag equivalence or successful build is assumed. Missing required tools must fail deployment rather than silently trigger an alternate route.

The environment schema's existing site roots, metadata files and extension/module rows can represent the fixed set. The trusted launcher validates source/recipe/platform and output bytes before constructing those rows; the worker validates its bound environment. A manifest is deployment provenance, not live runtime readiness. Fresh immutable publication, no failed-staging reuse, no concurrent writer and exact three-output checks provide a coherent contract under the accepted trusted deployment boundary. This review adds no hostile-root attestation requirement.

**Removal and later gates.** Current top-level cpp_extension/Warp imports and load_inline calls confirm that changing only one loader would be insufficient. V2 covers those dependencies, compiler-specific observer/driver branches, the direct ledger, live setuptools transaction and historical reader separation in one implementation/review packet. The parent's source closure, guardian, scope and exception guarantees remain requirements. Net deletion must be verified alongside replacement coverage; keeping obsolete tests green is not permission to retain live JIT.

The governing runtime-integration design and gap scan retain the same later work: real input/slot custody, Warp timeline/arena and concrete Warp compilation custody, ten observed warmups, distinct final readiness, final graph, one seal and guarded barrier, C2f correctness/failure acceptance and separate activation. The scope-convergence proposal and owner-directed systematic refactor remain consistent with this replacement. Moving compilation out of an already excluded timing phase supplies no performance result by itself.

## Verification, commands and limitations

Read-only local inspection used PowerShell `Get-Content -Encoding UTF8` and bounded indexed ranges, `rg`/`rg --files`, `Get-FileHash`, `Get-ChildItem`, and Git status/branch/revision/diff commands. Principal identity checks were:

```text
git status --short
git rev-parse HEAD
git branch --show-current
git diff --stat 0b8f4f527934e71cbb780c75dff9cc4ff251a33a HEAD
git diff --check 0b8f4f527934e71cbb780c75dff9cc4ff251a33a HEAD
Get-FileHash <candidate-v2> -Algorithm SHA256
Get-FileHash <original-review-v1> -Algorithm SHA256
Get-FileHash C:/Users/AG/新建文件夹/c4_assets/remaining_c2d_pinned_cpp_extension.py
```

One local `C:/Python314/python.exe -I -S -B -` invocation read `git show HEAD:<candidate>` through stdlib subprocess and computed hashlib SHA256 and line count. It imported no repository or third-party module and performed no probe/test execution.

Inspected source areas: native_bootstrap ordinary create/exec, call frames, runtime binding/init and failure boundaries; runtime_custody adoption/file/native-scope grammar; runtime_environment root/file/module validation; cuda_compilation's exact target/stream helpers and role issuance; cuda_extension_recipes; scene/LBVH/direct-light loader/preparation/workspace boundaries; and the preserved installed cpp_extension source. Governing predecessor documents were inspected for relevant requirements and replacement scope; this was not a fresh whole-C4 implementation audit.

Primary web sources were the pinned systemd v255 documentation and service/execute/exec-invoke sources, and PyTorch v2.9.1 CUDA source, linked above. They were read as source evidence, not a claim that the installed host has executed the proposed service. Some combined displays were truncated; the candidate, original report and relevant omitted decision/source sections were reread in bounded ranges. An overly broad assets filename listing also produced excessive truncated output; it changed no files and supplied no execution evidence.

No tests, builds, Ninja/compiler commands, backend imports, native observers/getters, GPU queries, service units, process fault experiments, package installation, remote writes or product entrypoints ran. No SSH connection was opened, no previous SSH handle resumed, and no signal was sent. All inspection commands completed with terminal results; no worker, supervisor, compiler, GPU job or background helper was launched. Short-lived shell/Git utility PIDs were not individually sampled, so this is bounded process hygiene rather than exhaustive OS attestation.

Only this outside-repository report was written. The candidate remained frozen and clean. No previous runtime output was promoted to evidence of a new execution. Installed-source/environment validation, manager fixtures, actual artifacts, native target1/ambient0 and partial-return/adoption failure matrices, numerical checks and affected real Go2 evidence remain the later implementation packet's obligations.

## Exact approved scope

Approval binds only design-request v2 at HEAD `3e31778766d7ea16ebe6b936dd00b9de0a0af5f0` and SHA256 `4f7cca3d448f5fd243f2786571193790dbaf2fc7604e755b86a2328e1c4dcc7d`. It permits implementing the bounded deployment-artifact replacement, ordinary guarded loading, exact factory binding and removal of superseded worker compilation machinery. Before real execution, the complete implemented packet and bounded commands still require the fresh exact-code preflight specified by the candidate.

This does not approve implemented CUDA artifact admission, a build/service/native run, checkpoint acceptance, Warp compilation, full C2d/C4, C2e/f activation, product PREPARE/submission, measurement/results/baselines, backend selection, losing-lane cleanup, or later C4c3/c4 and C4d/B3 gates. Existing accepted component scopes remain unchanged. Whole-C4 systematic refactor and independent acceptance are still required and unfinished.

The full report may be archived verbatim under the evidence-only workflow exception, preserving its exact verdict, identities, scope and limitations.
