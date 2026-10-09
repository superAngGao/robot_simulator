# Fixed CUDA artifacts: remove compilation from the optical worker

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: bounded replacement design request
Author: Codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: scope-convergence proposal v1; runtime-integration design/review v3;
  remaining-setup gap-scan v1; compilation-admission design/review v3;
  setuptools-admission design/review v2; systematic-refactor implementation-note v1
Owner Summary: Build the three CUDA extensions before launching a rendering worker.
  Load them as ordinary inventoried native modules in the existing runtime-init
  segment. Delete worker JIT and its special dependency/process machinery. Preserve
  real resource ownership, correctness, fair timing and every later C4 checkpoint.

## 1. Replacement and actual starting point

Candidate base is clean local `1a9f319708cdf1856fe73f1a0108acc78b9190e3`, branch
`implementation/c4-supervised-waits`. The primary isolated tree
`/data2/shared/ang.gao/robot_sim_cuda_compilation_wip` is clean at code
`92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af`; intervening commits are records/proposal.
This is a design candidate, not approval to execute its implementation.

On acceptance this document replaces the worker-JIT choice, generated direct
loader, compilation suffix and compiler-subprocess ownership in compilation-
admission v3, and removes the need for live setuptools-admission v2 in a worker.
Their historical results remain unchanged. It replaces the same-job-only cache
rule with reuse of an explicitly deployed immutable artifact set. It does not
relax the parent trusted-deployment boundary or introduce a hostile Python sandbox.

The accepted C2d3 input lifecycle, guardian, target/ambient rules, source-backed
scope observations, ordinary setup timeout/ACK, ten real warmups, final readiness,
seal/barrier order, independent review and downstream gates remain requirements.
This is the alternative already identified in remaining-setup gap-scan G1. It is
one route: no worker JIT fallback, alternate compiler daemon or new PID framework.

The current factories still call `load_inline`; PREPARE still rejects and Warp
timeline/arena are absent. Completing this replacement is therefore not completion
of C4 or permission to publish a speed comparison. The interrupted compiler
preflight v3 did not approve code92a8ac6 and is not revived by this document.

## 2. Build with existing tools, outside runtime

Add one deployment-only command in `tools/` with a fixed three-recipe selection,
not caller-provided code or commands. Keep `optics.cuda_extension_recipes` and the
three `_cuda_*_source` modules as the sole recipe/kernel sources. Use installed
`setuptools` plus Torch `CUDAExtension` / `BuildExtension` to write and build
`main.cpp` and `cuda.cu` in private directories. Do not use `load_inline`, patch a
Torch import hook, load the resulting extension, invoke kernels or initialize a
GPU during the build. Ordinary deployment Python may import its installed build
dependencies; the render-worker import mechanism is not a build sandbox.

The direct-light source gains its explicit six-entry `PYBIND11_MODULE` body,
using `torch::wrap_pybind_function` and the same function-name docstrings as the
installed Torch generator. Scene/LBVH bindings already exist. Preserve all kernel
bodies and existing names/signatures. Required Torch/CUDA headers formerly inserted
by `load_inline` become explicit in the sole source files. No copied generator,
second source body or private Torch build API. Retain `-O2` for C++/NVCC, CUDA-only
extension mode and Python module ABI. Standard BuildExtension supplies its own
required ABI/compiler flags; record actual emitted build inputs. Numeric behavior
is verified, not assumed from equivalent source names.

Keep the already frozen Python3.12.13, Torch2.9.1+cu128, Conda CUDA12.8, gcc/g++13,
H200 architecture9.0 (no PTX fallback), `MAX_JOBS=1` and disabled Torch backend
autoload. Set the exact compiler/environment paths; do not infer architecture by
enumerating GPUs. Launch through a clean explicit environment, with the admitted
library paths and private HOME/TMPDIR/build directories. The standard builder
does not exercise load_inline's shared PCH removal branch; do not modify shared
Torch include files. Verify this from the installed build implementation before
the first build. Missing tools or unsupported build behavior is a failed deployment,
not permission to install packages or switch APIs.

Run each fixed recipe as one transient **user service** using existing systemd:
`systemd-run --user --no-ask-password --wait --pipe --collect --service-type=exec`,
a unique recorded unit name, `ExitType=cgroup`, `Restart=no`, `RemainAfterExit=no`,
`RuntimeMaxSec=120s`, `TimeoutStartSec=120s`, `TimeoutStopSec=10s`,
`KillMode=control-group`, `SendSIGKILL=yes`. No SuccessExitStatus override, notify
extension, delegated cgroup or shell-supplied arbitrary command. The120s build
budget is retained from the previous design;10s is the bounded manager stop grace,
not a rendering deadline or renewed build budget. Use exact argv and file logs.

The manager owns descendants regardless of their process groups. Success requires
the waiting command's zero exit (unit Result=success), a completed expected output,
and no remaining members of this unit's cgroup. Capture the unit/cgroup identity
while it runs; after completion, an absent owned cgroup or empty membership is
required. A missing/unreadable observation is not an empty cgroup. Do not treat a
successful main-process exit alone as success. Keep command/output and failure
logs outside the transient unit, because `--collect` removes inactive units.
On timeout/failure never publish. On launcher loss the manager still owns the
bounded job; no launcher remains to publish its output. On uncertain cleanup keep
outputs private and report the exact unit; do not kill unrelated processes, retry
the build, or label a stuck kernel task reaped. This changes the *deployment*
owner; the rendering worker's existing failure protocol is unchanged.

Read-only primary-host observation on2026-10-09: systemd255.4-1ubuntu8.11, user
manager running, unified cgroup hierarchy and user bus present. No unit was
launched for this design. Upstream255 implementation keeps an ExitType=cgroup
service alive after its main exits while the group remains populated, and the
run command returns failure for a timeout result. This supports the design but
does not substitute for installed-host verification:
[service implementation](https://github.com/systemd/systemd/blob/v255/src/core/service.c),
[run result handling](https://github.com/systemd/systemd/blob/v255/src/run/run.c).

## 3. Immutable publication and provenance

The deployment launcher creates a fresh private root and three initially empty
recipe directories, using the existing checked creation/fchmod pattern on the
setgid shared filesystem. Build processes write only staging outputs. After all
three jobs have terminal success and empty groups, the launcher verifies output
files and atomically renames a complete artifact directory to a fresh immutable
deployment path. Do not replace an existing deployment, update a live symlink,
publish from the build child or reuse a failed staging directory. A crash before
publication leaves staging only; after publication the whole set is present.
Published files have no concurrent writer for any worker lifetime. This is the
existing trusted deployment assumption, not a root-resistant filesystem claim.

Use a canonical `q54.p12.4.cuda-artifact-set.v1` manifest with exact top-level
fields `schema_version, source, recipes, platform, environment, toolchain,
builds, artifacts`. Reuse existing canonical JSON/digest/file-row validation.

* `source`: commit plus sorted `{path,size,sha256}` for the deployment command,
  recipe module and three source modules. Bind source bytes; an unrelated later
  repo commit does not invalidate identical kernel/build source bytes.
* `recipes`: ordered `{name,entries,cpp_sha256,cuda_sha256,cflags,cuda_cflags}`
  for exactly the existing scene/LBVH/direct-light names and entries. Hash the
  UTF-8 source bytes actually written, including the explicit bindings/headers.
* `platform`: `{system,machine,python_version,soabi,ext_suffix,torch_version,
  torch_cuda_version,cxx11_abi,cuda_arch_list}`; exact deployed values, no ranges
  or architecture auto-detection. Driver/library compatibility continues to be
  bound by the worker runtime environment and real-device verification.
* `environment`: exact sorted string-to-string build environment; no secrets.
* `toolchain`: existing canonical inventory payload for actual resolved compiler,
  interpreter, dependency, header/link roots and symlink targets. Move/reuse the
  existing inventory producer outside the worker; do not duplicate that inventory
  in every runtime record or rehash all build headers inside a render worker.
* `builds`: ordered `{name,unit,argv,returncode,stdout,stderr,inputs}`. Log and
  generated-source/Ninja rows are relative `{path,size,sha256}` in the immutable
  deployment bundle. They are emitted inputs/logs, not an exhaustive claim about
  short-lived executed commands. Failed builds never produce a published manifest.
* `artifacts`: ordered `{name,path,size,sha256}` for exactly three regular,
  nonsymlink extension files at their standard ABI-suffixed top-level filenames.

The trusted cold launcher selects the explicit deployment path and manifest digest.
It verifies manifest/schema, actual artifact bytes, recipe/source and platform
matches before a worker launch. Derive the worker environment from this verified
set, with `cuda_artifacts` as one fixed `site` root, its manifest as a `metadata`
file, and exactly the three extension file/module rows (`phase=runtime`). Thus the
existing environment digest binds the selected manifest and outputs. Extend the
ordinary v5 file inventory; no setuptools v6 descriptor or new environment schema
is required. Validate all manifest/file/root cross-links; a caller-authored hash
or matching filename is not deployment authorization. A worker independently
verifies its bound environment bytes through existing admission.

Unchanged deployed sets may serve fresh sessions. Modules/owners are never reused
across processes/sessions. Runtime has no dependency on the compiler installation;
the build provenance is a deployment fact, not runtime entry/device readiness.
Missing, modified, wrong-ABI/source or partial artifacts reject before native load.

## 4. Reuse ordinary module loading and the existing six setup phases

The CUDA session loads all three fixed modules in `initialize_native_runtime`,
after the indexed Torch stream and `_RuntimeSetupBinding` exist, before completing
the existing wire3 runtime-init segment. Use three literal imports in fixed recipe
order through `_ExternalAdmission` / `_ExternalSourceLoader`. No dynamic generated
loader or direct ledger; these are ordinary file-backed modules in `sys.modules`.
External admission already pins the resolved path/bytes, loader/spec and retains a
create return before metadata/adoption can fail. Reject preloaded modules,
reloading, another root or an import outside this exact CUDA initialization interval.

For these three fixed extensions only, delegate create/exec through the existing
explicit Torch target/stream helpers extracted from the old compiler. Mark each
native leaf before invocation; retain a returned module in the external registry
inside the target/stream helper before leaving its contexts (context exit may
fail). Keep the existing module-return/file-call events. Retain original exceptions
and partial specs/loaders/modules/publications; failure poisons the session and
uses existing UNKNOWN/quarantine semantics. It cannot advance runtime_ready or
start another guard. Target1/ambient0 real observations cover create and exec,
including restoration before a following leaf. No new emergency sync is introduced.

Register a private `CudaExtensionSet` tied by identity to this source session,
controller, runtime binding and external admission. Each recipe holds actual
module/spec/delegate and entry objects; the ordinary completed file record and
each entry's module/name relationship must match before registration. Hold entries
and the set in the guardian before successful issuance. A partial set remains
held but unusable; copied fields or parsed evidence never register a live set.

Extend runtime-custody to v3, preserving v1/v2 historical meanings. Keep the six
phase IDs and guard order0..5. Add only (a) native-scope sites for the fixed
recipe create/exec leaves, at phase3 after runtime binding, and (b) one
`cuda_extensions_bound` journal event. Its details are exactly
`{owner_token,binding_token,recipes}`, with ordered recipe rows
`{name,module_token,spec_token,loader_token,entries}` and ordered entry rows
`{name,token}`. All objects must have prior actual adoption and the file records
must have completed create/exec with these exact tokens. Replay requires all six
matched native-scope pairs for successful CUDA binding, fixed entries, unique
non-aliasing owners, and original guard3 DISARMED before runtime-init completion.
Warp has no such event. In v3 CUDA, runtime-init cannot complete without it.
Reuse the current native-scope state/library/identity checks and external-file
grammar; no build steps, hook-restoration journal or compilation-selection suffix.
Input validation still ends at phase5 and explicitly is not final render readiness.

## 5. Factories and compatibility: one resolution route, no fallback

Captured preparation obtains the exact issued extension set from its session.
The scene, LBVH and direct-light prepare functions consume the corresponding
registered module and fixed entry identities through their private preparation
boundary; constructors retain `_PREPARED_EXTENSION_TOKEN` protection. Check session,
controller, target, stream, file metadata and entry identities before each prepare.
No bare caller-supplied module becomes authority. LBVH workspace sizing and factory
creation share the **same** resolved module and retain their current CUB query,
device/count bounds and out-style ABI. Registration does not call any optical entry.

Existing non-captured compatibility APIs resolve these same fixed module names from
the explicitly selected deployment with ordinary file imports and shared manifest/
entry checks, without claiming a captured session owner. Their default loader can
no longer compile. Keep captured resolution separate only for its existing live
authority requirement; share artifact/name/entry rules and all kernel/factory logic.
Do not introduce an optical-backend adapter. Remove runtime imports of
`torch.utils.cpp_extension`, its load_inline caches and compile kwargs plumbing.
CPU-only module imports remain possible. Warp-only compatibility imports become
lazy at their actual compatibility boundary; the native CUDA factory must not
initialize Warp as a side effect. Preserve approved C4c3 acceleration interop.

## 6. Removal, verification and remaining delivery

Delete current worker `cuda_compilation.py`, `setuptools_admission.py`, the direct
extension wrapper, their bootstrap phase/hook/root wiring, and compiler-specific
supervision/job launch paths. Move the six generic input phase constants out of
`compilation_contract.py`; remove its live build vocabulary. The standard render
supervisor, guardian and C2d3 publication/exception handling remain. Remove compiler/
setuptools-only current profile, driver branches and tests after their replacement
coverage passes. Preserve original evidence and historical catalog/profile files.
Historical compile/setuptools record readers, if retained for archived evidence,
are CPU-only under an explicit historical-reader boundary and are not imported by
the worker. Do not preserve live JIT code just to keep obsolete tests green.

One implementation/review packet covers deployment, loading, factory binding and
net deletion. Before real execution, fresh code preflight freezes that complete
packet and its bounded commands; no separate review project per syscall/helper.
Required evidence is proportional to the changed boundary:

1. CPU validation: malformed/mismatched/partial manifests, forbidden JIT imports,
   wrong session/entry and dependency-source closure; existing affected C2d3 replay,
   guardian and lifecycle tests. Check historical v1/v2 record interpretation.
2. Installed-manager acceptance before builds: normal success; a main that exits
   leaving a separate-PGID descendant (must wait); ignored-TERM descendant timeout;
   launcher loss. Record actual service/cgroup terminal facts and an unrelated
   sentinel remaining alive. Use short fixture budgets; never infer kill/reap from
   intended commands. This replaces the unexecuted custom pidfd campaign.
3. Build all three actual artifacts with the frozen toolchain. Verify publication,
   source/ABI/logs and no output import/GPU init by the builder. Real target1/ambient0
   worker loading plus module/entry/factory/workspace binding; reused immutable set
   in a new worker; missing/changed output and partial create/exec/adoption failures.
   Scope errors/hangs use the unchanged supervised failure protocol.
4. Renew affected real Go2 both-lane input/upload/readback and native-scope checks;
   then complete parent C2e/f timeline/arena, one-slot provisional graph, ten warmups,
   fresh final readiness/graph and guarded seal/barrier. Fixed artifact acceptance
   never substitutes for their real correctness/failure matrices.

Continue C4c3/c4 and C4d/B3 including joint reclaim/drain and the whole-C4 structural
refactor/independent acceptance. Warp NVRTC/module-cache/ModuleExec custody still
needs its concrete admission under the parent design; CUDA .so loading does not
solve it. Product PREPARE, measurement/results, backend selection and losing-lane
cleanup remain behind their existing gates. No performance claim follows from
moving compilation out of a phase that was already excluded from fair timing.
