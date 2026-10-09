# C2d compilation admission: fixed CUDA worker JIT and compiler ownership

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: bounded supplementary design request
Author: codex
Version: v1
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review
Related Files: runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-job gap-review v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3
Owner Summary: Choose worker JIT for the three existing CUDA extensions. Preserve their kernels and normal setup guards, add one private generated-extension admission boundary, and extend the dedicated supervisor to own compiler descendants. This request closes the CUDA part of G1 only. It does not approve Warp compilation, R_setup, allocation of the optical graph, warmup, final readiness, C2d/C4 completion, or product execution.

## 1. Candidate, scope and dependencies

Base is clean `8cad0e2` on `implementation/c4-supervised-waits`, following the
unchanged gap report archive. Whole C2d3 remains accepted at code/test `4da1fa7`,
production `f1ca45a`, review evidence `e5299e3`, acceptance archive `4c20cac`.
The gap report's identity paragraph calls `fc86182` the production identity;
that is the original native-run revision, not the later source-pin repair.
Preserve that report unchanged; the whole review v4 is authoritative on this
distinction. No code or native execution is part of this design candidate.

Deliver the fixed CUDA compilation and module-custody foundation after canonical
input validation. Successful foundation evidence stops at `compiled_cuda`, before
readiness certification. It is useful to subsequent parent implementation but
cannot be presented as full C2d or C4 acceptance. Warp's in-process NVRTC/module
cache/ModuleExec path requires its separate concrete admission definition before
execution; this design does not pretend a Python ExtensionFileLoader covers it.
Continue that work, one-slot construction, C2e, ten warmups and final composition
after this checkpoint. No backend selection or performance result follows.

The three existing source bodies and entry ABIs remain the only implementation.
Compatibility wrappers delegate to the same recipe definition. No offline
producer, artifact publisher, general build service, user callback, arbitrary
command, cgroup provisioner, package installation or additional daemon is added.

## 2. Route and fixed recipes

The captured CUDA worker calls the pinned Torch `load_inline` in its original
process under an ordinary setup ARM/ARMED segment. Compile one literal recipe per
segment in the order below, after the accepted input-validation boundary. A
successful segment includes generation, synchronous child completion, direct
native load, owner/entry audit and the supervisor completion check in section 6.
It ends through ordinary COMPLETE/DISARMED. Three segments are not retries or a
second setup epoch. No compile retry, changed architecture or renewed deadline
within an operation is allowed.

For the fixed foundation driver the six accepted input segments retain sequences
1..6, the three compilation segments are 7, 8, 9, and healthy checkpoint disposal
uses the existing sole terminal path at sequence 10. The driver freezes adoption
only after compiled owners are held, then uses the existing exact-target terminal
sync/hard-exit behavior. `compiled_cuda` is a completed setup prefix, not normal
optical rendering or release readiness. Failed compile segments retain the failed
original guard and cannot enter a replacement disposal ARM. No product entrypoint
is enabled by this driver.

| Recipe/module name | Repository source owner | Required entries, in order |
| --- | --- | --- |
| `robot_sim_cuda_scene_v1` | `optics.cuda_scene._CPP_SOURCE`, `_CUDA_SOURCE` | `update_triangles_world_aabb_out`, `update_planes_world_out` |
| `robot_sim_cuda_lbvh_native_v3` | `optics._cuda_lbvh_source.CPP_SOURCE`, `CUDA_SOURCE` | `cuda_lbvh_workspace_requirements`, `cuda_lbvh_rebuild_out` |
| `robot_sim_cuda_direct_light_v8` | `optics.cuda_direct_light._CPP_SOURCE`, `_CUDA_SOURCE` | `pinhole_camera_rays_out`, `first_hit_rays_out`, `shade_direct_light_no_shadow_out`, `shade_direct_light_no_shadow_diagnostics_out`, `shade_direct_light_with_shadows_out`, `shade_direct_light_with_shadows_diagnostics_out` |

All use `with_cuda=True`, `is_python_module=True`, `use_pch=False`,
`extra_cflags=['-O2']`, `extra_cuda_cflags=['-O2']`, `verbose=False`, no extra
include/link flags, no standalone/SYCL/HIP mode. Scene/LBVH supply their existing
pybind bodies; direct-light supplies the six-entry functions list above to
Torch's original generator. Do not copy its generator or CUDA source strings.
The recipe consumes the actual strings of the admitted, held repository modules,
checks their source-session identity, and freezes their UTF-8 bytes before build.
The internal builder takes a literal recipe ID, never source strings or callables
from callers. The legacy cached loader remains a wrapper over shared mechanics;
the captured path rejects a pre-existing lru_cache module and does not use it.

No constructor accepts a caller-provided module as authority. Later scene/LBVH
preparation consumes an exact locally registered compiled owner, validates its
session/controller/module/entry identities, and uses the existing private
`_PREPARED_EXTENSION_TOKEN`. The workspace query retains its actual CUB query and
validation; only module resolution is factored. This foundation does not execute
that query, create renderer buffers, or invoke any compiled optical entry.

## 3. Deployment, effects and cache eligibility

Use the existing robot_sim deployment on the primary host, Python 3.12.13,
Torch 2.9.1+cu128 and the existing Conda CUDA 12.8 toolchain. Before any
cpp_extension import, the trusted cold launcher fixes the environment below.
The worker audits it against the immutable run manifest; it cannot select a
different compiler/toolchain based on PATH discovery.

- `CUDA_HOME=/home/ang.gao/miniforge3/envs/robot_sim`; `CXX` is the resolved
  `/usr/bin/x86_64-linux-gnu-g++-13`; `CC` is the matching frozen gcc-13 path.
- `TORCH_CUDA_ARCH_LIST=9.0` for this H200-only checkpoint, no `native`, no PTX
  fallback, and no visible-device enumeration to choose architecture. Actual
  target properties must match major/minor 9/0 under the existing target scope.
- PATH contains the frozen Conda bin and fixed system compiler utility paths;
  its exact ordered value, resolved tools and symlink targets are recorded.
  `MAX_JOBS=1` bounds this first build's concurrency; it is not a performance gate.
- Preserve `TORCH_DEVICE_BACKEND_AUTOLOAD=0`. Reject compiler override/injection
  variables (`NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS`, `PYTORCH_NVCC`, `CFLAGS`,
  `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, `CPATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`,
  `LD_PRELOAD`) when present. Freeze the exact existing admitted LD_LIBRARY_PATH;
  library selection is not inferred from the Torch version label.
- Set TMPDIR and TORCH_EXTENSIONS_DIR to private run-owned directories. Every
  recipe gets a newly created, empty, mode-0700 directory beneath that run root;
  pass it explicitly as build_directory. Reject symlinked ancestors, existing
  recipe directories, output escape, or unlisted executable files. Retain outputs
  after failure for evidence; cleanup happens only after observed job closure.

The manifest inventories the existing interpreter/dependency closure plus build
tools, their symlink chains and the actual immutable compiler/include/link roots:
Conda CUDA binaries/headers/libraries/libdevice, Torch and Python headers/libraries,
host gcc/g++ executables and private helpers, binutils, system C/C++ headers and
startup/runtime libraries used by that deployment. Each file row is resolved
absolute path, size and SHA256; root and environment rows are canonical. Freeze
this concrete inventory in the implementation packet before any compiler run;
missing tool/root or changed bytes reject before use. Source inspection and the
existing read-only package metadata locate it; no unqualified compiler probing
outside a guarded compiler-capable segment. Actual emitted Ninja commands and
generated inputs are recorded and checked against the frozen recipe/tool roots.
This is trusted immutable deployment provenance, not hostile compiler attestation.

Torch's two shared files `torch/include/torch/extension.h.gch` and
`extension.h.sign` must both be absent (including dangling symlinks), before
cpp_extension use, immediately before each load_inline, and on return/failure.
The deployment contract excludes concurrent writers to these two paths for the
job lifetime. Reject presence without changing either file. With this invariant,
the pinned default PCH removal branch is a no-op. Do not monkeypatch PCH removal,
enable PCH, delete shared files, or claim the private build root alone isolates it.
If this invariant cannot hold on a host, compilation remains closed there until
a revised reviewed deployment policy; no automatic fallback.

First acceptance permits only fresh outputs generated in this same job. It does
not read the user's existing Torch cache or load artifacts from another session.
After one successful load, the exact registered owner may be reused within the
same session without recompiling. Repeated build is rejected. This is a bounded
cold-build route, not a permanent ban on caches. Cross-session reuse needs a later
explicit immutable provenance policy and its positive/negative proof; it is not
silently admitted by ExtensionVersioner, matching module name, or source hash.

## 4. One generated-extension admission boundary

Add a private build integration tied to the actual admitted
`torch.utils.cpp_extension` module and its pinned original
`_import_module_from_library` function. During the single synchronous build call
only, replace that exact module attribute with one fixed production adapter.
Before installing it, hold the original function/module/adapter and audit the
original identity/source; reject any earlier replacement. No caller hook,
configurable loader or generic import exemption is exposed. A process-local
single-entry registry rejects overlap, reentrancy, other threads and other
recipe names. Before restoration, require the attribute still is the adapter;
restore the exact original once, even on error. Restoration failure is sticky
failure and quarantine, never an excuse to declare a successful build.

The adapter requires the active setup ticket, exact session/controller, selected
recipe and directory, `is_python_module=True`, unchanged pinned dependency object
and the expected unsuffixed module name. Versioned `_vN` substitutions reject.
Require one regular nonsymlink `.so` at Torch's exact expected path, no earlier
module of that name in sys.modules or in the direct ledger, and successful
generation in the current private directory. Hold generated main.cpp/cuda.cu,
Ninja file, resulting shared-object bytes and their size/digests in the build
record. Their observed hashes are downstream build facts, not pre-existing
authority to load arbitrary matching files.

Reuse the existing source-session ownership/failure mechanics with a separate
direct-load record. Construct a standard ExtensionFileLoader and spec for this
exact path, with the same pinned origin/type/byte checks as external admission.
Hold intent, bytes, spec and actual delegate before module_from_spec/create;
hold every returned module immediately before exec or any audit that can fail;
then execute and audit the actual module/spec/loader/origin and fixed entries.
This route does not publish sys.modules artificially. Record `publication=direct`
and require later checks use the direct ledger; absence from sys.modules is not
absence of custody. Any actual unexpected publication is retained on failure.
Factor shared loader checks where useful; do not copy a second independent import
policy. The sole existing bootstrap importlib exception gets the exact additional
function/body policy, not a repository-wide dynamic-loader permission.

Successful admission registers an immutable private CompiledCudaOwner by exact
object identity with the source session and controller: recipe, generated record,
actual module, spec/delegate, fixed entry objects and their bound module, original
build dependency/function identities, target/runtime/stream roots, guard reference.
Consumers re-audit actual identities, not only a digest. Parsed evidence or copied
dataclass fields cannot register another owner. Cross-session, stale, replaced or
partially admitted owners reject before any next device leaf.

The whole admission stack and all returned roots belong to the existing guardian.
If create raises before a Python module is available, retain the exception,
traceback, intent/spec/delegate and any visible partial publications, mark UNKNOWN
using existing native-import failure semantics, and quarantine. The design does
not claim access to a native allocation that never returned. Exec, entry lookup,
owner registration, adoption or hook restoration failure retains available owners
before propagating; a swallowed dependency exception cannot recover the session.
No second emergency sync or ordinary module unload is added during failure unwind.

## 5. Scope and ordinary guarded lifecycle

Guardian/source/environment admission precede any Torch/build import. Add only the
actual cpp_extension and transitive CPU helper closure, including the pinned
hipify imports actually performed by _jit_compile. Split unused Warp compatibility
imports/native Torch preconditions where necessary in existing CUDA modules;
do not import Warp solely because an unused legacy checker bundles dependencies.
Preserve existing compatibility behavior through lazy wrappers over the same code.

Source policy/catalog updates cover every changed captured file and exact
additional loader boundary. Compilation is not assumed CUDA-incapable: library
load or runtime registration can touch native state. Before each new capable
boundary, retain operands and enter the existing exact-target leaf contract;
after it, verify target/ambient restoration before a following leaf. A descriptor
unsupported by the actual source-backed native observation remains UNKNOWN and
fails closed. Compile/import helpers may not invent device truth from arguments.

Use the existing target1/ambient0 CUDA runtime/stream and 120-second setup timeout,
1-second ACK timeout. A cold build too slow for this configuration is a real failed
run, not permission to restart the clock or reuse a late artifact. Any future
budget change requires an explicit evidence-backed reviewed configuration delta.
No compilation time is reported as optical sample time.

## 6. Compiler job ownership: dedicated supervisor and subreaping

Ninja 1.13.1 gives ordinary command children separate PGIDs. Therefore one killpg
is insufficient. This proposal adds a bounded Linux process facility to the
already dedicated compiler-capable supervisor, not to the optical worker or the
long-lived coordinator/pytest process. That supervisor has exactly one worker
tree, no unrelated child launch, one thread, and sole ownership of child waits.

Before launching the worker, it enables and reads back PR_SET_CHILD_SUBREAPER via
one private fixed-signature libc prctl adapter. Load only the exact inventoried
system libc; constants are PR_SET_CHILD_SUBREAPER=36 and
PR_GET_CHILD_SUBREAPER=37. The adapter accepts no arbitrary function, library or
prctl option, checks return/errno, retains its handle, and is confined to this
stdlib-only supervisor. This is an explicit supervisor native-OS boundary; it
does not extend the captured worker's ctypes/CDLL permission. Verify isolated
process identity, SIGCHLD default disposition, no SA_NOCLDWAIT, usable pidfd APIs,
procfs visibility and no existing children before worker spawn. Failure prevents
launch. No global subreaper setting or elevated privilege is requested.

Spawn the fixed cold worker with start_new_session=True, close unrelated FDs and
inherit only fixed channels. Record PID/starttime/SID and acquire its pidfd before
authorizing its first ARM. The original trusted worker/toolchain may create
separate PGIDs but cannot daemonize, change session, install a subreaper, use
CLONE_PARENT/non-SIGCHLD children, or delegate work to an external service.
Freeze that toolchain contract and later verify the actual packaged tools with
bounded process observation. Arbitrary hostile tools are outside this boundary.

The supervisor keeps a JobObligation independent of GuardSupervisorCore:
NEW -> LIVE -> DRAINING -> CLOSED, with sticky `failed` and `deadline_expired`.
LIVE starts at the actual worker launch. DRAINING starts on worker exit, guard
fault/parent loss or deadline expiry. CLOSED requires actual reaping/closure;
neither kill success nor an empty scan is sufficient. No transition returns from
DRAINING to LIVE. Duplicate terminal notifications only append previously missing
observations; they do not issue another guard or reset a deadline.

The job records each original guard's absolute operation deadline and each pending
ACK deadline as established by the existing core. While live they track its exact
obligation. On first fault/exit, freeze the earliest outstanding deadline; if the
core had no outstanding guard, use its existing terminal grace policy once, from
the first fault/exit time. EOF and leader exit cannot renew or erase this value.
Call core.observe_exit when exit is actually observed, preserving its meaning as
worker exit only; the outer job continues even though that core clears its fields.

One explicit wait owner uses waitid(WEXITED|WNOHANG|WNOWAIT) and waitpid on identified
direct children; no Popen.poll/wait/communicate/finalizer or other thread consumes
statuses. Set the retained Popen returncode from that actual wait once obtained.
Before reaping a waitable child, record its PID/starttime, parent/SID when still
available, actual wait status and whether it is the original worker or an adopted
descendant. A zombie retains its identity until that wait. Do not manufacture a
compiler return code for a process normally reaped by its own parent.

On timeout, record the existing worker kill request and send SIGKILL through its
held pidfd. Record attempt/error/observed exit separately. As parents exit, Linux
reparents their remaining descendants to this supervisor. Discover only its own
direct children through its proc task children list; verify PPID/starttime around
pidfd acquisition, hold handles, and signal only positively identified owned
children. Revisit until closure; a fork racing parent termination becomes an
adopted child, so session enumeration is diagnostic, not the closure proof.
Before expiry, continue observation/reaping without advancing the kill deadline.
If a child disappears during acquisition, rescan/reap; PID reuse or inconsistent
identity forbids a signal to that candidate. Handle exhaustion, inaccessible proc
data or signal denial records failure and retains known handles/evidence; it does
not authorize PID-only, UID/name-wide or unrelated-group termination.

The decisive failure/exit closure is: original worker actually reaped, no further
spawn by this dedicated supervisor, all observed adopted children reaped, and
waitpid(-1,WNOHANG) returns ECHILD with the unchanged subreaper/SIGCHLD policy.
Under the fixed descendant model any remaining descendant would have a live
ancestor child or have been adopted, so ECHILD is stronger than a proc snapshot.
Then check retained PID/start identities absent and drain inherited output FDs to
EOF. An empty proc scan alone is never sufficient. Persistent OS failure leaves
DRAINING/incomplete, keeps the owner alive, and forbids success/artifact cleanup;
an external harness timeout must preserve this incomplete result rather than
kill the supervisor and claim reaping. No promise of bounded kernel exit after
SIGKILL is made; the bound is the unchanged termination-attempt deadline.

Normal compiler-segment COMPLETE is intercepted before forwarding it to the core.
Only the three fixed compile sequences use this check. The trusted worker has
returned from synchronous load_inline and is blocked in COMPLETE; the inspected
subprocess.run/FileBaton/build path cannot keep spawning after that return.
Require no other member in the anchored worker SID, no adopted child of the
supervisor besides the live worker, and no retained still-live compiler identity.
This check observes quiescence while the worker waits; it does not infer it from
one arbitrary scan during an active compiler. Any residual or observation error
faults the still-armed guard and withholds DISARMED. On success, pass the unchanged
COMPLETE to the core and send its original DISARMED under the original ACK clock.
Do not disarm first and audit later. Normal whole-job exit still uses the reaping
and ECHILD closure above. No new wire role or compile-only success termination.

Factor the existing framed transport to accept only an exact private registered
compiler-job owner for this check; the legacy path without a compiler job retains
its current behavior. Do not expose a callable callback parameter. Immediately
before core.accept for a decoded COMPLETE, the fixed owner checks session/role/
sequence against its 7/8/9 table and performs the check above. Failed checks use
the existing core fault transition. ARM and ACK encoding, decoder limits, deadline
rules and output-fence semantics remain the single existing implementation.

## 7. Evidence and live-authority separation

Keep accepted C2d3 v1 evidence unchanged. Add separate canonical descriptive
records with explicit version strings; their parser never creates a live owner.
All objects reject unknown/missing/duplicate fields, bool-as-int, floats,
noncanonical encoding, unknown enums, invalid SHA256, sequence drift and dangling
references using shared canonical validators. Use the existing object-token
registry; do not build a second identity map for the same worker resources.

`q54.p12.4.cuda-compilation.v1` has exact root fields:
`schema_version, session, input_evidence_sha256, source_sha256,
environment_sha256, toolchain_sha256, target_ordinal, recipes, events, summary`.
Recipe rows have `recipe_id, guard_sequence, build_directory, source_files,
generated_files, artifact, entries, object_bindings`; file rows have
`path,size,sha256`, entry rows `name,object_token`, artifact is a file row,
and object_bindings has `module,spec,loader,compiled_owner` tokens. In an
incomplete prefix, generated_files/entries are ordered prefixes, artifact and
not-yet-created object tokens are null, never fabricated placeholders.
Events are `index,recipe_id,operation,object_token`, ordered transitions:
`build_entered -> generated -> create_entered -> module_returned -> exec_completed
-> entries_bound -> owner_registered -> hook_restored -> guard_completed`.
Each started recipe either follows this path or appends `failed` with the held
exception token and never resumes. `generated` means final helper reached after
build return, not merely main.cpp written. Summary is replayed
`{completed_recipes,failed,boundary}`; boundary is `incomplete` or `compiled_cuda`,
the latter requires all three exact recipes and joined successful guard records.
It still has no R_setup or final-readiness claim.

`q54.p12.4.compiler-job.v1` has exact root fields:
`schema_version,session,supervisor,worker,guard_evidence_sha256,events,summary`.
Process identities are `pid,starttime_ticks,sid`; supervisor/worker are required
once launched, otherwise null. Events are `index,at_ns,operation,details`.
Closed operation/detail variants are: `subreaper_ready {}`;
`worker_spawned {identity}`; `deadline_bound {sequence,deadline_ns}`;
`compile_quiescent {sequence}`; `draining {reason,deadline_ns}`;
`child_adopted {identity}`; `signal_attempted {identity,signal}`;
`signal_failed {identity,error_kind}`; `child_reaped {identity,wait_status}`;
`no_children {}`; `pipes_eof {}`; `observation_failed {error_kind}`;
`closed {}`. Signal is SIGKILL only; error_kind is
`missing,permission,identity_changed,proc_unavailable,resource_exhausted,other_os`;
reason is `worker_exit,guard_fault,parent_lost,deadline_expired,observation_failed`.
Summary is replayed `{state,failed,deadline_ns,worker_reaped,no_children,pipes_eof}`.
Identity reuse never merges process rows. at_ns is supervisor monotonic time;
deadline joins the original core trace, not worker timestamps. ECHILD is recorded
only after actual OS observation, and closed is allowed only by section 6.

The test/report aggregate joins the new records with original worker/lifecycle/
source/native-scope evidence by session, input digest, guard sequence and actual
same-run process observations. A codec proves consistency, not execution. Preserve
partial bytes and failures if write/parse/transport/exit fails. Do not synthesize a
missing tail or promote a prior completed prefix to current compilation success.

## 8. Required verification and implementation order

1. Implement fixed recipe factoring, schema/owner registry and supervisor logic;
   run meaningful CPU counterexamples through production boundaries. Fresh
   independent code preflight must approve exact source/policy/catalog and fixed
   runners before any new backend compilation or native observer execution.
2. In an isolated dedicated Linux supervisor, prove subreaper enable/readback,
   pidfd ownership, real separate-PGID descendants, leader-first exit, double
   fork/adoption, fork during cleanup, inherited-pipe hang, missing/failed COMPLETE,
   original deadline after leader exit, stale/disappearing identity, proc/handle/
   signal failure, ECHILD and actual reaping. An unrelated sentinel remains alive
   and unchanged. No compiler/backend import is needed for these process tests.
3. Prove real cold builds/admission of all three recipes, actual packaged Ninja
   process membership and output cessation, generated/source/toolchain provenance,
   target1/ambient0 before/after every newly admitted native boundary, all retained
   modules and entries, unchanged shared PCH paths, final owned OS/GPU residuals.
   Freeze and independently review the exact observer ABI/source/case matrix first.
4. Exercise wrong recipe/name/version/path/artifact/entry, dirty cache, PCH present
   with unchanged shared files (use an isolated fixture deployment, never create
   those files in shared Torch), injected create/exec/adoption/restoration failures,
   lock/compiler hang, owner copy/rebind/cross-session/repeated build, swallowed
   import failure and residual-at-COMPLETE. Show no subsequent device leaf or
   normal completion on failure and no second emergency or unsafe unload.
5. Independent whole checkpoint review covers the integrated three-recipe worker,
   custody, guard, supervisor and real evidence; a schema/process test subset is
   not its final acceptance. Then continue the remaining parent milestones.

No GPU compile has occurred for this candidate. Source snapshots underpinning
the design are pinned cpp_extension SHA256
`52ba4de5deedd894e2bdafc9b7f40526dbc2cc1e4c560bbae3b38d4a5dcab0c6`
and the unchanged gap report. Process semantics are the Linux
[subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html)
and [wait interface](https://man7.org/linux/man-pages/man2/waitpid.2.html);
actual host capability and behavior still require the tests above.

## 9. Maintenance and remaining gates

The direct adapter exists solely because pinned Torch bypasses finder admission.
Remove it if a later accepted deployment-artifact route replaces worker JIT; do
not keep both routes by default. Fixed recipe definitions and process ownership
are shared production mechanics, not test-only copies. At final backend choice,
delete the losing lane and its now-unused recipes/adapter/inventory/tests according
to the parent cleanup gate. Read-only historical reports remain evidence.

This design approval, if obtained, authorizes implementation and proportionate CPU
proof only until the native code-preflight gate above. It does not open PREPARE,
a3b/result/R2/S/host selection, sample tokens, final barrier, measurement or product
activation. C4 is complete only when its full approved composition and cleanup,
with their independent reviews, are complete.
