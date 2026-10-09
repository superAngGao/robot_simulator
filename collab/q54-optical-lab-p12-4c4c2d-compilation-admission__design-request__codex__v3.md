# C2d compilation admission: fixed CUDA worker JIT and compiler ownership

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: bounded supplementary design request
Author: codex
Version: v3
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review
Related Files: runtime-integration design-request/review v3; remaining-setup gap-scan v1; compilation-job gap-review v1; runtime-adoption design-request v2; guarded-custody repair-design v2; supervised-waits design-request v3
Owner Summary: Choose worker JIT for the three existing CUDA extensions. Preserve their kernels and normal setup guards, add one private generated-extension admission boundary, and extend the dedicated supervisor to own compiler descendants. This request closes the CUDA part of G1 only. It does not approve Warp compilation, R_setup, allocation of the optical graph, warmup, final readiness, C2d/C4 completion, or product execution.

## 1. Candidate, scope and dependencies

Base is clean `4597d51e3e001e0f7a2c5c2fd657489a11314dfc` on `implementation/c4-supervised-waits`, following the
unchanged independent design-review v2 archive. Whole C2d3 remains accepted at code/test `4da1fa7`,
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

Changes from v1 address independent R1 (actual zero-based wire sequence), R2
(independent post-exit expiry/actions), and R3 (exported continuation custody and
source-phase integration). The unchanged report and CPU counterexample remain
evidence. Nonblocking comments about create-return retention, command provenance,
and supervisor ABI/readback are incorporated below.
V2's independent review closed R1/R2 at design level and found one remaining
R3-continuation ambiguity. V3 adds the exact lifecycle-backed successful role
bindings in section7.2. Root-only partial retention remains legal but grants no
successful spec/loader/module/entry/compiled-owner binding. Prior versions and
both original counterexample probes remain unchanged.

The captured CUDA worker calls the pinned Torch `load_inline` in its original
process under an ordinary setup ARM/ARMED segment. Compile one literal recipe per
segment in the order below, after the accepted input-validation boundary. A
successful segment includes generation, synchronous child completion, direct
native load, owner/entry audit and the supervisor completion check in section 6.
It ends through ordinary COMPLETE/DISARMED. Three segments are not retries or a
second setup epoch. No compile retry, changed architecture or renewed deadline
within an operation is allowed.

For the fixed foundation driver the six accepted input segments retain sequences
0..5, the three compilation segments are 6, 7, 8, and healthy checkpoint disposal
uses the existing sole terminal path at sequence 9. The driver freezes adoption
only after compiled owners are held, then uses the existing exact-target terminal
sync/hard-exit behavior. `compiled_cuda` is a completed setup prefix, not normal
optical rendering or release readiness. Failed compile segments retain the failed
original guard and cannot enter a replacement disposal ARM. No product entrypoint
is enabled by this driver.

Define one production immutable phase table, consumed by both the driver and
supervisor: the existing six input phase IDs mapped to wire0..5, then
`cuda_scene_compile=6`, `cuda_lbvh_compile=7`, `cuda_direct_light_compile=8`,
and terminal disposal9. Human ordinal7 means wire6; no dummy guard is added.
The table must reject missing/duplicate/reordered phases and an unknown COMPLETE,
not merely apply extra checks when a supplied sequence happens to match.

| Recipe/module name | Repository source owner | Required entries, in order |
| --- | --- | --- |
| `robot_sim_cuda_scene_v1` | `optics._cuda_scene_source.CPP_SOURCE`, `CUDA_SOURCE` | `update_triangles_world_aabb_out`, `update_planes_world_out` |
| `robot_sim_cuda_lbvh_native_v3` | `optics._cuda_lbvh_source.CPP_SOURCE`, `CUDA_SOURCE` | `cuda_lbvh_workspace_requirements`, `cuda_lbvh_rebuild_out` |
| `robot_sim_cuda_direct_light_v8` | `optics._cuda_direct_light_source.CPP_SOURCE`, `CUDA_SOURCE` | `pinhole_camera_rays_out`, `first_hit_rays_out`, `shade_direct_light_no_shadow_out`, `shade_direct_light_no_shadow_diagnostics_out`, `shade_direct_light_with_shadows_out`, `shade_direct_light_with_shadows_diagnostics_out` |

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

Extract scene/direct-light source constants into the two pure source modules
named in the table. Keep their UTF-8 string contents byte-for-byte equal to the
old literals; legacy modules import aliases instead of retaining copied literals.
Keep the existing LBVH source module. A single pure
`optics.cuda_extension_recipes` definition owns names/entry order/flags and
references these constants; it imports neither Torch nor Warp. Legacy wrappers
and the private guarded compiler use that same recipe. This avoids importing
renderer/timeline/compatibility modules just to retrieve C++ strings. This stage
does not perform the formerly proposed lazy-compatibility refactor; later graph
preparation must handle its actual source closure under G2.

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
outside a guarded compiler-capable segment. Emitted build.ninja rules, generated
inputs and their expanded fixed recipe/tool paths are recorded as build inputs,
not mislabeled as observed executed commands. The supervisor records available
actual proc cmdline/exe observations for sampled compiler members separately;
short-lived processes can be missed. Pinned Torch's verbose=False path discards
successful Ninja stdout, so this design does not promise an exhaustive executed-
command log from that setting. Real process proof covers actual packaged-tool
membership/exit plus this trusted synchronous build path, not invented stdout.
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
wrap the exact delegate create call and retain its returned module before returning
to module_from_spec (whose metadata initialization can fail); retain it before
exec or any subsequent audit;
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
hipify imports actually performed by _jit_compile. The pure source/recipe closure
in section2 avoids importing unused Warp compatibility modules in this worker.
Leave those legacy modules' runtime behavior intact at this stage.

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

The same exact libc boundary also reads sigaction(SIGCHLD,NULL,&action) using
the pinned Linux x86_64 glibc ABI, solely to verify handler SIG_DFL and absence of
SA_NOCLDWAIT; it never installs a signal handler. Code preflight must freeze the
actual struct layout/constants from installed headers and an independent CPU
ABI probe. signal.getsignal alone does not satisfy this native flag readback.
All proc scanning/waiting/output draining is nonblocking and interleaves deadline
ticks; a full pipe or slow diagnostic scan cannot precede an already-due signal.

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

After each core transition, capture a `deadline_snapshot` referencing that exact
guard trace index: earliest deadline, source (operation/ack/terminal_grace), and
active sequence. Healthy DISARMED can snapshot null source/deadline/sequence;
this is the only normal clearing case. Before observe_exit, freeze the last
outstanding snapshot (including a pending ACK that expires sooner), or compute
the one existing terminal grace from observed exit when there was none. The job
records that frozen obligation in `draining`; subsequent snapshots cannot enlarge
it. A normal worker exit can close before that deadline with no expiry or signal.

One explicit wait owner uses waitid(WEXITED|WNOHANG|WNOWAIT) and waitpid on identified
direct children; no Popen.poll/wait/communicate/finalizer or other thread consumes
statuses. Set the retained Popen returncode from that actual wait once obtained.
Before reaping a waitable child, record its PID/starttime, parent/SID when still
available, actual wait status and whether it is the original worker or an adopted
descendant. A zombie retains its identity until that wait. Do not manufacture a
compiler return code for a process normally reaped by its own parent.

At the frozen deadline append the outer `deadline_expired` once, with its exact
deadline/source, and set failed/expired sticky. Two cases are distinct:

- Worker not yet observed exited: tick the existing core, record its single legal
  kill request when due, and mirror the actual SIGKILL pidfd attempt in the outer
  job. If exit wins the race, record the real exit and no fictitious signal.
- Worker already observed exited: its core trace is immutable after process_exited.
  Do not tick it for new events, record_kill_request, or signal that dead worker.
  The outer job alone expires and signals positively identified adopted children.

Each actual process identity receives at most one termination attempt. An earlier
attempt/error remains recorded; repeated ticks/EOF/exit observations do not renew
the deadline, duplicate the attempt, or clear failure. A newly adopted identity
after expiry is attempted immediately, using the same expired obligation. Record
attempt/error/observed exit separately. As parents exit, Linux
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

The job uses the same monotonic integer clock domain as the core. Backward,
out-of-domain or overflowing deadline arithmetic appends `clock_failed`, freezes
at the last valid timestamp, and reduces the outstanding deadline to no later
than that timestamp (never extends it). It enters DRAINING/failed, expires that
obligation and attempts only verified owned targets. While the worker core is
live its existing clock-fault/kill semantics also apply; after EXITED only the
outer trace changes. If evidence output fails, retain raw partial diagnostics and
custody; do not let serialization failure suppress already-due owned termination.

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
sequence against its 6/7/8 table and performs the check above. Failed checks use
the existing core fault transition. ARM and ACK encoding, decoder limits, deadline
rules and output-fence semantics remain the single existing implementation.

## 7. Evidence and live-authority separation

Keep accepted runtime-custody v2 output and v1 compatibility unchanged. New
canonical descriptive records never create live owners. Shared validators reject
unknown/missing/duplicate fields, bool-as-int, floats, noncanonical bytes, unknown
enums, invalid digests, sequence drift and dangling references.

### 7.1 Source continuation and the immutable input cut

Immediately after validate_native_input completes wire5, obtain its actual
successful RuntimeCustodyRecord v2 and retain its immutable bytes/digest, journal
length J and lifecycle count L. Require all six input phases, same prepared object
and source/controller/runtime identities, no failed source/import audit and no
readiness. This is the upstream input cut. Neither later imports nor compilation
can rewrite those bytes, their objects table or their six-phase summary.

Add fixed CUDA-only _SourceSession phases `compiling_cuda_scene`,
`cuda_scene_compiled`, `compiling_cuda_lbvh`, `cuda_lbvh_compiled`,
`compiling_cuda_direct_light`, `cuda_compiled`, in that order after input_validated.
Changing to each compiling phase occurs only inside its actual ARMED segment.
The completed phase requires the corresponding successful DISARMED. No caller
phase string, backtracking, lane switch, repeat or skipped predecessor is accepted.
The continued source allowlist is the previous input allowlist plus exactly
`optical_worker.cuda_compilation`, `optical_worker.cuda_compilation_evidence`,
`optical_worker.compilation_contract`, `optics.cuda_extension_recipes`,
`optics._cuda_scene_source`, `optics._cuda_lbvh_source`, and
`optics._cuda_direct_light_source`. The CPU-only contract can be loaded in the
existing custody prelude without an additional guard. The others execute under
wire6 before its build body. All added source bytes join the frozen catalog and
AST policy and must pass the same captured loader/check_loaded_modules audit;
none are dynamically discovered from disk after admission. Unexpected imports
reject. Existing core/source/adoption audits remain required after each import,
load and completed segment. The new pure closure avoids timeline/renderer imports.

The fixed expanded external environment is frozen before initial input setup;
new cpp_extension transitive modules have exact `phase=runtime` rules. External
admission stays in runtime phase, with unchanged file/native-global/publication/
replacement semantics. No environment rows are appended after the input cut.
The same _RuntimeObservation owns object tokens and immutable issued journal
occurrences throughout. It continues recording external/import/failure facts;
do not suppress audits or misrepresent these as already completed input work.

After the cut, use the new compilation checkpoint/parser, not the old six-phase
checkpoint on a widened journal. Preserve old last_validated as the input cut;
maintain a separate last validated *descriptive record*, not another token map.
Old v1/v2 parsers continue rejecting the new kinds. The new parser first validates
the immutable input record, then validates the suffix with the shared existing
external/file/component validators and an explicit compiler guard context.
Factor those common validators without widening the old accepted wire grammars.
Suffix common kinds allowed are external, file_call, native_global,
module_replacement, interpreter_alias, returned_owner (file producers only),
failure and native_scope; no new runtime binding/upload/storage/input-validation
or phase_completed event may appear in this suffix. Compiler success events use
wire `6 + completed_recipes`; terminal_sync uses the sole disposal wire9.
After failure, only existing failure/retention observations may extend the prefix.

Native-scope facts retain their existing row shape and native-state validation.
This downstream grammar adds only sites cuda_compile, cuda_extension_create and
cuda_extension_exec, with before/after pairs joined to their actual lifecycle
compile_or_jit operation, exact recipe, target/runtime/stream and thread. It also
permits the original terminal_sync site. It cannot admit an unrelated helper site
or infer successful native restoration from absence of observations. Every entered
new native leaf must have a completed pair before a successful segment completes;
failed prefixes preserve unmatched before observations. Actual observer semantics
remain subject to the code-preflight/real-device proof in section8.

### 7.2 One live registry, exported downstream objects and retention

At the start of the first compiler segment, before source/tool import or build,
create one private CompileCustodyRoot, hold it in the source session, and directly
adopt it under role `cuda_compile_custody`, operation `cuda_compile_custody`.
Its CPU-only class is defined in the already captured bootstrap, so construction
does not require importing an as-yet unadmitted compiler module.
It is the guardian-held mutable container for all available compiler/module roots,
including partial returns and exceptions; it lives through all three recipes and
terminal disposition. Its identity is separately registered, not caller supplied.
Failure before this root is adopted produces no compiler-success record; retain
the original input cut and truthful failed-guard/raw diagnostic prefix instead.

Allocate tokens only through the original _RuntimeObservation.token(owner).
During continuation, before a new token is first referenced by an event, append
the exact object to this root's strong retained list and issue a fixed
`compile_owner_retained` occurrence giving its exact slot and root token. This
is internal collection, not an arbitrary callback. Root construction has the
single bootstrap exception: its token is introduced by compile_root_registered
after direct adoption. Returned native module retention still happens inside the
delegate-create wrapper before importlib metadata initialization. Other existing
external/source holders remain intact; root retention supplements them and does
not grant admission or clear poison. A failed adoption can therefore leave a
returned object held through the already adopted root without inventing a direct
guardian record for that object.

The new record serializes an `objects` projection from this SAME registry. Include
exactly tokens referenced by its suffix journal or derived recipe facts, plus
their root/retention links. Sort by token. Exact row fields are
`token,origin,first_journal_index,guardian_sequence,retention_event_index`.
origin is `input` or `continuation`. An input-origin token must exist in the
upstream objects table and copy its first_journal_index/guardian_sequence exactly;
its retention_event_index is null because its original custody is already proved.
It cannot be reintroduced with another meaning. A continuation-origin token is
absent from that upstream table, first_journal_index is its actual first reference
in the global journal (>=J), and retention_event_index points to its first exact
compile_owner_retained event, whose slot holds this actual object. Except for the
root, this retention event is also its first journal occurrence. Root's retention
index is null and its exact direct guardian adoption is mandatory.

For new objects guardian_sequence is the actual frozen adoption sequence if
directly adopted, otherwise null. Non-null sequences must resolve uniquely to
the full lifecycle's real owner_adopted event before the relevant use, with the
expected role/operation/scope; distinct objects cannot share a direct adoption.
Null is legal only with a valid prior retention edge to the adopted root for
partial/not-yet-admitted objects, failure exceptions and other held diagnostics.
It is NOT legal for an object bound to a successful generated role below.
The collector checks actual object identity
against root slot and guardian owner arrays before exporting that relation. It
does not guess a Python/native ownership edge from a name or type. Parsing checks
these recorded edges for consistency; it is not independent proof of live custody.
Input reused tokens continue binding their actual upstream objects; compiled
entries/module/spec/loader/owner must have the role-specific events below, so a
valid but unrelated input token is not interchangeable with a generated object.

Successful generated roles require the following direct guardian adoption anchors.
The producer retains the object in the root FIRST, then directly adopts that exact
object with the literal role/operation below, then emits the successful role step.
All these adoptions use the existing `HOST_OWNER` constant (wire string
`"HOST_OWNER"`) as device_scope; this describes the Python owner,
not native device safety, which remains governed by the native leaf observations.
`recipe_id` is the full fixed extension name from section2. Operation strings are
the exact concatenations shown, within existing identifier limits; entry name is
one of the literal names for that recipe. No caller supplies role or operation.

| Successful field | Guardian role | Guardian operation_id | First successful step |
| --- | --- | --- | --- |
| spec_token | cuda_compile_spec | recipe_id + '.spec' | create_entered |
| loader_token (standard delegate) | cuda_compile_loader | recipe_id + '.loader' | create_entered |
| module_token | cuda_compile_module | recipe_id + '.module' | module_returned |
| each entry_token | cuda_compile_entry | recipe_id + '.entry.' + entry name | entries_bound |
| compiled_owner_token | cuda_compiled_owner | recipe_id + '.owner' | owner_registered |

For every non-null successful field, its object row must have a non-null
guardian_sequence resolving to a distinct real lifecycle owner_adopted occurrence
with exactly this role, operation and host scope. That occurrence must have the
same recipe's active setup guard and precede the first successful role step, not
merely exist later in the full trace. The collector matches the actual guardian
owner at that sequence to the same object registered for the token; the parser
joins that sequence to the independently supplied lifecycle facts. Wrong role,
entry name, recipe, guard, time or sequence rejects. All five role categories and
each named entry have distinct object tokens/adoption sequences for these fixed
recipes. Later steps and derived recipe rows repeat the originally bound tokens.
Previously imported input tokens cannot fill these new roles: their copied input
adoption rows cannot have the new recipe/role/guard anchors. Runtime, target,
stream and source tokens may still be reused from the input record as upstream
references; they do not become generated module roles.

The loader is the exact standard ExtensionFileLoader delegate, even when a fixed
private wrapper is interposed to retain create_module's returned object. The live
binding also retains that wrapper and audits the actual spec.loader/module loader
relationship to it and its delegate; strings naming the type cannot substitute
for these checks. The producer checks spec/delegate/module identity and origin,
and obtains each entry from that exact module's fixed attribute before adopting
it under its specific name. An entry bound to another attribute/module or a
replacement object rejects before a successful step. CompiledCudaOwner still
requires the same private issuance registry; direct adoption alone grants no use.

If any direct adoption fails, the root retains the original object/exception,
the recipe fails, and no successful step binds that failed role. A module already
returned can thus appear only as a retained object in a failed prefix when its
module adoption did not complete. Earlier successful bindings keep their original
valid anchors; failed repeats those bindings and never promotes the new object.
Row guardian_sequence may be null in such incomplete prefixes, and exception rows
can remain root-only. Serialization must not fabricate an adoption to retain or
describe an object. This keeps partial custody monotonic while making successful
roles distinguishable with input/lifecycle/retention facts held fixed.

### 7.3 Compilation record and its cut

`q54.p12.4.cuda-compilation.v1` exact root fields are:
`schema_version,session,input_evidence_sha256,source_sha256,environment_sha256,
toolchain_sha256,target_ordinal,observation_cut,recipes,journal,objects,components,
lifecycle,input_source_trace,source_trace,summary`.
observation_cut has `input_journal_length,input_lifecycle_event_count,
input_source_trace_length,journal_length,lifecycle_event_count,source_trace_length`. The first two equal
J/L; the last three are the actual immutable snapshot cut. journal is only the
suffix J..journal_length-1. lifecycle is a complete independently validated
GuardedLifecycleEvidence whose first L events match the input lifecycle exactly.
components is the same full native_globals/module_replacements/interpreter_alias
projection, including exact unchanged input prefixes and every observed suffix
occurrence. source_trace is the complete captured started/completed source trace,
with the original input_source_trace prefix preserved and each new executed module in the
fixed continued source closure. Current source/import audits and source snapshot
digest must match this cut. The canonical digest binds all these fields.
input_source_trace is frozen from the actual source session at the J/L input cut,
before opening continuation, and its count matches input_source_trace_length.
Its names/order/state must satisfy the original input source audit. It is a
descriptive companion of that cut, not an invented extra field in v2 input bytes.

Journal rows keep the existing shape
`index,kind,guard,lifecycle_event_index,details`; indices are global. Common kinds
use the existing schemas/validators under section7.1's continuation guard context.
New exact kind/detail variants are:

- compile_root_registered `{root_token}`.
- compile_owner_retained `{root_token,owner_token,slot}`; slots are contiguous,
  single-use, immutable after issuance; the same object is never assigned two slots.
- compile_step `{recipe_id,operation,module_token,spec_token,loader_token,
  compiled_owner_token,entry_tokens,exception_token}`.

Each compile_step operation is one of `build_entered,generated,create_entered,
module_returned,exec_completed,entries_bound,owner_registered,hook_restored,
guard_completed,failed`. Each successful recipe follows that exact order.
generated means the fixed final helper was reached after synchronous build return.
Before create_entered all four object tokens are null and entry_tokens is empty;
create_entered binds distinct retained spec/loader tokens, module_returned adds
the actual module token, entries_bound adds the ordered `{name,entry_token}` list,
owner_registered adds the actual immutable compiled-owner token. Later successful
steps repeat exactly the preceding bindings. exception_token is null except for
failed, when it is a held exception observed for this actual recipe failure.
Failed repeats only bindings already observed (possibly all null); objects that
returned but could not be admitted remain in retention occurrences, not invented
successful module/entry steps. This preserves partial create/exec/registration
prefixes. A repeated failure can add held diagnostics but never another successful
step; reusing an unrelated earlier exception token without this failure occurrence
rejects. guard_completed occurs after the actual matching DISARMED; other success
steps require the matching live setup guard. Common import failures also make
compiler failure sticky, even when dependency code catches the exception.

Recipe rows are derived snapshots with exact fields
`recipe_id,guard_sequence,build_directory,source_files,generated_files,artifact,
entries,object_bindings`. File rows are `path,size,sha256`, entries rows are
`name,entry_token`, object_bindings has
`module_token,spec_token,loader_token,compiled_owner_token`. They must equal the
journal's last observed bindings, never introduce unreferenced tokens. Uncreated
artifact/object fields are null; partial generated files are the observed list.
Summary is replayed `{completed_recipes,failed,boundary}`. compiled_cuda requires
all three recipes, complete direct-loader/hook/source/import/native audits and
joined successful guards; any missing/reordered/swapped record rejects. It gives
no readiness or execution authority. The separate supervisor join additionally
requires compile_quiescent for each completed recipe before its core COMPLETE.

Final evidence selection retains either a current validated compilation cut or
the actual prior validated compilation cut with a diagnostic; if none exists,
retain input cut and report compilation unavailable. Selection fields are
`schema_version,selection,record_sha256,selected_lifecycle_event_count,
observed_lifecycle_event_count,failure_diagnostic_present`, version
`q54.p12.4.cuda-compilation-selection.v1`, selection
`current,prior_validated_cut,unavailable`. Unavailable has null record/selected
count. Never call the old input record current compilation evidence. The old
C2d3 final() selection remains unchanged for input-only drivers.

### 7.4 Compiler job evidence

`q54.p12.4.compiler-job.v1` exact root fields are
`schema_version,session,supervisor,worker,guard_evidence_sha256,events,summary`.
Process identities are `pid,starttime_ticks,sid`; supervisor/worker are required
once launched, otherwise null. Events are `index,at_ns,operation,details`.
Closed operation/detail variants are:

- subreaper_ready `{}`; worker_spawned `{identity}`.
- deadline_snapshot `{guard_event_index,sequence,source,deadline_ns}`.
- compile_quiescent `{sequence}`; draining `{reason,source,deadline_ns}`.
- deadline_expired `{source,deadline_ns}`; clock_failed `{reason}`.
- child_adopted `{identity}`; signal_attempted `{identity,signal}`;
  signal_failed `{identity,error_kind}`; child_reaped `{identity,wait_status}`.
- no_children `{}`; pipes_eof `{}`; observation_failed `{error_kind}`; closed `{}`.

Deadline source is operation/ack/terminal_grace/clock_fault, null only for a live
no-obligation snapshot. Draining freezes a non-null source/deadline. Expiry is
single-use, at or after that deadline, makes failed sticky, and must precede any
signal_attempted. Clock failure reason is backward/out_of_domain/deadline_overflow;
it reduces the deadline as specified in section6 without inventing time. Signal
is SIGKILL only. Error_kind is missing/permission/identity_changed/proc_unavailable/
resource_exhausted/other_os. Draining reason is worker_exit/guard_fault/parent_lost/
deadline_expired/observation_failed/clock_failed. Later faults cannot create a
second draining event or renew its obligation; they append their specific event.
Summary is replayed
`{state,failed,deadline_ns,deadline_expired,worker_reaped,no_children,pipes_eof}`.
Nonzero worker exit, guard fault, expiry, clock/observation/signal error set failed;
closed with failed true means observed cleanup, never a successful compilation job.
Identity reuse never merges rows; events after CLOSED reject. ECHILD can be recorded
only after worker reaping, and closed requires ECHILD plus actual pipe EOF. Core
evidence remains byte-stable after its process_exited, even if outer expiry and
child-only signal/reaping events continue. An earlier pending-ACK deadline is
retained, not replaced by the longer operation deadline.

All cross-record joins use session, exact input/cut digest, source/environment,
guard identities and same-run process facts. Prefixes can describe incomplete
execution; no parser or test report synthesizes missing facts or issues authority.

## 8. Required verification and implementation order

1. Implement fixed recipe factoring, schema/owner registry and supervisor logic;
   run meaningful CPU counterexamples through production boundaries. Fresh
   independent code preflight must approve exact source/policy/catalog and fixed
   runners before any new backend compilation or native observer execution.
   Reproduce v1 R1/R2 with the preserved independent probe, then prove the new
   integrated driver uses wire0..5 unchanged, intercepts compile6/7/8 and terminates
   at9. Reject shifted/skipped/duplicate sequences. Prove leader-first exit with
   both operation and earlier pending-ACK deadlines: immutable exited-core bytes,
   outer expiry/child-only attempts, no duplicate signals or renewed deadline.
   For R3 exercise actual token collection/source continuation/parser together:
   legal reused/new-token positives and partial-create/adoption/exception prefixes;
   missing rows, wrong first occurrence, wrong guardian sequence/retention slot,
   cross-session, module/entry swaps and invented post-cut tokens must reject.
   Keep independent input and lifecycle fixtures fixed in these mutation tests.
   Specifically preserve v2's root-retention countermodel: a root-only successful
   binding must now reject. Add a legal lifecycle-anchored success and coherent
   spec/loader and same-recipe entry swaps, updating every repeated/derived binding
   while keeping lifecycle and retention anchors unchanged; both must reject.
   Also test valid unrelated retained/input token substitution, wrong recipe/role/
   entry operation, and failed direct adoption with the original returned object
   still held and no successful role occurrence. Schema models during design may
   illustrate these rules but do not count as actual collector/parser verification.
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
