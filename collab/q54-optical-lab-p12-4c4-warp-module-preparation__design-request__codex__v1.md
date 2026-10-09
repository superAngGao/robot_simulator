# Warp compilation and module custody for the existing setup driver

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded design supplement to parent C4c2 v3 section 5
Author: Codex
Version: v1
Date: 2026-10-09
Status: in_review
Related Files: q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md;
  q54-optical-lab-p12-4c4c2d-remaining-setup__gap-scan__codex__v1.md;
  q54-optical-lab-p12-4c4-fixed-cuda-artifacts__acceptance-review__codex__v1.md
Owner Summary: Use Warp's existing in-process compiler under existing setup
  guards, a fresh private cache per worker and explicit retention of the actual
  ModuleExec objects. Reuse the three existing optical kernel modules. This
  closes the remaining compilation-policy gap needed to construct the real
  render graph; it creates no compiler process manager or alternate renderer.

## Scope and inspected base

Authoring base is clean local 84ed0e7 (implementation/c4-supervised-waits).
Actual remote isolated checkout remains clean at
66240e4fedc24ad8a5109690e3b1e78b450e2f62. Fixed CUDA artifacts are accepted at
that code; parent C4 completion and rendering/performance are not accepted.
This supplement decides Warp cache origin, executable ownership and source
consumption, the open Warp portion of G1 in the accepted gap scan. The parent
already specifies two readiness objects, ten warmups, one final seal/barrier,
Warp R0/R1 transactions and common arena reuse; those contracts are unchanged.

Read-only copies of the installed Warp 1.12.0 source were freshly fetched from
/home/ang.gao/miniforge3/envs/robot_sim/lib/python3.12/site-packages/warp into
C:/Users/AG/新建文件夹/c4_assets/warp_setup_source. Hashes match the prior scan:

| File | SHA256 |
| --- | --- |
| _src/build.py | 5993e58b44a916fc12f35afde1cededc59ab1db7df7a7eecd75551191d9197df |
| _src/context.py | eb099d908c50effc3cacbf7ecc408be4187ca9a6a6f72e1e85c70676961524ee |
| _src/codegen.py | c5538f63618a64af90cf6e0eb629df0d47000a8fe7e57844a0c4b8fd6b41bbf2 |

Installed build.py:40-115 calls in-process NVRTC through the frozen native
runtime. Module.load in context.py:2935-3051 compiles or reuses files, then
stores a ModuleExec in module.execs keyed by device context and block size.
ModuleExec.__del__ unloads its native handle. Public load_module returns no
owner; its explicit Module argument and max_workers=0 select serial loading.
codegen.py:1171-1180 reads Python source via inspect/linecache. Current captured
get_source/get_data alone do not force linecache to use held bytes if the real
filename is readable. These are source facts, not executed compilation proof.

## One cache policy and one compiler route

The captured Warp worker creates one fresh directory with tempfile.mkdtemp,
prefix robot-sim-warp-, explicit parent /tmp on the currently selected Linux
deployment. This fixed path is an operational workspace choice, not a new
memory/performance limit. Creation is exclusive with the standard private
permissions; require a resolved nonsymlink directory and initial emptiness.
No caller-supplied existing cache, copied binary or shared user cache is admitted.
Retain its path in the adopted source session before any fallible configuration.

After importing Warp but before the existing wp.init(), set kernel_cache_dir
to that directory, cache_kernels=True, llvm_cuda=False,
use_precompiled_headers=False, cuda_output="cubin", load_module_max_workers=0,
and quiet=True. Reject preinitialized Warp and unexpected prior configuration;
record the original supported defaults and these explicit settings. Set all
configuration before optical kernel decorators execute. Disable PCH to keep
additional compiler cache artifacts out of implicit toolkit-selected locations.
The selected target is H200/sm_90; compilation must produce its native CUBIN,
using the existing module defaults (release, default optimization3,
fast_math=False, fuse_fp=True, block_dim256, backward enabled). Preserve all
other numerical and launch settings. Freeze the complete observed global and
per-module compilation options in the implementation's literal contract and
check them before compilation, recording and graph admission. No configuration
discovery or mutation during sample submission.

The standard Warp compiler alone creates source/binary/metadata files under
this fresh root. Cache reuse is permitted only in the same session after its
own observed successful compile, with the exact held ModuleExec and unchanged
module/options/artifact facts. A new worker always starts fresh. Cache files
are not Python extension modules and are not added to import admission.
Driver-internal caches remain within the existing trusted driver boundary;
they are not claimed to be enumerated or disabled by this directory policy.

Use only the ordinary installed Warp CUDA path. No subprocess/compiler service,
raw repository runtime.core/CDLL call, alternate LLVM backend, retry, fallback,
or general compilation callback is introduced. Compilation belongs to untimed
preparation in both lanes; this policy cannot justify charging it to one lane's
future renderer timing. Workers retain cache files on exit/failure for evidence.
They have no filesystem-cleanup destructor. After observed process exit, the
existing acceptance owner may archive the exact recorded directory; any later
deletion must check that exact owned path. This proposal introduces no sweeper.

## Frozen source and installed inputs

Before executing a captured repository module, the existing source loader
populates that module's exact origin in standard linecache from its already
held UTF-8 bytes. The entry uses mtime=None so checkcache cannot replace it
with current filesystem contents. Retain and audit its exact lines/content
and origin through the same session. Overwrite a preexisting stale entry at
first captured execution; after execution, any replacement/mutation is a
terminal source violation. Do not reread source from disk for Warp codegen,
create another loader, or grant general reflection/native-call exceptions.
This applies to captured repository sources so referenced functions receive
the same protection; it does not rewrite installed dependency source.

Extend the existing selected Warp runtime-environment profile with explicit
file rows for the actual compilation input closure. Use the existing offline
wheel inventory tool to author facts, including installed warp/native headers
and required native compiler support bytes; source-inspect actual includes and
loaded-library selection. The current profile contains only three native
source files and is insufficient to claim header closure. Commit a literal
reviewed file inventory; runtime directory scans cannot grant admission.
Reuse existing file kinds/schema/verification wherever sufficient. Any newly
loaded dependency follows existing external admission rules. No dependency
installation, version change, broad native-directory permission or toolchain
attestation system is authorized. The implementation review must inspect the
exact inventory and selected compiler flags before native acceptance.

## Ordered compilation and actual ownership

Add one private ordered preparation entry after the existing input_validated
phase. It requires the exact live Warp session, controller, runtime binding,
prepared arrays, target and stream; it is single-use, including after failure.
Import the existing optics.device_scene, optics.device_bvh and
optics.warp_native_render through their captured literal source closure. Reuse
their kernel/function objects; record all definitions the compiler consumes,
and separately identify the existing scene-update, leaf/internal-refit, raygen,
traversal/reset and four shade ABI entries needed by the later graph. Compilation
of other existing kernels in these same modules grants no execution permission.

For each of the three exact Warp Module objects:

1. Retain/adopt the module, its kernel/function owners and its live execs mapping
   before loading. Require no preexisting executable/failed build for that
   module; validate literal module identity, options, source and target binding.
2. Enter an ordinary nonnested setup guard using the existing configured budget.
   Mark the compile/load leaf as exact target before calling it. Within
   wp.ScopedDevice(target) and wp.ScopedStream(stream), call
   wp.load_module(module=actual_module, device=target, recursive=False,
   block_dim=256, max_workers=0). No new watchdog or restarted deadline.
3. Still inside those contexts, immediately retain the actual object in
   module.execs[(target.context,256)] before inspecting/adopting it further.
   Check its nonnull native handle, exact target, module hash and metadata.
   The preheld module/mapping retains any published executable even if the
   public load wrapper raises after publication but before returning.
4. Check the expected binary/metadata paths derived by the pinned Warp recipe,
   contained within this worker's cache, plus their bytes/digests, generated
   source, options and module hash. No arbitrary cache scan discovers modules.
   Successful guard completion must be observed before issuing compile facts.

A private session-held module-set record joins source/environment/cache origin,
three module identities, options, artifact facts, actual ModuleExec identities,
target/stream and successful guard identities. Construct it only after all
three succeed. Copies, parsed payloads, names/hashes or caller booleans cannot
issue it. Reuse the existing session adoption/identity registry patterns, not
a second guardian. The future graph holds this exact record and command
module_exec identities. Before each recording batch and final graph validation,
reject mapping replacement, unload, handle/hash/target/options/ABI drift or a
new variant. Check all newly recorded commands against the retained set.

Before a loader return, native temporary allocations and constructor cleanup
retain the existing trusted dependency exception-safety boundary. Python
retention cannot claim custody over an unpublished native handle. On any
failure, preserve the original exception, retain all available roots, mark
unknown scope/incomplete coverage as required and terminate under the original
failed guard: no COMPLETE, new emergency guard, retry or ordinary module unload.
On hang the existing supervisor kills the worker; the durable last worker
prefix may be incomplete and must not be fabricated as a terminal record.

Existing input-phase custody versions retain their meaning. Initial compilation
acceptance uses downstream raw observations and live owner checks bound to the
original lifecycle/guards. Any production serialized compilation record added
for later graph evidence must be versioned and descriptive only; parsing never
issues the live module-set authority or readiness. Do not reinterpret input
validation as final execution readiness.

## Integration and required verification

Once compilation succeeds, parent v3's sole readiness certifier can issue
truthful R_setup (warmup0). Existing factory preparation and command recording
must reuse these exact modules/executables. Record_cmd entry initialization
remains guarded and each returned command is retained before another fallible
leaf. It must not silently discover a new module/variant. Final factories use
NEW R_final after the ten real warmups, while retaining provisional owners.
Reuse NativeOpticalArena for the subsequent Warp R0/R1 graph; no duplicate slot
state machine, upload path, kernel implementation or runtime bridge.

Implementation/code review and native acceptance must cover:

- Captured held source is consumed even with stale preexisting linecache or
  changed on-disk source; later linecache/options/module/entry replacement fails.
- Cold real Go2 preparation at960x640 and1920x1080 on target1/ambient0 with
  actual source-backed native scope/restoration observations and all three
  compiled modules/artifact facts; same-session module reuse preserves owners;
  another worker has a distinct fresh cache. No optical-output claim yet.
- Preexisting cache/exec, outside-root or altered artifact, unsupported backend,
  changed target/stream/hash/ABI and copied/foreign record reject at their actual
  boundary. Tests must hit production validation, not parallel fixture logic.
- Original-guard compilation/load failure, failure after executable publication
  before public return, returned-owner adoption failure and hang. Validate all
  retained partial roots, zero next leaf after failure, no new sync/guard,
  actual worker/supervisor exit and unrelated-process preservation.
- Regress existing captured input paths, fixed CUDA preparation and source
  admission. CPU mocks do not establish CUDA scope or native compiler behavior.

The whole-C4 obligation remains: one-slot graph and Warp timeline/arena, ten
real warmups, final readiness/recording/seal/barrier, integrated CPU-oracle
render correctness and lifecycle matrices, C4c3/c4, C4d/B3, then systematic
refactor and fresh whole review. This supplement neither opens product PREPARE
nor benchmark/result/baseline/selection. No performance numbers are authorized
by compilation acceptance. Temporary tests/records follow existing cleanup
gates; the losing optical lane is removed only after the later backend decision.
