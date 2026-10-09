# CUDA compilation draft: integrated fixed driver and nested observer

Initiative: Q54 / P12.4 / C4c2 / remaining parent C2d
Stage: implementation note, not native preflight or acceptance
Author: codex
Version: v3
Date: 2026-10-09
Status: draft
Related Files: compilation-admission design-request/design-review v3; implementation-note v1/v2; systematic-refactor implementation-note v1
Owner Summary: A fixed compilation worker now reuses the existing six-phase input driver, then calls the production three-recipe continuation and terminal path. Its dedicated supervisor uses the compiler-tree owner, and the observer distinguishes nested loading from outer scope restoration. These are unreviewed source/CPU changes; the complete deployment and native verification remain outstanding.

## Work relative to 9ca40365642e48931997477e235c631c105012f6

`tests/support/cuda_compilation_process.py` adds only fixed worker/supervisor
invocations. The worker delegates to the existing runtime driver's explicit CUDA
compilation mode, preserving the original input-only invocation and six source
session calls. The continuation uses the shared production phase contract and
checks the terminal sequence after three completed compile guards. It preserves
the original input evidence and selects a separate compilation cut at termination;
it does not replay the new suffix through the old input-only grammar.

The supervisor launches the sole fixed worker through `_CompilerJob`, drives the
existing guard transport and the sole process owner until observed job closure,
and retains ownership across post-spawn initialization/transport/evidence errors.
It records the actual guard/job evidence, bounded output and available sampled
process metadata. It never calls Popen poll/wait/communicate, a numeric-PID kill,
killpg or an unrelated process cleanup. Persistent OS failure still prevents
closure; there is no new timeout that declares cleanup successful. Compilation
fault selectors and the real process-case driver remain unfinished.

The existing native observer gains the three fixed compilation helper wrappers.
Before/after the outer helper, it still requires ambient logical device0 and the
original stream state. Nested create/exec observations require the target logical
device1 and held stream, plus actual nesting. The same native getter implementation
and held input runtime objects serve both paths. No new ctypes ABI is introduced
by these wrappers, but changed observer source still requires renewed preflight.

Two process-adapter defects were found while connecting this path:

- The Popen assignment previously occurred after its constructor returned, despite
  the comment claiming partial-construction retention. The object is now held
  before initialization. Constructor failure after possible fork remains owned;
  pipe setup and identity checks cannot authorize an unanchored ARM.
- Job CLOSED previously preceded pidfd close. A close exception then attempted to
  append a failure to an already closed record. Closure now follows successful
  handle closure; an ambiguous close failure remains DRAINING and is not retried.

Process sampling now runs during active compilation as well as at COMPLETE, so
it can observe packaged compiler/Ninja processes while they exist. Sampling is
diagnostic, not an exhaustive executed-command log or a source of signal authority.

## Verification and precise limits

Windows C:/Python314/python.exe -B; plugin autoload and pytest cache disabled:

- The two new counterexamples were run against the exact original `launch` and
  `service_processes` AST bodies from Git9ca4036. Both failed: the constructor saw
  `job.process is None`; the close failure produced `event after compiler job
  closure`. This in-memory replay changed no tracked source and forked no process.
- The repaired process-unwind suite passed4 cases. Popen initialization and fd
  close are CPU stand-ins; no subreaper, pidfd, real reaping or signal proof follows.
- Actual observer wrapper/validation code passed the modeled nested success,
  wrong nested stream and missing outer restoration cases. The latter two poison
  the lifecycle before ordinary completion. Native getters are not executed.
- Combined process-unwind/job/transport/observer/compiler-unwind/custody checks:
  **65 passed, 9 skipped in2.54s**. The9 skips require the pinned Warp source file,
  absent on this Windows host. They are not a native pass.
- All three changed driver/adapter files parse as Python AST. `git diff --check`
  passed. Tests returned and their peers joined. No GPU/compiler/supervisor process
  was started, no remote tree changed and no shared dependency/PCH file changed.

No new native execution is authorized by this note. The source/catalog pins are
still intentionally stale pending the integrated candidate. Complete toolchain
inventory, external build-helper closure, actual collector/source/parser proof,
fixed failure cases, source-policy updates and fresh exact-code/ABI/observer
preflight remain necessary. Then run the required real Linux and CUDA matrices
and fresh whole-checkpoint review. The parent C4 objective and whole-C4 systematic
refactor remain active; this driver snapshot does not complete either.
