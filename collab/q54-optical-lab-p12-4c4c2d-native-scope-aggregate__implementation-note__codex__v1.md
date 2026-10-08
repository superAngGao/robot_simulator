# Native-scope v2 implementation and preflight candidate

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: implementation-note and code/native preflight request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: native-scope-aggregate design-request/review v2;
runtime-faults test-evidence v2; optical_worker/runtime_custody.py;
native_bootstrap.py; native_readback.py; tests/support/native_runtime_process.py;
tests/support/native_scope_acceptance.py; tests/gpu/test_native_scope_runtime.py
Owner Summary: Implement the approved descriptive observation and Warp readback
repair, then independently inspect exact native calls BEFORE running the new
positive Warp observer. Existing85 fault passes are old-production evidence;
new native observations have not executed. C2d3/C4 remain unaccepted.

Base is c243875 (packet gives full base/HEAD). Approved design is07a76ff,
SHA2569e678c7aaffa1147b49a6f1f3b946619f0c316e3bc68883ff0fa9a4d1ed9a0ad;
unchanged independent Approve report SHA256
563d8661a8b1463cef09921d07b58244c4b63f5d00d43421475798d21b5c3818.
No broader architecture/policy change is intended.

## Implementation

The shared parser explicitly accepts historical v1 and new v2; native_scope is
v2-only. Existing top-level/component/summary and sidecar/input versions remain.
The bootstrap emits v2 and still collects only CPU facts. The new event checks
exact per-lane representation, pinned native-library descriptor, existing
module/root/target/stream/binding tokens, original guard/phase, actual leaf
descriptor and lifecycle cut, before/after nesting, and narrow emergency scope.
Pristine stream observations precede retained return; after immediately follows
that actual return. Binding follows the exact binding journal/adoption. Reused,
future or stale leaves cannot be justified by journal counts. A completed normal
phase cannot retain an unfinished pair; a failed prefix can. Contradictory
completed native state is descriptive, not a positive verdict. A cold CUDA
logical derivation cannot be claimed if its raw/context prerequisites failed.

Warp readback adopts its retention list, declares indexed_validation_scope and
executes the original single readback body under runtime.ScopedDevice(target).
The held report returns only after actual scope exit. The CUDA body uses that
same factored implementation without adding a Warp scope. No extra ARM/deadline,
sync, allocation loop or validator. Warp's expected runtime-import/init leaf
count increases by one. Captured bootstrap/source policy/catalog are repinned.

The fixed sacrificial driver has an opt-in _NativeScopeObserver, not a production
callback. Default runtime_success/fault selectors do not install it. New fixed
selectors install wrappers after the original trusted baseline. They delegate
the original runtime leaves, owner retention, returned-owner append and binding
append, and then observe actual primitive state inside the existing ticket.
The observer retains libraries/functions/returned owners before fallible work.
All GPU queries remain in this test driver, never in the collector or parser.

CUDA uses the prior reviewed libcudart/C10 methods and queued-callback/seed/
unexecuted-Triton checks. Warp uses typed direct libcuda result-checked getters;
CUdevice handles are mapped through cuDeviceGet. Native current context and
default metadata are separate. The target native stream getter is used only
after actual get_stream returned the target's exact initialized _stream/_context.
Pinned source must substantiate that this original path established the native
context-info cache. Libraries require hash + existing mapping + RTLD_NOLOAD,
with no fallback load. Original Warp scopes are delegated once and observed
outside; helper snapshots occur inside. Failed bodies unwind without later
native queries. The one terminal callback gets before/after observations under
its original emergency ticket, before existing release.

The positive acceptance helper checks every actual relevant leaf occurrence,
mandatory scope/stream/terminal pairs, real full terminal attachment, same-run
external process identity, primary-context flags, lane-specific ambient state
and current-stream restoration. It never issues resource/readiness authority.
The opt-in GPU file collects16 cases: both lanes' four canonical workloads and
eight fixed observer/counterexample cases. Its live OS recorder is mandatory.
No case has executed; code/native preflight must precede the new native runs.

Two explicit negative controls deliberately alter only the reviewed boundary:
CUDA stream-only copy omits its outer indexed device scope; Warp readback omits
the new outer scope and executes the original body/helpers. They are fixed
sacrificial selectors, never positive delegation claims or product switches.
Other observer faults raise before a getter, after real stream retention, or
after a real allocation's snapshot append. A fixed raised getter fault is not
claimed to be an actual nonzero native CUresult; separate CPU models test that
result handling distinguishes errors from successful null pointers.

## Executed checks and honest limits

Local Python3.14: initial parser/manifest/readback83passed3.47s after correcting
a test-only missing fixture lane attribute (earlier2failed79passed preserved in
tool output). Source-policy/sources/bootstrap/runtime-binding/readback/parser/
manifest suite389passed13skipped148.88s. Skips include required POSIX/release,
symlink and configured real-Go2 cases; they are not passes. Later focused
observer/parser/readback56passed4.04s. Three observer CPU models cover a native
getter CUresult error versus null, wrong context despite normal scope exit,
and no complete snapshot/retry after a failed getter. They use the real guarded
controller and CPU observation journal but mock native state, never GPU proof.

Pinned primary Python3.12.13 ran a CPU-only scratch copy of the new codec against
historical actual provenance, inserting EXPLICITLY SYNTHETIC native rows.
Final native_scope_codec_cpu_v4.log: **32passed6.56s**, no NumPy/Torch/Warp import.
These exercise Warp/CUDA binding/state forms, lawful returned/adopted/nested/
emergency observations, the exact future-leaf/stale/same-kind attacks, missing/
crossed/duplicate pairs, object substitution and derived/observed confusion.
They do not migrate old evidence into new real executions. Old archive hash is
fixed; trusted source/case/interpreter/environment inputs stay fixed in attacks.

Earlier CPU scratch runs: v1 22passed2.01s; v2 19passed then fixture error because
an old CUDA request references an external profile rather than a repository
filename; corrected original-path/digest handling produced v3 28passed5.12s.
V4 includes exact stream-return ordering and both-lane forms. Raw logs/scripts
remain in sibling c4_assets; scratch is outside the remote candidate.

This candidate still needs independent code/native preflight, all16 actual
new native cases, required failures/persistence and affected regressions, then
fresh whole-C2d3 review. The preflight reviewer should test the implemented
joins, retained-return/failure ordering, ABI/source/cache feasibility and exact
producer delegation. Any required missing proof must stay explicit; author
test counts and earlier design approval are not implementation approval.

C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host,R2,
shipping remain CLOSED. C4 is the continuous objective, not this schema slice.
