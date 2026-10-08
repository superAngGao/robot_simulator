# C2d3: held native-global observations without fictitious creator ownership

Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design-request v2; native-self-alias design-request v1; cython-publication/owner-probe test-evidence v1; native-global-publication gap-scan v1; optical_worker/native_bootstrap.py; optical_worker/runtime_environment.py
Owner Summary: Add one separately described observation-and-custody relation for the three required fileless Cython globals. The exact trusted native interval is observable; an exact native instruction creator or Python owner path is not. This closes a real Go2 input prerequisite, not a renderer, benchmark, product activation or C2d3 acceptance gate.

## 1. Scope and trust

Implementation baseline: `62cf8ff0b1492163d166e243ffde7347fd4e03d9`.
The independent gap report is archived separately, unchanged. It is a gap scan,
not approval of this design or the incomplete C2d3 code.

Runtime-adoption v2 sections 3/4.2 remain authoritative for native exception
safety, file modules, creator-attribute publications and creator-self aliases.
This supplement adds a distinct `native_global` outcome. It does not weaken
the existing ownership relations or relabel an observation as one of them.

The trusted deployment boundary includes the exact immutable native binaries,
their dependencies and interpreter. We observe an object during a delegated
native interval and take our own strong reference. We do not prove which C
instruction created it, that a module exports it as a Python attribute, or that
the native runtime maintains any particular hidden reference. Arbitrary hostile
code running inside that same admitted binary/interpreter is outside this claim.
No native interposer, package patch, install, downgrade or new loader is added.

The required benefit is allowing real canonical Go2 input materialization under
honest custody so its actual upload/readback and failure tests can proceed.
This mechanism is untimed setup. Native-ready R0/R1/R2 boundaries, shipping,
renderer/acceleration attribution and all frozen gates are unchanged.

## 2. Evidence motivating the relation

The actual captured Go2 Warp 960x640 attempt at c8b404c failed with unlisted
module publication. Its failure graph held the exact `cython_runtime`,
`_cython_3_2_8` and `_cython_3_2_5` objects. All lacked ordinary file/spec/loader.
The worker exited 74 and was reaped without deadline expiry; no upload passed.
Direct attribute/GC probes did not demonstrate the existing owner relation.

Fresh independent CPU observations demonstrate why a return-only sys.modules
delta cannot establish an exact creator: each object can already be visible
when a nested dependency is requested before its loader runs. All observed
points are reachable at the current captured finder's common request entry,
including its stdlib branch; no general stdlib loader wrapper is needed.
These CPU observations support this bounded design but do not certify the
captured worker or GPU matrix.

The proposed literal deployment relations, all materialization/exec, are:

| Global key | Enclosing native interval |
| --- | --- |
| `cython_runtime` | `numpy.random._bounded_integers` |
| `_cython_3_2_8` | `numpy.random.bit_generator` |
| `_cython_3_2_5` | `scipy._lib._ccallback_c` |

Independent ordinary observations first saw them respectively at requests for
zlib, numpy.random._common and scipy._cyutility. These child names are observed
facts, not creators or a hard-coded import-order assumption for another process.
Implementation must demonstrate the frozen enclosing intervals using the actual
captured canonical paths before any successful admission claim. A different
required interval fails and requires a reviewed versioned table, never discovery
that adds permission. No new Cython version/name is admitted automatically.

## 3. Deployment schema

Advance the active environment schema to `q54.p12.4.runtime-environment.v4`.
Add the required top-level `native_globals` list; preserve existing row schemas.
Each new row has exactly `name, observer, operation, phase`, sorted uniquely by
name. `operation` is exactly `exec`; this change grants no create-time relation.
`observer` references an admitted non-package site-root extension file module.
The row phase equals the observer's file-module phase. Names are exact module
identifiers; no prefix, version pattern, alias of another rule, file/namespace/
publication/absence/disabled collision or fabricated parent is legal.

The binary path/hash and interpreter/environment provenance stay in the one
existing inventory. Do not duplicate hashes into a second policy authority.
Rows are frozen trusted launch configuration, unavailable through worker IPC.
Parsed rows/digests do not construct observed objects or permission registries.
Synthetic schema fixtures may use synthetic native delegates for local mechanics;
they never certify the deployed relations.

Old v1/v2/v3 profiles remain historical immutable files and are rejected by the
active v4 parser. Add versioned active profiles and update their selectors,
canonical digests and cleanup-ledger paths. Empty tables are explicit for
workloads that need no global. No automatic version migration or blanket module
inventory admission is permitted.

## 4. Single observation path and state machine

Extend the existing `_ExternalAdmission` and `_CapturedSourceFinder`; no second
finder, alternate materializer, user callback or general-purpose tracer.
Pin the supported execution model to the configured CPython worker and the single
import thread that configures admission; record its actual interpreter thread
identity without initializing a backend. Every supported checkpoint requires
that same identity. Observable foreign-thread entry or checkpoint reentrancy
poisons rather than sharing mutable stack state. Normal nested imports remain
legal. Native background mutations between checkpoints stay a trusted dependency
obligation, not a guarantee obtained from the GIL.

Maintain an internal identity-bound delegated-call frame for each external
create/exec, including Python file bodies. Each frame retains the exact loader,
delegate, spec, file record/module when available, operation, phase and parent.
This is execution state issued by the actual loader, not a supplied string.
A private identity registry retains each exact frame and its immutable binding;
copied or re-bound records cannot substitute. Frame states distinguish wrapper
checks, delegated execution, suspended request handling, returned and closed.
Finder handling suspends the current delegate after its entry checkpoint and
restores it only on successful return. Loader wrappers suspend their parent
while validating/recording. Mark return before fallible wrapper work. A finder
request during wrapper validation is not native execution and cannot establish
a first observation. An exact normal-return checkpoint is the other allowed
observation form. Exceptions never masquerade as normal returns.
Before each declared observer exec, record actual absence of its unobserved
global keys. This is the mandatory preceding absence checkpoint; a newly seen
value at that entry rejects. A frame is closed once and never reused.

Run one common native-global checkpoint:

1. At the captured finder's common request entry, before its stdlib/external/
   repository split or any child resolution/delegation.
2. At each external create/exec entry, before pushing its new frame.
3. Immediately after a delegated create/exec returns, while that exact frame
   is still current, before fallible return validation, explicit adoption,
   publication reconciliation or completion records. A returned native module
   itself must first be placed in the held file record, as already required.
4. At the existing segment audit, with no active frames; audit is validation
   only and cannot discover/admit a previously unobserved global.

These checkpoints examine only the finite frozen global-key set, not a complete
environment/hash traversal. The existing final audit still rejects other
unlisted publications. At each checkpoint, retain every newly available value
in the already-owned admission graph BEFORE any fallible validation/adoption/
event writing. A None, wrong-type or replacement value is retained then rejected;
the previous object remains held too. Failure intake retains all available
unlisted deltas under the existing rules. Never invoke module __getattr__.

States per row:

`ABSENT -> OBSERVED_PENDING -> ADMITTED`, or any state -> sticky session failure.

At cold configuration every configured global key must be absent, including
None. ABSENT may become OBSERVED_PENDING only at request entry or native exec
return while the CURRENT (innermost) external delegated frame is exactly the
row's native observer/exec/phase and actually executing. A Python frame or other
native interval cannot borrow a matching ancestor. Entry checkpoints attribute
observations to the still-current suspended frame, never the child about to
start. Create-return and no-active-frame observations fail.

Require exact ModuleType and raw name equal to the key, with file/spec/loader/
package raw values absent or None and no package path. These are format checks,
not provenance proof. Bind the pending record to the exact object and frame,
not only its name, id integer, hash or copied dictionary. Retain the native
loader/spec/module roots and explicitly adopt the global object only after it
is already held by admission. The record stores no `creator` or owner path.

Nested requests/delegated calls may observe the already-held pending object.
They must preserve its exact identity and cannot admit it, replace its original
frame or emit another first-observation record. A nested call that first creates
another permitted global gets its own frame. Shared later use by other
extensions is permitted only as unchanged identity reuse; it creates no new
permission or inferred native owner edge.

OBSERVED_PENDING becomes ADMITTED only at its own bound exec's successful
return, after all normal file/module/delegate checks, nested sticky-failure
checks and ordinary publication reconciliation succeed. Before recording that
file as completed, require every global assigned to its exec is present and
pending in that exact frame; then complete their admission once. A missing
required global is failure. A configured observer never executed may leave
its rows ABSENT; unused rows are not observed evidence.

Every later checkpoint requires pending/admitted keys still point to the exact
held object and preserve raw metadata and identity-bound records. Pending may
survive only while its original frame is on the live stack. At segment audit,
the stack is empty and no pending state is legal; requested observers must be
completed with admitted rows, unrequested observers' rows must remain absent.
Audit never reconstructs missing records or emits admission events. Repeated
audit is read-only. Removal, replacement, pending-frame copying, record removal,
wrong phase, changed rule/index/delegate, duplicate completion or table drift
fails sticky before successful normal segment completion.

## 5. Failure, limitations and lifecycle

A partial native exception or any checkpoint/adoption/record/audit failure uses
the current armed segment's existing quarantine and hard-exit path. Preserve
actual globals, original and late exceptions, available file roots and any
pending frame graph before unwinding. Later parent work after a swallowed
nested exception can append available custody but cannot admit another object
or clear failure. No COMPLETE, second ARM, retry, target sync or ordinary release
is allowed after a failed normal segment. Existing UNKNOWN native failure and
foreign-device scope rules remain conservative and unchanged.

Once poisoned, permitted terminal stdlib reporting must remain possible under
the existing frozen topology. It may extend failure retention but cannot run
normal global adoption or turn pending into admitted. Do not make a new live
check in the common finder block all terminal stdlib imports recursively.

The worker is the process-terminal owner of opaque native state, while the
guardian owns the exact exposed global objects through admission. A successful
observation does not prove internal native exception safety or prohibit an
unobserved transient sys.modules mutation within a trusted native call. Those
limitations are explicit, as for existing native self-alias checks; no native
sandbox, instruction tracing or hidden ownership proof is claimed. Observed
unsafe native cleanup remains a blocker under runtime-adoption section 3.

Normal success permits only the existing subsequent initialization/upload/
supervised validation-readback path. It does not seal the future optical graph.
Terminal success still requires the sole supervised target sync, authorized
whole-graph cleanup and actual exit/reap; global admission cannot authorize any
of those transitions on its own.

## 6. Versioned observation evidence

Add `q54.p12.4.native-global-observations.v1` as a CPU-safe descriptive record,
referenced downstream by the still-required final C2d3 runtime-custody schema.
Use existing canonical encoding and exact-key parsing. The record has exactly
`schema_version, environment_sha256, calls, events`. Event fields are exactly:

`index, operation, name, phase, observer, observer_call, call_sequence,
parent_call_sequence, absence_checkpoint, checkpoint, request_name, object_token`.

`absence_checkpoint` is exactly `observer_exec_entry`, referring to the same
call_sequence whose actual entry recorded that key absent. Calls are ordered
records with exact fields `sequence, parent_sequence, observer, operation, phase,
`absent_names, state`. operation is create or exec; state is started, returned,
completed or failed. Each record is a terminal snapshot of that actual frame;
sequence is contiguous, parent is null or smaller, absent_names is a sorted
unique list of this frame's required globals actually checked absent (empty
for non-observers/create). completed requires returned delegate plus successful
wrapper checks; failed/started never authorize admitted events. Issuance checks
the live private frame registry. Full aggregate custody separately binds these
calls to the external import and guard history. A parsed table alone cannot
prove that native code ran. Native-global replay checks each event against its
referenced call, name/phase/absence facts, parent relation and completion state.

`index` is contiguous from zero; `call_sequence` is the monotonic delegated
frame identity within this observed session, not a serialized capability.
`observer_call` is exec. `parent_call_sequence` is null or a smaller frame
sequence. `operation` is `observed_pending` or `admitted`. Observation checkpoint
is `request_entry` with the actual request_name or `exec_return` with null;
admission checkpoint is `exec_return` with null. object_token is an issuer-owned
positive integer assigned once to the exact held object; copies do not acquire
that identity. The same token/name/frame must persist from pending to admitted.

Parser/replay must check known exact rows/phases, contiguous ordering, one
pending then at most one admission per name, identical relation/token, no
cross-name token reuse, allowed checkpoints and no completion before observation.
Aggregate C2d3 replay must additionally cross-check actual file call/return and
guard sequence evidence and session/lane/target binding. A failed/killed prefix
may end pending and must never be accepted as completed runtime custody.
Never synthesize a file/spec/loader, native create/exec count, creator fact or
native ownership edge for a global. Parsing any record authorizes nothing.

No new result schema or product IPC is introduced. The dedicated record is a
component of the required runtime-custody evidence, not a second lifecycle
authority or a reason to defer whole-stage schema completion.

## 7. Verification and review boundary

Implementation must verify schema/version/collision/root rules without importing
a backend. Exercise the actual production loader/finder with fixed delegates:

- Exact return observation and request-entry observation under native exec;
  a nested child must not receive attribution for an already visible object.
- The real stdlib request branch observes before delegation; nested global
  publication belongs to its own innermost frame; repeated reuse emits no
  duplicate first observation or admission.
- Preloaded/early/None/wrong-type globals, wrong phase/observer/create/Python
  frame, matching ancestor with wrong inner frame, missing global on return,
  and same-metadata distinct-object insertion/replacement all fail.
- Delete/rebind sys.modules and pending/admitted records, copy frames, drift
  loader/spec/rules, erase a record before audit, or forge event order/token:
  all reject without late admission or a successful segment completion.
- Fault native return, global adoption, reconciliation and evidence recording;
  swallow an inner failure then publish in a parent. Sentinels prove actual
  returned globals and roots stay held through process-terminal disposition;
  terminal stdlib remains available and no new sync/ARM/COMPLETE occurs.

Then run real captured production materialization with both frozen Go2 assets,
both lanes and actual guardian/supervisor, recording native-global observations,
canonical identities and real input upload/readback. Complete the parent design's
canonical synthetic/dynamic, cold/dual-device, failure/hang and terminal matrices.
CPU delegates or a successful failure exit cannot replace these proofs.
Repin exact Git LF bootstrap/policy/catalog bytes and test the cold selector;
regress prior attribute/self-alias/optional/stdlib and C0 custody semantics.

Independent design approval authorizes only this bounded implementation.
There is no stand-alone CPU-subset C2d3 approval. Whole-stage code approval still
requires its full versioned runtime-custody schema and real mandatory matrix.
Keep C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host payload, R2 and shipping closed. C4 is not complete.

Retain only the selected worker's required dependency contract after selection.
New versioned fixtures and validation drivers follow the existing P12.4e cleanup
ledger; remove losing-lane permissions and temporary entries after equivalent
selected-worker failure coverage exists. Shared workflow/private instructions
and final backend-selection criteria are unchanged.
