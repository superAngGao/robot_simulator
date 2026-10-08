# C2d3 finite Torch replacement and interpreter-alias supplement

Initiative: q54-optical-lab-p12-4c4c2d-torch-publication
Stage: design-request
Author: codex
Version: v1
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2; native-global-publication design v1;
generated-stdlib design v1; torch-publication gap-scan/test-evidence v1;
optical_worker/native_bootstrap.py; runtime_environment.py
Owner Summary: Resolve the next actual cold Torch compatibility prerequisites
without weakening file-module identity or inventing creator paths. Preserve the
original executed module separately from its seven permitted exposed wrappers,
and observe one exact stdlib alias to the already held bootstrap main object.
No product or later C4 gate opens.

Base b88dbf1d5bf0a94951f3d0ecaa2bb4bdefd3ddc9, code parent8bc53bf.
Fresh gap session /root/torch_replacement_gap_v1 established G1/G2/G3. This
supplement requires fresh independent design approval before implementation.
It adds two explicitly distinct relations to runtime-adoption v2, preserving
its native trust, failure, scope and full-matrix requirements. It does not
expand the native-global observation contract.

## 1. Actual gap and finite deployment relation

Pinned Torch2.9.1+cu128 Python source replaces the sys.modules object for seven
file modules during their own exec. Six retain the original in raw m; VF retains
a native value in raw vf. Final wrappers have no raw file/loader/spec/package
path provenance. The original file source is still the execution globals owner.
torch.backends replaces itself before importing children; opt_einsum continues
assigning original globals after replacement. The current same-object rejection
is correct until this supplement is implemented and independently reviewed.

Environment schema v5 adds sorted exact-key `module_replacements` rows:
`name, phase, class_attribute, relation, native_root, native_attributes`.
Rows reference the existing admitted Python file row of the same name and reuse
its inventory bytes/hash. Phase must equal that row's phase and be runtime.
Only the following literal mappings are valid in this implementation; a profile
may contain all seven or none, and may contain them only with admitted torch,
torch._C and their original file rows. No extra name/class/path/relation and no
runtime-discovered rule is accepted. Existing historical profiles stay unchanged;
new v5 profiles carry an empty table for Warp.

| name | class_attribute | relation | native_root / native_attributes |
|---|---|---|---|
| torch._VF | VFModule | native_attribute | torch._C / [_VariableFunctions] |
| torch.backends | GenericModule | original_backlink | null / [] |
| torch.backends.cudnn | CudnnModule | original_backlink | null / [] |
| torch.backends.miopen | MiopenModule | original_backlink | null / [] |
| torch.backends.mkldnn | MkldnnModule | original_backlink | null / [] |
| torch.backends.opt_einsum | OptEinsumModule | original_backlink | null / [] |
| torch.backends.quantized | QuantizedEngine | original_backlink | null / [] |

These names cannot also be publication/global/namespace/absent/disabled entries.
Each row requires the corresponding file with its real package flag: VF false,
six backends true. Permission is finite deployment input, never inferred from
same-name objects. The eventual literal CUDA profile still requires complete
observed/source-audited dependencies, binaries, absent probes and review.

## 2. Two exact objects, one executed file body

Keep the file record's original module and exact spec/loader/delegate/source
binding for its entire lifetime. The existing create/exec frame always refers
to that original; do not overwrite it with the replacement. Separately retain
the exposed replacement, exact class, own exec frame and any native root/value.
This is an observed identity relation inside trusted pinned source execution,
not proof of the exact uninstrumented constructor instruction.

Before exec, sys.modules[name] must be the original returned by this loader;
replacement permission does not permit a preloaded or pre-exec wrapper.
For each first observed replacement require:

1. Exact own active Python-source exec frame, original/spec/loader identities,
   runtime phase, owner thread and delegate-execution context. Existing common
   wrapper markers distinguish a executing body from suspended/validation code.
2. Replacement is distinct from original. Its exact type is the raw original
   class_attribute object, whose metaclass is exactly type. Names and copied
   dictionaries cannot satisfy this identity check.
3. VF and QuantizedEngine directly derive from ModuleType. GenericModule and
   the other four PropModule wrappers directly derive from the exact PropModule
   in the held original torch.backends dictionary; PropModule directly derives
   from ModuleType. Validate actual bases/MRO with built-in type introspection,
   never custom descriptors. QuantizedEngine is not a PropModule subclass.
4. For original_backlink, the replacement's raw m is the exact original.
   For native_attribute, raw vf is the exact raw _VariableFunctions object on
   the held exact torch._C root, regardless of that value's Python type. The root
   remains subject to its original file custody; neither vf nor its annotation
   creates another module permission.
5. Hold immutable identity bindings to the original record/frame, replacement,
   class, required bases/MRO, raw constructor/forwarder function objects and
   their code objects, and native root/value when applicable. Functions' globals
   must be the exact held defining module dictionaries. Constructor/forwarder
   identities/code, original class attribute, bases, backlink/native value and
   published object must remain unchanged at later checks. Do not invoke them
   while validating. Freeze PropModule's forwarder on its defining original;
   QuantizedEngine and VF use their own defining originals.

Existing mutable records remain projections; a coherent rewrite of projections
and copied tokens cannot replace the immutable live binding. Later identity
substitution, deletion/None, second replacement, foreign frame/session/phase or
unlisted replacement poisons. A distinct same-class instance is rejected after
first observation even if it carries the same m/vf value. Transient mutation
entirely between supported boundaries inside trusted dependency code is not
claimed to be observable; no tracing/interposition is introduced.

## 3. Intake, pending execution and completion

Per row: original-created -> original-executing -> observed_pending -> admitted.
Any failed validation/call is sticky failure. The original remains the execution
object throughout. A never-executed optional row has no observations; an executed
listed row must actually replace itself before successful own return.

First-observation checkpoints are the existing common finder request-entry and
the exact original's exec-return, before other fallible return checks. At entry,
the currently executing delegate must be that row's own Python exec. An ancestor,
child, create call, wrapper/audit callback or foreign thread cannot first admit
a row. On return the supplied returning frame must be that own frame. Use a
distinct replacement checkpoint vocabulary; do not call Python replacement a
native-global observation.

At EVERY supported import/return/audit boundary, first retain every currently
available finite replacement-key value into the already adopted admission graph,
including invalid values. Then validate existing pending/admitted identities,
and only at an allowed first-observation boundary establish pending custody.
Retain originals/classes/native values/frames before controller adoption, event
serialization or metadata checks that can fail. Original attributes used to
find a class/native value must be read through raw built-in dictionary access.

While children import, a parent's existing pending replacement may satisfy its
parent package relationship. This is reuse of its bound identity, not a new
first observation or parent completion. A child's successful return cannot
complete its parent. At own successful return, after nested sticky failure,
metadata and other required checks, adopt the exact exposed owner and emit its
admitted event before completing the original file record. At audit all executed
rows must be admitted, no pending/call unfinished; audit is read-only and cannot
discover/repair missing replacement evidence. No new event may be observed for
a call after its admission has begun. Later stdlib diagnostics after poison only
retain available values; they cannot create successful observations.

## 4. Raw metadata and normal package lookup

Introspection uses ModuleType.__getattribute__(value, '__dict__') for module raw
dictionaries and type.__getattribute__ for class dictionary/bases/MRO. Required
exact type/class relations are checked before traversal. Audit never calls
wrapper __getattr__, arbitrary __getattribute__, ContextProp, dir or hasattr.
This does not forbid CPython's ordinary importer from using the pinned wrapper's
normal attribute forwarding as part of dependency behavior.

Original metadata remains the real file profile: exact name/spec/loader/origin,
package, file path and package search locations, using raw dictionary reads.
Replacement raw metadata is a separate profile: __name__ is the exact name;
__loader__, __spec__, __package__, __doc__ are present and None; __file__,
__path__, __cached__ and instance __getattr__/__getattribute__ keys are absent.
No implementation fills these fields to fabricate file provenance. Required
m/vf storage follows section2; unrelated behavioral attributes are not provenance.

For an admitted or pending replacement parent, resolve child file rules using
the original's frozen package path and require the incoming path sequence to
equal those exact contained locations. Reject raw shadow paths/spec/loader and
wrong original backlink. VF is non-package and grants no child path. This keeps
one resolver/loader; no Torch fallback importer is added.

CPython publishes a completed child on its parent after child exec_module returns.
Do not require that assignment inside the child's own return wrapper. At final
segment audit require each completed direct imported child to be the exact
exposed object in its parent's effective binding: for a backend wrapper parent,
raw wrapper child binding must be that child; if pinned parent source also wrote
the name into the held original (torch.backends from-import list), require that
raw original binding too. Freeze that exact from-import child list from pinned
torch/backends/__init__.py, not from runtime enumeration. For torch.backends on
torch and torch._VF on torch, validate the ordinary parent binding at this same
post-import audit. Later audits recheck these bindings. An original-only source
global before importlib publication is not misreported as an exposed binding.

## 5. Separate exact interpreter alias observation

Pinned CPython3.12 multiprocessing/__init__.py lines36-37 assigns __mp_main__ to
the existing __main__. It is not in sys.stdlib_module_names. The exact183 source
SHA is a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b.
This section adds an observed alias relation within the existing trusted stdlib
boundary, not a third-party file module, replacement, native observer or new
creator instruction claim. No multiprocessing process/spawn capability opens.

At cold installation require one exact ModuleType __main__ and no __mp_main__ or
multiprocessing. Bind and retain that original main object in the source session.
Capture the exact multiprocessing package source through the existing immutable
stdlib search topology: standard SourceFileLoader, resolved file under the
interpreter stdlib root, package location equal its contained directory, no
site/root override. Freeze path, held source bytes/hash and interpreter identity
in the session topology alongside existing generated-stdlib facts. Verify the
exact literal alias statement structurally in that pinned source; if absent or
different, reject this interpreter alias support. It must not match arbitrary
assignments or a string occurrence. No stdlib bytes are rewritten/executed early.

When the existing trusted finder resolves the exact multiprocessing name, hold
the exact returned spec/delegate against this descriptor before returning it.
Use its ordinary stdlib loader; no broad stdlib exec wrapper is introduced.
First alias observation is permitted at common request-entry, file exec-return
or final segment audit after that exact standard resolution. Require the loaded
multiprocessing object to be exact ModuleType, its raw spec/loader/file/package
and search path to match that held resolution/descriptor, and raw spec
_initializing false. This is a completed-import metadata observation within the
trusted interpreter boundary, not a newly claimed witnessed exec event. Bind the
actual observed multiprocessing object at first observation for later identity.

Before checking, retain any available __mp_main__, current __main__, publisher
module and spec into the already held session graph. Admit the alias only if
its actual value and current __main__ are both the exact cold main object. Emit
one interpreter_alias_observed fact; later checks are idempotent and require
unchanged main/publisher/spec/loader/alias and source hash. No replacement token
or file execution is minted. An absent alias before multiprocessing resolution
is permitted; after a completed observed multiprocessing import it is required.
A preloaded/injected alias, alias before resolution, wrong object with identical
metadata, replaced main/publisher, disappearance after observation or changed
stdlib descriptor poisons. A partially initializing stdlib publisher is not
admitted yet; if an alias is already available, retain it pending only until
that trusted import returns, and final audit rejects incomplete publisher.

Do not add __mp_* or this name to the generic stdlib classifier. The external
audit bypasses this single key only after the exact observed alias check. An
explicit missing-name import still resolves no invented file/body. Parsing the
alias observation cannot authorize a live module or retry a failed session.
Terminal stdlib reporting cannot make a new successful observation after poison.

## 6. Failure custody and evidence

The already adopted graph retains original, every observed exposed value,
class/base/function, native root/value, actual frame/spec/delegate and original
exception before fallible operations. Failure scans may enlarge this graph even
after poison or importlib deletion. Never discard a wrong value merely because
validation failed. No normal COMPLETE, fresh ARM, target sync, ordinary release,
retry or successful record after failed normal work. Original watchdog and
UNKNOWN/foreign-device quarantine rules, native internal exception-safety trust
and process-terminal ownership remain unchanged.

Add CPU-safe exact-key descriptive evidence (final field layout frozen in code):
environment/version/lane/phase and actual file-call reference; stable distinct
original/replacement/class/native-root/value tokens; checkpoint request-entry or
exec-return; ordered pending/admitted transitions and immutable relation enum.
File create/exec events stay on the original. Replacement events grant no file
provenance and native values need not be module tokens. Stdlib alias records
refer to the cold-main token, exact source descriptor/resolution and observed
publisher token/checkpoint, explicitly observation-only with no invented exec.

Issuance validates live immutable bindings against mutable event projections.
Replay cross-checks the actual import ledger and guard ordering; rejects absent
or invented originals/calls, wrong phase/parent, event/token substitution, same
call post-admission observation, duplicate completion, missing pending/owner
facts, admission after failure and forged summaries. Failed records are prefixes,
never rewritten success. Parsed records mint no runtime/source/storage/release
authority. These records must feed the final C2d3 runtime-custody aggregate;
they are not an alternate lifecycle authority or reason to postpone its delivery.

## 7. Validation and review boundary

Before bounded code approval: real production-loader CPU tests for both relation
forms and early pending backend parent/child order; original execution globals
stay original; callback-free auditing and actual incoming parent path checks.
Attack copied original/spec/class/dictionary/token/record, a fresh same-class
wrapper, wrong subclass/backlink/native root/value, unlisted eighth replacement,
early/preloaded/repeated/foreign-phase/thread/session, missing/late/second
replacement, parent binding drift and coherent projection rewrites. Test faults
after publication, before intake, in adoption/event/reconciliation, swallowed
nested error and terminal reporting; original and replacement sentinels remain
held through quarantine. No self-referential implementation-only test oracle.

Exercise the exact stdlib alias through a cold production session: standard
multiprocessing import, raw identity observation and repeated audit; reject
preload, injected copied main, replaced original/publisher/spec, late removal,
descriptor/source drift and poison-time observation. Keep all existing cold
finder/generated-stdlib/native-global tests and scope/guardian protocols.

Then actual captured Torch must demonstrate all seven relations, _C/ops/classes
publications, interpreter alias and original/exposed custody under the existing
supervisor, before a2 upload can claim success. No ordinary census or CPU test
substitutes for that proof. Full C2d3 approval still requires the entire frozen
both-lane canonical/replay/real-Go2 upload/readback, fault/hang/dual-GPU/terminal
matrix, final runtime-custody schema and fresh whole-stage code review.

AST policy/catalog changes are separately repinned to exact approved bytes.
No test hooks enter the production import closure, second upload implementation,
backend bridge, package fork/install or product selector is added. Update the
existing cleanup ledger: keep selected-backend custody long-term, collapse the
losing environment rules and fixed diagnostic entry only under P12.4e with
equivalent selected-worker tests. C2e/f, C4c3/c4, C4d/B3, product GPU preparation,
a3b, result/baseline, host payload, R2 and shipping remain closed.
