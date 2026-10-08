# C2d3 runtime-custody aggregate implementation boundary

Initiative: q54-optical-lab-p12-4c4c2d-runtime-custody
Stage: bounded schema implementation plan and gap preflight
Author: codex, implementation coordinator
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: runtime-adoption design v2 section7; guarded-custody design v2;
native-global-publication design v1; torch-publication design v1;
optical_worker/native_bootstrap.py; runtime_environment.py; guarded_evidence.py;
native_contract.py; tests/support/native_runtime_process.py
Owner Summary: Freeze the missing aggregate's event model before implementation.
This closes the approved requirement to bind actual import, runtime/storage and
guard histories. It adds no import permission, resource authority or product
entry. Review must identify ambiguities and missing failure behavior before code.

## Candidate and problem

Production baseline is a3d750bbaf79d760a6727e56146d277e1cb3b770, still clean on
primary183. Local0ae8df1 contains the approved test-fixture repair and evidence
archives; production/tests-support are identical to a3d750b. The final packet
pins this document's commit and digest. No aggregate implementation exists yet.
This v2 supersedes v1. Fresh gap-review v1 requested changes F1-F3; its complete
unchanged report is archived. The following field lists and explicit cut rules
replace the corresponding v1 promises; no historical record is reclassified.

The actual four CUDA normal workloads now pass. The captured positive ambient
case and current Warp synthetic/replay normal/terminal tests also pass, with
explicitly bounded evidence. Existing driver-observation.v0 dumps are diagnostic
snapshots; they are not this aggregate and will not be relabeled or converted
into authentic v1 worker records. Existing component parsers validate local
native-global/replacement/interpreter-alias rules but do not cross-bind all
imports, phase completions, runtime binding or physical prepared storage.

The approved runtime-adoption section7 explicitly requires a versioned CPU-safe
exact-key aggregate, descriptive only, retained with termination, and negative
replay tests. This plan specifies that implementation and its validation; it
does not reopen approved publication policy or add a new product prerequisite.

## One observer and one lifecycle authority

Add optical_worker/runtime_custody.py for CPU-only immutable payload parsing and
cross-validation. No backend imports, file discovery, native API, live authority
registry, synchronization, guardian operation or lifecycle replay copy exists
there. It delegates lifecycle validation to GuardedLifecycleEvidence and the
existing validate_custody_artifact, and component validation to runtime_environment.

The existing captured SourceSession owns one append-only runtime observation
journal and one descriptive object-token table. Create both before external
configuration and retain them through the already-adopted session. Every token
is a monotonically allocated positive integer for an exact held object; repeated
identity returns the same token. The table strongly retains objects to prevent
id reuse. Its integers are references within this record, never proof of native
ownership, constructor provenance or an authorization key.

Observation code stays in native_bootstrap at actual existing occurrences.
It may retain CPU metadata/returned objects but cannot perform extra device
queries, import probes, attribute-based discovery or another storage traversal.
Existing actual admission and exact-object registries remain the authority.
No observer may arm/complete/renew a segment or relax failed-session state.
Raw journal append is O(1) apart from copying that entry's bounded metadata;
full parse/replay/serialization occurs only at untimed checkpoints, never on
each import event or in C4 acquire. Object identity lookup uses id plus exact
retained-object comparison, not equality or hashing arbitrary objects.

## Version and top-level exact fields

`q54.p12.4.runtime-custody.v1` has exactly these fields:

- schema_version;
- source: {revision, catalog_sha256, snapshot_sha256};
- environment_sha256: null before successful environment binding, otherwise
  the actual bound RuntimeEnvironmentContract digest;
- configuration: original GuardConfiguration payload;
- lane_id: warp or cuda;
- observation_cut: {journal_length, lifecycle_event_count};
- journal: ordered entries defined below;
- objects: ordered {token, first_journal_index, guardian_sequence} rows;
- components: {native_globals, module_replacements, interpreter_alias}; each
  null only before its producer exists, otherwise the existing exact payload;
- lifecycle: exact existing GuardedLifecycleEvidence payload at the cut;
- summary: derived fields below, not trusted self-report.

Canonical JSON uses the existing protocol encoding. Ordered arrays carry order;
dictionary key order carries no semantics. All integers require exact int
(bool is rejected). Digests are lowercase64hex; revision is exact40hex. No
arbitrary object, bytes blob, callback, unknown key or open extension field.
The parser requires separately supplied exact RuntimeEnvironmentContract and
trusted launch inputs defined below, even for early environment-null cuts.
The environment digest must match when the record reports binding. This is
a reference to earlier trusted launch evidence, not embedded admission authority.
The source snapshot digest refers only upstream. No source/catalog/environment/
prepared manifest includes the final aggregate digest.

`objects` describes observed identity and optional direct guardian adoption.
guardian_sequence is null if the object is retained through the session/import
graph without its own guardian row, including objects returned before failing
adoption. A non-null sequence must name an existing actual owner_adopted row no
later than the observation cut. The emitter takes it from exact live identity
while the guardian table exists; it does not zip types/names or guess after
release. The parser does not infer that every null sequence is unowned or that
a non-null sequence proves native memory was individually freed.

## Journal envelope and event types

Every journal entry has exactly {index, kind, guard, lifecycle_event_index,
details}. index is zero-based contiguous. lifecycle_event_index is the last
actual lifecycle event present at occurrence, not a predicted index. guard is
the actual controller reference or null. References are nondecreasing; the
referenced lifecycle event must carry the same guard when non-null. Normal
execution facts other than source_bound/phase_completed require the original
successfully armed setup ticket, with no
intervening disarm/failure/termination; lifecycle replay remains authoritative.
Failure intake and postterminal observations cannot invent normal work.
phase_completed uses current null guard and explicitly references the original
normal_disarmed event in details; its envelope index is that actual latest event.
Failure/file_call-failed rows may carry the terminal reference but cannot append
new return/admission/completion or device work after sticky failure.

Kinds and exact details:

0. `source_bound`: {session_token, bootstrap_token, controller_token}. First
   journal entry, after actual guardian/source binding and before external work;
   guard null is permitted. Session and bootstrap tokens must reference their
   exact source-custody owner_adopted roles; controller is held through the
   session binding, without inventing a direct guardian adoption.

1. `environment_bound`: {environment_sha256, environment_token,
   admission_token}; once, in configure segment after actual verification and
   custody, before any external import. Failed verification cannot emit it.
2. `external`: {event, provenance}. event is the unchanged existing external
   event payload, including its existing index/operation/name/kind/phase. Keep
   richer disabled-optional event fields exactly as defined by that producer.
   Its index must be contiguous and its journal occurrence unique. provenance
   is discriminated below; it records actual held facts at this occurrence.
3. `file_call`: {sequence, parent_sequence, observer, operation, phase, state,
   module_token, spec_token, loader_token}.
   Occurs at each actual begin/return/complete/fail mutation of the existing
   native_calls ledger. No second call ledger creates permission. State follows
   started -> returned -> completed; started, returned or completed may then
   transition to failed. This last transition is a wrapper/observer failure;
   it does not erase a previously observed delegate return or completion.
   Failure intake may retain an already failed call without a new transition.
   spec/loader tokens identify the actual held call owners; create-start has
   null module_token, exec-start uses the previously returned file module;
   return/completion must reference the exact returned module. See returned_owner
   below for a return preceding any fallible return-processing or adoption.
   A started native call interrupted by process death remains started.
4. `native_global`: {event_index, token_bindings, adoption_event_index}; one occurrence for each
   existing component event, at its actual append site. Bindings are defined below.
5. `module_replacement`: {event_index, token_bindings}; likewise.
6. `interpreter_alias`: {event_index, token_bindings, publisher_token, spec_token,
   loader_token}; likewise, for the one alias. These owner tokens come from the
   exact retained alias binding/resolution at issuance, not the mapping table.
7. `runtime_binding`: {binding_token, controller_token, module_token,
   runtime_token, target_token, stream_token, source_token, stream_identity}.
   Emit after actual binding registration and adoption, still inside original
   runtime segment. stream_identity is the already-validated [ordinal,handle],
   not a fresh native getter. Failed registration emits no binding-success row.
8. `binding_consumed`: {binding_token}; exactly once, after successful actual
   consume/identity validation. A burn followed by rejection emits failure,
   never successful consumption. It must precede the first upload leaf.
9. `prepared_storage`: {prepared_token, source_token, ready_event_token,
   manifest, arrays, storage_sha256}. Use the exact already-registered storage
   entries after the existing storage audit succeeds. Full manifest is its
   existing CPU-safe payload. arrays is name-sorted and has exactly
   {name,array_token,storage_owner_token,signature}; storage_owner_token is null
   for Warp's existing signature model. signature has exactly the existing
   _PhysicalStorageSignature fields, with tuple fields represented as lists.
   storage_sha256 hashes this arrays payload using the existing canonical codec.
   No second physical inspection loop; expose a CPU projection from the actual
   native_contract registry after its normal check. Copies of this payload
   cannot register storage or recreate a live prepared wrapper.
10. `input_validated`: {prepared_token, manifest_sha256, content_sha256,
    topology_sha256, validated_arrays}. Emit only after the shared real readback
    validator returns successfully. Cross-check against prior manifest/storage
    and the exact original prepared object. It does not issue readiness.
11. `phase_completed`: {phase, disarmed_event_index}. phase enumerates
    setup_definitions, external_environment, canonical_materialization,
    runtime_initialization, native_upload, input_validation in that order.
    Emit after the original with-segment has completed, referring to its actual
    normal_disarmed event. No success from body return/COMPLETE send alone.
12. `failure`: {source, exception_token}. source enumerates source_session,
    external_admission, observer. Retain original exception before this fallible
    record. This row is allowed after guard failure/terminal transition; it
    conveys no new execution success. If observation itself cannot finish,
    retain the actual raw graph and select the last validated cut as defined
    below; do not invent a complete failure row.

13. `returned_owner`: {producer, call_sequence, leaf_event_index, object_token}.
    producer is file_create, runtime_stream or runtime_target. For file_create,
    call_sequence identifies its already-started create call and leaf_event_index
    is null. Emit immediately after retaining the actual delegate return in the
    existing admission.records, before fallible return_native_call or adoption.
    For runtime_stream/target, call_sequence is null, leaf_event_index names the
    original entered indexed-runtime leaf; emit immediately after the existing
    runtime_owners.append, before device metadata or adoption. Do not inspect a
    GPU object for this event; actual ordinal remains unknown until existing
    metadata/adoption succeeds. A failed adoption cannot invent a binding row.

Other journal kinds reject. No wall clock or cross-process timestamp is used
as execution authority. OS PID/start/ancestry and supervisor reap remain the
separate required actual process evidence.

## External provenance and cross-replay

Provenance is exact-key by event kind, never a free-form dictionary:

- file_module: {origin, delegate_kind, spec_token, loader_token, module_token}.
  delegate_kind is SourceFileLoader or ExtensionFileLoader from the actual
  pinned delegated loader, not the wrapper's invented class name. module_token
  is null before actual module return. Origin equals the actual resolved path
  and the bound environment row. self-alias absence events use this same file
  provenance but retain their distinct operation and zero publication meaning.
- absent_probe: {requester, requester_token}; the actual top import context,
  with no module/spec/loader/origin or fabricated native call.
- created_module/module_alias: {module_token, creator_token, publisher_token,
  relation, attributes}; exact held relationship from the successful existing
  admission. relation is creator_attribute or creator_self and must match the
  frozen environment rule. No name/prefix-only admission.
- namespace: {module_token, locations}; the actual verified namespace search
  list, with no file origin or delegated native/Python body claim.
- disabled_optional: {requester_token, fallback_token, token_bindings}; fallback
  is null before actual fallback retention. The unchanged richer event already
  carries its verified resolution, error/fallback and digest facts. Replay
  requires denial then matching fallback before requester completion; zero
  executable import events for the disabled name or descendants.
- failure: {exception_token}; an event named failed is a failure fact, not an
  admitted external name. It may refer to an unlisted attempted import.

The parser replays file_resolved -> create_started -> created -> exec_started ->
completed per name; prefixes may stop at any step. It checks actual file-call
start/return/complete transitions against these external transitions, phases and
nesting. A resolution without create is legal (e.g. deferred Triton probe).
Completed canonical/runtime/upload/readback phases require all calls they
started to have completed and required publication/fallback outcomes present.
Expected absences require an executing exact requester. Creator publication
requires its actual publisher exec interval and existing creator identity;
creator_self requires all three existing absence checkpoints. Namespace records
cannot be replayed as file execution. Extra admitted names reject.

Native-global/replacement/alias components in an issued record are passed to
their existing strict parsers, unchanged. A raw failure tail that these parsers
cannot represent follows the explicit prior-cut selection below; it is never
laundered through a weakened parser or relabeled as an issued complete record. Journal references cover their events exactly once, in existing order,
inside the matching real file-call interval (or allowed alias audit checkpoint).
At the chosen cut the file_call final states equal component native_calls
states, including completed-to-failed where the strict component remains valid;
inventing a
completed observer or swapping calls between guards must fail cross-replay.
No component's copied digest or locally valid history substitutes for that
cross-binding. The alias source/interpreter identity additionally matches the separately
supplied upstream launch inputs below, not an unrelated valid alias or a
repository snapshot which does not contain stdlib source bytes.

## Summary, partial failure and terminal attachment

Summary has exactly {last_completed_phase, external_event_count,
completed_file_count, successful_guard_sequences, binding_observed,
prepared_storage_observed, input_validation_observed,
final_execution_readiness_issued}. Derive all from replay, with the final field
always false. Reject forged counts, missing/reordered completion, incompatible
binding/prepared identity, fabricated native calls and contradictory digests.
Do not duplicate lifecycle disposition, dirty-scope, release or sync summaries.

The worker emits checkpoint snapshots through the existing atomic writer after
each successful phase and after a terminal attempt using the cut rules below. The observed cut may be a prefix, including a
worker killed during native work; never fill missing tail facts from expected
control flow. Persist the last complete snapshot if serialization/I/O fails.
Failed collector/parser work poisons normal continuation through the existing
session.fail/controller path and cannot itself initiate an extra sync or ARM.
The journal/token roots are already in the retained source-session graph before
native execution. A terminal record is a projection of that held graph, not a
new guardian adoption after freeze. Its generation cannot authorize release.

Keep the existing custody artifact unchanged. A new CPU attachment validator
takes (runtime_record, environment, trusted_launch_inputs, custody_artifact) and first calls existing
validate_custody_artifact. Require matching configuration/lane, and the runtime
lifecycle trace to be an exact prefix of the artifact worker trace (or equal).
If the artifact has no worker trace, validate the standalone record against its
trusted launch inputs but return attachment status unavailable_worker_trace.
Do not claim terminal cross-binding. Do not infer a worker record from a
supervisor-only trace. No claim that OS exit equals individual cudaFree calls.

## F1: explicit validated cuts and failure tails

No component parser is relaxed or silently version-reinterpreted. Existing
native-global/replacement v1 `admitted` events require completed calls. Actual
producer code can leave an admitted event with a failed call, or complete a
call and then fail its wrapper. Keep those facts in the existing raw failure
diagnostics and actual retained journal; they need not be an issuable aggregate.

After source binding, issue/store one validated CPU baseline cut (environment
null, no external producers). At each successful phase, the same worker builds
one candidate aggregate using the journal, available component projections and
actual lifecycle prefix, parses it strictly, retains its immutable encoded
record, then atomically persists it. Store last_validated_record only after the
entire parse succeeds. Failure in normal observation or parse poisons the
existing session and forbids advancing to the next phase, even if an older cut
exists. The old cut is evidence retention only, never a recovery checkpoint.

After sticky failure/terminal disposition, attempt one CPU projection of the
currently available data without imports, audits, identity-discovery callbacks
or device getters. It may issue a truthful partial aggregate if all existing
strict component and cross-record checks pass. If projection/append/parse cannot
produce a consistent record (including admitted-before-call-completion or a
half-initialized admission object), select the already-validated immutable
earlier cut unchanged. No trimming selected component events, rewriting call
states or synthesizing journal facts to make the current tail pass. Do not
retry aggregation with guessed subsets. If no baseline was successfully issued,
there is no aggregate record; the actual supervisor/worker evidence remains.

The fixed driver separately writes descriptive selection data with exactly
{schema_version, selection, runtime_record_sha256, selected_lifecycle_event_count,
observed_lifecycle_event_count, failure_diagnostic_present}; version
q54.p12.4.runtime-custody-selection.v1; selection is current, prior_validated_cut
or unavailable. Unavailable has null record digest/count. Counts come from the
actual selected immutable record and actual last available lifecycle trace;
they do not describe native events that occurred after that trace. A failed
write retains the earlier atomic file; supervisor-only evidence cannot invent
selection data. This sidecar grants no authority and is checked with the raw
failure diagnostics at acceptance. It does not change the main custody artifact.

Full raw component/journal tails and original exceptions stay held through the
session until hard exit. Existing external_failure.json and failure.json retain
available diagnostics; add a raw journal diagnostic under an explicitly
non-validating diagnostic version if its CPU serialization succeeds. It is not
fed to RuntimeCustodyRecord.from_payload and never passes as validated custody.
Unavailable/older aggregate coverage is explicit in each failed test and report.
The accepted aggregate does not pretend to enumerate all work before death.

Available component means: all fields required by its projection exist in the
actual partially initialized producer. Do not call its normal auditing getter
after failure. If any required field is missing or an immutable-issued-facts
check fails, choose the previous record; do not replace that present but broken
component with null in a fresh record. Null is allowed only in the baseline or
a cut before that producer's initialization completed. Alias initialization
precedes source binding, so baseline alias data exists unless baseline issuance
itself failed (then no record). No new adoption or terminal operation occurs
while serializing or selecting a prior record. If a terminal aggregate write/
parse fails after a successful authorized release, retain that actual release
fact in lifecycle evidence, select the earlier record, and report incomplete
aggregate output; do not pretend the release was undone or issue another sync.
This attempt does not satisfy the mandatory aggregate acceptance gate.

Tests freeze expected selection: return-before-adoption failure with a valid
component prefix should preserve the returned token in a current partial record;
admitted-then-observer-failure with an invalid strict component selects the
unchanged preceding validated cut plus raw failure diagnostics. After-call-
completion external-record failure accepts completed-to-failed only if the
unchanged strict components still accept it; otherwise prior_validated_cut.
This is an explicit narrowed issuance promise, not complete latest-tail proof.
Faults in each actual append/commit window must verify the chosen result and
full live retention, not just an exception or an old successful record.

## F2: occurrence-bound token joins

The global object table describes only objects referenced by committed journal
entries in the selected record. Tokens are positive, unique and monotonically
allocated, but need not be contiguous after a failed pending observation.
first_journal_index is the first actual reference in that journal; no forward
or unreferenced table rows. Pending tokens/objects remain strongly held even
when an interrupted append never publishes them; do not renumber later tokens.
Multiple module names or views of the same object use the same global token.
Different objects cannot share one global token. This is identity from `is`,
not matching type names, metadata, Python hashes or component-local integers.

Each token_bindings list has exact rows {local_token, object_token}, sorted by
local_token, unique within its component namespace. A native-global event maps
its object_token through the actual held native_global binding. A replacement
event maps every non-null original/replacement/class/native_root/native_value
through replacement_objects and its exact event bindings. Alias maps1 to the
actual captured main and2 to the actual held publisher. Disabled-optional maps
its non-null external-owner-N requester/fallback identities via the actual
retained list; those local strings are a separate namespace. No global lookup
is performed by interpreting a component integer as a global integer.

On repeated occurrences, the same namespace/local token must map identically.
Map exactly the local tokens used by that event, no extras or omissions.
Cross-check replacement original against the matching file call/module token;
native_root against its actual creator file token; native-global global token
against its exact original owner_adopted event: adoption_event_index names that
event, whose operation_id is native_global_observation and whose sequence equals
the object's guardian_sequence. The admitted event repeats the same prior
adoption reference; it does not claim a second adoption. Alias main matches
source_bound.bootstrap;
alias publisher mapping equals the separately recorded exact publisher_token;
the held spec/loader tokens are distinct resolution owners, not file modules.
Repeated alias uses preserve those same owners. The latter
has no fabricated external file_call, because stdlib loading is not captured
by the external adapter. Native-global records do not emit ordinary external
creator_publication events; their component event is their observed admission.
Cross-check their object token with any runtime/guardian use of that same owner,
without inventing an external file execution.

file_create returned_owner is the identity root for its subsequent file-call
return/module token, external created/module token, aliases and replacement
original. Runtime returned_owner supplies the stream/target token before
fallible metadata/adoption; a later runtime_binding must use that exact token.
CUDA target descriptor is CPU-constructed and recorded as a held runtime target;
it is not misclassified as a returned native allocation. Runtime module/root
tokens join existing external file/native-global identities. source/controller
tokens equal the source-bound identities. Source/upload/input token relations
are immutable across the whole record.

guardian_sequence is an optional direct-adoption fact at this cut, obtained
only by matching the live exact object against the existing guardian owners
and its corresponding issued record while the table exists. Before terminal
release, freeze these observed matches; never reconstruct them from type/role
or table order after release. Require the referenced owner_adopted event to
precede the cut and the expected role/scope for that use: source/bootstrap
source-custody roles; file/spec/delegate/publication external_import_owner;
binding runtime_setup_binding; prepared guarded_prepared_input; returned arrays/
events their existing native setup/readback roles and target scope. If an owner
already has an earlier legitimate adoption, preserve that row/role rather than
create a new one. Unadopted returned owners have null and remain explicitly held
through the source-session observer graph. No null-to-adopted fact is invented
after failure. Subsequent snapshots may add a real observed adoption without
changing object identity or previous journal entries.

## F3: separately trusted upstream validation inputs

All validation entrypoints take a trusted_launch_inputs CPU payload, independently
captured by the fixed trusted launcher before the attempted runtime and retained
as runtime_custody_inputs.json. It is not a worker request, live authority,
downstream receipt or configurable runtime. Exact fields:
{schema_version, source, lane_id, configuration, case_id, interpreter_alias}.
Version q54.p12.4.runtime-custody-inputs.v1. source has the same revision,
catalog_sha256, snapshot_sha256 fields as the runtime record, obtained using
the existing CPU source capture/verification for the exact candidate/lane.
configuration is the actual supervisor configuration; case_id is the fixed
canonical driver case. interpreter_alias has exactly {interpreter_identity,
resolved_executable, executable_sha256, source_path, source_sha256}.

The trusted launcher captures these interpreter values through the pinned
interpreter in a CPU-only -I -S -B preflight: the existing seven-element generated
identity, resolved executable path/hash, and the fixed stdlib multiprocessing
source path/bytes/hash with the existing fixed alias-statement check. No
multiprocessing/backend import is needed. Use a shared CPU projection/check
of existing bootstrap capture logic, not a second alias-admission policy.
These inputs are independent upstream deployment facts. The worker's existing
cold capture must equal them; mismatch poisons/rejects before external work.

The CPU parser reads no filesystem and runs no interpreter. It requires the
record's source fields/configuration/lane to equal supplied inputs. Alias
component identity/path/hash must equal the independently supplied descriptor.
The descriptor's executable spelling is retained exactly, while its resolved
executable path/hash equals environment.interpreter.path/sha256; version's first
three integers equal environment.interpreter.version. Other ABI/platform/
multiarch/stdlib-root/os-name values match the full captured seven-element
descriptor, not values guessed from the executable string. Existing lexical
alias validation still checks source location under that stdlib root.

Early records with environment null still require these upstream inputs and
source/alias matches. They do not claim environment verification happened.
After binding, the actual environment digest additionally must match. A source
or alias substitution can no longer pass by supplying another hash in the
record. Trusting an attacker who replaces both trusted launch inputs and all
execution evidence is outside this trusted-deployment model; no cryptographic
process attestation is claimed. Upstream files never reference the downstream
aggregate hash. Cross-attachment tests hold upstream inputs fixed.

## Shared manifest and storage projection boundary

Extract NativeArrayManifest and NativeInputManifest, their existing to_payload/
digest codec and new exact payload validation into a stdlib-only module,
optical_worker/native_manifest.py. native_input imports/re-exports those same
classes, preserving current public imports, canonical bytes and digests. No
duplicate dataclass/schema in runtime_custody and no early NumPy/materializer
import. Generic existing constructors remain compatible; the new C2d3 parser
requires the actual canonical fixed/replay contract, independently of readiness.

For this canonical path, semantics has exactly input_contract,
scene_contract_sha256, scene_id, each matching the top-level field; no recursive
arbitrary semantics object. Array rows have the existing seven fields and are
name-sorted/unique, c_contiguous true, exact nonnegative shape dimensions and
nbytes == product(shape)*dtype itemsize. Canonical frozen builders use <f4,
<i4 and<i8; parser supports these exact current canonical types for C2d3.
Validate exact manifest fields, scalar types, nullable hashes, array-name subsets,
fixed/replay alternatives and nonempty canonical case/source identities; do not
claim to recompute array content/topology from digest strings without array data.
input_validated must match the actual source manifest's content/topology hashes
and complete array-name list. Actual shared D2H/CPU validator supplies numerical
proof; parsing alone never does.

Add a CPU projection of the already-registered native_contract storage entries
after the existing storage audit, no additional native getter loop. It requires
the same exact registered subject/source/arrays/event identities. Signature
fields are device_ordinal, base_pointer, storage_identity, storage_nbytes,
active_start, active_count, dtype, shape, strides. Integer fields exclude bool;
counts/offsets/strides are nonnegative, ordinal equals target, shapes equal the
manifest, active_count equals manifest nbytes, active range fits storage.
Nonzero storage requires positive pointer/identity; zero-size storage permits
zero pointer and still preserves the actual observed identity. CUDA strides are
in elements; Warp strides are in bytes. Check C-contiguous progression for
nondegenerate dimensions using the corresponding units and dtype itemsize;
zero-sized arrays follow the existing registered signature without inventing
physical copies. Dtype strings must match the lane's existing canonical dtype
mapping, kept in a single shared literal metadata table. ready_event/source/
array/storage-owner tokens refer to the exact registry-held objects and their
prior actual adoption where applicable. Storage digest is descriptive only.

## Delivery and verification

Update only the required captured source closure/catalog/policy pins for the
CPU schema and observation sites. Add no arbitrary module, runtime capability,
source exception, product IPC field or test hook inside the production closure.
The fixed sacrificial driver writes new observed records; historical v0 files
remain historical and are not migrated into claimed v1 executions.

Meaningful CPU tests use actual guard/controller/Peer and actual import fixtures:
positive early/normal/failed cuts, all external outcome kinds and empty tables;
mutate name/origin/phase/requester/call/guard/owner/stream/storage/manifest,
missing and duplicate events, completion reorder, component transplantation,
forged counts, failed import swallowed by its parent, returned-before-adopt
failure, observer exception, terminal serialization failure and copied parsed
records failing to mint any binding/storage/release authority. Capture exact
native storage from existing registry fixtures; do not mock a second uploader.

Then run real both-lane asymmetric/replay/both-Go2 upload/readback, required
partial import/allocation/copy/event/audit/owner faults and guard/terminal/device
matrix using the actual new observed schema. CPU success is not C2d3 approval.
Freeze the complete candidate and request a fresh independent whole review;
repair/re-review findings, preserving complete reports and raw attempts.

Long-term retain selected-backend environment/source custody and this
descriptive schema. Losing-lane specifics and temporary validation entrypoints
follow the already-approved P12.4e cleanup gate; no lane is deleted now.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host,
R2 and shipping remain CLOSED. C4 remains the continuous objective.
