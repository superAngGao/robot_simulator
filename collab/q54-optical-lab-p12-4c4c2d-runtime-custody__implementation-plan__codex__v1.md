# C2d3 runtime-custody aggregate implementation boundary

Initiative: q54-optical-lab-p12-4c4c2d-runtime-custody
Stage: bounded schema implementation plan and gap preflight
Author: codex, implementation coordinator
Version: v1
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
primary183. Local1fd7b6c contains the approved test-fixture repair and evidence
archives; production/tests-support are identical to a3d750b. The final packet
pins this document's commit and digest. No aggregate implementation exists yet.

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
The parser requires the separately supplied exact RuntimeEnvironmentContract
when environment_sha256 is non-null and requires its digest to match. This is
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
execution facts require the original successfully armed setup ticket, with no
intervening disarm/failure/termination; lifecycle replay remains authoritative.
Failure intake and postterminal observations cannot invent normal work.

Kinds and exact details:

1. `environment_bound`: {environment_sha256, environment_token,
   admission_token}; once, in configure segment after actual verification and
   custody, before any external import. Failed verification cannot emit it.
2. `external`: {event, provenance}. event is the unchanged existing external
   event payload, including its existing index/operation/name/kind/phase. Keep
   richer disabled-optional event fields exactly as defined by that producer.
   Its index must be contiguous and its journal occurrence unique. provenance
   is discriminated below; it records actual held facts at this occurrence.
3. `file_call`: {sequence, parent_sequence, observer, operation, phase, state}.
   Occurs at each actual begin/return/complete/fail mutation of the existing
   native_calls ledger. No second call ledger creates permission. State follows
   started -> returned -> completed, or started/returned -> failed; subsequent
   failure intake may retain an already failed call without a new transition.
   A started native call interrupted by process death remains started.
4. `native_global`: {event_index}; one occurrence for each existing component
   observation/admission event, at its actual append site.
5. `module_replacement`: {event_index}; likewise.
6. `interpreter_alias`: {event_index}; likewise, for the one existing alias.
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
    retain the last good journal and do not invent a complete failure row.

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
- disabled_optional: {requester_token}; the unchanged richer event already
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

Native-global/replacement/alias components are passed to their existing strict
parsers. Journal references cover their events exactly once, in existing order,
inside the matching real file-call interval (or allowed alias audit checkpoint).
The file_call final states equal component native_calls states; inventing a
completed observer or swapping calls between guards must fail cross-replay.
No component's copied digest or locally valid history substitutes for that
cross-binding. The alias source/interpreter identity additionally matches the
actual frozen source snapshot/interpreter launch, not an unrelated valid alias.

## Summary, partial failure and terminal attachment

Summary has exactly {last_completed_phase, external_event_count,
completed_file_count, successful_guard_sequences, binding_observed,
prepared_storage_observed, input_validation_observed,
final_execution_readiness_issued}. Derive all from replay, with the final field
always false. Reject forged counts, missing/reordered completion, incompatible
binding/prepared identity, fabricated native calls and contradictory digests.
Do not duplicate lifecycle disposition, dirty-scope, release or sync summaries.

The worker emits checkpoint snapshots through the existing atomic writer after
each phase and terminal attempt. The observed cut may be a prefix, including a
worker killed during native work; never fill missing tail facts from expected
control flow. Persist the last complete snapshot if serialization/I/O fails.
Failed collector/parser work poisons normal continuation through the existing
session.fail/controller path and cannot itself initiate an extra sync or ARM.
The journal/token roots are already in the retained source-session graph before
native execution. A terminal record is a projection of that held graph, not a
new guardian adoption after freeze. Its generation cannot authorize release.

Keep the existing custody artifact unchanged. A new CPU attachment validator
takes (runtime_record, environment, custody_artifact) and first calls existing
validate_custody_artifact. Require matching configuration/lane, and the runtime
lifecycle trace to be an exact prefix of the artifact worker trace (or equal).
If the artifact has no worker trace, validate only the standalone partial record
and report no terminal cross-binding. Do not infer a worker record from a
supervisor-only trace. No claim that OS exit equals individual cudaFree calls.

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
