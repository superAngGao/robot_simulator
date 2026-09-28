Based on my comprehensive independent design review of C4c2 runtime integration proposal v1, I now provide the complete report:

---

# C4c2 Runtime Integration Design Review

**Initiative:** q54-optical-lab-p12-4c4c2-runtime-integration  
**Stage:** design-request  
**Author:** codex  
**Version:** v1  
**Date:** 2026-09-29  
**Reviewer:** claude (independent session)  
**Review Date:** 2026-09-29  
**Status:** Request changes

## Reviewer Session Identity

- Model: Claude Opus 5 (claude-opus-5[1m])
- Session: Fresh independent CLI session (prior thread limit reached)
- Review mode: Read-only design inspection, CPU reasoning only
- No GPU validation performed
- No repository modifications made

## Reviewed Artifact

- **Base commit:** c3d2a3720deb9752152a6b195065be8f10b63200
- **HEAD commit:** 7717556fa0614e0512973424c109df640de6b7f4
- **Design document:** collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v1.md
- **Verified SHA-256:** f4a6234a473c95d4d4c8cf0aedb82b01d76501678548043fc14e79641172794b
- **Tree status:** Clean working tree on branch implementation/c4-supervised-waits
- **Single file added:** The design document only (488 lines)

## Requirements Verified

Successfully located and read all required dependencies:
- Parent C4c design v7: collab/q54-optical-lab-p12-4c4c-warp-native-render-arena__design-request__codex__v7.md
- C4c2a native-contract-extraction design v1: collab/q54-optical-lab-p12-4c4c2a-native-contract-extraction__design-request__codex__v1.md
- C4c2b supervised-waits design v3: collab/q54-optical-lab-p12-4c4c2b-supervised-waits__design-request__codex__v3.md
- Camera-resolution design v1: collab/q54-optical-lab-p12-4c4c2-camera-resolution__design-request__codex__v1.md

Confirmed approved code ancestors exist in repository history.

## Observed Source State

Inspected relevant implementation modules to assess implementability:
- `optical_worker/wait_protocol.py`: C4c2b supervised-wait protocol exists (v1)
- `optical_worker/wait_supervision.py`: Supervisor state machine exists
- `optical_worker/camera_contract.py`: Asymmetric camera resolution exists (frozen digest matches)
- `optical_worker/lifecycle.py`: Setup lifecycle with device-scope tracking exists
- `optics/native_arena.py`: C4a control-plane contracts exist
- `optics/cuda_arena.py`: CUDA C4b arena implementation exists
- `optics/warp_native_render.py`: Warp kernel definitions exist

All C4c2a/C4c2b prerequisite contracts are present as independent CPU-safe modules.

## Summary

This design proposes the complete C4c2 runtime integration required to execute R0/R1 Warp native rendering through closed-worker supervision with guarded waits. It correctly identifies the integration gaps between approved pieces (C4c2a contracts, C4c2b protocol, camera resolution, parent C4c design) and presents a comprehensive proposal for materialization, supervision versioning, setup/warmup ordering, Warp timeline/arena, worker tokens, terminal observation, and failure custody.

The design demonstrates thorough understanding of the parent requirements and maintains consistent terminology with the approved foundation. However, several critical integration decisions require clarification or amendment before implementation can proceed safely.

## Verdict

**Request changes**

## Priority 1 Findings (Blocking)

### P1.1: Ambiguous warmup readiness transition creates authority cycle risk

**Location:** Section 5, steps 3-5  
**Trigger:** During setup, provisional readiness must be created before warmup (step 3), but "NEW final readiness" certification occurs after warmup (step 5)

**Issue:** The design states "create provisional readiness truthfully naming warmup0" (step 3), then "certify NEW final readiness... using observed compilation/ten-warmup facts" (step 5), with the directive to "Never edit provisional readiness/factory evidence or reuse its identity as final authority."

However, the mechanism for how warmup execution occurs between steps 3-5 is underspecified. If warmup requires readiness to execute (as implied by parent C4c v7's preparation requirements), and warmup must complete before final readiness exists, this creates a temporal dependency problem.

**Consequence:** Implementers face three problematic interpretations:
1. Execute warmup without readiness authority (violates parent requirement that all execution requires readiness)
2. Use provisional readiness to authorize warmup, then claim it was never "used" for authority (semantic confusion)
3. Create an undocumented third readiness state

Additionally, step 5 states readiness references "exact prepared arrays" but those arrays were already registered in step 2. The phrase "certify NEW final readiness for the same registered exact prepared arrays" suggests the arrays don't change, only the readiness evidence changes. But C4c2a section 8 requires "Input readiness describes its own uploads/warmup facts" - if warmup happened between steps 3-5, how does the original upload readiness from step 2 incorporate warmup facts that hadn't occurred yet?

**Repair direction:** Clarify the exact readiness state model:
- Option A: Provisional readiness is explicitly warmup-capable but not measurement-capable; define the state transition and authority scope precisely
- Option B: Setup has no readiness until after warmup; specify how warmup commands are authorized without readiness
- Option C: Single readiness with incremental certification after uploads, then extended after warmup

Specify exact field differences between provisional and final readiness evidence. Define which operations require which readiness state. Ensure the evidence DAG remains acyclic when readiness references "uploads/warmup facts" but warmup consumes readiness.

**Proof required:** After repair, independent implementation must show warmup execution authorization path and readiness evidence construction order with no circular references or authority backdating.

### P1.2: Guardian seal timing contradicts parent C4c v7 section 6.6.3

**Location:** Section 5 step 6, section 1 clarification paragraph  
**Trigger:** Attempting to reconcile seal timing between this design and parent

**Issue:** Parent C4c v7 section 6.6.3 states: "The guardian is created before CUDA-capable imports/init and incrementally owns every returned object... `SEAL` freezes the exact coverage manifest before the final setup quiescence barrier and token issuance."

This design section 5 step 6 states: "Run final target-device barrier under setup guard AFTER all final recording, JIT, event initialization and resource operations. Commit quiescence only after sync success AND DISARMED ACK, bound to exact epoch/target/final readiness/prepared graph and sealed coverage."

The temporal order conflict: parent says seal-then-barrier; this design says barrier-then-seal (implied by "sealed coverage" being an output of step 6's commit). Section 1's clarification attempts to resolve this: "Coverage becomes immutable BEFORE the final device barrier; no execution or token release is permitted until that supervised barrier succeeds."

However, "coverage becomes immutable" is not the same as "seal freezes coverage manifest." The parent's seal is a one-way operation that produces a coverage digest. If coverage is sealed before the barrier, the barrier cannot be included in sealed coverage. If the barrier is included in sealed coverage, sealing must occur after the barrier.

**Consequence:** Implementers cannot determine whether:
1. Guardian seal occurs before barrier (parent literal reading) - but then barrier execution isn't covered
2. Guardian seal occurs after barrier (this design reading) - but contradicts parent
3. Two seals exist (coverage freeze + final seal) - undocumented mechanism

This ambiguity directly affects failure handling: if the barrier fails or times out, is coverage sealed or unsealed? Can partial setup with failed barrier claim sealed status?

**Repair direction:** Explicitly reconcile with parent C4c v7. Recommended: distinguish "coverage freeze" (no more additions after step 5) from "coverage seal with barrier evidence" (produces final digest after step 6). Or explicitly amend parent design and mark the deviation.

**Proof required:** After repair, show exact coverage evidence construction with all fields and their sources, including whether barrier success is in or out of sealed coverage.

### P1.3: Section 3 row extension versioning conflicts with section 2 row stability claim

**Location:** Section 3 final paragraphs vs section 2 scope  
**Trigger:** Reading gate versioning requirements

**Issue:** Section 2 states: "No existing case ID/hash, scene/replay/fixture digest, row, threshold or gate is rewritten."

Section 3 concludes: "Gate/result versions increment from v12 to v13 for this binding extension... Add resolved_render_sequence_sha256 to each applicable case, and require bound_render_parameters plus hash in stage binding/receipt... Canonical row-semantic hashing includes the new sequence identity. Old v12 records cannot masquerade as v13."

These are incompatible. Adding required fields to rows and changing their semantic hash is rewriting the row definition. The six frozen R0 rows and two R1 rows mentioned in section 2 would need new identities under v13 schema, which contradicts the no-rewrite claim.

Furthermore, section 3 states "case/content hashes... remain unchanged" but earlier says "Add resolved_render_sequence_sha256 to each applicable case" - adding a field to a case structure changes its content hash unless "case" is being used ambiguously.

**Consequence:** Unclear whether:
1. Existing rows are grandfathered with optional camera binding (breaks "require bound_render_parameters")
2. Existing rows get v13 definitions with different hashes (contradicts "row... not rewritten")
3. New parallel v13 rows exist alongside v12 rows (doubles test matrix, unclear which is normative)

**Repair direction:** Clarify exact scope:
- If rows are immutable: specify that only NEW v13 rows are added, old v12 rows remain, and specify migration/deprecation plan
- If rows are versioned: admit that row semantic identity changes, specify v12→v13 transformation rules, update section 2 to reflect row redefinition scope

Distinguish "case content" (scene/replay data) from "case metadata" (camera resolution). If camera resolution is metadata added to case records, specify whether it changes the case hash or is stored separately.

**Proof required:** After repair, show one example row's complete v12 and v13 identity, confirm all hash inputs, and verify whether test matrices reference v12 or v13.

## Priority 2 Findings (Must address before implementation approval)

### P2.1: Resolved render sequence construction order underspecified for Go2

**Location:** Section 3 camera resolution for Go2  
**Trigger:** "Go2 at either resolution: actual imported visual bounds, extracted orbit builder, 120 frames..."

**Issue:** The design correctly freezes asymmetric camera via the approved camera contract with exact binary64 hex values. For Go2, it specifies "extracted orbit builder" but doesn't define from where. C4c2a section 1 notes "Go2 materialization still depends on the example MJCF importer; camera construction... need the subsequent C4c2 integration design."

This design (the subsequent C4c2 integration design) delegates to "extracted orbit builder" without defining its source, ownership, or whether it's CPU-only. The phrase "Retain its exact formula" suggests pre-existing code, but "extracted" suggests a move operation similar to C4c2a.

**Consequence:** Implementation cannot determine:
- Whether orbit builder extraction is in-scope for C4c2 or deferred
- Whether Go2 cameras can be resolved CPU-only or require GPU materialization
- Whether test matrix can run without Go2 assets vs must fail closed

**Repair direction:** Explicitly state whether orbit builder extraction is included in C4c2 scope or deferred to later slice. If included, specify source location and CPU-safety. If deferred, clarify that Go2 rows cannot execute until that extraction completes.

### P2.2: Warmup diagnostic setting conflicts with final profile binding

**Location:** Section 5 step 4 vs section 3 profile binding  
**Trigger:** "Run ten actual warmup transactions with the frozen row profile/shadows and pass diagnostics setting"

**Issue:** Section 3 binds "profile,shadows,diagnostics" from the frozen row/pass. Section 5 step 4 says warmup uses "pass diagnostics setting" which implies diagnostic mode is on during warmup. Section 6 states diagnostic vs primary arenas have different event allocations.

If warmup must run with row-frozen diagnostics mode, then:
- Primary (diagnostics-off) rows must warm up with diagnostics off (zero events)
- Diagnostic rows must warm up with diagnostics on (timing events allocated)

But step 5 creates "factories/bundles" after warmup using "observed compilation/ten-warmup facts." If diagnostic status changes between warmup and measurement, the factories observe different event configurations.

**Consequence:** Unclear whether diagnostic mode is:
1. Fixed per row (warmup matches measurement) - implies different warmup paths for primary/diagnostic
2. Chosen at preparation time (warmup uses one mode, measurement might differ) - invalidates "frozen row profile/shadows"
3. Always on for warmup, then final factories select subset - wasteful allocation

**Repair direction:** Clarify that diagnostic mode is row-frozen and warmup executes the exact same diagnostic configuration as measurement. Or explicitly allow warmup to differ from measurement and specify the validation that ensures compatibility.

### P2.3: S rows camera resolution explicitly unresolved but rejection point ambiguous

**Location:** Section 3 end, section 9 acceptance  
**Trigger:** "S camera resolution/adoption remains outside this scope; S rows still reject."

**Issue:** The design correctly excludes S rows but doesn't specify the rejection point. Section 3 says "S rows are explicitly unresolved/disabled" but also says unknown cases "fail" - S rows are known cases, just unresolved for camera.

Section 9 item 4 requires "R2/S/host reject before allocation" which groups S with R2 host rows, but those reject due to payload policy while S rejects due to camera resolution.

**Consequence:** Three possible rejection points with different evidence:
1. Camera resolution fails for S (no resolved sequence exists)
2. Row preparation rejects S explicitly by ID
3. Allocation phase rejects S alongside R2 host

Different rejection points produce different evidence and test requirements.

**Repair direction:** Specify exact rejection point: if S cannot resolve camera, say "camera resolution rejects S case IDs before any preparation." If S is separately blocked, say "S rows reject at preparation before allocation even though camera resolution is out of scope."

### P2.4: Section 7 token lifecycle "ACQUIRED never exported" conflicts with cancellation rules

**Location:** Section 7 token states and cancellation  
**Trigger:** "ACQUIRED is never exported. EXECUTE accepts only the next prepared sample"

**Issue:** The state flow shows PREPARED → ACQUIRED → SUBMISSION_POSSIBLE with "ACQUIRED is never exported." Cancellation section states "CANCEL_PREPARED_SAMPLE is legal only for an unacquired prepared token" but then "If an internal acquire has occurred but the first-submission latch is still false, the worker may use the same pre-submission cancel transition."

This creates ambiguity: if acquire happens internally before EXECUTE, the token is no longer "prepared" (it's acquired), but CANCEL_PREPARED_SAMPLE is the only cancel operation defined. The phrase "internal acquire" suggests acquire happens automatically, but the design also describes acquire as an atomic C4 operation with specific timing.

**Consequence:** Implementers cannot determine:
1. When exactly does prepared→acquired transition occur
2. Can CANCEL_PREPARED_SAMPLE operate on internally-acquired tokens (contradicts "unacquired" requirement)
3. Is there a hidden CANCEL_ACQUIRED operation or does cancellation reach through states

**Repair direction:** Clarify that EXECUTE performs acquire-then-enqueue atomically, so from external view tokens are either prepared-and-cancelable or acquired-and-uncancelable. Or define explicit ACQUIRED token state with its own cancellation rules.

### P2.5: Protocol v2 designation without v1 reference creates versioning ambiguity

**Location:** Section 4 and section 7  
**Trigger:** "Protocol v2 retains canonical framing/limits and eight operation names"

**Issue:** The design introduces "protocol v2" with "runtime-contract v3" but doesn't specify what v1 was or what changed. C4c2b describes a new guarded-wait protocol v1. Parent C4c design describes a different protocol. The relationship is unclear.

Section 4 says protocol v2 uses "the same eight request operations" but doesn't list them for v2, only references them. If v2 "retains" operations from v1, v1 must be defined.

**Consequence:** Cannot verify that v2 is actually a new version requiring new runtime-contract v3, or just a designation for "the protocol in this design." Cannot assess compatibility or determine what existing code (if any) implements v1.

**Repair direction:** Either cite the v1 protocol definition (if it exists) and enumerate changes, or clarify that this is the first versioned protocol and "v2" is a mistake (should be v1).

## Actual Verification Performed

### Source Code Inspection
- Confirmed C4c2a extracted contracts exist in `optical_worker/` with CPU-safe imports
- Confirmed C4c2b guarded-wait protocol v1 exists with MAX_SEQUENCE, guard configuration, and client/supervisor state machines
- Confirmed camera contract resolves asymmetric camera with frozen SHA-256 matching design
- Confirmed lifecycle.py exists with device-scope tracking and setup states matching parent C4c v7
- Confirmed cuda_arena.py implements C4b with C3 provenance validation
- Confirmed native_arena.py provides control-plane slot/lease/lifecycle contracts
- Observed warp_native_render.py exists with kernel definitions

### Document Cross-Reference
- Verified parent C4c v7 multi-GPU quarantine rules match this design's section 8 failure custody
- Verified C4c2b section 9 integration obligations match this design's claimed dependencies
- Verified camera resolution design camera digest matches this design's section 3 asymmetric reference
- Confirmed C4c2a extraction delivered the contracts claimed as prerequisites

### Design Internal Consistency
- Checked schema version progression (gate v12→v13, runtime-contract v2→v3, evidence schemas all v1)
- Verified boundary naming (R0/R1/R2, setup/resolve/emergency roles) consistent throughout
- Confirmed five implementation checkpoints (C2c-C2f + activation) mentioned in section 2 align with later decomposition
- Noted guardian/lifecycle/supervision terminology matches parent and prerequisites

## Unexecuted Checks (Limitations)

Cannot verify without GPU or running implementation:
- Warp factory/command/timeline implementations (section 6 Warp transaction) - not implemented yet
- Actual warmup execution and readiness transitions (P1.1 finding)
- Guardian seal and coverage evidence construction (P1.2 finding)
- Multi-GPU device-scope expansion and quarantine (section 8 custody rules)
- Real Go2 asset materialization and camera orbit builder (P2.1 finding)
- Actual supervisor ACK/DISARM sequences during setup (section 5 guarded segments)
- Performance characteristics of O(1) acquire claim (section 6 acquire cost)

Cannot verify without implementation review:
- Exact source allowlist for worker after C4c2 integration
- AST policy handling of ctypes usage in warp_native_render.py
- Actual runtime-contract v3 manifest JSON structure
- Complete token registry and trace buffer capacities for 120-600 frame tests

Cannot assess without baseline/result activation:
- Numerical correctness of Go2/dynamic cameras (deferred to later validation)
- Render result quality or performance (explicitly out of scope)
- Actual comparative timings between Warp R0/R1 paths

## High-Risk Areas Inspected

1. **Setup quiescence proof vs guardian seal**: Found P1.2 parent contradiction
2. **Warmup authority and readiness lifecycle**: Found P1.1 temporal dependency ambiguity
3. **Row versioning and gate compatibility**: Found P1.3 rewrite vs stability conflict
4. **Camera resolution CPU-safety and materialization dependencies**: Found P2.1 Go2 orbit builder gap
5. **Token lifecycle and cancellation boundaries**: Found P2.4 acquired state export ambiguity
6. **Device-scope expansion and emergency sync conditions**: Consistent with parent, no new issues
7. **Protocol versioning and supervised wait integration**: Found P2.5 v1/v2 ambiguity
8. **Evidence DAG acyclicity**: Depends on P1.1 resolution for readiness→warmup→readiness
9. **Failure custody and guardian transfer**: Correctly inherits parent, consistent with C4c2b
10. **Closed gates and activation boundaries**: Correctly sealed, multiple explicit reminders

## Approved Scope (After Required Repairs)

If findings are addressed, this design would authorize:
- **In scope for C4c2 implementation:**
  - Section 3: CPU materialization of workloads/assets/mjcf_visual/camera_presets to optical_worker
  - Section 3: Resolved render sequence v1 with frozen binary64 camera scalars and bound parameters v1
  - Section 4: Runtime-contract v3, protocol v2, guarded-wait v1 integration (after version clarification P2.5)
  - Section 5: Setup graph with guarded segments, ten warmup frames, provisional→final readiness (after P1.1 repair)
  - Section 5: Guardian seal and setup quiescence with target barrier (after P1.2 reconciliation)
  - Section 6: Warp transaction template with PREPARED→VALIDATED_OPEN→ENQUEUING→FINISHED states
  - Section 6: R0 3-stage, R1 5-stage transaction graphs with exact diagnostic phase pairs
  - Section 7: Worker token lifecycle with PREPARED→ACQUIRED→SUBMISSION_POSSIBLE→COMPLETION_RECORDED→RESOLVED_HELD→RECLAIMED
  - Section 7: PostReclaimIdleEvidence for clean close
  - Section 8: Failure custody with foreign/unknown device quarantine
  - Section 9: Specified test matrix for CPU moves, camera chain, authority, R0/R1/R2 rows, setup faults, transport loss, dual-GPU

- **Explicitly closed (correct exclusions):**
  - C4c3 static interop R2 (separate slice)
  - C4c4 closure review (separate review)
  - C4d/B3 joint lifecycle and host payload
  - Measurement integration and result activation
  - Shipping, a3b, baseline generation, backend selection
  - Product activation flags (all remain false)
  - Five R2 host-payload rows (reject before allocation until B3)
  - S rows (explicitly unresolved camera, after P2.3 clarification)

## Limitations

This review inspected design document internal consistency, cross-references to requirements, and observable source state. It cannot verify:
- GPU execution correctness
- Actual runtime performance or memory usage
- Complete source allowlist after integration (deferred to implementation)
- Numerical camera formula correctness (deferred to CPU tests specified in section 9)
- Complete fault injection matrix coverage (depends on implementation)

The review was conducted in CPU-only reasoning mode per review packet instructions. GPU validation, device-scope expansion, and multi-GPU quarantine tests remain mandatory during implementation code review as specified in section 9.

This is an independent design review session with no access to prior implementation conversation per collab/README.md isolation requirements.

## Conclusion

The C4c2 runtime integration design demonstrates thorough integration planning and correctly identifies the gaps between approved prerequisite pieces. The scope boundaries are well-defined, exclusions are explicit, and the test matrix is comprehensive.

However, three P1 findings create implementation ambiguity that must be resolved before code can proceed safely:
1. Warmup readiness authority and evidence construction order (P1.1)
2. Guardian seal timing relative to parent design (P1.2)  
3. Row versioning vs row stability claims (P1.3)

Five P2 findings require clarification to ensure complete implementation:
- Go2 camera orbit builder extraction scope (P2.1)
- Warmup diagnostic mode consistency (P2.2)
- S row rejection point (P2.3)
- Token acquired state and cancellation (P2.4)
- Protocol versioning baseline (P2.5)

After addressing these findings with explicit repairs, the design can provide a sound foundation for the five C4c2 implementation checkpoints (C2c-C2f + activation review).

---

*Review conducted 2026-09-29 by independent Claude Opus 5 session per collaborative review protocol. Report archived verbatim by design author.*
