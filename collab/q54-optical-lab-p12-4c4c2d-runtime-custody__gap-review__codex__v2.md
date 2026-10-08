# Independent runtime-custody schema and producer gap re-review

Initiative: q54-optical-lab-p12-4c4c2d-runtime-custody
Stage: independent design/gap re-review
Author: codex, independent reviewer /root/runtime_custody_schema_gap_v2
Version: v2
Date: 2026-10-08
Status: in_review
Related Files: runtime-custody implementation-plan v2; unchanged gap-review v1; runtime-adoption design v2 section 7; native_bootstrap.py; runtime_environment.py; native_contract.py; native_input.py; guarded_evidence.py; tests/support/native_runtime_process.py
Owner Summary: Request changes for two bounded source-binding specifications. The revised failure-cut policy, component token mappings and independent interpreter inputs address the original three findings in substance. However, the new text conflates the captured SourceSession with the later CanonicalNativeInput and omits the comparison between the independently captured case_id and the prepared manifest. Correct these exact joins before freezing the schema for implementation. This review does not claim an aggregate implementation or GPU acceptance exists.

## Verdict and inspected revision

**Request changes**, restricted to the candidate aggregate design. No implementation was edited or approved. This is a fresh independent review session, not the author session, with no subdelegation.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Review base: `0ae8df1377e288c30c86af6e3faf802a80b98d6d`.
- Frozen HEAD: `2acaf8362e0c9b17d35245ebc87f24c572a93b26`.
- Candidate only: `collab/q54-optical-lab-p12-4c4c2d-runtime-custody__implementation-plan__codex__v2.md`.
- Candidate SHA256: `aad17d8d58b04c17dd90058a2f60649a6fa063e7b10eec4d40c2c8bc66263ad7`.
- Prior report SHA256 independently checked: `bd114e897207ef8ecad4de367ebb12fff64d1e577b17b29a93836cf3382d7813`.
- Initial working tree clean. Base-to-HEAD diff contains only the 531-line v2 plan. Production and tests/support diff against `a3d750bbaf79d760a6727e56146d277e1cb3b770` is empty.
- Read-only remote: `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, clean detached `a3d750bbaf79d760a6727e56146d277e1cb3b770`.

AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, the complete candidate and complete unchanged prior report were read as UTF-8. The prior probe and final JSON were read and the original probe independently rerun. New scratch files are only in sibling `c4_assets`.

## Remaining findings

### R1 — Medium, blocking schema freeze: canonical source identity is equated with an earlier session identity

Locations: candidate lines 123–127 (`source_bound`), 156–178 (`runtime_binding`, `prepared_storage`, `input_validated`), and especially 403–405: “source/controller tokens equal the source-bound identities.” Actual producer: `native_bootstrap.py` 2167–2172, 2298–2299, 2404–2425, 2452–2465, 2513–2519; `native_contract.py` 280–281 and 331–339.

Trigger: the normal path reaches runtime binding. The source-bound event exposes only `session_token`, `bootstrap_token`, and `controller_token`. Its source custody objects are the captured `_SourceSession`, repository snapshot and initial bootstrap module. Canonical materialization later assigns a different object, `self.canonical_input`, and adopts it under `canonical_native_input`. `_RuntimeSetupBinding.source` and its authoritative registry tuple contain that exact canonical object. Upload passes that same object to the uploader; prepared registration requires `prepared.source is source` and `prepared.manifest is source.manifest`.

Therefore the runtime/prepared `source_token` cannot equal a source-bound session or bootstrap token under the candidate's own rule that different objects cannot share one global token. There is no canonical-input token in the source-bound event, and it cannot exist at that early occurrence. This is an identity distinction, not a naming preference. A literal equality check rejects every honest normal runtime/prepared record. Treating the token as the session instead loses the actual canonical-input join. Quietly reading “source-bound” as some unlisted later event requires the implementer to invent a schema rule.

Existing checks miss it because the strict component parsers describe imports, while lifecycle adoption rows do not encode `binding.source is prepared.source`. The new aggregate is supposed to supply this relation. The live production registry already distinguishes the two objects correctly; this finding is against the frozen text, not a production bug.

Repair direction: explicitly separate `session_token` from the canonical-input identity. Define `runtime_binding.source_token` and `prepared_storage.source_token` as the same actual `CanonicalNativeInput` object, and their relationship to the existing `canonical_native_input` adoption. State when its first journal reference becomes available. Either introduce an occurrence for the canonical source during materialization, or explicitly allow its first reference at runtime binding and cross-check the prior real adoption; neither needs another authority or device operation. Keep the controller equality with `source_bound.controller_token`. Do not require the later canonical object to be present in the pre-materialization baseline.

Proof needed: a CPU issuer test with distinct session, bootstrap, canonical source, binding and prepared sentinels; the honest identities receive distinct tokens where appropriate and the source token persists across runtime/upload/readback. Swapping only a runtime/prepared source reference to the session token, a second canonical source, or a foreign binding's source must reject. Early baseline and materialization-failure cuts must remain valid without inventing a canonical object or its adoption.

### R2 — Medium, blocking schema freeze: trusted case_id has no specified comparison to the emitted manifest

Locations: candidate lines 424–434 (trusted launch fields), 445–462 (explicit upstream comparisons), 474–485 (manifest validation), and 175–178 (input validation). Actual producers: `native_bootstrap.py` 2240–2304; `native_input.py` 450–467; `tests/support/native_runtime_process.py` 150–170.

Trigger: hold trusted launch inputs fixed for `go2_static_960x640_v1`, then substitute a complete internally consistent prepared manifest for `go2_static_1920x1080_v1`, updating the downstream manifest digest and related manifest fields consistently. Both are authorized canonical fixed cases and can have exactly the same input array bytes/layouts. The v2 text requires source revision/catalog/snapshot, lane, configuration and alias to match upstream inputs. Its manifest section requires a nonempty canonical case/source identity and internal semantics/hash/layout consistency. It never requires `manifest.case_id == trusted_launch_inputs.case_id`, nor defines a case check at materialization.

Consequence: an implementation can satisfy the listed comparisons yet accept evidence for a different fixed driver case. This is not an adversary replacing the trusted inputs; those remain unchanged. Matching storage extents, complete array-name lists and content/topology hashes cannot supply the missing case relation. Conversely, an implementer who rejects it must add an unstated cross-record rule. Since the purpose of these supplied inputs is to freeze upstream facts independently of the worker's record, this comparison belongs in the design.

Concrete retained-data evidence: the new CPU scratch probe reads the two existing fresh-reference Go2 manifest JSON files. Their `arrays` lists, `content_sha256` and `topology_sha256` are identical. Their manifest digests are respectively `ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12` and `4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e`. Differing fields are `case_id`, `case_sha256`, `fixed_render_fixture_sha256`, `scene_contract_sha256`, and `semantics`. Thus the complete valid manifest for the other case preserves the proposed array/storage checks. This is a design counterexample using retained data, not a claim that a nonexistent aggregate parser was executed or that the old runs were incorrect.

Repair direction: require the actual materialization case and every emitted manifest's `case_id` to equal the separately captured fixed launch case. Freeze any additional case-hash validation against the existing canonical case definition through a single shared CPU source, or explicitly delimit what is producer-verified versus offline-replay-verified. Do not add filesystem discovery, a backend import, or another independently maintained case policy to the parser. For early cuts before a manifest exists, specify that case execution is not yet observed; do not synthesize it from the trusted launch intent.

Proof needed: both real Go2 manifests pass for their own trusted inputs; cross-case attachment fails with trusted inputs held fixed, even though arrays/content/topology match. Test a one-field case substitution and a fully consistent alternate manifest with updated downstream hashes. Early baseline/failed pre-materialization records remain honest and accepted under their actual limited coverage.

## Independent disposition of original F1–F3

**Original F1: closed at the design-policy level.** V2 explicitly permits `completed -> failed`; it preserves previous return/completion entries. It keeps existing strict component parsers unchanged, specifies an immutable validated baseline and phase cuts, and chooses an unchanged earlier cut for admitted-but-failed component tails or half-initialized producers. It forbids trimming or guessed subset retries, retains raw tails and original errors, distinguishes unavailable coverage, and prevents an old cut from resuming execution. The original probe still produces the same producer/parser behavior, as expected: direct actual `fail_native_call` changes a completed frame to failed; native-global admitted/failed and replacement admitted/failed prefixes reject. V2 now has a truthful, explicit selection for those cases. The exact append/adoption/swallowed-failure CPU tests remain required during implementation; design closure is not their execution.

**Original F2: the missing component/return mapping is substantively addressed; the identity section is not fully closed because of R1.** V2 gives occurrence-bound local-to-global mappings, namespace separation, stable repeated mappings, exact first-reference rules, returned-before-adopt observations, and native-global adoption-event joins. Actual producers retain the relevant objects: replacement_objects, native-global event bindings, captured alias descriptor/resolution, admission.records and runtime_owners. `_SourceSession.bootstrap_owner` is indeed the captured `__main__`, so the alias-main-to-bootstrap join is correct. Independent integer namespaces in the rerun still differ as expected; they must not be treated as common global integers. The new R1 is a specific inconsistency in the later source join, not a reason to discard these repairs.

**Original F3: closed for its original source/interpreter gap.** All entrypoints now take independently captured upstream inputs. The seven-element interpreter identity, executable resolution/hash and fixed multiprocessing source path/hash can be captured using shared existing CPU bootstrap logic under the pinned interpreter. Parser comparisons need no filesystem discovery or backend import. Early environment-null records still have a defined trusted comparison input. The rerun confirms that an alias source-hash substitution continues to pass only the existing local alias parser; the specified new upstream equality would reject it. R2 is an additional unbound field introduced in that trusted input contract, not a claim that the original descriptor repair failed.

## Other high-risk checks and bounded comments

The one-observer/one-lifecycle-authority architecture remains appropriate. `GuardedLifecycleEvidence` and `validate_custody_artifact` retain state/disposition authority; the aggregate and selection sidecar remain descriptive. Exact lifecycle-prefix attachment can work for early and terminal records, and missing worker trace is explicitly unavailable rather than fabricated. Source/manifest/environment digests stay upstream, avoiding cyclic evidence.

The actual source-bound baseline is feasible once the required CPU modules are admitted in the captured prelude. Alias capture occurs before binding and the component can carry an empty event list. This observation does not authorize calling its current auditing getter after terminal failure: the proposed non-auditing CPU projection must read retained issued facts. Normal file creation orders external create_started, call started, retained return, call returned, external created, call completed. Exec orders external exec_started, call started, returned, wrapper reconciliation, call completed, external completed. Preserve these distinct sequences in implementation; the wrapper can still reconcile after delegated return. A postterminal observer cannot transform that into new successful work.

The shared `native_manifest.py` extraction is a feasible single-source boundary. Existing manifest classes currently reside in NumPy-importing native_input; simply importing that module for early replay would violate the proposed boundary. V2 now explicitly extracts and re-exports the same classes and codec. Existing `_PREPARED_STORAGE` holds the exact arrays, ready event, storage owners and frozen signatures after registration. Projecting these after the existing storage recheck needs no extra device getter. The frozen signature layout and CUDA element-stride versus Warp byte-stride distinction match the source. Generic constructors must remain compatible while the C2d3 parser enforces the narrower canonical path. Tests should also recompute the manifest content digest from the serialized array-manifest list using the shared codec where applicable; this checks descriptor consistency and does not pretend to rehash device bytes.

Non-blocking implementation follow-up: persist and validate the selected-record/selection-sidecar digest relationship across partial writes. The main record and sidecar are separate atomic files; an in-memory validated record may be newer than the durable record when its write fails. Readers must not accept a sidecar naming absent bytes or silently claim that a stale sidecar describes a newer file. Test failure between the two writes and preserve the current design's incomplete-output/closed-gate outcome. This does not require making the sidecar lifecycle authority.

## Verification, commands and results

1. Local Git revision/status/branch, base-to-HEAD stat, candidate/prior-report SHA256, and production/tests-support baseline diff. No candidate edits, tracked outputs, installs or tests that exercise GPU paths.
2. UTF-8 reads and `rg` callsite searches covering importer create/exec, native call mutation, global/replacement issuers, alias capture/observation, source custody, canonical materialization, runtime registration/consume, upload/storage registry, lifecycle replay and driver atomic persistence. Reviewed runtime-adoption section 7 and relevant predecessor custody/publication contracts. Read the current-Warp-Go2, captured-ambient-and-terminal, and CUDA-normal-workloads v1 reports as scoped claims.
3. `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && git rev-parse HEAD && git status --short && git branch --show-current'`: expected clean detached production revision.
4. Piped the unchanged `runtime_custody_schema_gap_probe_v1.py` over SSH stdin into `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B - /data2/shared/ang.gao /data2/shared/ang.gao/robot_sim_c2d3_binding_wip /data2/shared/ang.gao/183_cuda_go2_author_proposed_v1.json`. No remote file write. Fresh output retained as `runtime_custody_schema_gap_probe_v2_rerun.json`.
5. Rerun on CPython 3.12.13: untouched native-global/replacement/alias components and custody artifact accepted; admitted/failed native-global prefix rejected with `invalid native global admission`; replacement prefix rejected with `replacement admission differs from observation`; changed alias source digest accepted by its local parser; direct production completed-to-failed mutation observed. Backend-import set empty. Environment digest `79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919`.
6. Fresh probe confirmed historical raw report/worker/custody hashes respectively `f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20`, `e32db1fb74a300216bcba4f4b1dd926e3276cfeeb8b48437b52950bd27b8afc6`, `f18a834b1519688be73603ff512bea825f158f42e586722edafd9a41ed86bcde`. Schema remains `q54.c2d3.driver-observation.v0`; these are not new aggregate worker executions.
7. `C:/Python314/python.exe -I -S -B runtime_custody_schema_case_probe_v2.py`: standard-library read-only comparison of retained CPU manifest files, output retained as `runtime_custody_schema_case_probe_v2.json`. Results stated in R2. No rematerialization or numerical GPU rerun is implied.

The new artifacts are the report, original-probe rerun JSON, and case probe script/JSON in `C:/Users/AG/新建文件夹/c4_assets`. The original probe and original final result were left unchanged. No GPU API/query/reset, package installation, process signal, candidate source edit or remote write occurred.

Final revision/status verification after report creation: local HEAD and candidate SHA256 were unchanged and the local tree remained clean; remote HEAD remained the pinned clean detached production revision. Both status checks completed successfully.

## Scope and remaining gates

No aggregate implementation exists, so this review cannot execute its future parser/issuer/attachment suite or prove any new fault behavior. R1 and R2 need a corrected frozen design and fresh independent re-review. Original cut-selection and identity repairs need their stated live CPU fault/mutation tests during implementation, followed by real new-schema observations.

Existing both-lane four-normal-case/ambient/Warp-terminal evidence is useful bounded evidence, not aggregate acceptance. Full import/init/allocation/copy/event/audit/owner faults, hangs, ACK/EOF, foreign/UNKNOWN/dual-GPU, cleanup/missing-exit coverage and fresh whole C2d3 review remain required. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping remain CLOSED. No backend-selection or optical performance conclusion is made.
