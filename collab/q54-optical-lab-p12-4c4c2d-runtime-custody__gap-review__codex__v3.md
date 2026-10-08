# Independent runtime-custody schema and producer gap re-review

Initiative: q54-optical-lab-p12-4c4c2d-runtime-custody
Stage: independent design/gap re-review
Author: codex, independent reviewer /root/runtime_custody_schema_gap_v3
Version: v3
Date: 2026-10-08
Status: accepted for the bounded design scope below
Related Files: runtime-custody implementation-plan v3; unchanged gap-review v1/v2; runtime-adoption design v2 section 7; native_bootstrap.py; runtime_environment.py; native_contract.py; native_input.py; guarded_evidence.py; guarded_lifecycle.py; lifecycle.py; tests/support/native_runtime_process.py
Owner Summary: Approve the exact v3 aggregate design for implementation. R1 and R2 are closed at design scope, and the F1–F3 repairs remain intact. No substantive design finding remains from this review. The aggregate does not yet exist, so this is neither implementation approval nor new real-device or whole-C2d3 acceptance.

## Verdict and exact inspected state

**Approve**, restricted to the frozen aggregate schema, observation placement, upstream comparisons, failure-cut selection and attachment/persistence contract in implementation-plan v3. This was a fresh independent reviewer session, with no candidate authorship, candidate edit or subdelegation.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Base: `2b3fe90be49a6b3a3d7ef43f0e5ca08e37db0f2c`.
- Frozen HEAD: `ab24a7fe6619362875596fd8c3694c0756b5ef4c`.
- Candidate only: `collab/q54-optical-lab-p12-4c4c2d-runtime-custody__implementation-plan__codex__v3.md`.
- Candidate SHA256: `6dc1fa68fef830b5c5f631128cedb78a35f79ee31767879424945550fa483beb`.
- Prior gap-review v2 SHA256 independently checked: `9b4f6d0b1ca538976f6af3e27b421d538ae4c7410c39a78a4cc73c17e38bd858`.
- Initial working tree clean. Base-to-HEAD diff contains only the 587-line candidate. Production and tests/support diff against `a3d750bbaf79d760a6727e56146d277e1cb3b770` is empty.
- Read-only remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip`, clean detached `a3d750bbaf79d760a6727e56146d277e1cb3b770`.

AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md were read completely as UTF-8. The complete candidate and prior v1/v2 reports were read, and the v2-to-v3 text diff was independently inspected. Original repro scripts/results and the author extraction script/log were inspected as claims to verify. All new files are in sibling `c4_assets`; no implementation was changed.

## Findings and prior-finding disposition

No substantive findings remain in this bounded design review.

### R1 — Closed at design scope: session and canonical source have distinct identities

Candidate lines 414–427 now explicitly distinguish the captured SourceSession from the later CanonicalNativeInput. Only the controller joins directly to source_bound.controller_token. The runtime/prepared source token first appears at runtime_binding and names the prior actual canonical_native_input host adoption; it must differ from all three source_bound identities. The prepared wrapper is distinct, and the canonical identity persists through upload and validation. Baseline and failed materialization cuts do not invent it.

This matches the actual producer. `native_bootstrap.py` 2167–2172 adopts session/snapshot/bootstrap before materialization. Lines 2298–2299 later assign and adopt the canonical source. Lines 2404–2427 place both the session and that canonical source into the issued binding/registry and adopt the binding. Lines 2452–2465 require binding.session to be the session while binding.source and session.canonical_input are the exact registered source. `native_contract.py` 280–281 and 331–349 separately bind the exact prepared wrapper, canonical source, manifest and registered arrays/event. These are different objects and relations, as v3 now specifies.

The distinct-token rule and prior adoption check make the original literal-equality contradiction impossible. The required sentinel and substitution tests at candidate lines 573–577 are appropriate implementation obligations. They have not been executed against a nonexistent aggregate. As additional bounded regression evidence, this review ran existing production binding/state tests with actual guard protocol and simulated backend leaves: 30 passed, 2 actual CPU-materialization cases deliberately deselected. They exercise canonical-source drift, copied binding rejection, consume-once behavior, returned stream retention after adoption failure, and upload/readback rejection before helper entry. They do not test the new aggregate issuer.

### R2 — Closed at design scope: completed case and every prepared manifest join to launch intent

Candidate lines 180–184 and 467–480 now require the completed materialization's actual canonical_input.manifest.case_id and every prepared manifest.case_id to equal the independently captured trusted launch case; prepared case also equals the preceding observed materialization case. The text explicitly rejects a complete consistent alternate manifest with updated downstream hashes. Before successful materialization, case intent is not relabeled as observed execution.

The comparison has a real producer source: materialize_native_source constructs the canonical object through the existing fixed/replay builders and returns its manifest only after the guarded segment succeeds. `native_input.py` 358–467 validates the packed-scene/fixed/replay relationship and emits the manifest from the canonical case. V3 explicitly leaves canonical case-hash recomputation with that live producer and requires descriptor/hash consistency offline. It no longer implies that the offline parser has an independent case-definition oracle.

Fresh execution of the original case probe reconfirmed that the retained Go2 960 and 1080 manifests have identical arrays, content and topology but different case_id, case_sha256, fixed_render_fixture_sha256, scene_contract_sha256 and semantics. Their manifest digests remain `ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12` and `4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e`. V3's equality against fixed upstream inputs excludes this original counterexample independently of storage similarity. Both own-case acceptance and cross-case/single-field mutation tests remain explicitly required during implementation.

### Original F1 — Failure-cut repair remains closed at design scope

The journal continues to permit completed-to-failed while preserving earlier return/completion entries. The existing component parsers remain strict; admitted-then-failed or partially initialized components select the unchanged earlier validated record when the current projection cannot pass. The baseline, immutable last_validated_record, no guessed-subset retry, retained raw tails, sticky failure and no continuation from a prior cut remain explicit.

I inspected actual create/exec ordering at `native_bootstrap.py` 540–616, call mutations at 1029–1071, and native-global append/adoption/completion at 1073–1194. Native-global adoption precedes its pending event; its admitted event precedes final file-call completion. The unchanged repro again observed actual completed-to-failed mutation and rejection of the constructed admitted/failed prefixes by both strict parsers. Those facts are expected and remain handled by the v3 cut-selection contract, not by weakening the parsers or rewriting history.

### Original F2 — Identity mappings and return retention remain closed at design scope

Occurrence-bound namespace mappings, exact used-token coverage, stable repeated mappings, first-reference rules, native-global adoption references and held alias resolution owners remain present. R1's corrected canonical join completes the affected later-source relation. The three component token namespaces remain independent in actual data; native-global token 1, replacement original token 1 and alias main token 1 denote different owners. V3 does not equate those integers.

File module return is retained in admission.records before fallible return_native_call/adoption. Runtime stream/target is retained in runtime_owners before metadata/adoption. The returned_owner event sites and null adoption facts reflect those actual windows. The table itself remains descriptive; live registries, not a parsed token or copied digest, control bindings/storage/readiness.

### Original F3 — Independent upstream inputs remain closed at design scope

All entrypoints continue to require independently captured source/lane/configuration/case/interpreter inputs, including early environment-null records. Alias identity/path/hash compare to that descriptor, and environment executable/hash/version comparisons are specified separately. Existing bootstrap capture reads the fixed multiprocessing source without importing it and obtains the seven-part interpreter identity. Parser-side filesystem discovery is prohibited.

The remote rerun again accepted an alias source-hash substitution only in the old local alias parser. This is its documented descriptive limitation; v3's separately supplied unchanged descriptor supplies the missing comparison. No source snapshot is claimed to contain stdlib bytes, and no upstream digest depends on the aggregate.

## Affected invariants and remaining implementation obligations

The added partial-write contract (lines 350–356) closes the prior non-blocking persistence concern. Selected bytes precede the selection sidecar; readers compare the actual durable file's digest and lifecycle count. Missing/mismatching/stale output is incomplete and cannot become a valid attachment by inference. In-memory validation is not durable persistence. Tests must fault between the two atomic writes and between validation and the first write. The existing driver writer uses a temporary file, flush/fsync and replace; its two-file failure boundary must be exercised in the eventual implementation. No new transactional or lifecycle authority is inferred from this sidecar.

Manifest extraction remains a feasible single-source boundary: the two classes/codec currently live in NumPy-importing native_input. V3 explicitly moves/re-exports the same classes through a stdlib-only module and preserves generic constructors while imposing narrower canonical replay validation. Independent rerun of the supplied exact-AST extraction probe produced byte-identical payloads/digests for all four retained CPU manifests and imported no backend. This is extraction feasibility only.

The added content_sha256 recomputation matches the actual producer at `native_input.py` 442. A new scratch probe executed the exact existing codec AST, recomputed the array-list content digest for all four retained manifests, and confirmed that changing one array's content digest changes that composite digest. This checks descriptor consistency; it does not read native bytes or supply numerical validation.

Storage signatures/projected owners are available in the actual `_PREPARED_STORAGE` registry after registration and the existing recheck. V3's field names, units (CUDA element strides/Warp byte strides), source/event/array identity joins and extent checks match `native_contract.py` 153–231 and 272–357. No additional physical getter pass is required for projection. Its shared dtype metadata and zero-sized array handling still need actual implementation tests.

Concrete application of the existing earlier-adoption exception: the upload core adopts the prepared wrapper as cuda_a2_prepared_input (`native_setup.py` 324) or warp_a2_prepared_input (438), before SourceSession calls _hold_runtime_owner with guarded_prepared_input. The latter preserves the actual prior row at `native_bootstrap.py` 2315–2318. Candidate lines 438–439 expressly preserve an earlier legitimate row/role and therefore cover this normal path. Replay must accept that actual lane role and must not require a fabricated guarded_prepared_input re-adoption. The coordinator independently raised this source fact near report completion; I verified the actual sites and found the existing exception sufficient. This is a concrete implementation reminder, not a blocking schema change or permission to accept arbitrary roles.

Lifecycle authority remains the existing GuardedLifecycleEvidence/validate_custody_artifact path. I checked normal_disarmed recording, guardian adoption/release, terminal intent and existing supervisor/worker joins. The journal's phase completion uses the post-segment null current guard while referring to the actual disarm event carrying the completed guard; these are deliberately distinct. Failure serialization cannot undo an actual release, initiate another sync, or adopt after freeze. Guardian owner-table release is not proof of individual native deallocation; observer/session roots remain held through process exit.

The exact lifecycle-prefix attachment and unavailable_worker_trace outcome remain conservative. Historical v0 observations cannot be migrated into claimed v1 executions. Required raw failure retention, swallowed nested failure tests, mutation tests and original-deadline supervision evidence remain part of implementation acceptance, not waived by this approval.

## Verification and artifacts

Commands executed, using the stated local repository unless noted:

1. `git rev-parse HEAD`, `git status --porcelain=v1`, base-to-HEAD diff/stat, production/tests-support diff against a3d750b, SHA256 checks, UTF-8 reads and `rg` callsite searches. The candidate is the sole base-to-HEAD change.
2. `ssh tileops-dev 'cd /data2/shared/ang.gao/robot_sim_c2d3_binding_wip && git rev-parse HEAD && git status --short && git branch --show-current'`: clean detached pinned production revision.
3. Piped unchanged `runtime_custody_schema_gap_probe_v1.py` over SSH stdin into `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B - /data2/shared/ang.gao /data2/shared/ang.gao/robot_sim_c2d3_binding_wip /data2/shared/ang.gao/183_cuda_go2_author_proposed_v1.json`. Fresh output: `runtime_custody_schema_gap_probe_v3_rerun.json`. No remote write.
4. Remote CPython 3.12.13: original native-global/replacement/alias/custody records accepted; admitted/failed global prefix rejected with `invalid native global admission`; corresponding replacement prefix rejected with `replacement admission differs from observation`; local alias digest substitution accepted; actual fail_native_call completed input became failed. Backend import set empty. Environment digest `79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919`.
5. Historical input report/worker/custody raw hashes again equal `f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20`, `e32db1fb74a300216bcba4f4b1dd926e3276cfeeb8b48437b52950bd27b8afc6`, and `f18a834b1519688be73603ff512bea825f158f42e586722edafd9a41ed86bcde`. Schema remains q54.c2d3.driver-observation.v0.
6. Local `C:/Python314/python.exe -I -S -B` executions of original case probe and author extraction probe, with fresh outputs `runtime_custody_schema_case_probe_v3_rerun.json` and `runtime_custody_manifest_feasibility_v3_rerun.json`. New exact-codec scratch probe/output: `runtime_custody_manifest_content_probe_v3.py` and `.json`. All passed.
7. With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_guarded_runtime_binding.py -k 'not session_materializes' --basetemp=C:/Users/AG/新建文件夹/c4_assets/schema_v3_binding_pytest_tmp`. Result: **30 passed, 2 deselected in 2.71s**. Log: `runtime_custody_binding_tests_v3.log`. Tests use simulated backend objects and real guard protocol; no real GPU call occurred. Two materialization cases were out of this selected binding regression, not reported as passed.

Linux deployment parsing was run on Linux with the pinned interpreter because the existing profile parser is host-dependent. No cross-platform aggregate implementation is claimed. One exploratory rg invocation used unsupported Windows wildcard paths, and a read targeted absent tests/conftest.py; corrected file searches established there is no conftest.py. These were inspection-tool errors, not test or candidate failures. The no-index design diff returned its normal exit 1 for differing files.

The original repros/results were preserved. New artifacts are only the named sibling scratch files, the pytest scratch directory and this report. No package installation, GPU execution/query/reset, process signal, candidate edit or remote file mutation occurred.

## Scope and limitations

This approval permits implementation of the exact v3 design. No aggregate parser/issuer/attachment implementation exists yet, so its mandatory live CPU mutation/failure tests and new-schema runtime executions cannot have passed. The existing tests/probes substantiate dependency behavior and design feasibility only.

Current-warp-go2, captured-ambient-and-terminal and cuda-normal-workloads v1 remain bounded prior evidence. Their both-lane normal results are not new executions by this reviewer and do not approve the aggregate, fault matrix, performance or backend choice. Required full import/init/allocation/copy/event/audit/owner faults, hangs, ACK/EOF, foreign/UNKNOWN/dual-GPU, cleanup/missing-exit coverage and fresh whole-C2d3 review remain outstanding.

C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host, R2 and shipping remain CLOSED. No parent-stage approval or product activation is implied.

Final integrity checks after report creation: local HEAD remained `ab24a7fe6619362875596fd8c3694c0756b5ef4c`, branch `implementation/c4-supervised-waits`, candidate digest unchanged, and working tree clean. Remote HEAD remained the pinned `a3d750bbaf79d760a6727e56146d277e1cb3b770`, clean and detached. Both checks completed successfully. The final report-only additions do not touch either checkout.
