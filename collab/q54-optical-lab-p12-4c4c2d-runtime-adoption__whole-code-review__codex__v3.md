# Independent whole C2d3 code and acceptance re-review

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: independent whole code/acceptance re-review
Author: codex, independent reviewer /root/c2d3_whole_review_v3
Version: v3
Date: 2026-10-09 (Asia/Shanghai)
Status: in_review — Request changes
Related Files: runtime-integration design/review v3; runtime-adoption design/review v2; runtime-custody implementation-plan/gap-review v3; native-scope-aggregate design/review v2; native-scope-cache design/review v1; whole-code-review v1/v2; array-adoption-join implementation-note v1
Owner Summary: **Request changes.** The named-array exchange counterexample now rejects, but moving its shared helper changed lifecycle.py without updating the separate release-pinned regex exception. The exact candidate cannot pass captured cold source verification. Independent current-source and cold-entry tests reproduce this regression. Required repaired fault/terminal matrices also remain incomplete at this review cut; no whole C2d3 acceptance is issued.

## Identity, independence and scope

Fresh reviewer session `/root/c2d3_whole_review_v3`, supplied the review packet, requirements and prior original reports/repros without the implementation conversation. I authored no candidate code. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md completely, correcting the initial PowerShell Chinese display with explicit UTF-8. I read both complete original whole reports, the governing primary designs/reviews, current progress boundaries, relevant repair/evidence documents and actual source/test changes. Oversized combined outputs were followed by focused reads.

Repository `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`; branch `implementation/c4-supervised-waits`; clean frozen HEAD `edba56a5633c0512b2387639b5644bad72696da4`. Whole C2d3 base is `ff2209e9fdcb1815a0e17a7dd97c399fc1e536e3`, following approved C2d2 code `0dd684867a8d60a69978e878b26d98169d2b9b1c`. Latest repair base is `83cfef6`, the unchanged v2-report archive. Prior runtime code remains `fc8618268dd8f3fb10e29e32626219500751c41a`; prior review HEAD was `9c1899898bf753644f691c5340336c35ca964a9b`. Whole-base delta has 196 files; repair delta has six files. This review was not restricted to those six files.

AGENTS expressly authorizes independent review agents. Two fresh `fork_turns="none"` read-only subreviews assisted: `/root/c2d3_whole_review_v3/bootstrap` for complete environment/import/source-policy/publication paths, and `/root/c2d3_whole_review_v3/native_flow` for canonical materialization, binding, upload/storage/readback/lifecycle and fixed native observer. I reviewed custody replay/issuance/attachment, acceptance and original evidence, independently reproduced the new blocker and old aggregate attacks, and combined the findings. Their bounded conclusions are not stage approvals.

The new Linux CPU tree `/data2/shared/ang.gao/robot_sim_c2d3_array_join_wip` was independently checked clean at edba56a after replay. The separate fc86182 GPU tree and active matrix were not mutated or signaled. Candidate remained frozen throughout.

## Blocking finding

### R1 — P2: moved helper leaves the lifecycle source exception pinned to old bytes, preventing every current cold source capture

Locations: `optical_worker/lifecycle.py:131`, `optical_worker/native_source_policy.py:49` and `:219–220`; catalog `optical_worker/data/native_sources/v12.json`; actual callers `native_sources.py:315`, `native_bootstrap.py:3365` and `tests/support/runtime_custody_preflight.py:21`.

The helper move changes the exact Git LF lifecycle source SHA256 to:

`2ca76eac3b72fc99f22d6c671e51a9a86724609575b0d61c5b101d3f7a6334dc`.

The v12 catalog correctly contains that digest. However, `_REGEX_SOURCE_PINS["optical_worker/lifecycle.py"]` remains:

`2770a0aff05c899f830890edcd4bb14297bd198c87d0936c1808d86067362a43`.

`validate_native_source_bytes` requires both the supplied catalog digest and the independently pinned full-byte regex exception. It therefore raises `ClosedWorkerProtocolError: regex exception source is not the reviewed release`. The production CPU source capture invokes that validator for this mandatory source in both lanes. Captured bootstrap and trusted-launch preflight fail before runtime admission or any GPU leaf. This is an availability/packaging regression in the delivered supported path, not a GPU fault, a Windows newline issue or an unsafe-admission bypass.

Independent minimal reproduction `../c4_assets/whole_v3_source_pin_repro.py` reads `git show HEAD:optical_worker/lifecycle.py`, asserts the bytes are LF and their digest equals the current catalog, and calls the actual production validator with that exact digest. It rejects as above. The same candidate validator accepts the old fc86182 Git lifecycle bytes whose digest equals the old policy pin. The moved function's AST, including signature/docstring/body, compares equal to the original; unchanged algorithm semantics do not satisfy an exact-byte source exception. The independent bootstrap reviewer additionally tested all 78 catalog Git blobs: lifecycle is the sole failing blob.

Actual cold entry separately fails: `test_native_bootstrap.py::test_actual_source_entry[warp]` exits78 with `source_bootstrap_failed: ClosedWorkerProtocolError: regex exception source is not the reviewed release`. Both lane entry variants fail in the complete run. My five `test_runtime_custody_native_failure_cold` cases fail in `capture_runtime_custody_inputs` before reaching their intended fault windows, with the full trace into the same production check. These are required regressions failing on the frozen candidate, not just a hypothetical mismatch.

Why the supplied positive subset missed it: array-record replay imports the CPU parser and validates old execution inputs; it does not capture or cold-load the new candidate's source release. Most direct-module tests likewise do not exercise captured entry. The existing policy test at `tests/unit/optics/test_native_source_policy.py:172–175` explicitly uses `git show HEAD:{path}`. After this finding, the coordinator disclosed that the supplied 684-pass run was performed on the repair working tree before committing, while Git HEAD was still83cfef6. That is the author's provenance clarification, not a newly observed execution by this reviewer; the test source independently confirms why such a run can test old Git lifecycle bytes. It is not an exact edba56a release check. The frozen-HEAD policy test fails in our run.

Repair direction: update the existing narrow exact-release exception for the reviewed lifecycle bytes and consistently refresh its dependent policy/catalog pins. Preserve the same regex-only permission and reject modified/unreviewed bytes; do not weaken verification, bypass cold capture, normalize source at runtime or duplicate the array-ID algorithm. Check the whole current catalog through actual capture and cold entry after freezing the revised commit. Existing release/cold tests already exercise this requirement; adding a second verifier is unnecessary.

Required proof after repair: exact committed candidate passes both-lane captured source verification and trusted CPU preflight, complete affected source/bootstrap/alias/cold-failure suites, and the original named-array attack still rejects while original records and legitimate failure cuts pass. Preserve these failures and rerun independent review. Native behavior is unchanged by this helper move, but that does not prove the moved source can be admitted. New GPU execution cannot repair this deterministic preflight defect and must not be launched before required code/native preflight.

## Previous whole-review findings and comments

- **V2 R1, named array-token exchange:** repaired at parser/source and independently replayed actual-record level. `prepared_storage()` now requires each array's adoption operation to equal the sole `_array_operation_id(lane + "_allocate", array.name)`. Setup imports that same lifecycle function; it has no copied implementation. I replayed all16 actual fc86182 records with the edba parser on Linux. Originals roundtrip, selection/attachment and same-run identity joins pass; all eight positives pass native occurrence acceptance. On all eight, the original token exchange with original lifecycle/object table/manifest/signatures/trusted inputs fixed rejects with `runtime adoption operation mismatch`. Duplicate-array-token controls also reject. This closes the specific contradiction but not the new cold-entry regression.
- **V1 F1, foreign upload event:** independently rerun by native-flow reviewer using the actual production cores/controller and original saved CPU counterexample with its prior exception-catching wrapper. CUDA and Warp reject a record-time foreign0 event; dirty set becomes `[0,1]`; event waits, normal COMPLETE, emergency sync and terminal ARM remain zero; quarantine retains owners and release is unauthorized. Relevant complete upload/readback suites pass. This is a CPU double fault proof, not a newly executed native injection.
- **V1 F2, omitted external completion:** source phase completion requires every begun file admission to finish; resolution-only probes remain permitted. I independently deleted NumPy's external completion from each of eight current actual positives and recomputed descriptive indices/counts as in the original attack, holding trusted inputs/lifecycle fixed. All eight reject with `phase completed with incomplete external file admission`.
- **V1 F3, twelve regressions:** the specific replacement/environment/contract repairs remain present and the relevant complete suites pass in this review. This does not erase the new source/cold failures. The old combined Linux 1058-pass/45-fail run remains a real failed orchestration attempt; isolated cold-trio262 is separate evidence, not a replacement statement that the combined run passed.
- **C1/C2:** the cleanup repair includes current v12 and the previously omitted test entries, preserving P12.4e conditions. The older intrinsic-module-name aggregate request versus later exact five-field grammar remains explicitly documented; no intrinsic-name aggregate field or independent CUDA storage-owner native identity is claimed.

For storage joins, canonical source is an earlier adopted distinct source and remains equal through binding/storage; prepared/event roles and scope are distinct; array names/order/layout match the exact manifest; operation IDs now prevent a directly adopted array from satisfying a different name. CUDA storage-owner objects are projected from the live registered storage graph, but the descriptive schema has no separately observed native owner-to-array relationship. I do not infer such a proof from tokens or the rejected unrelated-owner control. Warp storage-owner tokens remain null. Parsing continues to grant no live binding/storage/readiness authority.

## Whole-chain inspection and boundaries

The source review covered the complete C2d3 call/data/state/resource chain rather than assuming the old whole verdict was proof. The existing captured loader/finder, fixed deployment tables and guardian/controller remain the single supported path. File/spec/delegate identity, creator and publisher intervals, native-global roots, replacements, interpreter aliases, returned-before-adopt retention and sticky failures were traced. Exact catalog and narrow AST policy form separate obligations; their newly inconsistent lifecycle pins are the identified defect.

Canonical source still comes from the production materializer; exact runtime binding is registered and single-use. C0 and guarded callers use the shared uploader with separate abort policies. Destination/source/view/stream retention precedes copies; storage validation reuses the actual registry rather than minting readiness. Readback holds returned host objects, validates event scope, rechecks storage and uses the shared CPU comparison. Warp's enclosing real ScopedDevice is part of the accounted original setup segment. No duplicate uploader, validator, guardian or readiness authority was found.

Custody code delegates lifecycle/component validation, checks guard/phase ordering, exact prior adoptions and occurrence-bound tokens, external admission completion, actual latest native leaves, pair nesting, separate journal/lifecycle index domains, source/case/trusted interpreter joins and exact worker-prefix attachment. Immutable issued bytes, prior validated cuts and record-before-sidecar persistence preserve evidence without recovering execution from an old cut. The observer retains original functions, returned owners and pinned mapped-library handles, uses fixed ABI/result checks, differentiates successful null from errors and gates Warp's potentially initializing stream getter on the audited original-path/cache prerequisites. CPU models exercise those checks; no native getter was executed by this reviewer.

Normal DISARMED, terminal emergency sync/release authorization, actual cleanup and observed process exit remain distinct. Foreign/UNKNOWN/failed guard history cannot be repaired by a new target sync. Native dependency/driver internal exception safety remains the approved trusted boundary, not something Python retention proves. No product PREPARE, compile/record/render/warmup/final readiness, measured sample or result authority was opened.

## Executed verification and retained evidence

Local tools: UTF-8 reads, rg, Git revision/status/full-base and repair diffs, SHA256 checks, stdlib AST/archive scripts. Full-base production/test/tool `git diff --check` passed. Catalog comparison against fc86182 independently confirms only lifecycle/setup/custody source digests changed; catalog schema, lanes/paths and policy metadata did not. That unchanged policy is insufficient for the lifecycle byte change.

Local Python `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONDONTWRITEBYTECODE=1` and separate basetemp/log files outside the candidate:

| Executor / complete selection | Actual result |
| --- | --- |
| Main: runtime_custody, runtime_native_scope_codec, runtime_custody_native_failure_cold, runtime_failure_observations, native_runtime_driver_process | **5 failed, 25 passed, 36 skipped / 6.33s**, exit1. Five failures are explicit source-preflight errors. Skips require POSIX/retained evidence and are not passes. |
| Bootstrap subreview: native_bootstrap, native_sources, native_source_policy, runtime_environment, runtime_import_admission, native_global_publication, torch_module_replacement, generated_stdlib, interpreter_alias | **76 failed, 544 passed, 29 skipped / 122.98s**. Failures: bootstrap32, sources12, policy1, alias31. Many never reach their intended scenarios. 56 surface as UTF-8 decode errors on non-UTF8 child stderr containing Chinese paths; do not misreport all76 as directly displaying the pin error. Direct cold entry, explicit policy failure and all-Git-LF reproduction independently establish the source defect. |
| Native-flow subreview: native_upload_core, guarded_runtime_binding, guarded_native_readback, native_manifest, native_contract_extraction, native_scope_observer, native_input_observations, guarded_lifecycle | **320 passed, 9 skipped / 24.46s**. Nine skipped exact-Device model cases initially lacked a local Linux source path. |
| Native-flow: complete native_scope_observer with retained pinned context source supplied after independent SHA check | **33 passed / 2.91s**. This reruns overlapping cases and covers those nine CPU models; counts are not additive unique coverage. |

Read-only primary SSH CPU replay used `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B -` and the edba CPU checkout. Script `whole_v3_storage_replay.py` is the original v2 repro with only the checkout changed and explicit rejection assertions added. Result:16 original records parsed/attached,8 original token exchanges rejected,8 missing-completion attacks rejected, zero NumPy/Torch/Warp imports, exit0. Actual records retain fc86182 execution/source/trusted-input provenance; this is not edba GPU execution and does not validate new-source capture.

Archives independently hashed and raw summaries/stdout/residuals inspected:

| Archive | Bytes / SHA256 | Observed scope |
| --- | --- | --- |
| c2d3-fc86182-native-scope-v2.tar.gz | 7823610 / `d872f90f699fea8a1ea3b6a3aecc2c6a481bf34d4936b9a4de732cbaaef23130` | Raw16pass/1183.04s,8 positives/8 intended negatives,16 live identity chains,34 owned PID references absent in recorded OS/GPU residuals. Independently replayed as above. |
| c2d3-edba56a-array-join-cpu-v1.tar.gz | 13577 / `1704801cde136b1cfb2200a925e0f0556f814006c65b38e0700f89e9d2ba94ba` | Author CPU runner:16 new originals/8 exchange rejects,113 historical originals,48pass/1056deselected/112.01s on16 old/new positives,336 source-file hashes retained unchanged;4 owned runner/child PIDs absent in recorded residuals. Guardian-sequence and token exchanges are the new test variants. This is replay, not source preflight/native execution. |

The complete current fc86182 **85 fault matrix and subsequent12 terminal matrix were not available at this report cut**. Partial progress was supplied as running status only and is not acceptance evidence. Their full outcomes, unchanged selectors/deadlines, attachments/failure cuts, ancestry/reap and residuals must still be audited. Historical e585956/7cde88785/12, previous mutation archives and the author's113-record compatibility replay remain historical/bounded evidence. I did not independently rerun113 in this round or re-audit every prior archive, and do not claim the missing current matrix gates closed. The proven source blocker justifies this early Request changes without waiting for them.

Artifacts under `../c4_assets`:

- `whole_v3_source_pin_repro.py`: SHA256 `d4369ebb56112eb739183f4039247043fcd535b7ffc10207cac9fc106cbfeef6`; `.json`: `52341723bfa082e4aa6c4af1a7afec466ae2c1384ba8fd99081ef9baf102f2b3`.
- `whole_v3_custody_cpu.log`: `bbce216443c7fae5461a3be3e4aad6138e905069a49a2f9a20552986a29502fa`.
- `whole_v3_bootstrap_cpu.log`: `723d6c4463a0a0e6873b494fa6d97664261f6035869a1fe98c8e9b98797fbb5a`; `whole_v3_bootstrap_git_pin_repro.log`: `14cd36368860c7906b27d10fbf1ba6c06c98f4e6a5f8054a3bed6aaa0c2fb39c`.
- `whole_v3_native_flow_cpu.log`: `dce74ffd612e8f45d8bd30c422b36d7e618a093507c1abc7a166b2a0896ccef1`; observer log `bdbc35805973d54b292dfcef9a680d3c4f4f2ccedd0e53d2106848f77c9407b7`; event repro log `cf208fcaf8bb1dfb426c3c90d10b6eb2f291bff9278d639401236b80b277f5af`.
- `whole_v3_storage_replay.py`: `dc0d4eae4db8f33d6c425357b4fcbdba096a3e721828e5de1eca7a820c67fd7a`; output `.json`: `8069effe6f5ad4569fec2334f7d4c10bf161b17a301f0134bf38d1c28954952c`. PowerShell initially redirected output as UTF-16; it was rewritten as UTF-8 JSON outside the candidate, preserving values. Original v2 repro SHA independently remains `e32110b9ac1184cdca8e7695d717ae806db759a6143e557be3ae44d7b892f740`.

## Hygiene, limits and precise verdict

All reviewer artifacts and test packaging are outside tracked candidate files. Local direct pin probe PID39200/parent34652 completed; remote replay PID3433645/parent3433644 completed and subsequent exact ps query returned header only. Scoped local Win32_Process queries found no Python/SSH command referencing this round's tests/repros. CPU event peers closed/joined. Not every short-lived local utility/pytest descendant start identity was sampled; these are bounded checks, not complete OS attestation. No reviewer GPU work, native getter, compiler, package install, process signal, reset, fallback host, private-context edit, publication or tracked mutation occurred. The ongoing fc86182 matrix was untouched.

**Verdict: Request changes** for whole C2d3 candidate `edba56a5633c0512b2387639b5644bad72696da4`. Repair the source-policy inconsistency, obtain exact committed cold-source proof and independent re-review, and finish/audit the required runtime matrices before any future whole approval. This report neither approves a repair-only subset as C2d3 nor invalidates the correctly attributed fc86182 native evidence. Native behavior equivalence and parser compatibility are useful bounded evidence; they cannot substitute for a working captured candidate entry.

Still CLOSED: whole C2d/C4 completion; compile/record/render/warmup, final graph/seal/quiescence and R_setup/R_final; prepared sample/optical execution authority; C2e/f; C4c3/c4; C4d/B3; product GPU PREPARE/native dispatch; a3b; result/baseline; host payload/R2/S; shipping/activation; performance verdict/backend selection. No losing-lane deletion or later implementation authority is issued. The complete unchanged report may be archived under the evidence-only exception without changing its verdict or scope.
