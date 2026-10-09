# Independent bounded R0 fixed-storage native acceptance review v1

Initiative: Q54 / P12.4 / C4 bounded R0 fixed render storage
Stage: independent native evidence acceptance
Author: codex, fresh reviewer /root/fixed_storage_native_acceptance_v1
Version: v1
Date: 2026-10-10 (Asia/Shanghai)
Status: accepted
Related Files: runtime-integration design-request codex v3 sections 5, 8, 9; remaining-setup gap-scan codex v1 G2; fixed-render-storage code-review codex v1; warp-slot-construction code-review codex v2; fixed-storage-native-support code-review codex v1
Owner Summary: Approve the bounded 14-case native storage evidence at exact 3d6692bb31145d70b4b841dbbc82f1183cd6a58f. All raw cases independently pass frozen acceptance and production custody replay. Fresh CPU checks: 150 passed, 41 host-dependent skips; additional checks verify all 148 positive output metadata bindings and reject 64 evidence alterations. No substantive finding. This does not approve the later shared resource-scope implementation, graph/rendering, whole G2, or C4.

## Independence, source identity and verdict

Reviewer session `/root/fixed_storage_native_acceptance_v1` received a fresh bounded review packet without the implementation conversation. I authored no candidate code and spawned no agents. I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the relevant parent requirements, G2, and predecessor reports. Author summaries and earlier approval were claims to verify, not substitutes for this review's raw replay.

**Verdict: Approve**, solely for the observed native R0 storage construction, scope/restoration, partial owner retention, guarded failure disposal and supervised hang termination in the matrix below, against:

- Native source: `3d6692bb31145d70b4b841dbbc82f1183cd6a58f`, tree `ac0e9bc7ecdea313e356a99f4321bb23b7891922`.
- Original clean detached checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_fixed_storage_wip`. Initial and final observed HEAD matched; both status reads were empty.
- Production storage predecessor: `12c161969a6393722510cb3c373e2a0e3c25a637`; allocation predecessor: `fdb86a03bc771e9422d7b5b04e0179b2a792684f`. An independent diff between 12c1619 and 3d6692b under optical_worker/ and optics/ was empty.
- Main repository `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design` stayed clean at `6300531a300d68dd558d155979e0ef86e3e78dfb`. It was read only; its later code is not what this campaign executed.

In particular, this campaign does **not** prove the changed common resource-scope implementation introduced later at 275095e/8315b00 or its descendants. Neither that newer code nor the failed graph campaign at 50143c6 is storage acceptance evidence here.

## Exact evidence and provenance

Raw directories are `/data2/shared/ang.gao/c4-fixed-storage-author-v1` through `v14`. Local archive:

`C:/Users/AG/新建文件夹/c4_assets/c4-fixed-storage-author-evidence-v1.tar.gz`

SHA256, independently computed locally and remotely:

`0c236e6ea39392af2e74f7e450ed492d52979ca4c734bfd5b578f0067d57e326`

I extracted the archive outside the candidate and compared **all 392 archived regular files, totaling 344,959,308 uncompressed bytes**, byte for byte against the remote original files, including the mapped archived author scripts and environment profiles. All matched. The adjacent author JSON was not the replay input.

I inspected the original batch script. It calls the reviewed `run_case` serially with literal case/selector pairs, checks exact revision and clean status before every case, uses the original supervisor and preserves raw results. It does not replace the controller or supervisor. Every raw request independently binds the exact revision, catalog digest `cf71b91af50a026636691656b094430091337720d813421e29dc3c0cf8da6124`, target ordinal 1, 1000 ms ACK timeout and 120000 ms operation timeout. All 14 guard sessions are distinct; the request configuration exactly equals the corresponding supervisor configuration.

Environment profiles were parsed and their current files independently verified with the frozen `RuntimeEnvironmentContract.verify_files` on the remote CPU:

- Warp `183_warp_go2_author_v4.json`: `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb`.
- CUDA `c4-fixed-cuda-environment-v2.json`: `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`.

Raw source observations identify the pinned Linux CPython 3.12.13 environment. The evidence retains the H200 host's GPU mapping and original source/backend admission records. I verified all seven CUDA factory-binding records name the three expected captured modules, retain exact factory bindings and target1 workspace requirements, and preserve before/after state. For all seven Warp cases, all three module source digests matched current exact-source bytes and all **63 retained cache files** matched recorded sizes and hashes. Their seven separate `/tmp/robot-sim-warp-*` caches remain present. These checks bind the existing compiled prerequisites to this storage campaign; they do not reissue a separate compilation approval or claim a native rerun.

## Independent replay and observed matrix

Scratch implementation and extracted snapshot are under:

`C:/Users/AG/新建文件夹/c4_assets/fixed_storage_native_acceptance_v1/`

I inspected the earlier Warp review's AST technique, then wrote `replay.py` for this exact source. It compiles the frozen `run_case` verification tail beginning with `storage_case`, ending immediately before its return. It excludes launch, writes and cleanup. The tail uses the original `_validate_fixed_storage`, `RuntimeCustodyRecord.from_payload`, runtime-custody selection and attachment validators. Independently, the script calls `validate_custody_artifact`, checks stored/embedded artifact equality, request/source/profile/configuration binding, positive/fault matrix identity, PID relationships, unique sessions, observed exit and absence of ordinary Python cleanup markers. It also checks hang elapsed time and exact retained prefix identity.

The script ran via SSH stdin with `/home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12 -B -`, importing only CPU verification/support definitions. It invoked no worker, native initialization, CUDA getter, GPU allocation or rendering. Tool session 15105 completed with exit 0 and `ALL 14 RAW CASES PASS FROZEN RUNNER AND PRODUCTION CUSTODY REPLAY`.

| Raw versions | Cases | Output arrays retained/observed | Normal completes / total ARMs (Warp; CUDA) | Worker exit | Terminal syncs |
| --- | --- | --- | --- | --- | --- |
| 1–4 | Both lanes, 960×640 and 1920×1080, diagnostics off | 14 each | 12/13; 8/9 | 0 | 1 each |
| 5–8 | Both lanes, both resolutions, diagnostics on | 23 each | 12/13; 8/9 | 0 | 1 each |
| 9–10 | Both lanes, 960×640, second allocation boundary raises | 1 each | 11/12; 7/8 | 74 | 0 |
| 11–12 | Both lanes, 960×640, second returned owner pre-adoption raises | 2 each | 11/12; 7/8 | 74 | 0 |
| 13–14 | Both lanes, 960×640, second allocation boundary stalls | 1 each in boundary marker | 11/12; 7/8 | -9 | 0 in retained prefix |

All 14 custody selections are `current`; the original selected bytes, trusted launch inputs and lifecycle attachment parse and replay successfully. Positive input validation is observed. The runtime-custody record remains an upstream input-phase record; its summary is not being relabeled as a storage receipt. Storage observations are downstream diagnostics checked with the actual production storage path and lifecycle evidence.

For hangs, operation-deadline expiry occurs **120004.813108 ms** after the last Warp ARM and **120000.381064 ms** after the CUDA ARM. The traces then record kill request and observed process exit -9. No terminal worker.json, failure.json or successful storage binding exists. The combined worker component equals the previously persisted prefix.json exactly: it is honest partial evidence with RUNNING/IDLE prefix state and `custody: none`, not a fabricated final quarantine or worker-success record.

## High-risk behavior checked

**Actual allocation and input reuse.** I traced `SourceSession.prepare_fixed_render_storage` through the existing guarded segment, exact original upload/validation and compilation checks, `_prepare_fixed_render_storage`, and `_output_array_allocation`. The observer calls the production session method, not a duplicate allocator. Its empty wrapper observes native scope before delegating to the original native empty, then immediately returns the actual array. Production appends the array to the pre-adopted `slot_array_custody` list before metadata inspection and lifecycle adoption. The diagnostic hook checks that guardian-held list and retains primitive metadata rather than output array owners of its own. Successful observations bind output object IDs and all borrowed prepared-input IDs. No second upload is used.

The normal validator checks count and first output shape. As an additional independent check, `independent_checks.py` checks **all 148 positive arrays** against the inspected fixed output ABI: exact semantic inventory, each array's shape, dtype and allocation operation hash, matched by actual output ID. This accommodates CUDA's allocation order differing from tensor_items order without relaxing either order's meaning. Ordinary and diagnostic arrays pass at both resolutions and in both lanes.

**Real target/stream and ambient restoration.** The observer uses the established native driver/runtime queries under the live original guard. At every attempted output allocation, evidence records target1, the held target stream and exactly primary context 1 active. Warp current/default device is 1 inside the allocation scope; CUDA logical/raw device is 1. Every positive before/after state is exactly equal: Warp restores default0/no current context, while CUDA restores logical0/raw1 and the prior target stream, consistent with its established deferred-device behavior. No failed-ticket native restoration query or failure-path restoration claim is made.

**Partial failures and ownership.** Four raise cases preserve the original RuntimeError text in both relevant diagnostics. The production public-operation observer marks the original setup guard failed; the evidence contains one failed guard at sequence 11 (Warp) or 7 (CUDA), full quarantine, transferred guardian custody, no authorized release, no new leaf after the injection boundary, no further normal COMPLETE, no new ARM and no emergency synchronization. The second returned-owner case has two retained array IDs before adoption fails; the allocation-boundary case has the first returned array retained. The original known enclosing owners remain in the retained graph. The inspected exception observer does only CPU root inspection after guard failure.

**Positive terminal behavior.** Successful storage completes its setup guard, then follows the existing one-sync emergency disposal. Its additional emergency ARM is not counted as a normal COMPLETE. Raw lifecycle replay establishes successful sync and completed release, and supervisor evidence separately establishes exit0. The successful storage path issues no command recording, rendering, sample, warmup, final readiness or product capability. The hardcoded zero diagnostic fields alone are not proof of that exclusion; the inspected production call path supplies the constraint.

## Fresh CPU checks and independent negative controls

From an isolated 3d source archive, with `C:/Python314/python.exe -B`:

```text
-m pytest -q -rs -p no:cacheprovider
  tests/unit/optics/test_prepared_fixed_render_inputs.py
  tests/unit/optics/test_native_output_allocation.py
  tests/unit/optics/test_native_scope_observer.py
  tests/unit/optics/test_warp_failure_diagnostics.py
  tests/unit/optics/test_cuda_artifact_runtime.py
  tests/unit/optics/test_runtime_native_scope_codec.py
150 passed, 41 skipped in 23.70s
```

Tool session 60808 exited 0. The skips are nine unavailable pinned Warp source-text model cases and 32 POSIX deployment codec cases on Windows. They are not native passes. The actual campaign records were subsequently replayed on POSIX as described above.

`independent_checks.py` also makes seven isolated mutations to each of eight positive raw storage records: lost retention, wrong stream, foreign active context, wrong output device, changed restored stream, replaced input identity and wrong guard. All 56 are rejected by the original storage validator. Eight forged normal-completion summaries are rejected by production lifecycle replay. Total: **64 independent negative controls rejected**, with the unaltered data accepted. Original evidence files are never edited.

Inspection/scratch issues were corrected transparently: the repository's principles file is under collab/, not root; initial Git archive extraction inherited Windows CRLF conversion, so catalog-byte comparison failed and the snapshot was regenerated with `git -c core.autocrlf=false archive` before byte/provenance checks. CPU tests had run against the same source with newline-only conversion. Windows intentionally rejects POSIX absolute deployment paths, so complete record replay ran on the original POSIX host. The first scratch mutation harness caught the wrong exception class, and an attempted CUDA restoration mutation initially assigned its already-zero stream; these were corrected to `GuardProtocolError` and a guaranteed changed value. None is a candidate failure or a weakened production check.

## Findings, limits and remaining gates

No blocking or substantive non-blocking finding was identified in this bounded evidence set.

The reviewed faults are explicit test-only boundary injection: the second allocation raise/stall happens before that second original native empty call; pre-adoption failure happens after its returned owner is production-retained. They demonstrate guarded failure and hang containment at those production boundaries. They do not claim an observed real CUDA OOM, a native allocator internally hanging, or unknown internal partial native allocations. Fault cases use 960×640 with diagnostics off. Success evidence covers the two stated resolutions and diagnostic modes. No extrapolation to other rows or failure matrices is implied.

I did not execute a new native campaign. This is fresh independent acceptance of the supplied actual-device raw campaign plus newly executed CPU checks and raw replay. Native handle and object IDs are observed per-process bindings, not cryptographic proof against an adversarial operating system; the trust boundary is the source-frozen original launcher/worker and verified environment. Current filesystem verification corroborates retained provenance but does not reconstruct arbitrary historical host changes.

Whole G2 still includes more than fixed R0 storage. R1 mutable slots, complete resource/command/event graph, per-frame ABI binding, reset and real warmups, final readiness/origin, final recording/seal/barrier and setup admission remain outside this verdict. The later common-scope extraction requires its own evidence. Complete rendering, native graph acceptance, C2e/f, full C4 implementation/refactor/validation and fresh whole review, product PREPARE/sample activation, R2/S/host delivery, performance/results/baselines and shipping remain closed under their own requirements. The earlier compilation approval and the failed newer graph campaign do not open any of these gates.

## Process hygiene and artifacts

All 29 author parent/worker/supervisor PIDs were independently derived from original parent.json/identity.json/result.json relationships and found absent under /proc. They are:

```text
3198545 3198679 3198685 3202999 3203000 3212182 3212183
3217607 3217608 3226688 3226690 3231461 3231478 3243539
3243548 3250849 3250857 3260353 3260355 3264904 3264905
3302061 3302062 3308087 3308103 3317138 3317140 3331416 3331417
```

Reviewer replay PID3351830/parent3351826 and provenance inspection PID3353864/parent3353860 exited and were subsequently confirmed absent. The final cache inspection PID3356834/parent3356826 completed with observed SSH/tool exit0 and both were subsequently confirmed absent. The local pytest invocation completed with exit0; it exposed a tool session rather than a retained OS PID. A final scoped Win32_Process query found no Python command matching the reviewed pytest invocation. No test peer teardown failed. I launched no native worker/supervisor, performed no GPU operation, sent no process signal and did not touch unrelated GPU jobs. Archived GPU process snapshots are retained historical context, not a claim that unrelated jobs must be absent.

Scratch scripts `replay.py`, `remote_audit.py`, `cache_check.py`, `independent_checks.py`, the local raw extraction, exact-source snapshot and audit JSON are outside the candidate. Remote scripts were delivered over stdin with bytecode disabled; no remote script or implementation file was written. `remote_audit.json` records the full file/profile/PID audit and `cache_check.json` records cache/source checks and final clean 3d source state. This full report may be archived verbatim as evidence for only the exact source and bounded scope stated above.
