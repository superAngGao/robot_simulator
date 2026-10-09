# Fixed render warmup: bounded native evidence review v1

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: independent bounded native-evidence acceptance review
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5/5.1 and approved review v3; fixed-render-warmup code-review codex v2; warmup-bootstrap-repair code-review codex v1; warmup-readback-checkpoint code-review codex v1
Owner Summary: Accept only the eight positive fixed Go2 setup-warmup lifecycle executions and terminal exits at exact native revision 7fe32a58b9e4bee1b6d354f42ceed942f6e9cb92. Independent strict replay, custody checks, archive integrity, mutations and process checks pass. A reproducible 1920x1080 numerical disagreement remains unresolved in both lanes. No optical correctness, fault-matrix, performance, activation or whole-C4 gate opens.

Reviewer session: `/root/render_warmup_native_acceptance_v2`, fresh independent session with the task packet and repository, without implementation conversation. No implementation authorship, candidate edits or delegated agents.

Verdict: **Approve with comments**, exclusively for the eight positive native setup-lifecycle cases below and accuracy of the reported diagnostic calculations. No substantive blocking finding remains in that bounded lifecycle scope. Comment C1 is an unresolved blocker for numerical correctness acceptance, which is explicitly not approved. This is not an approval of complete fixed rendering or C4.

## Exact candidate and evidence

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, clean HEAD `9067e114ed609928904bf3cd95b871554bb15d26`. Native repository: `tileops-dev:/data2/shared/ang.gao/robot_sim_render_warmup_wip`, clean detached HEAD `7fe32a58b9e4bee1b6d354f42ceed942f6e9cb92`. The local delta after native HEAD contains only PROGRESS.md and the unchanged readback checkpoint review record; it does not change executable source. Initial and final identity/tree checks matched. The separate later geometry diagnostic candidate is outside this review and was not used as evidence.

Frozen archive: `c4_assets/c4-render-warmup-native-evidence-v3.tar.gz`, SHA-256 `75447f0f29738fed6a12326f8c3e53f2771d91e2ac8e7dd04ef46d6ddeb531ad`. Independently checked the remote archive hash and all 277 regular members against their original remote bytes, including eight raw case directories, runner/archive scripts, batch record and both environment profiles. Checked that every file in each original case directory is included, with no duplicate member names. The local downloaded archive has the same digest. Author summaries and archive code were inspected as claims; original validators and reviewer assertions determined acceptance.

Read the complete collaboration protocol and review principles, accepted design sections, relevant predecessor reviews, production warmup authority/driver and Warp transaction source, the live observation/readback helper, and unchanged acceptance/custody validators. No native worker was launched by the reviewer. Existing native executions were inspected read-only; CPU replay and one fresh CPU oracle diagnostic were performed separately.

## Eight positive lifecycle executions

All cases use `direct_light_full`, shadows enabled, real target ordinal 1 and foreign ambient ordinal 0, one single-lane worker per case. Target UUID from each mapping is `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`; recorded driver is 595.71.05. Exact lane environments and their current files were independently verified below.

| Raw suffix | Lane | Resolution | Diagnostics | Normal completions | Supervisor / worker PID |
| --- | --- | --- | --- | --- | --- |
| v3 | Warp | 960x640 | off | 25 | 3479832 / 3479833 |
| v4 | CUDA | 960x640 | off | 21 | 3483768 / 3483770 |
| v5 | Warp | 1920x1080 | off | 25 | 3490944 / 3490949 |
| v6 | CUDA | 1920x1080 | off | 21 | 3495555 / 3495562 |
| v7 | Warp | 960x640 | on | 25 | 3505499 / 3505500 |
| v8 | CUDA | 960x640 | on | 21 | 3511640 / 3511642 |
| v9 | Warp | 1920x1080 | on | 25 | 3518966 / 3518968 |
| v10 | CUDA | 1920x1080 | on | 21 | 3523347 / 3523355 |

Each original frame sequence has ten distinct retained transactions and terminal objects, ordered frame indices 0..9, ten actual terminal waits, original event identity, and ten successful normal guard handshakes. Warp executes five recorded launches per frame, including reset kernels; CUDA executes three original stage entries per frame. Diagnostic off has no diagnostic events; diagnostic on has three phase pairs in addition to the terminal record. The actual leaf ordering and returned observations match the corresponding original raw lifecycle descriptors. Every frame's last descriptor precedes its unique normal_disarmed, which precedes the next frame's first descriptor. Shared event reuse therefore follows completion and DISARMED in the recorded executions.

Original graph/storage validation passes: ten frame bundles, retained events and command owners, 42 validated input arrays, correct output dimensions, matching compiled-module or issued CUDA-factory identity, and unchanged warmup0 provisional readiness. Inside each observed native leaf the held target stream is current and only primary context 1 is active. Before/after scope snapshots match. Warp restores default device 0 and no current context. CUDA restores logical device 0 while raw device remains 1, its frozen lazy-restoration contract; this report does not claim CUDA raw device 0 restoration.

Production source checks live exact session/controller/guardian/graph/runtime/stream and registered transaction identity before each leaf. The observer calls the original source-session entry; it does not manufacture completion authority. Warp's immutable actual plan is validated and executed. CUDA's warmup_launch returned flag records successful declaration-context return before the native entry, not kernel completion; the real terminal wait and original guard handshake supply completion evidence. Owners remain in original guardian custody. Completion evidence is locally issued only after all ten completed tickets; provisional readiness is not relabelled as final readiness.

The final RGB copy has its own successful original setup guard and a refreshed validated runtime checkpoint. Every image has exact frame10 dimensions/byte count, matching SHA-256 and finite float32 values. All four diagnostic-on images are byte-for-byte identical to their corresponding diagnostic-off images. This proves diagnostic-output consistency for these images, not numerical correctness.

Every case independently satisfies the complete positive terminal disposition: supervisor EXITED/complete and observed worker return 0, no stderr, deadline expiry, kill request or pending ACK; exactly one additional emergency ARM, one successful target terminal synchronization, completed release, guardian RELEASED and worker disposition aborted_synced_exit. The emergency ticket remains ARMED through terminal intent; no emergency normal-completion credit is fabricated. No failure, fault or ordinary-atexit marker exists. The single terminal disposal sync is not the future final setup barrier.

## Independent verification

1. Executed unchanged `validate_custody_artifact`, `validate_runtime_custody_selection`, `RuntimeCustodyRecord.from_payload` with original trusted launch inputs and environment, and `validate_runtime_custody_attachment` for every case. All eight attachments are `worker_prefix_matched`. Compared combined artifact with separately archived worker and supervisor output; cross-checked request/result/revision/catalog, lane/case/mode and parent/supervisor/worker identities. Replayed `_validate_fixed_storage`, `_validate_render_graph` and `_validate_render_warmups` on original raw bytes and full traces. All eight pass.

2. Applied thirteen independent alterations per case to reviewer-only copies: wrong completion count, reversed frame indices, duplicate transaction identity (the script labels this foreign_transaction), replaced terminal-event identity, wrong guard sequence, broken ambient restoration, foreign held-stream ordinal, removed terminal wait, removed DISARMED, changed RGB hash, forged terminal-sync summary, forged normal-completion summary and changed trusted source revision. All **104 alterations rejected** by original validators. No original bytes were changed. These serialized consistency probes do not prove resistance to arbitrary trusted-process Python reflection or establish a hostile-input attestation boundary.

3. Fresh cold CPU source preflight, through original `capture_runtime_custody_inputs`, exactly reproduced archived launch inputs for both lanes. Warp source snapshot: `d66ea150c1b1e5f96ef2e42d4fd290d8298f047916939de819073c773e823235`; CUDA: `ae525bb3c3cb8c2ff9fc532aa93a8de690a0f7bbc5e0e8f77bad8beca63ec649`. Catalog SHA-256: `f6208cb33d29476878925e81f4f7e3c0665ca84f718aac4ff042f35b5a5fc70a`. `RuntimeEnvironmentContract.verify_files` passes for Warp profile `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb` and CUDA profile `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`. Replay/integrity processes explicitly assert no torch, warp, numpy or multiprocessing import.

4. Independently recomputed full RGB pair differences and archived CPU sample errors using NumPy locally. Read the original CPU diagnostic's frozen scene/sequence/scalar construction, then reran its original 54-ray outlier/neighbour CPU rendering against exact remote source and existing frozen Go2 assets into reviewer-exclusive scratch. The fresh indices, CPU RGB and selected GPU RGB arrays exactly equal the supplied CPU54 archive. This is a fresh rerun of the original diagnostic through the existing CPU executor, not a second independently implemented optical oracle.

5. Preserved old failed runs. Independently hashed failure archives v1 `7cc7eb78fa664aa1439fbc61730700996433189cceaeafc8653ca8d5dd64b67c` and v2 `0b7e668a1717f52683874ea831624e2aae0b38849d4b0b27cc3d7e78c115b8b1`; both original supervisor outputs record worker return 74. Neither receives retroactive acceptance from this campaign.

Reproduction programs and complete output reside outside the candidate under `c4_assets`: `warmup_native_review_replay_v1.py` and its final log; `warmup_native_review_integrity_v1.py/.log`; `warmup_native_review_archive_v1.py/.json`; `warmup_native_review_numbers_v1.py/.json`; `warmup_native_review_cpu_stdin_v1.py`, `warmup_native_review_cpu_v1.log`, and `warmup_native_review_cpu_1920_v1/`. Remote replay/integrity command: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B -` over SSH. CPU optical diagnostic uses the existing environment's `python -B -`; local arithmetic uses `C:/Python314/python.exe -B`. Final replay tool session 62822 reached exit 0. Earlier partial replay inspected completed cases while the untouched author batch continued; final replay was repeated against the complete frozen campaign.

Reviewer tooling limitations: initial root REVIEW_PRINCIPLES.md lookup failed before the actual collab file was read fully; an exploratory shell lookup used incorrect interpreter paths before locating the existing miniforge environment; a Windows rg glob invocation was invalid. The first local CPU54 attempt exited before rendering because trimesh is absent. No dependency was installed: the fresh remote CPU rerun resolved that environment limitation and completed exit 0. Reviewer and coordinator independently copied the same remote archive to the same local basename, briefly causing a coordinator sharing violation. Both transfers finished; no scp process remained, both observed the expected final SHA-256, and the reviewer additionally verified all 277 local member digests against the independent remote inventory. No further transfer occurred. These are tooling/environment issues, not candidate failures, and were not counted as passes.

## C1 — unresolved numerical correctness disagreement

Severity: **P1 for numerical correctness acceptance; non-blocking only for this expressly separate lifecycle-evidence approval.** Location/trigger: frame10 `warmup_rgb.f32` in the 1920x1080 Warp/CUDA cases v5/v6 and their byte-identical diagnostic variants v9/v10. Exact kernel/root-cause location is not established by RGB-only evidence.

At 960x640, full-image Warp/CUDA maximum absolute RGB difference is `2.384185791015625e-7`; the archived 420-ray CPU sample has maximum `6.825475442839846e-6` for either lane, independently recomputed here. This sample is not a full-image CPU acceptance result.

At 1920x1080, six pixels exceed the diagnostic screening value `1e-4`, with full-image maximum cross-lane difference `1.2056253775954247`. Fresh 54-ray post-hoc CPU outlier/neighbour rendering reproduces maximum CPU difference `1.2056406666813424` for CUDA and `0.10956230499053238` for Warp. The six original outlier indices and maximum RGB errors versus CPU are:

| Pixel index | CUDA error | Warp error |
| --- | --- | --- |
| 782417 | 1.2056406667 | 0.0000152891 |
| 901306 | 0.0331984712 | 0.0000411393 |
| 976396 | 0.0000000026 | 0.0700501338 |
| 985873 | 0.5564471102 | 0.0000008949 |
| 1026413 | 0.0000000084 | 0.1095623050 |
| 1143492 | 0.1906471997 | 0 |

Thus four CUDA and two Warp outlier pixels materially disagree with this CPU diagnostic. Valid guard/custody completion cannot establish optical agreement. Existing `_validate_render_warmups` checks dimensions, finiteness, byte identity, call order and custody, and intentionally has no CPU-oracle numerical assertion; its pass cannot detect this issue. The `1e-4` screening value is not introduced as a product tolerance or acceptance gate.

Required follow-up before optical correctness acceptance: preserve these exact repro inputs and pixels; investigate actual rays, first-hit geometry/identity and shadow/shading decisions against the frozen CPU contract; establish the cause and any policy implications, repair within separately reviewed scope, and provide fresh source-frozen native plus CPU-oracle evidence covering the affected pixels and the broader authorized correctness matrix. Do not loosen thresholds or label discontinuities harmless without evidence. The post-hoc selection is useful for diagnosis and cannot stand in for unbiased image/channel coverage. No root cause, fix or later diagnostic-support candidate is approved here.

## Remaining gates and process hygiene

No native warmup partial-submit/record/wait/ACK/readback failure matrix, hang/reaping fault matrix, full numerical image/channel/frame acceptance, sentinel/reset acceptance, performance or backend comparison is established. Final readiness/origin, final factories/commands/reset, C3 registration, single coverage seal/final target barrier, measured arena/timeline, R1 and complete lifecycle integration remain unfinished or closed. Product PREPARE/sample/lease/IPC activation, R2/S/host delivery/external holds, result/performance/baseline/backend-selection/shipping authority and whole-C4 acceptance remain closed. The later systematic whole-C4 refactor and fresh independent review remain required.

At Unix time `1791578981.5705543`, the independent final archive check found author parent 3479660 and all sixteen supervisor/worker PIDs in the table absent from `/proc`. Current source remained clean and unchanged. Reviewer remote CPU diagnostic PID 3519126 completed exit 0 and was independently found absent; replay and integrity commands completed terminally. At Unix time `1791579106.206153`, final scoped checks found reviewer process/parent IDs 3517490, 3517477, 3517951, 3517948, 3519126, 3525110, 3525100, 3535178, 3535176 and 3535850 absent, and native HEAD/tree unchanged. The local Python command-line query found no remaining warmup_native_review process; local HEAD/tree also remained unchanged. Short inspection processes returned terminal tool results without exhaustive individual child-PID sampling. No manual signal, GPU reset, GPU worker restart, unrelated-process cleanup or native GPU job was performed by the reviewer. Original remote raw campaign files and source stayed untouched; only separate local/remote reviewer scratch, the local archive copy and this report were written. The final check is recorded in `warmup_native_review_hygiene_v1.json`.

The coordinator may archive this complete report verbatim under collab in an evidence-only commit. That record cannot change its exact approval scope, numerical blocker or remaining gates.
