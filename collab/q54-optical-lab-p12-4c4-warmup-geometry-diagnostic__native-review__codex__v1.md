# Warmup geometry diagnostic: independent native-evidence review v1

Initiative: Q54 / P12.4 / C4 render integration
Stage: bounded independent native-evidence review
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design-request codex v3 sections 5 and 9; warmup-geometry-diagnostic implementation-note and code-review codex v1; fixed-render-warmup native-review codex v1
Owner Summary: Approve the two original native geometry diagnostic executions and their bounded first-hit diagnostic evidence, with the unresolved numerical correctness comment retained. Each lane completed ten warmups and nine same-frame guarded copies, then the original terminal disposal and observed exit. Six substantial RGB discrepancies reproduce; first-hit outputs already differ at all six. This does not establish a numerical root cause, full geometry/RGB correctness, or whole-C4 acceptance. Two additional normal discrepancies have identical RGB and must not be hidden by an RGB-only screen.

Reviewer session: `/root/geometry_native_evidence_review_v1`, fresh independent Codex session given the task packet and repository, without implementation conversation. No implementation authorship, candidate edits, delegated agents, GPU workers, dependency installation, or changes to source/raw evidence.

Verdict: **Approve with comments**, solely for the two positive diagnostic executions, existing warmup/readback/terminal lifecycle, and diagnostic statements supported below. No new blocking implementation or evidence-integrity finding was established in that bounded scope. Comment C1 remains a **P1 blocker for numerical correctness acceptance**, explicitly outside this approval. The words “localized to first-hit outputs” mean divergence is already observable before shading; they do not mean the intersection algorithm is the proven exclusive cause or that all geometry differences are confined to six pixels.

## Exact source and evidence

- Base: `9067e114ed609928904bf3cd95b871554bb15d26`.
- Frozen reviewed executable/support HEAD: `5a5a17cdd4e75c01016e1e71738aed99d9dab8fe`, tree `f1eeafc1b7ff8833af579ce9797ce05c7fd08f1c`.
- Local frozen checkout: `C:/Users/AG/新建文件夹/c4_render_execution_next`; remote checkout: `tileops-dev:/data2/shared/ang.gao/robot_sim_render_geometry_wip`. Both clean and unchanged on initial/final checks.
- Primary checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, clean HEAD `8cce978b73901eeac42a9e2fdc7e861a23a57890`. Its delta after 5a contains PROGRESS.md and three archived reports only; no executable/support source changes. It is not represented as the original native execution revision.
- Full base-to-5a diff inspected: two support drivers, their unit file, and the implementation note. No production kernel, numerical parameter, tolerance, catalog, guard budget, or product gate change. `git diff --check base 5a` passed.
- Original remote directories: `/data2/shared/ang.gao/c4-render-geometry-author-v1` (Warp) and `v2` (CUDA), and `c4-render-geometry-batch-v1.json`.
- Archive: `/data2/shared/ang.gao/c4-render-geometry-native-evidence-v1.tar.gz`, independently recomputed SHA-256 `619167158ae9da9a99112a6ce22a191fb033645dbcf2bb714a4c810ae33e8f3a`. All **91 regular members** independently matched their current original bytes, including complete case-file inventories, runner/archive scripts, batch and environment profiles. No duplicate regular member names or non-file/non-directory members. Companion JSON and archive-author assertions were inspected as claims, not substituted for original replay.

Cold original CPU source preflight freshly reproduced each archived `runtime_custody_inputs.json`. Catalog SHA-256 is `f6208cb33d29476878925e81f4f7e3c0665ca84f718aac4ff042f35b5a5fc70a`; source snapshots are Warp `b07421578384b6d6d78eb08e29e4bbc0595ccdbc77cd3699b9441a7c2ddf2881`, CUDA `efd1ca26cbbc013525fd16e2b559e7bfc54d042ae2c0dacdf1fb737dde139f93`. Original `RuntimeEnvironmentContract.verify_files` passed against Warp `02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb` and CUDA `a0b6cf8e0feaad6e6bd4ea75849bce623ff91f73ef3626f97f4ae9f6d785d3ce`.

## Actual lifecycle and copies

Read complete shared review instructions, required design sections, implementation note/prior reviews, the actual diff, original runner/archive/extractor scripts, production warmup authority/transactions, native resource scope, readback body, original acceptance validators and runtime-custody parsing/attachment. Code establishes the data source and operation semantics; actual artifacts and strict replay establish the observed execution sequence. Neither hashes nor caller-supplied counts alone are treated as native authority.

Both cases are real Go2 `1920x1080`, `direct_light_full`, shadows on, `primary_unprofiled`, target ordinal1/ambient0. Recorded target UUID is `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`, driver `595.71.05`.

| Lane / directory | Warmup guards | Readback guard | Normal completions / arms | Supervisor / worker |
| --- | --- | --- | --- | --- |
| Warp / v1 | 14..23 | setup24 | 25 / 26 | 3547058 / 3547065 |
| CUDA / v2 | 10..19 | setup20 | 21 / 22 | 3550988 / 3550989 |

Fresh calls to `validate_custody_artifact`, `validate_runtime_custody_selection`, `RuntimeCustodyRecord.from_payload`, `validate_runtime_custody_attachment`, `_validate_fixed_storage`, `_validate_render_graph` and `_validate_render_warmups` all passed on the original bytes. Supervisor raw artifacts equal the separately saved custody artifacts. Batch and case results agree. Current runtime selection contains all 4,381 Warp / 6,042 CUDA lifecycle events, not a stale failure prefix; journal cuts contain 11,446 / 19,320 entries.

Each case records ten distinct original transactions and terminal objects, frame indices0..9, ten terminal waits, unique completed guards, correct graph/bundle/event identity and target stream/context observations. Warp has five launch leaves per unprofiled frame; CUDA has three, followed by terminal record/wait. Each frame's final descriptor precedes its DISARMED, which precedes the next frame's first descriptor. Original production `warmup.py:166` waits on the real terminal event and requires the completed guard ticket before reusing the event. Provisional readiness remains warmup0; neither final execution readiness nor a sample token is issued by this diagnostic.

The actual `native_runtime_process.py:1112` readback body reads `session.fixed_render_storage[3]`, after the tenth physical completion, in this exact order:

`rgb, origins, directions, hit_mask, range_m, position_world, normal_world, numeric_instance_id, material_index`.

Warp calls each original array's `.numpy()`; CUDA calls `.detach().cpu()` and then converts the returned host tensor to NumPy. All nine ordered D2H descriptors, nine host-owner adoptions and one subsequent normal DISARMED are present under the same lane-specific readback guard. The held owner list is adopted before the body, each host is retained before conversion/serialization/next copy, and the current post-readback checkpoint follows successful context/guard exit. All bytes, native shapes/dtypes and manifest hashes validate; RGB and geometry both name frame10 and the same guard. There is no intervening render, reset or extra kernel to reconstruct geometry. Native scope uses the existing target and stream binding; it does not add a separate synchronization policy.

The existing observer restores Warp ambient device0/no current context. CUDA preserves its frozen lazy restoration contract: logical0/raw1. This is not a claim of raw CUDA device0 restoration. Readback does not have a separate per-copy hardware profiler trace; the trusted frozen execution path, custody operation records and resulting original buffers are the evidence available here.

Both original workers and supervisors exited0, with empty stderr, no failure or ordinary-atexit marker, no watchdog expiry or kill request, and original `aborted_synced_exit` disposition. One emergency target synchronization and completed release occur during final sacrificial disposal; this is an exit path, not a reusable success state. Original source uses `os._exit`. CPU replay also rejected removed copies, reordered copies and removed readback DISARMED for each lane. These six mutations test serialized consistency, not arbitrary compromise of the trusted native process.

## Diagnostic calculations and C1

**C1: unresolved numerical correctness, P1 for its separate acceptance gate.** Trigger/location: frame10 native RGB/geometry buffers of these two 1920 cases. Existing lifecycle validators check execution/custody/bytes/shape and deliberately do not enforce CPU optical agreement, so successful lifecycle validation cannot detect or resolve this failure. The proposed direction remains investigation and a separately reviewed numerical repair, with preserved repros and fresh CPU/native validation under unchanged frozen tolerances.

Independent full-frame binary comparison finds RGB maximum absolute difference `1.2056253775954247`, mean `9.481629192419044e-7`, and exactly the original six pixels above the **diagnostic filter** `1e-4`. This filter is not an acceptance tolerance. Both new RGB files are byte-identical to their prior original warmup v5/v6 counterparts. All values in the two six-pixel extracts independently reproduce from the raw binary channels, including their full-file hashes.

| Index | Warp / CUDA instance | Warp / CUDA range | Max RGB difference |
| --- | --- | --- | --- |
| 782417 | 6 / 6 | 1.7280921936 / 1.8124505281 | 1.2056253776 |
| 901306 | 30 / 30 | 1.7531722784 / 1.7548371553 | 0.0332396105 |
| 976396 | 15 / 14 | 1.4505175352 / 1.4518601894 | 0.0700501353 |
| 985873 | 6 / 6 | 1.6117191315 / 1.6956703663 | 0.5564480051 |
| 1026413 | 6 / 5 | 1.4972925186 / 1.4098925591 | 0.1095623001 |
| 1143492 | 3 / 6 | 1.4328759909 / 1.4328759909 | 0.1906471997 |

The first five also have different normals. At the last, normal/range agree but numeric instance and material differ (material3 versus5). Thus first-hit data already diverges at every RGB outlier. It is not accurate to claim identical rays: origins and hit masks agree over the whole image, but directions differ at 971,723 pixels, maximum component difference `2.384185791015625e-7`, including all six. Small direction differences and different arithmetic may affect discontinuous predicates; these measurements do not determine the causal mechanism.

Full-frame inspection also shows that six RGB outliers do **not** exhaust geometry differences. Numeric instance and material disagree only at indices976396,1026413,1143492. Normals disagree at seven hit pixels: the first five RGB outliers plus **1116499 and 1116503**, whose max normal differences are `1.8273425698280334` and `1.2757190763950348`, despite exactly equal RGB. There are 656,857 differing range values and 909,908 differing position vectors, many tiny. These counts are descriptive, not pass/fail metrics. Both lanes have 132,022 misses with nonfinite range/position/normal sentinel values; none of those channels has nonfinite values on hits. Matching NaNs are treated as matching sentinels in these comparisons, consistent with the original reset source. This inspection is not cross-generation sentinel acceptance.

Fresh CPU replay ran the original `check_cpu_with_native_rays_1920_v1.py` and `check_cpu_fixed_fixture_intersections_v1.py`, after inspecting their actual construction, against the frozen source and existing frozen Go2 assets. It used an exclusive scratch directory and six **Warp-returned** binary32 rays converted losslessly to binary64, plus the script's normalized-ray comparison. Both scripts exited0. Fresh JSON objects exactly equal the supplied `cpu_sample_diagnostic.json` and `fixture_intersection_diagnostic.json`; every fresh NPZ array exactly equals the original NPZ array.

CPU native-ray instance IDs are `[6,30,14,6,5,3]` for both raw and normalized versions. The CPU-vs-Warp RGB max is `0.10956230777247106`. The exact encoded fixed-fixture diagnostic independently yields nearest instance IDs `[6,30,14,6,5,3]`, with fixture SHA-256 `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0`. At1143492, its first two intersections have exactly equal binary64 distances for instances3 and6, ordered by source key. These are reproducible diagnostics through the original CPU routines, not a second independently implemented optical oracle. The fixture script reconstructs vertices from encoded v0/edge arrays and calls scalar CPU intersection routines; it does not emulate GPU instructions, compiler contraction, BVH traversal or tie arithmetic.

The coordinator supplied later scratch float32/float64 predicate investigations while this review ran. Those extra arithmetic scripts/results were **not independently reviewed or replayed here**, are not part of the frozen native archive, and add no acceptance claim. This report neither endorses a specific epsilon/FMA/tie root cause nor rejects those hypotheses. A repair still needs causal evidence plus fresh native and CPU validation of affected outputs, the two additional normal discrepancies and the broader authorized correctness matrix. Changing tolerances or calling discontinuities harmless is not justified by these results.

## Fresh commands, coverage and limits

Environment: remote `/home/ang.gao/miniforge3/envs/robot_sim/bin/python` (recorded Python3.12.13), `-B`; local PowerShell for reads, Git and report files. No new native worker/supervisor was launched. CPU-only original source-preflight subprocesses were invoked and synchronously reaped by their original helper.

Reviewer scripts are outside both candidates in local `c4_assets` and remote `/data2/shared/ang.gao`:

- `geometry_evidence_review_probe_v1.py`: archive/member integrity, fresh source/environment verification, strict original replay, ordering/exit assertions, six replay mutations, exact extractor verification and full-array comparison. Final SHA-256 `397764ca9bd18e0e32c2f6319ef113c6941e89858b800ee1225612dc31393fd1`. Remote command: `python -B /data2/shared/ang.gao/geometry_evidence_review_probe_v1.py /data2/shared/ang.gao/geometry_evidence_review_probe_v1.json`; final tool session43314 exited0. Its JSON was copied to local `c4_assets`.
- `geometry_cpu_review_replay_v1.py`: isolated fresh original CPU replay and exact JSON/NPZ equality; SHA-256 `e45c0ccd3a85b128b4029d17b26d74522996a72bedd04c360eb9415c26633e19`. Remote command: `python -B /data2/shared/ang.gao/geometry_cpu_review_replay_v1.py`; session63668 exited0. Scratch `/data2/shared/ang.gao/geometry-cpu-review-v1-ck0ocyva` retains outputs/stdout/stderr/ledger; local `geometry_cpu_review_result_v1.json` contains the outcome. The script passes that scratch, frozen repository, `/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2`, and original Warp outlier JSON to each original diagnostic.
- `geometry_review_details_v1.py`: readback host-adoption/cut/source inspection and full-frame normal-difference indices, terminal exit0.

Inspection limitations/tooling corrections: broad console outputs were truncated and relevant code/design sections reread in bounded ranges; initial Chinese decoding corrected with UTF-8; two speculative rg paths/globs failed before actual files were found. Two early scratch probe runs exited1 on the reviewer's overly broad assumption that all geometry values are finite: miss range is infinite and miss position/normal is NaN by the original reset. The scratch comparison was corrected to distinguish hit data and matching miss sentinels; this was not a candidate failure or source change. The final successful run retains no NaN comparison maxima. An attempted v2 `author_process_observation.json` read confirmed that supplemental live start-tick file is absent there; this is reported below rather than fabricated.

This is read-only acceptance of already captured real executions, not a fresh GPU reproduction. No fault/hang matrix, native partial-readback failure, all-row/profile/R1 correctness matrix, full-frame CPU oracle, separate CUDA-ray CPU replay, or numerical fix was executed in this review. Prior unit/code reviews are context, not newly run tests. Same-frame originals provide first-hit diagnostics; they do not prove complete shading/shadow correctness or absence of further geometry differences.

## Actual process hygiene and remaining scope

Original parent PID3546893; Warp supervisor/worker3547058/3547065, CUDA3550988/3550989. Identity/parent/result/raw-supervisor records agree and original supervisors record worker exit0. Warp's supplemental live observation additionally records parent/supervisor/worker start ticks244370879/244371570/244371581. CUDA has no equivalent supplemental live start-tick observation; its launch parentage, raw exit and current absence are the available evidence, not a retrospectively invented start-identity proof.

Reviewer archive/replay attempts: PID3569510 (start244460216) exit1, PID3570017 (244461865) exit1, PID3570580 (244463571) exit0, final PID3572198 (244469147) exit0. CPU harness PID3571744/parent3571743/start244468130; children3571813/start244468139 and3572821/start244471160 both exit0, synchronously waited and checked absent. The original short CPU-preflight subprocess PIDs were not separately sampled; their helper uses `subprocess.run` and returned successfully. Short SSH/scp/inspection commands completed terminally without individual PID ledgers.

At `2026-10-10T05:14:12+08:00`, scoped remote `ps` found none of the original parent/worker/supervisor or recorded reviewer PIDs. A command-line search found only its own inspection shell, no remaining reviewed script/CPU-scratch/native-runner command. GPU query showed only pre-existing PID2238410 `nvidia-cuda-mps-server` on ambient GPU0 UUID `GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274`, matching both original after-case GPU records; no target-GPU process appeared. No signal, broad cleanup or reset was issued. No unresolved owned residual/hang was observed. This is scoped cleanup evidence, not a claim about unrelated processes or all possible historical hangs.

Only reviewer scripts/results/report and exclusive CPU scratch were written outside the candidates; original source and native/CPU artifacts were preserved. Both frozen checkouts and primary remained clean at final check. This bounded approval opens no numerical correctness, performance, activation, baseline/backend-selection, result publication, final readiness/origin/recording/reset/arena/barrier, full lifecycle/fault matrix, R1, host delivery/R2/S, or whole-C4 completion gate. The prior RGB P1 and later systematic C4 review obligations remain unresolved. Archive this report verbatim; an evidence-only archive cannot broaden its verdict or scope.
