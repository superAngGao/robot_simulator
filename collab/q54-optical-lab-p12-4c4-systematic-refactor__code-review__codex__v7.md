```text
Initiative: Q54 / P12.4 / C4
Stage: Independent CODE review — coherent R1 preparation/final composition and F1 repair
Author: Codex, fresh independent reviewer /root
Version: v2
Date: 2026-10-10, Asia/Shanghai
Status: accepted
Related Files:
  runtime-integration design-request/review codex v3
  systematic-refactor gap-review codex v1
  systematic-refactor code-review codex v5/v6
  systematic-refactor implementation-note codex v8/v9
  PROGRESS.md; production and test changes from the M1 base
Owner Summary: Approve the exact bounded R1 candidate. The original F1
  counterexamples now reject through normal final admission in separate
  fresh real-device executions. No new substantive finding remains.
  C2f, whole-C4 acceptance and all later product gates remain closed.
```

**Verdict: Approve.**

Approval covers the coherent R1 dynamic preparation and dormant final graph composition, including the F1 repair, at the exact revisions below. It does not approve final sample execution or full C2f/C4 acceptance.

**Independent session and frozen revisions**

Fresh session: **`01a1248e-3bb3-7850-9565-295f071e3b77`**, agent `/root`.

I received the review packet without the implementation conversation, read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` fully, and inspected the requirements, associated design review, delivery-order review, M1 approval, previous F1 report and original reproduction artifacts. Author notes and prior verdicts were treated as claims to verify.

No agents were spawned. No candidate files were edited. This complete report is returned directly to the CLI output destination for unchanged archival.

| Item | Verified value |
|---|---|
| Local candidate | `C:/Users/AG/新建文件夹/c4_dynamic_graph` |
| Local branch | `implementation/c4-dynamic-graph` |
| Approved M1 base and merge-base | `c3472f22f1d52bfb6c853bf95cef4040cf666316` |
| Local HEAD | `fb49e7eba847d2168a0de81d791357c64c3f9e1c` |
| Local tree | `36258c53217872982050f79e67bf7aa44a13fc13` |
| Primary remote | `tileops-dev:/data2/shared/ang.gao/robot_sim_dynamic_graph_13d17dd` |
| Remote HEAD, detached | `13d17dd293f9c79028260b856e99e39d5f344da5` |
| Remote tree | `675c26ae960a96d6b4f87c53d7cdc3e4b2f58295` |
| Working trees | Clean at initial and final checks |

Local HEAD differs from remote `13d17dd` only in `PROGRESS.md` and implementation-note v9. Production and test sources are equivalent. The last production change is `13d17dd`.

The complete base-to-HEAD delta contains 29 files, 2,858 insertions and 533 deletions. Review covered that coherent change, rather than only the repair tests.

**Findings and original F1 disposition**

No new substantive blocking or nonblocking finding was established.

**F1 — previous P2 blocking finding: closed for this revision.**

I inspected the original reviewer script, mutation record, accepted final graph and execution log. Those artifacts demonstrate the earlier invalid admission; they were preserved and were not counted as fresh execution.

The repair’s actual call chain is:

1. `_warp_command_recording()` retains each returned native command before constructing its descriptor from the exact recording inputs.
2. Scene and refit bundles freeze native snapshots through the existing, unchanged `_WarpRenderDescriptor` and `_render_launch_snapshot()`.
3. `_validate_dynamic_stages()` validates both bundles and their replay/dataflow relationships.
4. `_require_final_graph()` invokes this validation during normal final-barrier admission, and again after the barrier before publication.

Relevant locations:

- [Shared recording and descriptor capture](/C:/Users/AG/新建文件夹/c4_dynamic_graph/optical_worker/native_setup.py:1499)
- [Scene native snapshots](/C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/device_scene.py:261)
- [Refit binding validation](/C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/device_bvh.py:180)
- [Shared dynamic-stage validation](/C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/warp_timeline.py:54)
- [Final graph validation](/C:/Users/AG/新建文件夹/c4_dynamic_graph/optical_worker/native_setup.py:1015)

Inspection covers actual kernel/device identity, launch settings and dimensions, parameter counts and addresses, packed array pointers/shapes/strides, scalar values and frozen executable/hook identities. Strict command/descriptor pairing and frozen snapshots detect command replacement, count changes and reordering. These checks remain in untimed validation.

I independently reran the original two mutations **separately** against current production code. Only scratch copies of the test launchers were changed. Neither test directly invoked the validator.

| Fresh case | Actual mutation | Observed result |
|---|---|---|
| Scene frame 0 | `set_param_at_index(7, frame1.transforms.x_world_r)` | Rejected by normal final admission |
| Refit frame 1 | `set_param_at_index(5, refit.bvh.bounds_max)` | Rejected by normal final admission |

Observed native addresses:

| Operand | Before | After |
|---|---:|---:|
| Scene translation | `43486553600` | `43486553612` |
| Refit bounds-min output | `43486567936` | `43486568448` |

Both raised:

```text
ValueError: recorded array descriptor differs from exact bound storage
```

Both executions established:

- Worker exit **74**, observed by the supervisor.
- No final barrier, quiescence or final admission.
- No affected final-command submission.
- **26** completed normal guards and **27** total ARMs.
- One healthy-IDLE emergency target synchronization, successful release and terminal exit.
- Emergency guard remained armed through terminal handling; no emergency COMPLETE/DISARMED.
- No deadline expiry or kill request.

The original failure was admission of a changed dormant graph. These fresh runs directly close that failure without requiring execution of corrupted commands.

Evidence: [fresh reproduction log](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/repro.log>). Complete raw cases remain under `/data2/shared/ang.gao/c4-r1-review-v2-scratch/fresh_original_scene_pointer` and `fresh_original_refit_pointer`.

**Coherent R1 implementation assessment**

The inspected composition preserves the accepted architecture:

- Warp uses prepared replay views, existing scene-update kernels and immutable leaf4/sort topology with slot-owned mutable bounds. Initial CPU topology construction remains untimed setup.
- CUDA uses its existing scene-update and LBVH-rebuild stages.
- Both lanes retain `NativeOpticalArena` as the slot/generation/FIFO/hold authority.
- Shared frame binding serves provisional and final graphs. Final factories and readiness have distinct identities while reusing the permitted warmed storage.
- Final frame selection starts at sample/replay index zero after warmups 0–9.
- Recording, upload/readback, allocation and event initialization use existing custody mechanisms. Returned owners remain held before later fallible operations.
- R1 preserves five phase positions. Diagnostic initialization precedes terminal initialization; diagnostics-off allocates no diagnostic pairs.
- Guardian coverage seals once before the final target barrier. Successful synchronization and DISARMED are required before admission.
- Armed setup failures retain the original failed guard and do not acquire an emergency retry.
- Acquire continues through fixed identity/state checks; the new packed-command inspection is not added to acquire.

The repair introduces no numerical kernel, slot state machine, ABI-inspection framework or generic backend adapter. CUDA input assembly and logical stage-input selection are consolidated rather than copied.

The unchanged renderer inspector’s name remains renderer-specific, but this naming does not create a second implementation or a blocking maintenance concern.

**Independent verification**

| Check | Result |
|---|---|
| Six core CPU suites | **136 passed**, 209.87 seconds |
| Five affected CPU suites | **176 passed**, 86.92 seconds |
| Fresh original native mutations | **2/2 rejected correctly** |
| New matrix custody/runtime replay | **14/14 records validated** |
| CPU numerical recomputation | **8/8 passed** |
| Historical targeted fault replay | **12 records validated**, historical executions only |
| Source catalog/body/policy checks | **90 entries passed** |
| Archive integrity | All **716** manifest-listed bodies matched |
| Full-delta `git diff --check` | Passed |

The CPU executions had **no failures or skips**. They used:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1
--basetemp=<authorized local scratch>/...
```

Core suites under `tests/unit/optics`:

```text
test_dynamic_command_preparation.py
test_dynamic_arena_binding.py
test_warp_native_command_contract.py
test_warp_arena.py
test_native_final_graph.py
test_native_render_graph_support.py
```

Affected suites:

```text
test_native_render_warmup.py
test_native_upload_core.py
test_guarded_native_readback.py
test_cuda_arena_contract.py
test_cuda_arena_resolution.py
```

These include packed pointer/scalar/dispatch/kernel/shape/parameter-address negatives, replay root/offset checks, custody failures, readiness/authority rejection, barrier/ACK behavior and the 1/120/600-frame acquire work-count regression. CPU native doubles remain CPU evidence.

Logs: [core tests](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/cpu.log>), [affected tests](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/cpu_affected.log>).

Remote commands used `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B` with these scratch scripts:

```text
review_remote.py audit
review_remote.py repro
recompute_numeric.py <candidate> <new-matrix> 8
final_remote_audit.py
close_review.py
```

`recompute_numeric.py` retains the inspected author checker’s calculations and redirects its output into reviewer scratch. It does not overwrite the original numerical report.

**Evidence integrity and interpretation**

Catalog v17 has 90 entries and SHA256:

```text
e532beb68548a1dbc515cf7af59028053ac9292edbd70401b90db4c138363097
```

I validated individual committed bodies against their digests and production AST policy, verified local/remote production equivalence, and checked the remote working bodies against committed HEAD.

All three scene-kernel and two refit-kernel ASTs are unchanged from M1. CUDA numerical source files and `warp_native_render.py`, including the reused inspector, are byte-identical to M1.

The local and remote v2 archives match:

```text
Size:    184751064 bytes
Members: 717
SHA256:  94e1af07deba62234ba48295bd4113abf2611d5e0ec213955ddafb36d4cd3058
```

Members are unique regular files with safe relative paths. All 716 manifest-listed bodies matched their sizes and digests, both inside the local archive and in the remote raw evidence.

For the new matrix, I replayed combined custody, runtime-custody selection and attachment, and inspected original failures, mutation markers, partial final state, graph/storage/warmup bindings and supervisor exits.

The six native mutations cover scene/refit pointers, scene environment scalar, scene/refit dispatch and scene kernel. Their recorded rejection follows normal final admission. They show no final admission or affected submission and one legal emergency synchronization/release.

The eight positive cases cover both mandatory R1 rows, both lanes and primary/diagnostic modes. They establish ten setup warmups, three dormant final plans, final barrier/admission, acquire/cancel and zero final sample submissions.

The earlier twelve targeted fault records still replay as quarantine under their original failed guard, without another synchronization. This was evidence replay, not fresh fault execution. Earlier failed diagnostics remain failed historical runs and are not relabeled as passes.

**Numerical and environment limits**

Independent recomputation matches all eight author numerical case records:

- All **614,400 rays** of setup warmup 10, replay index 9.
- Exact hit-mask, instance-ID and material comparisons.
- Ray tolerances `1e-6`; floating output tolerances `1e-5`.
- Maximum observed floating error: **`7.298419912871168e-08`**.
- Existing overflow, stack-depth and traversal-counter invariants pass.

This does not establish final-sample correctness, all replay frames, occlusion or internal-tree traversal. The canonical fixture has one triangle.

Local environment: Python 3.14.0, NumPy 2.5.2, pytest 9.1.1; no local Warp execution.

Remote environment: Python 3.12.13, Warp 1.12.0, Torch 2.9.1+cu128, NumPy 2.5.1. Fresh native runs used H200 physical GPU1, UUID `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`, with all GPUs visible.

Captured evidence supports target1/ambient0 behavior. CUDA’s recorded ambient state includes logical device0 and raw device1; it does not prove raw-device0 restoration.

I did not rerun the full repository, the legacy nine-case GPU suite, all positive native cases, or the full watchdog/parent-loss/C2f matrix. The legacy masked physical1/logical0 run is not dual-GPU proof. No performance conclusion is drawn.

Inspection-command encoding/quoting errors were corrected: initial UTF-8 display, a remote process-filter expression, PowerShell’s unquoted tree revision syntax and one package-version probe. These were reviewer-tool errors, not candidate failures.

**Process closure and preservation**

Reviewer CLI: PID **46072**, parent **44604**, started `2026-10-10 14:44:23`.

Local pytest processes **10852** and **45804** completed successfully and were absent at final inspection. Source/archive audit PID **40912** also completed and was absent.

Fresh native reproduction driver: **1505078**, parent **1505077**.

| Case | Supervisor | Worker | Observed exit |
|---|---:|---:|---:|
| Scene pointer | 1505216 | 1505221 | 74 |
| Refit pointer | 1510210 | 1510222 | 74 |

Both worker/supervisor pairs were absent before continuing. Final scoped checks confirmed all recorded author matrix/archive and reviewer run PIDs absent, no scratch-scoped Python residual, and no owned PID in GPU-process records. Closure checker **1522822** subsequently exited and was confirmed absent.

The numerical replay and short-lived inspection utilities completed synchronously; their individual PIDs were not all sampled. This is scoped closure evidence, not an exhaustive OS attestation.

All tool sessions reached terminal results. Unrelated GPU processes were preserved. No process signals, resets, installs, candidate edits or evidence deletion occurred. Original reviewer evidence and prior archives remain intact.

Evidence: [source/archive audit](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/source_archive.log>), [remote evidence audit](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/final_audit.log>), [process closure](</C:/Users/AG/新建文件夹/c4_assets/r1_review_v2_scratch/closure.log>).

**Exact approved scope and closed gates**

Approved: **the R1 dynamic preparation, supervised setup warmups, dormant final composition, C3/arena binding, final preparation barrier/admission and F1 repair at local `fb49e7e`, with production/test code equivalent to remote `13d17dd`, relative to M1 `c3472f2`.**

No substantive repair is requested for this bounded scope.

C2f must still integrate actual worker execution, guarded resolution, release/reclaim and post-reclaim close/ACK/observed exit, with the full declared correctness and failure matrix. Whole-C4 systematic refactoring remains mandatory and has not been waived. Existing structural follow-ups and P12.4e transitional-harness cleanup obligations remain outstanding at their applicable checkpoints.

Still closed: **product PREPARE, final measured submission, C2f and full C2/C4 acceptance, activation, R2/S/host delivery, external holds/depth two, C4c3/c4, C4d/B3, result/baseline/a3b, performance results, backend selection and shipping.**