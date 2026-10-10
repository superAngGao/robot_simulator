```text
Initiative: Q54 / P12.4 / C4
Stage: Independent CODE review — R1 dynamic preparation and final graph
Author: Codex, fresh independent reviewer /root
Version: v1
Date: 2026-10-10, Asia/Shanghai
Status: in_review
Related Files:
  collab/q54-optical-lab-p12-4c4c2-runtime-integration__design-request__codex__v3.md
  collab/q54-optical-lab-p12-4c4c2-runtime-integration__review__codex__v3.md
  collab/q54-optical-lab-p12-4c4-systematic-refactor__gap-review__codex__v1.md
  collab/q54-optical-lab-p12-4c4-systematic-refactor__code-review__codex__v5.md
  systematic-refactor implementation-note v7/v8; PROGRESS.md
Owner Summary: Request changes. A real-device counterexample shows that
  Warp final admission accepts changed scene/refit packed command operands.
  Positive preparation evidence and numerical claims otherwise checked out.
  Product and later-stage gates remain closed.
```

**Verdict: Request changes.**

One blocking finding remains in the bounded R1 scope. The defect concerns validation of the actual dormant final graph; it does not depend on opening C2f or executing a final sample.

**Independence and revisions**

Fresh reviewer session: **`01a1246a-22f1-7702-b7be-618ec7caa515`**, agent `/root`.

I received the review packet without implementation conversation, read `AGENTS.md`, `collab/README.md` and `collab/REVIEW_PRINCIPLES.md` completely, and inspected the requirements, associated independent design review, delivery-order review, M1 approval, author notes and complete R1 delta. I did not spawn agents, edit candidate files or implement fixes.

| Item | Verified value |
|---|---|
| Local candidate | `C:/Users/AG/新建文件夹/c4_dynamic_graph` |
| Local branch | `implementation/c4-dynamic-graph` |
| Base and merge-base | `c3472f22f1d52bfb6c853bf95cef4040cf666316` |
| Local HEAD | `ba394c6f4b45632d09d65362caa4456458d1a81b` |
| Local tree | `86e95acc47443684a15b384639284eb1257100f3` |
| Primary remote | `/data2/shared/ang.gao/robot_sim_dynamic_graph_98315da` |
| Remote HEAD, detached | `98315dac353490161742c44062f5794d31d4720a` |
| Remote tree | `83e6da691bc2408b5c157a8979b7e42a135e84ef` |
| Working trees | Both clean at initial and final checks |

The approved R0 revision `640cf9f24490e9bf7d259c37ec9adfb0412ac1ba` is an ancestor of the M1 base.

The revision claims are accurate:

- Local HEAD differs from remote `98315da` only in `PROGRESS.md` and implementation-note v8.
- Changes after production revision `626f6320eda3bbfb527be49390c696012873e33d` are confined to tests, test drivers and documentation/evidence.
- `98315da` changes only the test driver’s channel lookup, using existing buffer inventories.
- The complete base-to-HEAD delta contains 27 files, 2,265 insertions and 529 deletions.

**F1 — P2, blocking: Warp final admission does not validate the actual dynamic command operands**

Primary locations:

- [Dynamic-stage validation](C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/warp_timeline.py:54)
- [Final graph validation](C:/Users/AG/新建文件夹/c4_dynamic_graph/optical_worker/native_setup.py:1015)
- [Scene bundle validation](C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/device_scene.py:333)
- [Refit command bundle](C:/Users/AG/新建文件夹/c4_dynamic_graph/optics/device_bvh.py:130)

`_validate_dynamic_stages()` checks scene/replay Python objects and the world/BVH relationships. `_validate_replay_views()` correctly checks the selected view’s original allocation and offset. However, these checks do not establish that the actual recorded scene command still consumes that view.

The scene bundle’s signature records command identities, owners and descriptive evidence, without inspecting packed command arguments. The refit bundle likewise lacks equivalent packed-command validation. The renderer already performs this kind of inspection through `_render_launch_snapshot()`.

I reproduced the gap through the real captured Warp preparation path on the primary host. Only copies of the two test launchers were modified under the authorized remote scratch directory. Production source, catalog, kernels, supervisor semantics and candidate checkout remained unchanged.

After `original_prepare()` constructed the final graph, but before final barrier/admission:

1. I called the existing Warp command API:
   ```python
   scene_command.set_param_at_index(7, frame1.transforms.x_world_r)
   ```
   This changed final frame 0’s actual translation operand to frame 1. Its Python `transforms.transform_frame_index` remained 0.

2. On another final frame, I called:
   ```python
   refit_command.set_param_at_index(5, refit.bvh.bounds_max)
   ```
   This redirected the leaf-refit command’s `bounds_min` output to the existing `bounds_max` allocation.

Observed native addresses:

| Operand | Before | After |
|---|---:|---:|
| Frame-0 scene translation | `43486553600` | `43486553612` |
| Refit bounds-min output | `43486567936` | `43486568448` |

The scene bundle’s `validate_bound()` accepted the changed command. The unchanged production final validation and barrier then accepted the graph:

- Final readiness and quiescence were published.
- Three dormant plans remained present.
- Acquire/cancel succeeded.
- Worker exited **0**.
- Supervisor observed exit, with **27 normal completions, 28 ARMs**, no deadline expiry and no kill request.
- Final sample submissions remained **zero**.

The mutated commands were **not launched**. The demonstrated failure is admission of commands whose native operands disagree with the graph’s declared replay/dataflow. Executing those commands would read the wrong replay translation or write the wrong bounds array; I did not execute them to demonstrate downstream corruption.

This is an internal binding-validation counterexample required by the accepted descriptor/dataflow contract, not a claim of an IPC escape or hostile-Python sandbox vulnerability. The new R1 composition depends on these existing leaf bundles, so their weaker validation must be addressed at this checkpoint.

**Why current checks miss it:** replay tests alter view metadata/root/offset, while native positive cases leave recorded arguments intact. Module/executable identity checks still pass after `Launch.set_param_at_index()`. C3 facts describe the Python storage graph, which also remains unchanged.

**Required repair:** extend or reuse the existing read-only recorded-command inspection for scene and refit commands. Validate actual kernel/dispatch and packed operands against their frozen expected inputs and outputs during untimed preparation and final validation. Preserve shared recording/execution, existing custody and fixed-cost acquire; do not introduce another kernel or adapter framework.

**Required proof after repair:** independently rerun these two mutations as separate cases and show rejection before final admission, with no affected command submission and conservative custody. Include relevant scalar/environment-index, dispatch and output-pointer mutations, retain valid R1 preparation, and obtain a fresh review of the revised code.

Reproduction and evidence:

- [Reproduction generator/runner](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/run_command_mutation.py)
- [Native run result](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/command_mutation.log)
- [Observed operand changes](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/review_command_mutation.json)
- [Accepted final graph](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/final_graph_binding.json)

Complete raw reproduction records remain at:

`/data2/shared/ang.gao/c4-r1-review-v1-scratch/native_command_mutation`

No second substantive blocking finding was established.

**Architecture, ownership and gate inspection**

The implementation follows the intended shared architecture:

- Both lanes use `NativeOpticalArena` for slot, generation, FIFO and hold control.
- Warp reuses the existing scene-update and leaf4/sort refit kernels; CUDA reuses scene update and LBVH rebuild.
- Initial Warp topology construction remains untimed setup. Transactions do not perform a CPU rebuild.
- Mutable world arrays and refit bounds are represented as slot resources; immutable topology and prepared replay allocations are retained as session resources.
- Recording, transfer and event initialization use the existing custody helpers. Returned native owners are retained before subsequent fallible work.
- Diagnostic initialization precedes the terminal initialization record. R1 has five diagnostic pairs; diagnostics-off has none.
- Setup and final factories/readiness remain distinct. Final plans start from replay/sample index zero after warmups 0–9.
- Normal failure handling retains the original failed guard and does not manufacture an emergency retry.

I found no second numerical implementation, slot state machine or general backend framework.

Product gates remain closed. The worker still rejects `PREPARE_FROZEN_ROW`; captured arena enqueue reaches the rejecting `latch_measured_sample_submission()`. Internal final admission accepts the frozen R0/R1 scope, while final specs require one native-only slot. No activation, backend-selection or result gate was changed.

**Executed verification**

Local tests used `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider`, `PYTHONDONTWRITEBYTECODE=1`, and scratch-only basetemps.

| Verification | Observed result |
|---|---|
| Eight core CPU suites | **265 passed**, 278.45 seconds |
| Five affected CPU suites | **120 passed, 4 skipped**, 71.19 seconds |
| Full-delta `git diff --check` | Passed |
| All 90 committed production source entries | Digests and AST admission passed |
| All 90 remote working source bodies | Equal catalog digests and committed HEAD bytes |
| Warp numerical kernel AST comparison | Three scene and two refit kernels unchanged |
| CUDA numerical source comparison | Scene, LBVH and direct-light source bytes unchanged |
| Original/corrected matrix custody replay | All **28 raw records** validated |
| Corrected matrix CPU numerical recomputation | All **8 cases** passed |
| Fresh real-device mutation reproduction | Invalid final graph accepted; F1 reproduced |

Core suites:

```text
test_dynamic_command_preparation.py
test_dynamic_arena_binding.py
test_warp_arena.py
test_native_final_graph.py
test_native_render_warmup.py
test_native_upload_core.py
test_cuda_arena_contract.py
test_cuda_arena_resolution.py
```

Affected suites:

```text
test_guarded_native_readback.py
test_native_render_graph_support.py
test_prepared_fixed_render_inputs.py
test_warp_native_command_contract.py
test_native_sources.py
```

All are under `tests/unit/optics`. The four skips were two unconfigured real-Go2 asset cases and two POSIX release-byte checks. They are not passes. The previously reported graph-observer fixture failure is repaired in this candidate, and its tests passed.

Logs: [core tests](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/cpu_core.log), [affected tests](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/cpu_affected.log).

The CPU tests use native doubles where documented; they are not native-execution proof.

Remote verification used SSH `tileops-dev` and `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B`. The native reproduction used the frozen Warp environment and target physical GPU1 H200, UUID `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`, with all GPUs visible. No dependencies were installed and no performance measurement was performed.

Reviewer scripts executed remotely:

```text
audit_native.py
run_numeric.py
run_command_mutation.py
final_audit.py
```

They reside under `/data2/shared/ang.gao/c4-r1-review-v1-scratch`, with corresponding local scratch copies.

Inspection-script errors—a missing keyword argument on the first AST-audit invocation and shell quoting/path issues—were corrected. They were tool/reviewer-script failures, not candidate test failures.

**Evidence verification and interpretation**

Source catalog v17 contains **90 entries**. I checked individual source bodies and AST policy rather than accepting only its hash. The literal lane inventories remain exact.

Catalog SHA256:

`b9db06834328433d24ee67b697acb0a2d68d2c736e12c2c0a2eb2f9316a39dca`

The matrix archive was complete locally and remotely:

- Size: **245,895,053 bytes**
- Members: **1,195**, unique, with no unsafe paths or link members
- SHA256: **`7fa37ad6cad63cc79e9a1f66d74457003db12830e2bf73307351130451073849`**
- All **1,194** manifest-listed files matched their sizes and content digests.

I replayed custody artifacts, runtime-custody selection and attachment against the candidate validators, checked request/environment bindings and supervisor-observed exits, and validated applicable storage, graph and warmup records.

| Campaign | Original driver outcome |
|---|---|
| `f15aa62b11` | **16/20 passed:** four primary positives and twelve expected-fault cases |
| `98315dac35` | **8/8 passed:** both rows, both lanes, both pass modes |

The four original diagnostic cases remain failed executions. Their `AttributeError` records are preserved; successful custody replay does not turn them into passes.

The twelve fault cases reached the recorded native-return/adoption boundaries. Their workers exited 74 with quarantine, no emergency synchronization and no resource-release authorization. Each retained one more ARM than normal completion. Final-barrier faults did not publish admission or quiescence and retained sealed coverage.

For the corrected positives, evidence establishes ten actual setup warmups, three dormant final plans, final barrier/admission, acquire/cancel and zero final sample submissions. Warp’s five phases contain separate reset/kernel commands; CUDA executes its existing three stages covering the same five phase positions.

I independently reran the supplied CPU oracle program after inspecting it, changing its output destination to scratch. The recomputed eight case records exactly matched the author’s numerical case records:

- **614,400 rays** per checked frame.
- Exact hit-mask, instance-ID and material comparisons.
- Rays use `1e-6` absolute/relative tolerance; output floating channels use `1e-5`.
- Maximum observed floating error: **`7.298419912871168e-08`**.
- Existing overflow, stack-depth and traversal-counter invariants passed.

This compares **warmup frame 10, replay index 9**, not all frames or final samples. The full-shadow fixture has one triangle and no occlusion; its 6,393 directional/unoccluded shadow rays do not prove internal-node traversal or occluded-shadow behavior.

Evidence: [custody audit](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/audit_native.log), [numerical recomputation](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/cpu_numeric_recomputed.json), [fault/source/hygiene audit](C:/Users/AG/新建文件夹/c4_assets/r1_review_v1_scratch/final_audit.log).

The earlier primary archive also matched its stated size and SHA256:

`4ebd2fe77850de49b437a0693b8751273a76da8c5127aa5345bf433f5dd6c302`

I inspected its retained earlier failures and later replay records. I did not relabel the original failed Warp observers or failed parent validator as successful driver executions.

**Coverage gaps and residual risks**

The positive matrix provides substantial evidence for this preparation checkpoint, but it does not close F1.

Other limits remain explicit:

- No final-sample rendering, final-output numerical comparison, or integrated worker submission/resolve/reclaim/close was exercised.
- Native numerical comparison covers only the last setup warmup. Wrap-index tests use CPU metadata doubles.
- The 1/120/600-frame acquire regression exercised the inherited arena fixture; it is not a native R1 campaign at those sizes.
- The canonical R1 fixture does not exercise shadow occlusion, multiple internal BVH levels or richer geometry.
- I did not rerun the entire repository, the legacy nine-case GPU suite, or the full parent-death/transport-loss/watchdog matrix.
- Broader C4 structural obligations and C2f acceptance remain outstanding. They are not waived, and completing whole C4 is not a condition imposed on this bounded review.

**Process hygiene and candidate preservation**

Reviewer CLI PID: **45132**, parent **45968**, creation time `2026-10-10 14:04:57`.

The first local pytest process was **44212**, parent **43368**; it exited and was absent at final inspection. The second pytest execution completed with exit 0, and the final scratch-scoped process query found no residual test process. Its individual PID was not sampled.

Fresh native reproduction:

- Driver **1211878**, parent **1211877**
- Supervisor **1212050**
- Worker **1212062**, observed exit **0**

The driver, worker and supervisor were confirmed absent before review completion. Remote audit/numerical processes **1204425**, **1216211** and final audit **1238132** were also absent at the final check.

All **58** recorded author matrix PIDs were independently confirmed absent before the reviewer GPU round and again afterward. GPU-process inspection found no recorded owned residual. Other GPU processes were left untouched.

All tool sessions reached terminal results. No process signals, GPU resets, history deletion, candidate edits or unresolved owned hangs occurred. Local and remote HEAD/status remained exact and clean.

**Scope and gates**

This report does **not approve the R1 candidate**, pending F1 repair and fresh independent re-review. Existing R0 and M1 approvals retain their original scope.

Still closed: product PREPARE, measured sample submission, C2f and full C2/C4 acceptance, activation, R2/S/host delivery, external holds/depth two, C4c3/c4, C4d/B3, backend selection, performance results, a3b, result/baseline generation and shipping.