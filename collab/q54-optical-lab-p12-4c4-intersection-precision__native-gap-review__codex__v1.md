# Independent native evidence and numerical-gap review v1

Initiative: Q54 / P12.4 / C4 intersection precision
Stage: bounded independent native-evidence review and numerical gap assessment
Author: codex
Version: v1
Date: 2026-10-10
Status: accepted
Related Files: AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md; runtime-integration design/review v3 section9.2; camera-resolution design v1; intersection-precision code-review v3 and release-review v4; intersection-precision native-evidence author v1
Owner Summary: Approve the four source-frozen positive native executions and their bounded artifact/lifecycle evidence. Numerical correctness remains Request changes. The remaining failures are both ray/primitive selection sensitivity and precision already lost in source packing/fixture derivation. Independent CPU replay additionally proves that rounding the authoritative camera's origin and ray to float32 changes the selected surface at782417. Standardizing the current float32 rays or widening only intersection arithmetic is not a sufficient repair. A scoped, independently reviewed numerical-input representation amendment must precede implementation; preserve the original oracle, assets, workload, tolerances and closed gates.

Reviewer session: `/root/intersection_native_precision_gap_v1`, fresh independent Codex session supplied a bounded packet and repository access, without the implementation conversation. No candidate authorship/edits, subagents, GPU launches, native builds, installs, cleanup or signals. This is independent role/context isolation, not a separate vendor/security boundary.

## Verdict and exact scope

**Approve with comments** for the four original positive Warp/CUDA Go2 executions, compiled artifact custody, ten warmups followed by same-frame geometry/RGB copies, and observed terminal disposal/exit. No new blocking native-evidence integrity finding was established in this bounded scope.

**Request changes for numerical correctness**, with F1 and F2 below unresolved. This is not optical acceptance. Agreement between lanes does not substitute for the existing CPU oracle. The report approves neither a proposed replacement representation nor a particular kernel rewrite.

- Original numerical-repair base: `8cce978b73901eeac42a9e2fdc7e861a23a57890`.
- Approved numerical source: `a42dbe0154e7c97dcd23a1de2514636551079d74`.
- Frozen native/release HEAD: `812ca29030a45d802addf10ca5811a4354ce0eb9`; tree `e6894e90698e41c052b6f3805bc595074c4d78a6`.
- Candidate: `C:/Users/AG/新建文件夹/c4_intersection_precision`; remote `tileops-dev:/data2/shared/ang.gao/robot_sim_intersection_precision_wip`. Initial/final HEAD and clean status independently checked. The a42-to-812 delta is the three-file catalog/policy/evidence change, with no numerical-source change; diff whitespace check passed.
- Main evidence checkout: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, independently observed clean `6ed9730985a78016bef146bbef53bc51b50198d4`.
- Accepted runtime design v3 independently hashes to `a76bbe8963064876895ba56eebe72221d941af97f0c84722189b505d6536067`. Section9.2 requires actual CPU rays/geometry/IDs/material/RGB/intensity with unchanged C1 tolerances and explicitly refuses to infer GPU arithmetic from binary32 input agreement.

I read the shared instructions and review principles in full, the relevant approved runtime/camera requirements and review scopes, author evidence, prior numerical/release/native reports, actual materializer/packer, native raygen/intersection/normal paths, readback implementation, archive/run scripts and replay validators. Earlier reports and author assertions were treated as claims and scope constraints.

During finalization the author advanced main to clean `6ad973b568308d6a931c4517cd757a1c356702f8`, adding PROGRESS and a diagnosis note; I checked its HEAD/status/file summary. The original native note was inspected at6ed9730, and this report's production/native conclusions remain bound to812ca29. The later two-face representation matrix was reported by the author but not independently replayed here and supplies no additional approval claim.

## Original evidence, compilation and lifecycle

Fresh local and remote SHA-256 calculations agree:

| Archive | SHA-256 | Independent member-to-original verification |
| --- | --- | --- |
| `c4-intersection-geometry-native-evidence-v2.tar.gz` | `02ed978f890352fc5a52c88f3915db6605295ffc12b6c347de40045ccf987698` | 193 regular members, including18 Warp cache files |
| `c4-intersection-cuda-build-evidence-v1.tar.gz` | `c90fc21deb24556f1021c32be5f74dbc9550e765e5d5bf6c2aaf29195a948c34` | 51 regular members |
| `c4-intersection-cpu-diagnostics-v1.tar.gz` | `d61c2382ffb15cd065499a7d560e5e65b6163096c172cba557ef7747c559dda5` | 25 regular members |

No duplicate regular names or members other than regular files/directories. Every regular member above was hashed against its original remote file. Native archive inventories cover all four raw case directories. Each Warp cached source/cubin/meta file also matches the size/digest in its actual module-binding record, not just a companion claim. Both generated renderer sources contain the widened arithmetic and `wp::cast` lowering; their bound sm90 cubins are present. No new JIT/native execution was launched by this reviewer, and no disassembly-level equivalence claim is made.

Fresh production `verify_deployment` passes against the current candidate's source files, all three original CUDA binaries, recipes/build inputs/logs/terminal records. Manifest SHA is `8497e8410d17addb6e6e4739465bc6e636afc775ca181541c039023c2b1fab6d`. Its source revision correctly remains **a42dbe0**, whereas workers execute release **812ca29**. The verification call takes expected platform from the validated manifest, so it is an offline source/artifact consistency replay, not an independent new live ABI probe. Original real native execution, pinned runtime file verification and custody supply the bounded runtime evidence.

| Original output | Lane/resolution | Worker / supervisor | Normal completions / arms | Readback guard |
| --- | --- | --- | --- | --- |
| `c4-intersection-geometry-author-v2` | Warp1920x1080 | 3639235 /3639232 | 25 /26 | setup24 |
| `v3` | CUDA1920x1080 | 3643200 /3643199 | 21 /22 | setup20 |
| `v4` | Warp960x640 | 3650069 /3650063 | 25 /26 | setup24 |
| `v5` | CUDA960x640 | 3654592 /3654590 | 21 /22 | setup20 |

For **each** case, fresh cold `capture_runtime_custody_inputs` reproduces the original payload; `RuntimeEnvironmentContract.verify_files`, `validate_custody_artifact`, `validate_runtime_custody_selection`, `RuntimeCustodyRecord.from_payload`, `validate_runtime_custody_attachment`, `_validate_fixed_storage`, `_validate_render_graph` and `_validate_render_warmups` pass on original bytes. Raw supervisor artifact equals the separately saved artifact; batch/results/identity/parent records agree. The same source catalog43b90be4 and lane-specific source snapshots are retained. Real target1/ambient0, direct_light_full, shadows on and primary_unprofiled remain the run's scope.

Actual source `tests/support/native_runtime_process.py:1112` reads original `session.fixed_render_storage[3]` after the tenth completed warmup, in order:

`rgb, origins, directions, hit_mask, range_m, position_world, normal_world, numeric_instance_id, material_index`.

Warp performs the actual `.numpy()` copies; CUDA uses `.detach().cpu()` before host NumPy conversion. The original guarded host custody retains each returned owner before the next copy. All nine ordered D2H descriptors occur under the same readback guard, before its one normal DISARMED. Each warmup's last descriptor precedes DISARMED and the following frame's first descriptor. Fresh replay rejects removed copies, reordered copies and removed readback DISARMED in every case:12 negative replay controls. These are serialized-consistency controls within the trusted execution model, not proof against arbitrary native-process compromise.

All original workers/supervisors report exit0, no stderr/failure/ordinary-atexit record, no deadline expiry or kill request. Each terminal path performs the original one emergency synchronization followed by completed release and `aborted_synced_exit`. This is sacrificial terminal disposal, not reusable normal shutdown or final readiness. Positive execution does not exercise integrated fault/hang behavior. Per-copy hardware profiler traces are not supplied; source, actual custody/operation records and resulting buffers are the available readback evidence.

## Independent full-array and CPU observations

All raw channel hashes/shapes/dtypes were independently read and compared, with fixed `atol=rtol=1e-5`, rays `1e-6`, and exact hit/instance/material IDs. Equal miss sentinels are handled separately. The old `1e-4` RGB filter is not used as an acceptance threshold.

| Captured scope | 1920x1080 | 960x640 |
| --- | --- | --- |
| Hit/instance/material differences | 0 /0 /0 | 0 /0 /0 |
| Different normal pixels | 1:782417 | 0 |
| Maximum normal difference | 0.0016724737361073494 | 0 |
| Maximum RGB difference | 0.0006107091903686523 | 2.384185791015625e-7 |
| RGB /normal pixels failing cross-lane tolerance | 1 /1, both782417 | 0 /0 |
| Maximum direction difference | 2.384185791015625e-7 | 2.384185791015625e-7 |
| Maximum range difference | 2.09808349609375e-5 | 1.52587890625e-5 |
| Maximum position difference | 1.9073486328125e-5 | 1.1444091796875e-5 |
| Miss count in each lane | 132022 | 80946 |

Range/position maxima above1e-5 do not by themselves fail the combined absolute/relative rule: zero captured pixels fail that rule in either pair. There are971723 /282717 direction vectors with some unequal component, but none fails the ray tolerance. Origins are identical. Every hit channel is finite. Every miss has positive infinite range, NaN position/normal and zero instance/material, matching the actual reset source. This does not establish cross-generation sentinel behavior. Warp1920 origin/direction bytes are identical to the original pre-repair run.

The old maximum cross-lane RGB error1.2056253775954247 is removed; all original six RGB outliers and two normal-only pixels were retained in examination. Neither that improvement nor complete960 cross-lane agreement proves CPU correctness. Intensity is not one of these nine copied channels.

Fresh original CPU scripts reproduce both lanes' eight-pixel `cpu_sample_diagnostic.json` and every NPZ array exactly. Fresh fixture replays used all eight pixels per lane; the original archived fixture diagnosis contains only residual pixel782417, whose complete row matches exactly. Thus this review adds seven fixture rows per lane rather than pretending the original archive contained them. The frozen fixture hash reproduces as `8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0`. Warp's original arithmetic JSON also reproduces exactly.

The initial author “canonical double ray” is generated by the CPU camera routine from **already quantized native float32 scalar payloads**. It must not be described as the authoritative binary64 camera. I independently reconstructed the actual frozen frame9 binary64 camera and replayed the subsequently supplied original CPU executor script using the frozen binary64 camera and lighting. Its JSON exactly reproduces. Against this authoritative eight-pixel CPU result:

| Lane | Normal failures | RGB failures | Maximum RGB error |
| --- | --- | --- | --- |
| Warp | 782417,901306 | 901306 | 4.112737104212949e-5 |
| CUDA | 782417,901306 | 782417,901306 | 0.0006238465172372365 |

The eight selected hit/instance/material IDs, ranges and positions pass their respective rules. This is a direct counterexample to correctness, not a full-image CPU acceptance run.

## F1 — P1: ray representation still changes the selected surface

Locations: `optics/warp_native_render.py:614`, `optics/_cuda_direct_light_source.py:261`, the scalar binding in `optical_worker/render_parameters.py`, and the fixed ray-storage/launch ABI. Trigger:1920 frame10 pixel782417 near an original mesh triangle boundary.

Exact actual GPU rays converted losslessly to binary64 and intersected with the frozen encoded fixture select primitive53234 for Warp and28312 for CUDA. Both have instance6/material5, so exact instance/material checks cannot detect the change. Their stored normals and shaded colors reproduce the cross-lane discrepancy. This localizes the residual to different ray inputs/surface selection; it does not implicate a new remaining double-intersection error or prove absence of other errors.

More decisively, authoritative CPU camera origin and direction at782417 are:

```text
origin    [1.3614383505833265,-0.8202649149268576,0.797271845249641]
direction [-0.8217804379707481,0.5006385842330298,-0.27209909912705116]
```

The original CPU executor on original geometry selects the normal of53234. Rounding those same origin/direction once to float32, converting losslessly back to double and executing the same CPU scene selects the normal of28312; maximum RGB change is about0.00061033. My separate two-triangle fixture check independently produces the same primitive switch. That check is limited to those two candidates; the full CPU executor replay is the stronger original-scene observation. Both direction and origin were rounded in this experiment, so it does not separately identify which rounding is necessary/sufficient.

Consequences: ray tolerance alone permits a discontinuous normal/RGB change that violates the separately required optical tolerance. Making the lanes agree on a single **rounded float32** canonical ray would not preserve this existing CPU result. Changing `rsqrtf` alone, increasing intersection precision after the ray has been rounded, widening an epsilon, smoothing normals or accepting the pixel as “boundary noise” is not established as a valid repair. Existing lifecycle validators intentionally do not enforce optical agreement, and ordinary well-conditioned unit scenes miss this counterexample.

Required direction/proof: specify a common camera-to-intersection numerical path that preserves enough authoritative camera/origin/direction information through the boundary predicate. A reviewed higher-precision internal ray representation, or a comparably justified precision-preserving calculation using authoritative inputs, is needed for consideration. Float32 presentation/output need not automatically imply float32 intersection input. Freeze evaluation/rounding/contraction semantics where consequential and prove the original782417 case plus unchanged canonical-camera/IDs/normal/RGB rules on the required scene matrix. No native algorithm is approved by this report.

## F2 — P1: packed geometry loses normal information before GPU arithmetic

Locations: `optics/scene_contract.py:243` and265 (local vertices rounded to float32), `optical_worker/native_input.py:1213` (float32 world transform, subtraction/cross/normalization), and both triangle helpers' direct use of the stored fixture normal (`warp_native_render.py:931`; `_cuda_direct_light_source.py:397`). Trigger: pixel901306, primitive346916; both lanes choose the same primitive and return exactly the stored normal.

Stored normal `[0.9007425904273987,0.17109498381614685,-0.3992357552051544]` differs from original CPU `[0.900718743935694,0.17145981755928835,-0.39913315483111816]` by0.00036483374314150163. Recomputing the normal in double from encoded edges still differs0.00036476589415207483. Computing double edges/normal from existing packed local vertices reduces the maximum to7.027663032155296e-5, still outside the fixed rule.

My independent source-versus-packed 2x2 check separates vertex and transform effects at this primitive:

| Vertex source /rotation | Maximum normal error versus original CPU |
| --- | --- |
| Original local vertices /original rotation | 7.69e-14 |
| Original local vertices /packed float32 rotation, widened | 4.69e-9 |
| Packed float32 vertices, widened /original rotation | 7.02803e-5 |
| Packed float32 vertices and rotation, widened | 7.02766e-5 |

Rounding the accurately source-derived final normal to float32 has error9.02e-9 here. Therefore this failure is not evidence that all normals, transforms, outputs or every arithmetic operation need float64. It is evidence that widening after independently rounding nearby local vertices cannot restore lost edge information; subsequent world-coordinate subtraction adds another loss.

I also inspected and freshly replayed the later all-triangle representation exploration, with exact JSON equality. It covers all376139 admitted source triangles of this static imported scene. The maximum final float32 normal errors versus source-derived CPU world normals are0.0037789534950759818 for widened packed vertices,3.980613605267247e-5 for source edges rounded to float32 then double transform, and6.130744889532025e-8 for source unit normals rounded to float32 then double transform. Counts above the exploratory absolute1e-5 filter are49212,21,0 respectively. That filter is diagnostic, not a new acceptance rule. This calculation is static CPU arithmetic, not device, dynamic-transform, intersection, lighting/shadow or rendered-image proof.

Required direction/proof: preserve well-conditioned primitive quantities derived from the original geometric source before destructive coordinate rounding. Source-derived unit normals in float32 are a supported candidate for the normal channel; source-derived edges alone are not enough to claim normal accuracy across even this fixture. The design must jointly specify position/edge data, normals, transforms, bounds and original primitive/source ordering. Repairing normals only does not address F1 or prove geometry/shadow correctness. Do not add one-pixel branches, alternate CPU fallbacks or unbounded diagnostic infrastructure to the renderer.

## Smallest coherent next decision and identity consequences

Proceed with **one scoped numerical-source/representation design amendment**, rather than another pixel-specific intersection patch. Keep the original CPU geometric/camera definition as the oracle and the frozen asset revision/content, case intent, camera trajectory/resolution, pixel convention, materials/lights, profiles/shadows, sample order, physical limits, tie/source-order rules, workload rows and tolerances. Describe exactly what precision is retained, where each rounding occurs, which arrays are authoritative, and how the two native lanes consume one semantic definition without independent drifting materializers.

The design must explicitly resolve the present contradiction: frozen packed scene/fixture bytes are currently designated logical-content identities, yet the existing unquantized CPU oracle exposes information those bytes discarded. The current camera design also explicitly chooses binary64-to-binary32 native scalar conversion and float32 native outputs. Correctness cannot be obtained by silently treating new precision as already admitted under these identities.

Necessary identity accounting depends on the selected implementation, but includes:

1. If derived geometry/normals/bounds change, version/review the materialization and R0 fixture contract and every resulting digest; preserve original artifacts. New content must not masquerade under8354993f or other old fixture hashes.
2. If packed source quantities are added/replaced, review packed-scene schema/semantics/dtype/manifests, source-contract digest and mappings to original primitive IDs. Determine explicitly whether packed bytes remain “logical source” or a versioned numerical representation of unchanged original assets. Do not simply regenerate hashes and call a different workload equivalent.
3. If authoritative binary64 camera values reach native computation or internal ray dtypes/layout change, amend native-scalar/ray ABI and preparation, resource sizes/alignment, graph bindings, readback descriptors, custody/source catalogs, CUDA artifact recipes and acceptance schemas as actually affected. Keep the authoritative binary64 camera values/trajectory unchanged; any digest changed solely because it embeds a new scene identity needs an explained migration, not a recomputed camera from different rounded bounds.
4. R0 and R1 must share the chosen source semantics. New precision cannot be an R0-only hidden bypass while dynamic preparation/refit/shadow paths retain an inconsistent source. Retain ownership/readiness/arena/lifetime checks for any changed allocation or ABI, and account fairly for the actual preparation/work boundary in both lanes.

These are review obligations, not permission to implement every listed change or add a new infrastructure layer. The evidence supports targeted preservation of source quantities and internal rays; it does **not** prove float64-everywhere necessary. Select the smallest representation whose original CPU counterexamples and required matrix demonstrably pass. If the current ABI/fixture identity is immutable under existing authority, the correctness gate remains closed pending the scoped design decision; do not weaken the oracle/tolerance or refreeze a more convenient workload.

After design approval, use the existing canonical materializer/binding paths, replace superseded logic coherently and add causal regressions for rounding-before-subtraction, boundary rays, physical cutoff/tie and conservative bounds. Require independent source/code review, fresh artifact/native execution, full retained failures, complete required CPU comparison and all affected lifecycle/fault/profile/shadow/R0/R1 coverage. Eight pixels and one static normal census cannot stand in for that matrix.

## Commands, limitations and process hygiene

Remote CPU environment: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B`, pinned Python3.12.13, frozen assets `/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2`. Local reads/hashes: PowerShell, Git and `C:/Python314/python.exe -B`. No dependency or source mutation.

Fresh scripts/results retained outside candidates in local `c4_assets` and remote `/data2/shared/ang.gao`:

The remote invocation prefix was `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B`; exact executed script/argument suffixes were:

```text
/data2/shared/ang.gao/intersection_native_gap_probe_v1.py /data2/shared/ang.gao/intersection_native_gap_probe_v1.json
/data2/shared/ang.gao/intersection_native_artifact_probe_v1.py
/data2/shared/ang.gao/intersection_native_cpu_replay_v1.py
/data2/shared/ang.gao/intersection_native_cpu_replay_v1.py /data2/shared/ang.gao/intersection-review-cpu-v1-zb68box_
/data2/shared/ang.gao/probe_source_representation_v1.py /data2/shared/ang.gao/robot_sim_intersection_precision_wip /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 /data2/shared/ang.gao/intersection_review_source_representation_v1.json
/data2/shared/ang.gao/check_authoritative_camera_pixels_v1.py /data2/shared/ang.gao/robot_sim_intersection_precision_wip /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 /data2/shared/ang.gao/intersection_review_authoritative_camera_v1.json
/data2/shared/ang.gao/intersection_native_gap_final_v1.py
```

- `intersection_native_gap_probe_v1.py` /JSON: all native/CPU archive members, source/environment/custody replays,12 negative controls, all arrays; tool17282 exit0, PID3668696/start244862113.
- `intersection_native_artifact_probe_v1.py` /JSON:51 build members, actual deployment verification and generated Warp source; final PID3673939 exit0.
- `intersection_native_cpu_replay_v1.py` /JSON: original CPU/fixture/arithmetic replays and independent vertex/rotation/camera experiments. Exclusive scratch `/data2/shared/ang.gao/intersection-review-cpu-v1-zb68box_`; final PID3674016/start244881777, tool55026 exit0. CPU scripts were executed synchronously via runpy in this process, not launched as detached jobs.
- Original `probe_source_representation_v1.py` with frozen repo/assets and output `intersection_review_source_representation_v1.json`, tool63442 exit0; script SHA `569cdb6e8929d54ebe41adcd3166e2457899477efa6fa8b39f57377291b9e7a4`.
- Original `check_authoritative_camera_pixels_v1.py` with frozen repo/assets and output `intersection_review_authoritative_camera_v1.json`, tool76021 exit0; script SHA `a3e71da94bc61971569301ed9ae794199a88ac4491efc6c228adc5be788647b0`.
- `intersection_native_gap_final_v1.py` /JSON: exact supplemental replay equality, fixed-tolerance eight-pixel canonical CPU comparisons, miss sentinels and scoped process check; PID3679272 exit0. This result records final script digests.

The original supplemental source/camera scripts and their outputs arrived during review; they were inspected and independently rerun, not accepted from the coordinator's conclusion. They are outside the three original archives and must be preserved distinctly as supplemental CPU diagnoses. No new full-frame CPU render, native GPU reproduction, profile/shadow matrix, dynamicR1, intensity check, fault matrix, performance measurement or device arithmetic emulation was executed.

Tooling corrections are not candidate failures: initial wrong-path/glob reads and several broad outputs were corrected with bounded UTF-8 reads. Artifact scratch initially expected an incorrect generated helper-name token (exit1); actual generated source was inspected and the token assertion corrected. CPU harness PID3670858/start244870518 exited1 because it initially compared its new eight-row fixture output with an original one-row diagnostic; existing fresh outputs were retained and the exact common-row comparison corrected before continuation, without rerunning/claiming a different native job. The final sentinel scratch initially assumed-1 IDs and exited1; actual frozen source writes zero, and the source-correct zero/inf/NaN assertions pass. None changed the candidate, raw evidence, tolerance or numerical result.

At `2026-10-10T06:24:14.500070+08:00`, all nine original campaign PIDs, archive reviewer3668696, CPU reviewer3670858/3674016 and artifact reviewer3673939 were absent. Scoped command-line inspection found no remaining native runner or any numerical reviewer/supplemental script. The final checker and short SSH/scp commands ended terminally. Cold-preflight helpers use synchronous subprocess calls; individual short child PIDs and the two supplemental CPU script PIDs were not separately sampled. No GPU worker, supervisor, service or compiler was launched by this reviewer. GPU query showed only pre-existing ambient0 PID2238410 `nvidia-cuda-mps-server`, UUID `GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274`; no target GPU process. No signals or broad cleanup were issued. This is scoped present-time hygiene, not a retrospective proof of all historical start identities.

All full-C4 integration/fault/profile/R1/refactor/whole-review obligations remain pending. No final readiness, reusable shutdown, host delivery/R2/S, activation/default, result/baseline, backend selection or performance gate opens. Archive this report verbatim through the shared evidence-only exception; that archival act cannot broaden its approval or close F1/F2.
