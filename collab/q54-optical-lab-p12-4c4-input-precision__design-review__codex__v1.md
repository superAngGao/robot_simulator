# Independent design review: source-normal and camera input precision v1

Initiative: Q54 / P12.4 / C4 numerical input representation
Stage: independent bounded design review
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: input-precision design-request/diagnosis v1; runtime-integration design-request/review v3; camera-resolution design v1; intersection-precision native-gap-review v1, code-review v3 and release-review v4; optics/scene_contract.py; optical_worker/native_input.py; optical_worker/camera_contract.py; optical_worker/render_parameters.py; optics/device_scene.py; optics/_cuda_scene_source.py; optics/warp_native_render.py; optics/warp_execution.py; optics/device.py; optics/cuda_direct_light.py; optics/_cuda_direct_light_source.py
Owner Summary: Request changes to close two migration-contract gaps: the asymmetric camera's still-current binary32-native identity, and the disposition of the legacy Warp brute-force normal path. The proposed source-normal/f64-query approach is feasible and supported by independently repeated CPU arithmetic checks. This is not numerical acceptance, and no implementation or later gate is approved.

## Verdict and identity

**Request changes**, with two P2 design findings below. Neither finding requires making all geometry/shading binary64, changing the original oracle, or relaxing a tolerance. This verdict does not assert that the proposed native arithmetic has failed: it has not been implemented or executed in this review.

Reviewer session: `/root/input_precision_design_v1`, a fresh independent Codex session receiving the bounded packet and repository access without the implementation conversation. I did not author or edit the candidate and spawned no subagents. This provides role/context isolation, not a different vendor or security boundary. Later author messages supplied an optional CPU ray-order script and acknowledged the identified migration issues; those messages are not approval evidence. No proposed v2 was reviewed here.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base and independently observed merge-base: `6ff82b182e310335380b0923eca1ab1bad88e983`.
- Frozen HEAD: `13f2e885de3500a67e9649b374b78ea3c68d77b9`.
- Sole candidate delta: `collab/q54-optical-lab-p12-4c4-input-precision__design-request__codex__v1.md`, 224 added lines.
- Candidate SHA-256: `06c92c35d53394d6041fea4d08adf4214441dcd6a273907e78276981baa402df`.
- Initial and final tracked/untracked status: clean. HEAD and candidate hash remained unchanged; base-to-HEAD `git diff --check` passed.
- Separate original native candidate: local `c4_intersection_precision` and remote `tileops-dev:/data2/shared/ang.gao/robot_sim_intersection_precision_wip`, independently observed clean `812ca29030a45d802addf10ca5811a4354ce0eb9`. Remote CPU source-normal replay used that unchanged source, not a new implementation of this design.

I read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md in full (rereading the principles as UTF-8 after PowerShell's initial display encoding was wrong). I inspected the candidate, predecessor camera/runtime requirements, runtime review scope, diagnosis and native-gap findings, numerical/release review scopes, and affected source/call paths. Prior approvals remain approvals only of their stated revisions and scopes.

## F1 — P2: asymmetric camera identity still freezes the superseded native encoding

Locations: candidate section 3 and section 4 lines 146–164; `optical_worker/camera_contract.py:13–17,45–65`; `optical_worker/render_parameters.py:130–134,298–311,354–361`; camera-resolution design v1 sections 2–3.

The amendment correctly replaces bound-render-parameters v1 with v2 and defines the 17 binary64 camera/query arguments. It does not give a migration disposition for the upstream asymmetric resolved-camera record. The actual current production resolver returns `q54.p12.4.resolved-camera.v1`, digest `04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef`, whose hashed payload explicitly declares `native_scalar_encoding = binary64_to_binary32_nearest_even_v1`. Its validator accepts exactly that payload. The resolved sequence names `asymmetric_resolved_camera_v1` and materialization invokes this resolver.

Trigger: regenerate the asymmetric sequence and bind it to the new binary64 native camera as directed by v1. Section 4 says that only the sequence's embedded scene identity changes. Retaining the current camera identity therefore retains a normative binary32-native declaration while executing binary64 camera arguments. Updating the camera record and its resolution ID would instead require an additional identity change not admitted by that literal sequence-migration rule. The design needs to choose and record this migration, rather than leave the implementer to reinterpret the old hashed declaration as nonbinding history.

Why existing checks miss it: the current exact camera validator positively enforces the old encoding and digest; the new bound-record checks could correctly validate binary64 launch bits without detecting this inconsistent upstream declaration. Camera numeric values and CPU ray checks can all pass, so numerical checks do not close this identity issue. My CPU probe imported the actual resolver and confirmed the old payload/digest; this is not inferred solely from documentation.

Required direction: explicitly version the affected asymmetric resolved-camera contract/encoding and resolution identifier (or define a comparably explicit historical-only separation), preserving every original binary64 camera value and old artifact. Adjust the allowed sequence identity changes accordingly. State which current records are accepted/rejected and keep the new exact digests in the implementation migration table. Do not alter the rational rotation, trajectory, dimensions, or pixel convention.

Proof after repair: before/after payloads show unchanged numeric camera fields and an explicit encoding/identity migration; independent digest recomputation; the current sequence references the intended current resolution; old/mixed camera-resolution/bound-record combinations fail before launch; actual camera scalar bytes remain binary64. Review the amended design before implementation.

## F2 — P2: compatibility replacement does not settle the brute-force normal ABI

Locations: candidate section 2, section 3's resident-ray compatibility requirement, and section 4 lines 174–188; `optics/warp_execution.py:39–174`; `optics/warp_native_render.py:638–775`; `optics/device.py:42–103`.

The R0 fixed materializer and device-scene update normal paths have a concrete new source and argument position. The broader compatibility wording promises removal of superseded rounded-vertex normal derivations, no second implementation for old result-capable callers, and compatibility executors using the current kernels. One actual current result-capable entry point remains outside the specified normal migration: `GpuBruteForceOpticalExecutor.execute` uploads only the rounded world triangles to `_brute_force_first_hit_kernel`, which reconstructs and normalizes their cross product. This kernel is in the same renderer module affected by the contraction change, and its ray arguments are among the existing float32 paths to be migrated.

Trigger: migrate compatibility ray allocations/kernel annotations to binary64 and apply the module option, while implementing only the normal inputs enumerated in section 2. The brute-force entry point still loses source-normal information and preserves the very derivation section 4 says to remove. Alternatively, an implementer can add a new normal argument or retire/exclude this entry point, but v1 does not choose its source semantics, ABI position, or scope disposition. This is a migration ambiguity; no new brute-force numerical GPU failure was executed or claimed.

The missing dependency is not unavailable data. The existing sole host workload builder already computes `HostOpticalPrimitiveWorkload.triangle_normals_world` from original world vertices before float32 rounding, preserving its existing filtering/order. The executor currently ignores this field. Thus a small, concrete consumer migration is available without inventing another source-normal materializer. Its world-normal semantics differ from the new packed local-normal/rigid-transform path and should be stated explicitly if this compatibility entry remains supported.

Why existing checks miss it: all frozen R0/R1 native-path checks and the new packed-array census can pass without executing `GpuBruteForceOpticalExecutor`; checking that it imports the same kernel module does not show that it consumes source normals. Old tests expecting its current rounded-vertex behavior would not establish the new representation contract.

Required direction: enumerate the affected compatibility first-hit/render entry points and give this brute-force path an explicit bounded disposition. If retained under the new precision contract, specify use of its existing source-derived world-normal array, the exact added kernel argument/order, upload/retention/validation, unchanged geometry/degeneracy/filter/order behavior and hit-facing flip; remove the redundant normal reconstruction. If intentionally excluded or retired, say exactly which guarantees and callable entries remain, reconciling the broad replacement language without implicitly expanding this C4 slice. Do not add a second packer or fallback implementation.

Proof after repair: source/data-flow and ABI tests trace the actual compatibility entry to its source normals and binary64 rays; a thin/cancellation-prone triangle distinguishes the old reconstructed normal from the supplied source normal; wrong shape/dtype/row ordering is rejected at the applicable boundary; numerical/query/geometry regressions cover retained compatibility paths. The production-native custody trust model must not be falsely attributed to a legacy API that does not possess that authority.

## High-risk areas checked and otherwise coherent

- Source quantity: `_pack_registry_arrays` has the original binary64 triangle vertices and existing binary64 cross-product degeneracy filter before each float32 vertex row. It can derive one local unit-normal row without changing primitive filtering/enumeration, source keys or existing arrays. Empty `(0,3)`, nonfinite/zero rejection, immutability, manifests and owner coverage are expressly required. The source normal is geometric, not a smoothed mesh normal.
- R0/R1 semantics: `_fixed_instance_transform` currently composes float32 world/body/geometry transforms; Warp and CUDA scene kernels also compose their retained transforms. The proposal widens the already-composed nine components for applying/normalizing the stored normal, keeps translation out, preserves orientation/flip, and restricts the operation to the rigid contract. It does not accidentally demand double vertex transforms. Both Warp scene-update variants and CUDA's triangle update need the new argument. R0's independent world-edge degeneracy check remains intact.
- Geometry/bounds: R0 derives world vertices then edges, Warp's AABB update derives/transforms local edges, and CUDA derives edges from transformed points. The proposal preserves those existing per-path formulas rather than pretending they are bit-identical. It requires unchanged CPU geometry/topology/bounds and still requires full R1/oracle correctness. Source-normal accuracy alone does not prove intersection or conservative-bound correctness.
- Scalar/query flow: the existing 24-scalar order does split as 17 camera/query plus 7 shading scalars. Source rays are already binary64 in `OpticalRaySensorSpec`. The current downcasts exist in both resident-ray/camera compatibility executors, `WarpCameraParameters`, CUDA camera construction, buffer layouts, native setup decoding, native source loads and helper annotations. The design expressly forbids these intermediates for the new query inputs.
- Typed custody: actual Warp packed-launch checking currently selects `ctypes.c_float` for Python floats and serializes `<f`; it must select the new type by authoritative argument role and inspect double physical bytes. `validate_resolved_parameters`, CUDA bound-scalar evidence, native setup decoding, buffer layouts/intervals and readback descriptors all require the same split. These are implementation obligations covered by the design's typed-binding rule, not satisfied by changing only JSON hashes. Shadow-off ABI's omitted bias argument and float32 shading outputs stay relevant. Readback must expose f8 ray inputs.
- Intersection boundaries: current common triangle/AABB helpers widen float32 operands internally, but their input annotations/parameters still narrow rays. Primary plane arithmetic also needs the prescribed double evaluation. Retaining double all the way through predicates, explicit widening of shadow inputs, and preserving float32 hit-distance ordering/conservative stack quantization is implementable. Mixed Warp scalar types and float32 position outputs require explicit casts during implementation; real JIT remains mandatory. Original physical-limit/tie/BVH regressions must be rerun.
- Compilation: the repository has one renderer module and fixed CUDA direct-light recipe. Renderer-specific `fuse_fp=False` plus `--fmad=false` can be bound without changing scene/BVH module options or introducing another runtime precision knob. Existing `warp_setup` currently freezes `fuse_fp=True`, so its renderer-specific exception, artifact hashes and option-negative tests must actually change. This affects all renderer kernels and all shading regressions; no speed claim follows.
- Resource/timing/migration: the extra ray storage is correctly 24*N bytes per pair. New source normal storage must also appear in scene manifests/uploads. Allocation adoption, typed graph/ownership checks, failure/quarantine rules and readbacks remain required. Both lanes pay the same ray storage/stage boundary; CPU source-normal preparation stays outside timing and R1 transformation stays inside scene update. Other explicitly named scene/fixture/native-input/render/CUDA recipe versions, old-record rejection, preserved historical artifacts, false activation values and unchanged logical rows are coherent subject to F1/F2.
- Verification scope: section 5 correctly separates representation feasibility from full original CPU comparisons and prohibits treating eight pixels, lane agreement or a static census as optical acceptance. The native-gap report's original F1/F2 remain unresolved numerical acceptance findings until new implementation/native/full required oracle evidence closes them. Earlier numerical-source/release/native-evidence approvals do not do so.

## Executed evidence and commands

Read-only source inspection used `rg --files`, `rg -n`, bounded UTF-8 reads, Git status/revision/branch/merge-base/diff/check, and `Get-FileHash`. Some early guessed paths were absent and some broad outputs were truncated; relevant existing files/ranges were subsequently located and read. Those were inspection-tool corrections, not candidate failures.

New external driver:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/input_precision_design_review_probe_v1.py
```

Driver SHA-256: `4a6692e06bb28989e58892f97add9fe210226f937e15ff862585e64a614a90f7`.

It verified the diagnostics ZIP SHA-256 `0a864fe19809dc7c2386593173fc508f6cc3406eaf9519fd9626a5f071b76c99`, six unique regular file names, and exact equality of all six members to their separately supplied local scripts/results. It imported the actual asymmetric camera resolver and checked its native-encoding declaration. It synchronously launched the inspected original `probe_proposed_camera_order_v1.py` against the frozen design checkout, with output `input_precision_review_camera_order_v1.json`.

The ray-order probe passed, exit 0: both Go2 resolutions at sample indices 0,9,119,120, all 10,752,000 ray rows; maximum component difference `3.3306690738754696e-16`, zero failures of the existing component tolerance. At 1920 pixel782417/sample9, the proposed direction exactly equals the original CPU direction. Result SHA-256: `2076a58fd1fc8a2b2396789e539db92a1b5ba0dbb8c4a0f9459c109ccbd87186`. Script SHA-256: `e46443f3f8373632af638364ba77ac5bd045ef16d23ce8eb0815e0704fb3bb8d`. Separate NumPy ufuncs model the proposed uncontracted double order; they are not native device execution or an image test.

The driver's subsequent local all-triangle source probe exited 1 because this local Python lacks `trimesh` (`mujoco_menagerie_robot_preview.py requires trimesh`). The driver consequently exited 1 after recording its ledger. No dependency was installed and no test/code failure was concealed. Its completed camera result is retained independently of this environment failure.

The same inspected source-normal script was then run through a new CPU-only wrapper on the primary host:

```text
scp C:/Users/AG/新建文件夹/c4_assets/input_precision_design_remote_probe_v1.py tileops-dev:/data2/shared/ang.gao/input_precision_design_remote_probe_v1.py
ssh tileops-dev /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B /data2/shared/ang.gao/input_precision_design_remote_probe_v1.py
```

Wrapper SHA-256: `ac16b348f03d4254046737c53751693316d4f3d14cb91833f009b5e769194eb5`. It verifies remote HEAD/clean status and invokes via `runpy`, in the same process:

```text
/data2/shared/ang.gao/probe_source_representation_v1.py
  /data2/shared/ang.gao/robot_sim_intersection_precision_wip
  /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2
  /data2/shared/ang.gao/input_precision_review_source_normals_v1.json
```

Remote execution completed exit 0. All 376139 admitted triangles reproduced the original result exactly. Maximum normal errors: widened packed vertices `0.0037789534950759818`; source edges rounded once `3.980613605267247e-5`; source local normal rounded once then double transform/normalize `6.130744889532025e-8`. Diagnostic counts above absolute1e-5 are 49212,21,0 respectively. This is a diagnostic threshold, not a new acceptance rule. Primitive346916's source-normal path is included. Downloaded `input_precision_review_source_normals_remote_v1.json` and original `c4-source-representation-probe-v1.json` both hash to `753283734c27e78929c06336b612c56e261d8ce78b90ac0f5bfc0114d0c7492b`.

The original camera-eight-pixel and two-face scripts/results were inspected as archived diagnostic evidence, but not re-executed by this session. I did not repeat the prior raw native/build-archive audit, deployment verification, full image render, R1 normal replay, negative-control lifecycle suite or native numerical tests. Their historical results are not new executions here.

## Process hygiene, limitations and closed gates

Local driver PID7272, parent33496, synchronously waited camera PID14052 (06:33:20.945776–06:33:24.799966 +08:00, exit0) and local source probe PID36216 (06:33:24.800370–06:33:25.252167, exit1). Ledger/logs remain under `c4_assets/input_precision_design_review_probe_v1.json` and `input_precision_review_probe_*.log`. These are recorded launch/end times, not sampled OS creation times. At06:34:52.1927955 +08:00, Win32_Process inspection found none of the recorded PIDs or matching reviewer Python command lines remaining.

Remote wrapper PID3692757, parent3692756, `/proc` start tick244956180, ran06:33:51.769813–06:34:02.486453 +08:00, exit0. It used synchronous Git children and in-process `runpy`; no worker/supervisor/background service was launched. A later `ps -p 3692757 -o pid,ppid,lstart,args` returned only its header/exit1, confirming absence at that check. The contemporaneous GPU process query showed only pre-existing PID2238410 `nvidia-cuda-mps-server`, UUID `GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274`. This is scoped present-time evidence, not a claim about every unrelated process. The author subsequently reported its separate CPU frame job PID3693595; that is not a reviewer child or reviewer acceptance evidence.

No GPU job, native build/JIT, install, signal, process cleanup, GPU reset, candidate edit or commit occurred. Short inspection/scp/ssh commands returned terminally; individual short transport/Git child PIDs were not separately logged. No unresolved owned residual or hang was observed. Only external reviewer scripts, logs, JSON results and this report were created. Original archives/scripts/results were preserved.

This design review does not establish CUDA/Warp compilation, generated f64 arithmetic or contraction behavior, full image normals/IDs/material/RGB/intensity correctness, dynamic transform/shadow/profile coverage, conservative geometry bounds, actual allocation/ABI/admission/lifecycle fault acceptance, memory feasibility at scale, throughput or reusable shutdown. All of those remain implementation/native/integration obligations under the unchanged parent matrix.

No implementation is authorized by this verdict until a revised frozen design resolves F1/F2 and receives independent review. No numerical correctness, full-C4, result/baseline, performance, backend-selection, shipping/default activation, R2/host/S/physics publication, arena/barrier completion, C4c3/c4, C4d/B3, systematic refactor or whole-C4 review gate opens. This complete report may be archived verbatim under the shared evidence-only exception; archival cannot change its verdict or scope.
