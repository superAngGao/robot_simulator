# Independent design re-review: source-normal and camera input precision v2

Initiative: Q54 / P12.4 / C4 numerical input representation
Stage: independent bounded design re-review
Author: codex
Version: v2-review
Date: 2026-10-10
Status: accepted
Related Files: input-precision design-request v2 and design-review v1; runtime-integration design-request/review v3; camera-resolution design v1; intersection-precision native-gap-review v1; input-precision diagnosis v1; optics/scene_contract.py; optical_worker/native_input.py; optical_worker/camera_contract.py; optical_worker/render_parameters.py; optics/device.py; optics/warp_execution.py; optics/warp_native_render.py; optics/device_scene.py; optics/_cuda_scene_source.py; optics/cuda_direct_light.py; optics/cuda_extension_recipes.py; benchmarks/optical/schema.py; optical_worker/native_source_policy.py
Owner Summary: Approve this exact bounded design amendment. Both prior design findings are closed by explicit migration contracts supported by actual source inspection and fresh CPU feasibility probes. This permits implementation only. The existing numerical correctness findings, native acceptance and all later C4/product gates remain open work or closed authority as before.

## Verdict and reviewed identity

**Approve**, for the exact v2 design and bounded implementation scope below. No substantive blocking or non-blocking design finding remains in this re-review. This is not approval of an implemented normal/ray ABI, numerical correctness, or full C4.

Reviewer session: `/root/input_precision_design_v2`, a fresh independent Codex review session receiving the bounded task packet and repository access without the implementation conversation. I did not author or edit the candidate and spawned no agents. This is role/context isolation, not a different vendor or security boundary. Author packet assertions and prior reviews were checked against the source; they were not substituted for this review.

- Repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base and independently observed merge-base: `13f2e885de3500a67e9649b374b78ea3c68d77b9`.
- Frozen HEAD: `db062cd603464b9b3088739a2aca68acb2a44769`.
- Candidate: `collab/q54-optical-lab-p12-4c4-input-precision__design-request__codex__v2.md`.
- Independently verified candidate SHA-256: `76397d6d7068a5b8d211fc8e6f1b788014f02104818e08a3f65321c8a87b6a9c`.
- Base-to-HEAD delta: candidate v2, 283 added lines, and complete prior design-review v1, 119 added lines. No source implementation or tests change.
- Initial and final tracked/untracked status: clean; HEAD and candidate digest unchanged. Base-to-HEAD `git diff --check` passed.

I read AGENTS.md, collab/README.md and REVIEW_PRINCIPLES.md in full, the full v2 proposal and prior review, camera-resolution design, input-precision diagnosis and native-gap review. I inspected runtime-integration v3's camera/materialization/schema/9.2 requirements and its independent review scope, and the actual affected source/call paths. Some broad output was truncated or initially displayed with the wrong encoding; relevant sections were reread as bounded UTF-8 output. Prior accepted designs retain their original authority except for the explicitly amended representation rules.

## Prior F1: closed at design level

V2 section 4, lines 191–212, explicitly migrates the asymmetric resolved-camera schema to `q54.p12.4.resolved-camera.v2`, its native encoding to `binary64_native_no_rounding_v1`, and its sequence resolution identifier to `asymmetric_resolved_camera_v2`. It preserves every numeric/convention field and exact rational-derived binary64 rotation. It permits exactly these additional changes in the sequence migration, instead of requiring only a scene-identity change. Current resolver/validator accepts only v2; old and mixed camera/resolution/bound-parameter combinations fail before launch. Section 5 requires independent digest and numeric-preservation evidence.

Actual `camera_contract.py` currently freezes v1 and `binary64_to_binary32_nearest_even_v1`; `render_parameters.py::_resolution_ids` names `asymmetric_resolved_camera_v1`, and materialization invokes the production resolver. Thus the revised disposition addresses the real upstream inconsistency, not merely a bound-record label. Existing strict canonical validation can implement the migration without another camera resolver or altering its geometry.

The fresh CPU probe imported the actual current resolver. Its digest is `04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef`. Applying only the two specified resolved-camera payload changes yields independently calculated SHA-256 `3cee982411e6a7c02b81dda2929ead53ea5895ed00513b8e67d3d91dcf1d46ef`; every other field is identical. This is an external calculation of the proposed payload, not a new production resolver or proof that old/mixed records already reject. The implementation must freeze/reproduce this and all dependent sequence/row identities and execute the required negative controls.

## Prior F2: closed at design level

V2 section 3, lines 147–177, names retained compatibility executors and explicitly adds the already-existing `HostOpticalPrimitiveWorkload.triangle_normals_world` immediately after `triangles` in `_brute_force_first_hit_kernel` and its launch tuple. It specifies float32 contiguous `(N,3)`, matching row counts, finite nonzero validation, native owner retention in result resources, unchanged builder filtering/order, source world-normal semantics, hit-facing flip, removal of the returned rounded-vertex reconstructed normal, and retained geometry degeneracy/predicate/order behavior. Rays follow the binary64-query rule.

Actual `device.py::build_host_optical_primitive_workload` transforms original binary64 vertices and computes/normalizes their cross product before rounding the final normal. `warp_execution.py::GpuBruteForceOpticalExecutor.execute` currently ignores that field, uploads rounded triangles, and retains the uploaded input resources in its result. `warp_native_render.py` currently reconstructs the returned normal from rounded triangle vertices. The proposed argument position is unambiguous (zero-based index 9 in this current kernel ABI), and the source quantity is already available from the sole builder.

A fresh CPU production-builder probe used a degenerate face followed by two opposite-winding cancellation-prone faces. The builder retained original primitive keys `[1,2]`, matching numeric IDs `[17,17]`, and exact source-derived float32 normals. The first source normal is `[0,-0.9170700311660767,0.39872610569000244]`; reconstructing from the builder's rounded vertices gives `[0,-0.9175556302070618,0.3976074457168579]`, maximum component difference `0.0011186599731445312`. Empty role-filtered output is correctly `(0,3)` float32. This establishes both dependency availability and a causal regression fixture that distinguishes the two consumers. It does not execute the future kernel or prove native row/flip behavior.

The revised trust boundary is appropriate: ordered construction and actual upload/launch tests establish correspondence within the legacy path; caller-visible arrays do not acquire captured-session C4 custody authority. Section 5 expressly requires row-swap-sensitive numerical tests and actual compatibility entry coverage. Native validation, owner lifetime and hit-facing behavior remain implementation acceptance work.

## Whole bounded amendment and affected invariants

- The sole packer has original binary64 local vertices and the existing degeneracy filter before rounding. Adding one local unit normal does not require new enumeration/filtering, a second packer or changed source assets. V2 specifies empty inputs, finite/nonzero rejection, immutability, manifests and typed custody. Geometry/bounds/topology remain subject to before/after equality and original CPU acceptance.
- R0 `_derive_fixed_fixture` has the existing composed float32 rotation; both Warp scene-update variants and CUDA's triangle update have the corresponding transform/data path. Widening that composed rotation and stored local normal for transform/normalization is feasible. Translation stays out. R0's independent rounded-world-edge degeneracy check stays. Existing per-path vertex/edge formulas are preserved, not asserted bit-identical across backends. R1 remains rigid-only and requires real dynamic correctness evidence.
- The actual scalar order has 17 camera/query values followed by 7 shading values. V2 defines exact separate f64/f32 byte encodings and current-entry rejection of old records. Resident input rays are already available in binary64. Array allocation/dtype/layout, camera conversion, helper annotations, scalar decoding and readback must all migrate; widening inside the helper after a float32 ABI would still be wrong and is expressly prohibited.
- Actual Warp launch validation currently selects `ctypes.c_float` for Python floats and compares `<f` bytes. Its revised authoritative role-based typing must inspect real double bytes. CUDA's scalar evidence and buffer validators currently freeze float32 too. These are covered by the actual typed-binding requirement, not fulfilled by new JSON hashes. Source admission currently permits `c_float` but not `c_double`; any necessary narrow admission change and updated function/source pins belong to the required source/admission review. No broad reflective permission follows.
- Renderer contraction settings have one existing Warp module and one fixed CUDA direct-light recipe to change. `warp_setup` currently freezes `fuse_fp=True`; V2 explicitly requires a renderer-specific exception while leaving scene/BVH options intact. Source recipes, captured artifacts and compatibility consumers must move together. Real JIT/build/packed-ABI and shading regressions are mandatory; CPU evaluation cannot prove device compiler behavior.
- Common triangle/AABB helpers already widen geometry but currently accept float32 rays/limits. Preserving double query operands through predicates, retaining strict physical cutoffs, returned float32 hit ordering and conservative box-entry quantization is a coherent bounded change. Shadow construction and shading remain float32, widened at helper calls. Any remaining original-oracle failure is still a failure, not implicit authority to change this policy.
- Actual gate/result v13, row-semantics v8 and C3 binding v6 exist. V2 explicitly advances these to v14/v9/v7, replaces the gate's old all-binary32 declaration with the shared 17/7 encoding, and requires the implementation's exact migration/digest table. Scene/fixture/native-input/Warp/CUDA recipe changes are enumerated. Unchanged logical rows and activation=false remain required; old records cannot be relabeled as new executions.
- Additional ray storage is correctly `24*N` bytes per pair. New normal uploads and enlarged ray allocations must enter existing manifests, OOM accounting, typed graph and ownership coverage. Preparation/timing boundaries are symmetric, R1 normal transformation remains measured, and returned owners must be adopted before further fallible work. Existing failure/quarantine/disposal rules are inherited rather than replaced.
- Superseded result-capable paths are explicitly replaced and old incompatible ABIs reject. Historical evidence remains historical; no live dual precision mode or new comparator is introduced. Existing selected-backend cleanup and whole-C4 obligations remain.

## Fresh verification and evidence

Read-only inspection used `rg --files`, `rg -n`, bounded UTF-8 reads, `git status --short`, `git rev-parse HEAD`, `git branch --show-current`, `git merge-base`, `git diff --stat/--numstat/--name-status/--check`, and `Get-FileHash`. Some guessed paths and one PowerShell wildcard search were invalid; actual owners were subsequently located with repository-wide searches. These were inspection-tool errors, not source failures.

Executed external CPU command, from the candidate directory:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4_assets/input_precision_design_review_probe_v2.py
```

Final script SHA-256: `ea6f42ce9730f680933b3e828e705280f4e9f855e3ec65eaf66e97db6b3a2365`.
JSON evidence SHA-256: `ef8b923f0efa094e5350f9dccef8d0d4a96c211680096c88238ba0ac7db4e4ea`.
Environment: Python 3.14.0, NumPy 2.5.2. Final exit 0.

Besides the identity/builder probes above, the driver checked diagnostics ZIP SHA-256 `0a864fe19809dc7c2386593173fc508f6cc3406eaf9519fd9626a5f071b76c99`, six unique members, and exact byte equality of every member to the supplied local script/result. I inspected the source-normal census script and its original result scope. That census was independently repeated by the previous reviewer; it was not re-executed here and is not claimed as fresh v2 evidence.

The driver freshly ran the inspected original `probe_proposed_camera_order_v1.py` in-process with `runpy`, using frozen sequence artifacts from this candidate and a new exclusive output `input_precision_review_camera_order_v2.json`. Both resolutions at sample indices 0,9,119,120 cover 10,752,000 rays. Maximum component error versus actual original CPU ray construction was `3.3306690738754696e-16`; zero rows fail the existing ray component tolerance. At 1920 pixel 782417/sample 9, proposed and original CPU direction match exactly. New result SHA-256 is `2076a58fd1fc8a2b2396789e539db92a1b5ba0dbb8c4a0f9459c109ccbd87186`, matching the independently retained v1 result after a new execution. Script SHA-256 is `e46443f3f8373632af638364ba77ac5bd045ef16d23ce8eb0815e0704fb3bb8d`.

The first scratch probe exited 1 before producing output: my synthetic triangle used a 1:2 small-edge ratio that happened to survive float32 rounding, so the assertion that it distinguished the representations failed. I changed only external scratch data to a 1:2.3 ratio and reran successfully. This is a probe-construction correction, not a candidate fix or concealed numerical failure. No original archived evidence was edited.

## Process hygiene, limitations and approval boundary

Final CPU driver PID 36244, parent 47340, ran from `2026-10-09T22:40:56.239586+00:00` to `2026-10-09T22:40:59.789141+00:00`, exit 0. These are program-recorded times, not OS creation-time observations. `runpy` launched no child process. Initial failed probe PID was not separately sampled; its tool returned terminal exit 1. A subsequent Win32_Process query for the recorded PID and exact probe Python command-line substring returned no matching process. All inspection shells returned terminally; short shell/Git PIDs were not individually recorded. No unresolved owned process or hang was observed.

No SSH, remote process, GPU job, native build/JIT, installation, signal, cleanup, candidate edit or commit occurred. The author's separate original 1920 CPU frame diagnostic was not touched, inspected for acceptance, or adopted as reviewer evidence. During finalization the coordinator reported that its full-frame comparison completed and found shared instance/material, range/position, normal and RGB failures against the old native candidate, with intensity unavailable in those native readbacks. I did not independently inspect or reproduce this new author evidence. It reinforces the already-explicit numerical adequacy uncertainty: the two original counterexamples are not claimed to exhaust failures, and no claim is made that this proposed representation resolves all geometry or shading failures. No remote/GPU hygiene claim is made; this session launched no such work. Only the external scratch probe, its new JSON/camera result and this report were created. `-B` and `sys.dont_write_bytecode` prevented bytecode artifacts in the candidate.

The CPU probes establish design feasibility only. They do not establish future production rejection of stale/mixed records, actual double launch bytes, device arithmetic/contraction, new native normal row custody, successful build/JIT, source closure, allocation failures/lifetimes, R1 transforms, shadow/profile correctness, full image normal/IDs/material/RGB/intensity agreement, performance or memory feasibility at scale. Local `trimesh` is unavailable and no install was attempted. The original all-triangle and full native matrices were not rerun. Those are mandatory implementation/native/integration checkpoints, not waived by this approval.

Approval authorizes implementing exactly v2's source-local float32 normal and binary64 camera/query representation amendment for the bounded R0/R1 paths and named retained compatibility entries, followed by fresh independent code/native review and original CPU correctness acceptance. Native-gap-review v1's numerical F1/F2 are not closed by this design verdict. If later full comparisons expose another failure, preserve it and diagnose it under the original contract; this design gives no automatic permission for additional precision/schema/tolerance changes.

Still closed: numerical/full-optical acceptance, performance/result/baseline authority, backend selection, shipping/default activation, R2/host/S/physics publication and full-C4 completion. Final graph/arena/barrier integration, lifecycle matrices, C4c3/c4, C4d/B3, systematic whole-C4 refactor and fresh whole review remain required. Recording this report verbatim through the shared evidence-only exception cannot change its scope, verdict or gates. The final report SHA-256 is supplied separately in the reviewer handoff to avoid a self-referential digest.
