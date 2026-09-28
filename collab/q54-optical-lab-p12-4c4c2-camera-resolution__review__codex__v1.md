# Independent design review: C4c2 asymmetric camera resolution

Initiative: q54-optical-lab-p12-4c4c2-camera-resolution
Stage: review
Author: codex
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: collab/q54-optical-lab-p12-4c4c2-camera-resolution__design-request__codex__v1.md; optical_worker/cases.py; sensing/optical.py; physics/spatial.py; optics/execution.py; benchmarks/optical/workloads.py
Owner Summary: Approve this numerical camera decision and its bounded CPU contract implementation plan. Independent exact arithmetic and real canonical-scene CPU execution on Windows and Linux reproduce the matrix, ray convention and four-hit observations. No runtime, product, result or baseline gate is opened.

## Identity, isolation and inspected snapshot

Reviewer session: `/root/c4_camera_resolution_design_review_v1`, a fresh independent review session receiving a bounded task packet rather than the implementation conversation. I did not author the candidate. This is Codex context/role isolation, not a claim of a different vendor or a separate security boundary.

Actual base / merge-base: `ccfeb82673ddefdc58af1dbe4c90e809a4031c4d`.

Actual HEAD: `848b5a2abf24f15021b1dcab4c859beefce031ed`.

Local repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, branch `implementation/c4-supervised-waits`. Initial and final `git status --porcelain=v1` were empty. Base-to-HEAD diff contains only the new 203-line design request. `git diff --check` passed.

Candidate SHA-256: `811027332afcbfd37b23a77c177b9e46169c3cd6db7668b5bffce8759b85a1b6`.

Linux checkout: `/data3/ang.gao/robot_sim_c4_camera_design_848b5a2` on `ang.gao@192.168.80.101`, detached at the same full HEAD, initially and finally clean. Used the packet's explicitly authorized fallback after its reported primary-host timeout; I did not independently repeat that timeout. The optional host-routing decision file is absent in this snapshot; AGENTS.md itself permits the fallback.

Read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md fully. Read the C4c2a section 8 and C4c2b v3 section 9 follow-on obligations, and relevant C4c v7 camera, stage-evidence and exact-provenance sections. Author observations were hypotheses, not approval evidence. The author's external exploratory script was neither used nor executed. No source edits, installs, pushes, GPU execution, benchmark timing or baseline production occurred.

## Verdict

**Approve.** No substantive findings and no unresolved blocking evidence gap for this design-only scope.

The decision is explicit about being a new numerical resolution, not recovery of a previously frozen numerical camera. It preserves the old case/content identities, gives a unique executable convention and exact scalar encoding, admits only the known canonical record, and explicitly leaves live provenance and runtime integration to their separately reviewed design. The planned acceptance tests are capable of distinguishing the high-risk wrong conventions and malformed payloads.

This is not an implementation approval: the proposed camera module does not exist in this candidate, and its resolver/validator/import-closure/immutability tests still have to be implemented and independently reviewed.

## Independently verified facts

1. **Case and scene preservation.** The current case has width 7, height 3, intrinsics `(5.25, 7.75, 1.3, 0.8)`, translation `(0.31, -0.27, 0.43)`, range 50 and role `rgb`. History at 231a541 contains the symbolic rotation, and the case relocation at 953cbe3 does not resolve it. The candidate changes no case, materializer, replay, fixture, gate or runtime source. The canonical materializer creates the existing 21 triangles; it was not replaced by a hand-written review fixture.

2. **Mathematical convention.** `SpatialTransform.from_rpy` uses `rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)`. `build_pinhole_camera_rays` uses integer meshgrid coordinates, row-major flattening, multiplication by `R.T` for row-stored vectors and normalization after rotation; this equals the proposed column-vector camera-to-world mapping. A separately written generic-axis Fraction composition exactly matches all nine rational entries. Its exact `R.T R` is identity and exact determinant is 1. Each Fraction rounds to the specified binary64 hex value. The binary64 orthogonality maximum error is `2.5680016054740313e-17` on both hosts.

3. **Full-ray observability.** An independent scalar nested loop over all 21 pixels matches the actual production host ray generator with maximum absolute direction error `2.220446049250313e-16`. Origins match all 21 repeated translations; all directions are unique. The full comparison includes boundaries 0/6/7/13/14/20 and misses rather than observing only the four hits.

4. **Real CPU geometry and shading.** The canonical registry, production OpticalSceneCache, CpuDirectLightOpticalExecutor and a second brute-force CpuReferenceOpticalExecutor reproduce hits `[4,11,16,17]` and material ordinals `[5,12,16,18]`. These are three image rows, columns 4/2/3, and 17 misses. Hit positions, normals, RGB and intensity are finite. Miss ranges are positive infinity, positions/normals NaN, material ordinals zero, and default-background RGB/intensity zero. CPU BVH and brute-force hit/range outputs agree. The minimum hit barycentric coordinate is `0.07028714183448068` on Windows and `0.07028714183448075` on Linux, not an edge/tie construction.

5. **Quantization and negative controls.** Binary32 quantization of all camera scalar inputs via stdlib struct preserves the four hit/material identities when fed through the host ray generator. Transposition or column-major reinterpretation yields hit `[14]`, material `[4]`; identity yields hits `[7,8,10,14,15,17]`; reversed composition yields `[2,10,15]`. Each negative fails full ray equality as well. This establishes input-quantization robustness, not GPU arithmetic parity.

6. **Native ABI feasibility.** Inspected WarpCameraParameters, `_render_f32`, its from_spec mapping, the native pinhole kernel and bound-command scalar validation in optics/warp_native_render.py. Current scalar normalization uses stdlib binary32 packing, row-major rotation fields and explicit `R00` through `R22` ABI arguments. The design fits those capabilities. Native raygen's arithmetic/normalization sequence differs from CPU host generation; the candidate correctly does not promise bit identity and leaves real-device parity mandatory.

7. **Schema and trust boundary.** The payload has an exact finite field set, explicit encodings, primitive integer/string/list values and no executable authority. Independent construction with the existing canonical encoder produces a unique digest; the existing IPC decoder accepts that canonical representation and rejects duplicate JSON keys. The new API consumes an exact dict, so raw-JSON duplicate rejection is not silently assigned to dict validation. Unknown case/rotation and any changed canonical value must reject rather than fall back. The current stdlib-safe cases module and package boundary permit this implementation without importing a renderer. The design's required exact types matter: plain Python equality alone would not reject bool/int/float substitutions.

8. **Integration boundary.** The existing gate has no resolved-camera record. Worker preparation remains disabled, and no source or gate delta alters that. Existing Warp render evidence's camera digest describes normalized camera/query parameters; it is not automatically this new resolved-camera digest. Section 6 requires the later versioned integration to bind both logical resolved identity and actual camera/ABI facts to row, prepared plan, C3 stage evidence and receipt. Its pre-enqueue wrong-matrix/correct-digest, wrong-row and altered-ABI negative requirements prevent treating this CPU record as execution authority. C2a section 8 and C2b v3 section 9 remain operative.

Independent reproduced identities:

| Identity | SHA-256 |
| --- | --- |
| Existing case | `3f6ce0ac1d59d0c0d0e5ea57f497c308a9d5c4b2a5c19462c900627fd99d0181` |
| Existing scene | `ab3b06cd48e0315ce5d33c0f81a880e71fe5df7369c805a2ae85f5c0197b9e35` |
| Existing fixed fixture | `4fcece950264cb2ebb78cf1ec06ad52f5a151b3d2cf2970314f1cae5a2aa7907` |
| Proposed resolved-camera canonical payload | `04ab7bc2571b8396f06d36b7baea66224868001894c4ce7a5ea12e4faecda1ef` |

The last digest is an independent design calculation, not a newly issued runtime receipt. It should agree with the implementation's frozen test fixture and reviewed evidence.

## Executed verification and reproducibility

Inspection commands included `git status --porcelain=v1`, `git rev-parse HEAD`, `git branch --show-current`, `git merge-base <base> HEAD`, `git diff --stat/--name-status/--check <base> HEAD`, Get-FileHash, targeted `rg`/Get-Content, and `git show` of the historical case definition/relocation. Initial mistaken root REVIEW_PRINCIPLES and several guessed source paths were discovery misses, corrected through the actual collab path and repository search; they were not test failures. The optional routing file was confirmed absent.

Independent scratch probe, created with apply_patch outside the repository:

`C:/Users/AG/新建文件夹/c4-camera-independent-review-probe-v1.py`

Final probe SHA-256 on both hosts:
`96d31339699800a8d077ec8574d7017ad6c73e477c1368a87e53178f0d0586ce`.

Commands run successfully:

```text
C:/Python314/python.exe -B C:/Users/AG/新建文件夹/c4-camera-independent-review-probe-v1.py C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design

C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_direct_light_executor.py tests/unit/optics/test_scene_contract.py tests/unit/optics/test_native_contract_extraction.py

scp -q -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 C:/Users/AG/新建文件夹/c4-camera-independent-review-probe-v1.py ang.gao@192.168.80.101:/tmp/c4-camera-independent-review-probe-v1.py

ssh -o BatchMode=yes -o ConnectTimeout=8 -i C:/Users/AG/.ssh/tileops_ed25519 ang.gao@192.168.80.101 "/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim python -B /tmp/c4-camera-independent-review-probe-v1.py /data3/ang.gao/robot_sim_c4_camera_design_848b5a2"
```

Local environment: Python 3.14.0, NumPy 2.5.2, pytest 9.1.1. Probe exit 0; targeted suite **53 passed in 5.09 seconds**. This includes existing cold import/real packer/fixture/replay checks, shared-contract regressions and frozen product/gate-source assertions.

Linux environment: conda robot_sim, Python 3.12.13, NumPy 2.5.1. Independent probe exit 0, same exact identities and expected controls. The original probe ran on both hosts, then the reviewer added actual IPC canonical-decode/duplicate-key assertions and reran the final version on both hosts; the final hash above identifies those runs. All numerical observations reported above are from new executions, not author outputs or cached evidence.

The only review writes were the external scratch probe, its `/tmp` copy, and this external report. `-B`/`sys.dont_write_bytecode` and disabled pytest cache were used; both tracked/untracked repository status checks remained clean. No material deletion occurred.

## Limitations, follow-through and exact approval scope

Approved: the v1 numerical decision at the stated candidate digest, and implementation of the single stdlib-only resolved-camera definition/resolver/validator plus the bounded CPU tests and durable-input cleanup-ledger entry specified by it. Only asymmetric_synthetic_7x3_v1 is resolved. No claim is made that four hits cover all lighting, shadows, geometry degeneracy or product workloads.

Still required at implementation review: every field/type/malformed-value mutation, immutable construction and payload-copy isolation, source-case/rotation drift rejection, no default camera, actual cold-import attempts, independent numeric/hash fixtures, complete content regressions and Linux CPU acceptance. This design review demonstrates feasibility; it does not pretend those nonexistent APIs have already passed tests.

Still closed: runtime adoption and C4c2 runtime integration; C4c3, C4c4, C4d, B3, product worker GPU preparation/submission, host-payload sealing, shipping, a3b, native results/baselines and backend selection. No gate/row/threshold/content identity rewrite, result publication or retroactive result validity is authorized. C4 remains incomplete.

Before any runtime implementation, a separate complete integration design must satisfy C2a section 8, C2b v3 section 9 and this design's section 6, including materializer ownership/import closure, remaining camera resolutions, versioned provenance/ABI binding, readiness/warmup/seal ordering, live exact-object authority, timeline/lease composition, supervised-failure custody, same-slice real GPU fault matrices and cleanup. Real GPU parity, native execution and all those lifecycle/security properties were deliberately not exercised here. No broad Go2 asset or GPU regressions were needed for a one-document camera-resolution design.
