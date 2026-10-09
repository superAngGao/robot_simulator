Initiative: Q54 / P12.4 / C4 numerical correctness  
Stage: independent full-CPU evidence and gap review  
Author: codex  
Version: v1  
Date: 2026-10-10  
Status: accepted, evidence-only scope  
Reviewer session: `/root/full_cpu_numerical_gap_v1`  
Verdict: **Approve with comments — the frozen diagnosis and PROGRESS delta accurately establish unresolved numerical failures. This does not approve a revised implementation or revised numerical design.**

Owner Summary: The reported failures reproduce. Source filtering, determinant rejection and nearest-hit ordering differ materially from the original CPU oracle. Preserving those native policies prevents the source-normal/ray-only amendment from being a complete remedy. A coherent amendment should preserve source query geometry and camera precision, apply the CPU’s world-geometry predicates and ordering, and retain existing output/shading precision unless further evidence requires changing it. No new generic infrastructure is justified.

Reviewed identities and isolation:

- Candidate: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`, HEAD `d88c6b64dfc58c99424270b546ef96a9c6ebdf91`.
- Base: `02e5c1c701b7221175a8b6f3a81628ad5eb305a3`.
- Delta: only `PROGRESS.md` and `collab/q54-optical-lab-p12-4c4-full-cpu-frame__diagnosis__codex__v1.md`; 130 insertions, two deletions.
- Candidate working tree clean at initial and final checks; unchanged throughout review.
- Original code: local `c4_intersection_precision` and remote `/data2/shared/ang.gao/robot_sim_intersection_precision_wip`, both clean at `812ca29030a45d802addf10ca5811a4354ce0eb9`.
- Approved input-precision design v2: `db062cd603464b9b3088739a2aca68acb2a44769`; inspected design SHA256 `76397d6d7068a5b8d211fc8e6f1b788014f02104818e08a3f65321c8a87b6a9c`.
- Frozen diagnosis SHA256: `e9c858c0ae59d466608fa461e5a9e919817741ccf042a32b11395695ef64df93`.
- Read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, design-request/review v2, relevant prior native-gap evidence, original CPU/native source and supplied scripts.
- The unfinished `c4_input_precision` worktree was excluded. No implementation, tracked candidate file or CPU oracle was modified. No GPU work or installation was launched.

**Evidence integrity and oracle fidelity**

Both supplied archives independently match their specified SHA256:

- CPU archive: `f6698bee3e11b7e0a2bb24b7abc28a6e74531b88b369bd10ae103051a6e92936`.
- Native archive: `02ed978f890352fc5a52c88f3915db6605295ffc12b6c347de40045ccf987698`.

A fresh local archive audit found 5,263 regular CPU-archive files and verified all 5,250 chunk hashes against their archived manifests. Archived runner/comparator scripts match their supplied local copies. The native archive contains 193 regular files; all 36 raw readback-channel hashes across v2–v5 match their archived metadata.

Fresh remote asset preflight verified all 25 files against the frozen asset contract:

- Asset revision: `da76818e269b82289eba39808e2fb91d679d6994`.
- Aggregate SHA256: `479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`.

The runner uses the original `CpuDirectLightOpticalExecutor`, original source scene, frozen frame-10 camera/lighting and shadows. Source inspection confirms that the renderer evaluates each ray independently; chunk boundaries do not change intersection or shading state. Restoring each subset’s original direction array correctly avoids the sensor constructor’s second normalization.

Fresh independent checks established:

- Complete contiguous coverage of 2,073,600 and 614,400 pixels.
- Every saved origin/direction equals the corresponding freshly generated original CPU camera value exactly.
- Packed scene and fixed-fixture identities reproduce the frozen workload identities.
- Original CPU shaded execution reproduces every saved compared channel exactly for 48 selected 1920 pixels and 24 selected 960 pixels, including all instance/range/material failures, representative normal/RGB failures, the first/last pixels and the 511/512 boundary.
- Re-executing those selections in seven-ray subsets yields identical results.
- CPU misses retain infinite range and NaN position/normal. Native intensity remains unavailable and uncompared.

Fresh full saved-array comparisons reproduce the diagnosis:

| Resolution | Lane | Instance | Material | Range / position | Normal | RGB |
|---|---|---:|---:|---:|---:|---:|
| 1920×1080 | Warp | 36 | 1 | 3 / 3 | 2,375 | 507 |
| 1920×1080 | CUDA | 36 | 1 | 3 / 3 | 2,375 | 507 |
| 960×640 | Warp | 13 | 2 | 4 / 4 | 576 | 144 |
| 960×640 | CUDA | 13 | 2 | 4 / 4 | 576 | 143 |

Hit masks match. Origins/directions pass the existing ray tolerance. These comparisons use exact discrete channels, `atol=rtol=1e-6` for rays and `atol=rtol=1e-5` for remaining compared floating channels. No tolerance changed.

**Numerical findings**

There is no blocking factual defect in the frozen documentation delta. The following P1 numerical findings remain blockers to numerical acceptance and contradict treating v2’s preserved native policies as a complete repair.

1. **P1 — Native source filtering removes CPU-visible nearest faces.**

   Locations: `optics/scene_contract.py:241`, `optics/scene.py:286`, and fixed materialization in `optical_worker/native_input.py:1219`.

   The original CPU snapshot admits 396,778 triangles; native packing retains 376,139. Independent materialization confirms 20,639 omitted CPU source faces.

   At 1920 pixel `999527`, CPU nearest key `21474879918` has range `1.3957116418519675`; native returns approximately `1.4999752044677734`. At `1016744`, CPU nearest key `21474891830` has range `1.4260190626571585`; native returns approximately `1.4611616134643555`. Both nearest source faces are absent from native packing.

   The native local-area filter is `1e-8`; the CPU world-area filter is `1e-12`. Camera or normal preservation cannot recover a face that was never packed. Cross-lane agreement misses this shared omission.

   Repair direction: preserve the source-face population needed by the original CPU world predicate, with stable source keys and explicit world-degeneracy handling. Required proof includes these original pixels, source membership and transformed/degenerate cases. Merely reducing a constant while retaining all other representations is not established as sufficient.

2. **P1 — Native determinant rejection discards retained CPU-visible nearest faces.**

   Locations: `optics/execution.py:645`, `optics/warp_native_render.py:916`, `optics/_cuda_direct_light_source.py:369`.

   The frozen note correctly leaves pixel `970674` unresolved at its recorded evidence boundary. Fresh independent work narrows it further: nearest retained key `21474880202` has CPU range `1.419132939189466`, while native returns approximately `1.4682166576385498`.

   Its encoded-geometry determinant magnitude is:

   - Original camera ray: `9.869380553664255e-9`.
   - Actual saved Warp ray: `9.869380491265986e-9`.

   Both are below native `1e-8` and above CPU `1e-12`. Fresh host evaluation of the actual Warp helper body rejects this face.

   The same examination finds determinant rejection for the nearest retained faces of all four 960 range failures:

   | Pixel | CPU nearest key | Encoded determinant with actual Warp ray |
   |---:|---:|---:|
   | 232793 | 98784258485 | `8.433005199338972e-9` |
   | 269225 | 128849027151 | `7.405999012876171e-9` |
   | 299129 | 12884902983 | `6.417966109513807e-9` |
   | 326962 | 21474894645 | `6.664753902517183e-9` |

   Actual Warp helper-body host evaluations reject all four nearest faces. At `232793`, this accompanies an instance change from CPU instance 24 to native instance 1 and a range change from approximately 1.87249 to 3.10729 metres.

   This establishes a concrete predicate conflict. It does not constitute GPU traversal replay or prove that no additional native cause exists.

   Repair direction: align the actual intersection predicate with the original CPU contract, including world-area validity. Recheck physical/shadow cutoffs and conservative BVH pruning separately rather than changing a shared epsilon indiscriminately.

3. **P1 — Native tie handling can select a farther, different instance.**

   Locations: `optics/execution.py:866`, `optics/warp_native_render.py:777`, native triangle return conversion at approximately line 933, and corresponding CUDA helpers.

   CPU ordering uses binary64 hit distances and `1e-9`; native helpers round distances to float32 before the `1e-5` comparator.

   Independent nearest-hit reconstruction at pixel `891790` finds:

   - Key `21474842648`, instance 6: `1.6056378750401445`.
   - Lower key `17179871039`, instance 5: `1.6056450124999637`.

   Native returns instance 5. The actual Warp comparator’s host body accepts the farther lower-key candidate over the closer candidate and refuses the reverse replacement.

   All 36 supplied 1920 instance-failure candidate pairs meet the documented different-instance/lower-key condition with separation between `1e-9` and `1e-5`. The full supplied 39-pixel geometry isolation was freshly rerun and reproduced its original JSON byte-for-byte, including original/encoded geometry and both ray variants. This supports the diagnosis without claiming a native traversal replay of every choice.

   Repair direction: preserve sufficient precision through internal nearest-hit comparison and pruning, adopting the original CPU ordering rule. Lowering the epsilon while keeping float32 internal distances cannot generally reproduce CPU ordering when distinct distances round to the same float32 value. Final range output can remain float32 subject to existing output tolerances.

4. **P1 — Geometry precision and the location of degeneracy filtering must be resolved together.**

   Independent census of the unchanged registry finds:

   - 398,432 source faces.
   - 396,778 CPU world-admitted faces.
   - 12 faces whose binary64 local-area and world-area admission decisions differ at `1e-12`.
   - Eight CPU world-admitted faces cross to at-or-below `1e-12` after local float32 rounding.
   - Six cross that threshold after the current float32 world calculation.

   These are **threshold crossings**, not evidence that eight or six faces become exactly zero-area. For example, key `21474881787` has original world area magnitude `1.0000000002032285e-12`; rounded local geometry gives `9.632294961647858e-13`.

   A once-local `1e-12` filter is therefore not an exact substitute for the CPU’s world-area predicate, even on this frozen asset. The census is a representation/admission counterexample; it does not establish that those particular near-threshold faces cause the reported full-frame errors.

**Supported bounded repair scope**

The new evidence requires revisiting v2’s requirements to preserve native filtering, determinant thresholds and float32 hit ordering. It does not invalidate its original camera/normal counterexamples, and it does not itself approve another design.

A coherent amendment can:

- Preserve original camera/query precision and source geometry through the world intersection representation, including original transform inputs; widening already-rounded vertices or transforms is insufficient.
- Retain source faces in immutable packing and apply the original CPU world-area rule during world materialization/query. Keeping all source faces is a straightforward way to avoid the demonstrated lossy local filter. The design must define harmless no-hit behavior for world-degenerate faces, conservative finite bounds and stable source-key/instance/material correspondence.
- Use the original CPU determinant and nearest-hit semantics, with sufficiently precise internal hit distances and conservative traversal. Preserve established physical and shadow endpoint rules.
- Derive world geometric normals from the same preserved world geometry and round only the existing normal output. Under that coherent geometry representation, v2’s extra source-local normal array can be superseded rather than maintained as another independent representation.
- Keep current final-output, material/light and shading precision unless concrete remaining failures require a further change. These findings do not justify blanket float64 shading, extra high-precision output buffers, new precision knobs, CPU fallback or per-pixel exceptions.

This direction requires a bounded design amendment covering its actual layouts, affected R0/R1 and retained compatibility paths, source-key mapping, materialization and derived identity migrations. It need not introduce another lifecycle framework, allocator, comparator service or parallel implementation family.

A future implementation must prove actual device behavior and full CPU correctness independently. The current evidence does not show that geometry/policy changes alone eliminate every normal/RGB discrepancy.

**Commands, fresh results and artifacts**

Remote interpreter: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`, invoked with `-B` over `tileops-dev`. Reviewer scripts and outputs are preserved under local `C:/Users/AG/新建文件夹/c4_assets/` and remote `/data2/shared/ang.gao/`, except the local archive audit.

Executed successfully:

```text
python -B independent_fullcpu_archive_20261010_v1.py

/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B \
  /data2/shared/ang.gao/independent_fullcpu_gap_review_20261010_v1.py \
  /data2/shared/ang.gao/independent_fullcpu_gap_review_20261010_v1.json

/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B \
  /data2/shared/ang.gao/independent_fullcpu_supplement_20261010_v1.py \
  /data2/shared/ang.gao/independent_fullcpu_supplement_20261010_v1.json

/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B \
  /data2/shared/ang.gao/probe_full_frame_geometry_failures_v2.py \
  /data2/shared/ang.gao/robot_sim_intersection_precision_wip \
  /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 \
  go2_static_1920x1080_v1 \
  /data2/shared/ang.gao/independent_fullcpu_geometry_replay_20261010_v1.json

/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B \
  /data2/shared/ang.gao/independent_fullcpu_assets_20261010_v1.py
```

Reviewer output SHA256:

- Full comparison/reproduction JSON: `ca25c7d65516ca1fc262ebc0b08b7bf9437d1f1165ce6bbb15f01d29cc56797f`.
- Supplemental census/helper JSON: `017cbb6850be1a155efff8b50d7bda1c01fd436d3628648c4e8b232210d36991`.
- Archive-audit JSON: `307437dc18b292ed57c814f925278e54f12164066b40b4bf9f54fe6a1832d917`.
- Fresh geometry replay: `1391ac726bdd89124192b139323d30ec35e2a1578f80eaa3f9c074c56c3a918a`, identical to the supplied original result.

Script SHA256:

- Full comparison/reproduction: `57460d167b50ae2ada556cb643c3267b61ba47ffd34f6e11c4455cf864330838`.
- Supplemental census/helpers: `ee4cd8235995efddc5cf11e36f7f6b15a17869bea7883f1459c0791824ae82b4`.
- Archive audit: `5ed29c2234075690a3c1a57d4dbe0f5712e4f834b2b67aa357f46281ceaf5817`.
- Asset preflight: `8e988cc98bbce16266fd5ebacb37ca2d1a2a64b53b11675a2c344b3c50961450`.

All substantive review executions returned exit 0. Scratch creation and downloaded scratch outputs were the only filesystem changes. Bytecode generation was disabled. The original and candidate tracked trees remained clean.

**Limitations and non-blocking comments**

The full saved arrays were independently checked; all 2,688,000 pixels were not freshly rendered again. Fresh original shaded rendering covered the 72 selected pixels, with original source inspection, exact full-camera reconstruction and complete archive checks supporting the chunked evidence.

No new Warp JIT, CUDA build, GPU execution, hardware trace, R1 matrix, other-frame/profile matrix, lifecycle campaign or performance run was performed. Actual Warp Python-body evaluation is CPU source evidence, not native-device execution.

The coordinator supplied additional minimum-precision and shading-isolation observations during review. They were not adopted as independent approval evidence. The threshold and ordering conclusions above come from this session’s own checks.

Non-blocking disposition for the documentation: retain the frozen v1 record unchanged and record the new determinant/960/census evidence separately. Its statement that `970674` remained undiagnosed accurately describes the evidence available to that frozen record.

**Exact bounded verdict**

**Approve with comments** for recording the frozen diagnosis and PROGRESS delta as evidence of unresolved original-CPU numerical failures and the need for a bounded amendment. **No numerical acceptance, new design approval or implementation approval is granted.**

Whole C4 acceptance, product/result/baseline authority, backend selection, performance claims, final graph/lifecycle integration and all later gates remain closed. The required systematic whole-C4 refactor and fresh independent whole review remain outstanding.
