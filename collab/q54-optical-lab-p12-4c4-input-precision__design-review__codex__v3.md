# Independent design review: source geometry and CPU query policy v3

Initiative: Q54 / P12.4 / C4 numerical input and intersection semantics
Stage: independent bounded design review
Author: codex
Version: v3-review
Date: 2026-10-10
Status: in_review
Related Files: input-precision design-request v3; design-request/design-review v2; full-cpu-frame diagnosis/gap-review v1; runtime-integration design-request/review v3; original scene, packing, fixture, host-BVH, Warp and CUDA query/update sources
Owner Summary: Request changes for one concrete R0 materialization defect. The proposed representation, policy migration and existing resource integration are otherwise coherent, but the mandated new host transform evaluation order changes original CPU world-degeneracy decisions on the frozen asset. Preserve the original CPU world materialization for the common R0 fixture; do not leave this known counterexample to a future acceptance run.

Verdict: **Request changes**. One substantive blocking finding, F1, is below. This is a design review only, with no implementation, numerical acceptance, performance or whole-C4 approval.

Reviewer session: `/root/unified_geometry_design_v3`, freshly delegated with the review packet and repository access, without the implementation conversation. I did not author or modify the candidate, inspect the excluded input-precision WIP, or spawn another reviewer. The coordinator supplied the scope/authority and asked that the independently discovered transform counterexample be judged against the actual design promises; it supplied no numerical result used as evidence.

## Reviewed identity and authority

- Candidate repository: `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design`.
- Branch: `implementation/c4-supervised-waits`.
- Base and observed merge-base: `d88c6b64dfc58c99424270b546ef96a9c6ebdf91`.
- Frozen HEAD: `2ea95ea4aa3d96c6f43af40527de3afd50cda6fb`.
- Candidate: `collab/q54-optical-lab-p12-4c4-input-precision__design-request__codex__v3.md`.
- Independently verified candidate SHA256: `e6bf584e226cce829f652bb14df44fff5f07dba472f33e5f7396e067a231f4c5`.
- Initial and final candidate working trees were clean; HEAD and digest remained unchanged.
- Original runtime: local `C:/Users/AG/新建文件夹/c4_intersection_precision` and remote `/data2/shared/ang.gao/robot_sim_intersection_precision_wip`, both observed clean at `812ca29030a45d802addf10ca5811a4354ce0eb9`.
- Candidate source under `optics`, `optical_worker`, `benchmarks` and `tests` has no delta from that original runtime. The base-to-HEAD delta is the v3 proposal, prior complete evidence review and PROGRESS, with no implementation change.

I read AGENTS.md, collab/README.md and collab/REVIEW_PRINCIPLES.md, the complete proposal, prior v2 request/review and full-CPU diagnosis/gap-review, plus relevant inherited runtime-integration requirements. Prior approvals and author diagnoses were context, not proof of this design's adequacy. Original assets, oracle, cameras, logical workload, tolerances, timing boundaries and independent native ownership remain authoritative. No later gate is opened by this review.

## F1 — P1: the required R0 host evaluation order changes the authoritative CPU world predicate

Location: proposal section 2, especially lines 74–81 (the new ordered three-term transform and requirement that the fixed host materializer use the same order), together with lines 84–97 (world-area rejection). Actual original sources: `optics/scene.py::transform_points`, `_build_instance_snapshot`, and `optics/execution.py::_intersect_triangle_scalar`; the existing replacement target is `optical_worker/native_input.py::_derive_fixed_fixture`.

The proposal correctly removes the lossy local filter and preserves original binary64 vertices and transforms. However, it additionally requires the common R0 CPU materializer to replace the original CPU matrix evaluation with `(a0*b0 + a1*b1) + a2*b2`, then translation, without contraction. Original `transform_points` uses `points @ R.T + r` on the original mesh vertices. These are not numerically interchangeable at the hard world-area cutoff, even on the frozen Go2 source.

A fresh CPU probe imported the original registry and obtained each instance's transform from the actual original `OpticalSceneCache` snapshot. It called the unchanged `transform_points` on each original mesh vertex array before face indexing, then compared that result with the exact scalar operation order prescribed by section 2. It found:

- 398,432 original source faces.
- 396,778 faces admitted by original CPU world-area geometry.
- 396,782 faces admitted after the prescribed ordered transform.
- 20 differing world-area decisions: eight CPU-admitted faces become rejected and twelve CPU-rejected faces become admitted.
- 108,567 different world-coordinate components, maximum absolute difference `5.551115123125783e-17`.

Concrete source key `47244640296` has original CPU cross-product norm `1.0000000000163711e-12`; the prescribed ordered world geometry has norm `9.999999999991179e-13`. A ray from the original face centroid plus `0.01` times its original unit normal, directed along the negative normal with physical limit 1, is accepted by the unchanged original `_intersect_triangle_scalar` on original world vertices and rejected by that same helper on the prescribed world vertices. The saved evidence contains exact original/proposed coordinates and rays. Source key `47244640273` exhibits the converse: original norm `9.999999999917472e-13`, ordered norm `1.000000000030742e-12`, original no-hit and ordered hit.

Consequence: retaining every source row no longer guarantees preserving the original CPU's world-degeneracy decision. A correctly implemented v3 R0 materializer would already create this known discrete acceptance difference. This is not a claim that the 20 faces explain a frozen full-frame pixel error, nor a demand for universal bitwise identity of all CPU/device arithmetic. It is a concrete, avoidable change to the common **CPU-prepared R0 fixture**, at exactly the discontinuous world-area boundary that this amendment declares authoritative. The existing preparation boundary permits both native lanes to receive the original CPU-derived binary64 world geometry independently; R0 does not need the new native evaluation order.

Why existing evidence/checks do not close it: the supplied minimum-precision census uses original matrix-derived world vertices and compares float32 alternatives. It does not evaluate the proposal's new host arithmetic. Same-lane agreement, all-source enumeration and outward bounds also pass for the changed geometry. Section 7 promises threshold and full-image checks, but discovering this known prescribed mismatch later cannot itself authorize changing the frozen host arithmetic, CPU oracle or cutoff.

Minimal correction: explicitly preserve the unchanged original CPU composition/vertex-transform semantics and result for the sole common R0 fixed materializer, with binary64 world edges derived from those vertices. Remove the requirement that this host materializer use the new native scalar evaluation order. Independently upload that same immutable fixture to both lanes and retain the proposed outward float32 bounds, all-source rows and world-query tests. This is not CPU rendering fallback and does not justify another normal array or policy service. Keep the R1 source float32 inputs unchanged; native scene-update evaluation remains a separate, explicit numerical implementation whose adequacy must be checked under the original matrix. Do not infer permission to relax R1 tests or to modify the oracle from this R0 correction. If a different resolution is chosen, freeze its precise arithmetic and authority in the revised design before implementation.

The revised design must also identify the authoritative source route for this correction. Current `CanonicalFixedRenderFixtureHost` derives from `PackedOpticalSceneHost`; the packed per-face `(N,9)` array no longer carries the original unique mesh vertex/index layout. Merely writing “call transform_points” separately on each packed triangle is not proof of reproducing the original mesh-wide computation. Preserve or reconstruct the needed source structure through the canonical materialization owner with explicit layout/provenance and matching tests; do not introduce caller-injected world arrays, a second drifting geometry authority, or an assumed equivalence of different matrix shapes. The coordinator highlighted this integration constraint after the independent counterexample was reported; inspection of the actual packed input and fixture signature supports it. This report does not approve any particular revised indexed-geometry layout that has not yet been frozen and reviewed.

Required proof after amendment: reproduce all 20 recorded keys through the actual revised R0 materializer, comparing world vertices/edges and world-area admission with unchanged original CPU materialization; include both directions of the crossing and the retained degenerate rows. Check the two explicit ray outcomes above, source-key/instance/material correspondence, outward bound containment and independent identical fixture inputs. Retain original-pixel, R1, compatibility and full-channel native checks already required by section 7. A new design revision needs fresh independent review before implementation of the amended scope.

## Other high-risk areas inspected

No additional blocking design finding was established in these areas:

- **Source population and mapping:** the actual packer owns original binary64 local geometry before its `1e-8` filter/cast. Removing that filter, keeping every original face and retaining `(instance_index << 32) | original_face_index` while densely regenerating packed ordinals is implementable. The proposal correctly distinguishes stable keys from new topology/global rows and removes the superseded local-normal array rather than retaining two normal sources.
- **Query precision and policy:** original CPU triangle helpers use world cross-norm and determinant cutoffs of `1e-12`, and `_update_reference_hit` uses binary64 distances with `1e-9`. Current native helpers return float32 distances and compare at `1e-5`. The proposed fixed constants, double candidate/best/stack distances, physical-limit-before-rounding and final-position formation address actual call-flow defects. Fresh execution of the unchanged CPU comparator picks the nearer key 20 over farther key 10 for distances 1 and 1.00000002 in both visit orders. Full CPU comparisons still determine adequacy; merely matching a comparator does not prove complete traversal equivalence.
- **Bounds and retained degenerates:** the existing CPU median builder accepts zero-extent boxes and arbitrary retained source rows. A fresh external model of the exact proposed outward rule contained every proposed binary64 Go2 world vertex. Every leaf/internal union remained conservative, all primitive IDs were represented once, and keys were unique. The unchanged leaf-size-4/sort builder produced 262,107 nodes, depth 17. Empty input produced no nodes; five coincident point boxes produced one valid leaf. Signed values, zero and subnormal rounding directions were also checked. This establishes CPU construction feasibility, not future GPU slab arithmetic or native refit/LBVH correctness.
- **R0/R1 and compatibility coverage:** the actual fixed materializer, both Warp triangle update forms, CUDA scene source, `device.py` host workload and all six named executors contain affected casts/filters. The proposal names their migration together, preserves R1 replay float32 producer values and layouts, and requires derived edges from transformed vertices. Brute-force can derive query normals from its newly preserved world vertices without v2's extra argument. Existing ABI order and ownership remain applicable, with geometry/ray dtypes changed.
- **Binding and ownership:** real Warp launch validation currently chooses `ctypes.c_float`; source admission currently names `c_float` but not `c_double`. Array specs, CUDA pointer/dtype checks, graph descriptors and actual readback also encode float32. The inherited v2 camera/scalar contract plus v3 owner/consumer update requirements cover these real dependencies; metadata-only relabeling cannot satisfy them. The narrow admission change, typed byte checks, immediate adoption and partial-failure custody tests remain implementation obligations. Existing lifecycle, quarantine, waits, leases and disposal are retained.
- **Identity and compilation:** scene/fixture/native-input, recorded Warp ABI, CUDA scene/direct-light recipes, gate/result, row semantics and C3 binding are explicitly advanced. Camera v2 and 17-f64/7-f32 binding rules remain inherited, including asymmetric resolution-ID rejection. Scene-update contraction is explicitly disabled in addition to rendering; unchanged LBVH bounds/topology ABI does not itself require a new dtype/layout. Derived scene, fixture, replay-parent, camera-sequence and topology hashes must be regenerated and inspected, without relabeling old execution. Original output/delivery formats and activation=false remain fixed.
- **Shading and continuation:** retaining float32 shading/output and current shadow-origin construction is a bounded proposal, not proven adequacy. The inspected shading-isolation script faithfully describes only selectively rounded original CPU first-hit outputs at prior RGB-failure pixels; it does not emulate all native shadow math. Section 7 correctly retains mandatory original CPU full-channel comparisons, including previously missing native intensity. No extra output buffer, blanket float64 shading, new lifecycle framework or precision option is justified by the supplied evidence.

## Fresh verification, artifacts and limitations

Read-only inspection used `rg`, bounded UTF-8 reads, Git revision/status/diff/merge-base commands and SHA256 checks. All six supplemental script/result digests in proposal section 8 matched the local files. Their script semantics were inspected where relevant; their prior outputs were not represented as fresh executions. The large raw full-frame archives and full rendering matrix were not re-executed in this design session; their prior reports were read with that limitation.

CPU probe environment: `tileops-dev`, `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B`, NumPy 2.5.1, unchanged original runtime and existing Go2 asset directory. No GPU context, native JIT/build, install or implementation edit was requested or launched.

Final evidence command:

```text
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B /data2/shared/ang.gao/unified_geometry_design_probe_20261010_v3.py /data2/shared/ang.gao/unified_geometry_design_probe_20261010_v3_coordinates.json
```

The script/result are preserved both in local `C:/Users/AG/新建文件夹/c4_assets/` and remote `/data2/shared/ang.gao/`. Final SHA256 values are recorded in the final verification addendum below. The external probe models prescribed arithmetic; it is not an edited production materializer or a native implementation acceptance test.

The first attempt passed a string to an API requiring `Path` and terminated before producing evidence; this external probe error was corrected. An initial successful model used batched face-array matrix multiplication. The probe was then strengthened to call actual original `transform_points` on mesh vertices and snapshot transforms, with identical census results, and finally saved full counterexample coordinates/rays. No candidate or original evidence file was altered. A shell quoting error in one `pgrep` invocation was corrected with a simpler exact script-name search. Candidate-only `git diff --check` passed; the full delta reports only eight intentional Markdown two-space line endings in the verbatim prior evidence review, not a design defect.

No Warp/CUDA execution, real-device typed ABI proof, full original-image rerender, R1 replay matrix, compatibility native matrix, allocation-failure campaign, hardware trace or performance measurement was performed. These remain mandatory implementation/native checkpoints. CPU bounds feasibility does not establish native nextafter/bit-operation behavior, real-device conservative traversal, memory feasibility, or exact hit-facing/shading behavior after final normal rounding.

## Scope and disposition

The candidate remains frozen and unapproved for v3 implementation pending F1. Preserve this complete report and its counterexample artifacts. The existing v2 approval does not authorize silently implementing the new geometry/policy amendment or its correction.

Still closed: numerical/full-optical acceptance, result/performance/baseline authority, backend selection, shipping/default activation, R2/host/S/physics publication and whole-C4 completion. Final graph/arena/barrier integration, remaining lifecycle matrices, C4c3/c4, C4d/B3, systematic whole-C4 refactor and fresh independent whole review remain outstanding. Recording this report verbatim cannot change the verdict or reopen those gates.

## Final verification addendum

- Final probe script SHA256: `214ac4959b23891048b5d56a1dca078f520cf49ac180f3065203097b2042ffbb`.
- Full coordinate/ray result SHA256: `4729c1eec40147590b2e626a5cfe979abd18fea9a6abddd0d7564c2a27e47140`.
- Final CPU probe PID `3752635`, program-recorded start/end Unix times `1791587531.4135506` / `1791587545.5663588`, terminal exit 0. Earlier successful probe PIDs `3748423` and `3750086` also returned exit 0; the preliminary failed API-type attempt returned terminal exit 1.
- A final remote `pgrep -af unified_geometry_design_probe_20261010_v3.py` returned no match (exit 1); explicit checks of earlier recorded PIDs also found no process. There were no probe children, pending owned sessions or observed hangs. No signal or process cleanup was performed. Short inspection/transfer process IDs and the failed first probe PID were not individually sampled.
- Final candidate HEAD remained `2ea95ea4aa3d96c6f43af40527de3afd50cda6fb`, with clean status and unchanged candidate digest. Original local/remote source stayed at the clean frozen runtime revision. Files created or updated by this review were only its uniquely named external scratch scripts/results/log and this report.
- The report's own SHA256 is supplied separately to avoid a self-referential digest.
