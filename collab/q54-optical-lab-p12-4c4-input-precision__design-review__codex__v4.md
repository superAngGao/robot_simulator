Initiative: Q54 / P12.4 / C4 numerical input and intersection semantics  
Stage: fresh independent DESIGN follow-up review  
Author: codex  
Version: v4-review  
Date: 2026-10-10  
Status: accepted for bounded design scope, with comments  
Reviewer session: `01a122f8-290a-7dd1-9606-ae073fdbf9b5` (`/root`)  
Related Files: input-precision design-request v4, design-request/design-review v3, inherited v2 camera design, full-cpu-frame diagnosis/gap-review v1, original runtime sources  
Owner Summary: The revised indexed-source contract closes prior F1 at design level. No substantive design finding remains. Implementation and native correctness remain unproved and require the specified independent checkpoints.

**Verdict: Approve with comments — design only.**

Approval binds the exact candidate and revision below. It permits implementation of this bounded amendment; it does not approve an implementation, numerical results, performance, backend selection, shipping, activation or whole-C4 completion.

**Checked identity and repository state**

| Item | Observed value |
|---|---|
| Candidate repository | `C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design` |
| Branch | `implementation/c4-supervised-waits` |
| Initial and final HEAD | `056b40c3a365e8a6b4960ac2ef1dced4732994de` |
| Base and observed merge-base | `2ea95ea4aa3d96c6f43af40527de3afd50cda6fb` |
| Candidate | `collab/q54-optical-lab-p12-4c4-input-precision__design-request__codex__v4.md` |
| Initial and final SHA256 | `df1035b0213fe0c48110439222c014f1a81af86f50996254d88634e520ecb9df` |
| Base-to-HEAD delta | Three documentation files: v4 proposal, complete v3 review, and PROGRESS; 456 insertions, two deletions |
| Tracked working tree/index | No changes; working-tree and cached diffs empty |
| Whitespace validation | `git diff --check` passed |

Initial full status reported no entries but warned that `.pytest_cache/` was inaccessible. Final tracked-only status was clean. Therefore, complete untracked-directory inspection is not claimed.

Original-runtime Git commands refused the local checkout because of dubious ownership. Without changing configuration, I read its worktree metadata:

- Branch: `implementation/c4-intersection-precision`.
- HEAD: `812ca29030a45d802addf10ca5811a4354ce0eb9`.
- Independently compared 460 tracked files under `optics`, `optical_worker`, `physics`, `benchmarks` and `tests` against that commit’s Git blobs, normalizing checkout CRLF to LF: **zero missing files and zero mismatches**.
- Candidate Git history also showed no delta in those directories between the original runtime commit and candidate HEAD.

This establishes the inspected source identity; it does not substitute for a complete original-checkout Git status.

I fully read AGENTS.md, collab/README.md, collab/REVIEW_PRINCIPLES.md, v4, prior v3 design and complete review, inherited v2 camera requirements, and the full-CPU diagnosis/gap-review v1. I inspected the relevant original CPU, packing, fixture, scene-update, query, compatibility, binding and ownership sources.

**Prior F1 disposition: closed at design level**

Prior F1 concerned an avoidable change in the common CPU-prepared R0 fixture: v3 prescribed scalar transform arithmetic that changed 20 original CPU world-area decisions.

The correction is concrete:

- V4 §2, lines 49–80, preserves complete original mesh vertex blocks, including unused vertices, with geometry offsets and original face indices.
- Lines 99–109 require original `SpatialTransform.__matmul__` composition, one original `transform_points` call per complete mesh block per instance, then face gathering and original per-face NumPy edge/cross/norm operations.
- Lines 112–116 keep this inside the sole derived-only materializer over the immutable packed source owner, with no separately injected world geometry.
- Lines 118–127 separate native R1 arithmetic and explicitly retain its original-CPU acceptance obligation.
- Section 7 requires all 20 crossing keys and both explicit centroid-ray cases through the eventual actual R0 materializer.

Original `scene.py` confirms the reference sequence: compose transforms, transform the complete mesh with `points @ R.T + r`, gather faces, and evaluate each face’s cross-product norm. `physics/spatial.py` supplies the specified original composition. The indexed layout retains the structure the existing per-face packed array loses.

Fresh execution of the unchanged original CPU intersection helper reproduced the saved outcomes for **all 20 crossing records**, with no outcome mismatches:

| Source key | Original CPU area magnitude | V3 ordered area magnitude | Original geometry hit | V3 geometry hit |
|---|---:|---:|---|---|
| `47244640296` | `1.0000000000163711e-12` | `9.999999999991179e-13` | Yes | No |
| `47244640273` | `9.999999999917472e-13` | `1.000000000030742e-12` | No | Yes |

A separate fresh synthetic indexed-layout experiment used the original registry, snapshot, composition, transform, CPU primitive packer and host BVH builder. It covered three shared mesh blocks, six instances, unused vertices, a plane’s empty offset interval, opposite winding and retained degenerate faces:

- 8,202 vertices, including 8,193 unused vertices.
- Offsets `[0, 0, 3, 10, 8202]`.
- 18 retained source faces; 12 CPU-admitted and six degenerate.
- Exact transformed-world equality and identical admitted source-key sequences.
- Conservative outward float32 bounds, complete BVH primitive coverage and valid internal unions.
- Eleven BVH nodes; empty input produced zero nodes.

These checks support the design’s source route independently of the author’s verdict. They are not execution of a v4 production materializer, which does not yet exist.

**Findings and affected invariants**

**No substantive design findings were established.** The following high-risk areas were checked:

- **Singular source authority and mapping.** Complete geometry blocks replace duplicated per-face source vertices. Offsets, indices, transforms and source-key mappings belong to the same source identity. Validation covers geometry-local intervals, instance correspondence and unchecked narrowing. Stable source keys remain distinct from regenerated dense primitive ordinals.
- **R0/R1 separation.** R0 preserves the original CPU materialization route. R1 retains exact existing float32 producer samples, widened before native composition, without claiming recovered precision or CPU-bitwise arithmetic equivalence.
- **Degeneracy and bounds.** All source rows remain represented. World-area rejection governs hits; zero shading normals cannot become an alternative validity authority. Degenerate rows retain valid bounds. Outward float32 bounds and binary64 traversal distances are explicitly specified.
- **Query policy.** Original CPU code confirms the `1e-12` area/determinant policy and `1e-9` nearest-hit comparator. V4 preserves double candidate/best distances through comparison and physical acceptance. A fresh original-comparator check selected nearer key 20 for distances `1` and `1.00000002` in both visit orders.
- **Compatibility and ABI.** The actual CUDA wrapper and declaration currently take the old vertex array first. V4 §6 specifies replacing that position with vertices followed by indices, retaining subsequent order, for CUDA and both Warp update forms. Offsets remain custodied metadata. All six retained executors migrate, and brute-force avoids the superseded extra-normal argument.
- **Camera and typed binding.** The inherited 17-f64/7-f32 scalar contract, actual f8 ray readback, asymmetric-camera migration and old/mixed-record rejection remain required. Existing float32 allocations, CUDA checks, Warp ctypes binding and source-admission restrictions are real implementation dependencies covered by the migration requirements.
- **Identity and custody.** Scene, fixture, native-input, recorded-command, recipe, gate/result and binding migrations are enumerated. Existing exact-owner checks, allocation adoption, partial-failure handling, waits, quarantine, leases and disposal remain applicable. No new lifecycle framework is introduced.
- **Fairness and scope.** Both lanes receive independently owned uploads of the same R0 fixture. R1 update work remains inside its measured boundary. Changed bytes and primitive counts must be accounted for. Shading/output formats, original workload, cameras, assets, tolerances and activation restrictions remain fixed.

**Fresh commands, evidence and results**

Repository checks included:

```text
git rev-parse HEAD
git branch --show-current
git status --porcelain=v1
git status --porcelain=v1 --untracked-files=no
git merge-base 2ea95ea4aa3d96c6f43af40527de3afd50cda6fb HEAD
git diff --stat 2ea95ea4aa3d96c6f43af40527de3afd50cda6fb HEAD
git diff --check 2ea95ea4aa3d96c6f43af40527de3afd50cda6fb HEAD
git diff --exit-code
git diff --cached --exit-code
```

I also inspected the complete v3-to-v4 document diff and used bounded UTF-8 reads, `rg`, `Get-FileHash`, and Git tree/blob comparisons.

Fresh CPU programs ran through PowerShell here-strings piped to:

```text
C:/Python314/python.exe -B -
```

Environment: Python `3.14.0`, NumPy `2.5.2`. Successful programs performed the source audit, 20-record original-helper replay, nearest-pair check, synthetic indexed-layout/BVH experiment and asset-byte audit. They returned exit 0 and wrote no artifacts.

All 25 local Go2 manifest files matched their recorded sizes and SHA256 values. Recomputed aggregate:

`479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`

The manifest names source revision `da76818e269b82289eba39808e2fb91d679d6994`. This was a byte audit, not a fresh complete Git/remote asset-preflight certification.

The four packet evidence digests matched:

| Artifact | SHA256 |
|---|---|
| `unified_geometry_design_probe_20261010_v3.py` | `214ac4959b23891048b5d56a1dca078f520cf49ac180f3065203097b2042ffbb` |
| `unified_geometry_design_probe_20261010_v3_coordinates.json` | `4729c1eec40147590b2e626a5cfe979abd18fea9a6abddd0d7564c2a27e47140` |
| `probe_indexed_source_materialization_v1.py` | `253ecfdc05f4999d476aac84a3fb410790afc0301e4817b049b5fa2a9fa8363c` |
| `c4-indexed-source-materialization-v1.json` | `a97cf37e388c966c6d4965ac5fc8757e5a6ba822c92c4889422106dfeaf88077` |

The author’s feasibility script and result were inspected. Its complete Go2 census remains author evidence, not a fresh successful execution by this reviewer.

**C1 — non-blocking verification limitation**

Location: v4 §7, particularly the actual-materializer reproduction requirement.

The full original Go2 reconstruction could not run in this session:

- `ssh -o BatchMode=yes -o ConnectTimeout=10 tileops-dev ...` failed with `Could not resolve hostname tileops-dev`.
- Authorized fallback `ssh ... ang.gao@192.168.80.101 ...` failed with `connect to host 192.168.80.101 port 22: Permission denied`.
- Local original importer execution stopped with `mujoco_menagerie_robot_preview.py requires trimesh`.

No escalation, installation or configuration change was attempted. No alternate mesh loader or substituted oracle was used.

Consequently, I do **not** claim a fresh complete 398,432-face reconstruction, reproduction of all original mesh blocks from assets, or verification of the author’s complete world-coordinate census. Design closure rests on the explicit corrected source/operation contract, original-source inspection, fresh saved-counterexample execution and synthetic feasibility checks.

Disposition: retain §7’s existing obligation to reproduce all 20 cases through the actual implementation, plus full-source correspondence and native validation. No design repair is requested. Synthetic checks and the author’s feasibility result cannot close that later gate.

An initial audit attempt also failed on Python’s default GBK decoding; retrying with explicit UTF-8 succeeded. This was an inspection-script error, not a candidate failure.

**Remaining gates and process hygiene**

Still required:

- Actual v4 packing/materialization, identity migration and negative validation tests.
- Fresh Warp JIT and CUDA compilation, typed launch/readback proof, compatibility coverage and supervised R0/R1 execution.
- Original CPU comparisons across all required channels, including intensity, profiles, shadows, frames, sentinels and diagnostics.
- Affected allocation/partial-failure and lifecycle campaigns.
- Final graph/arena/barrier integration, remaining lifecycle matrices, C4c3/c4, C4d/B3, systematic whole-C4 refactor and fresh whole review.

No GPU execution, native build, performance measurement or full-frame rerender occurred.

I did not author or modify the candidate, resume or fork another session, read conversation rollout files, use implementation-chat context, spawn agents, inspect the excluded `c4_input_precision` worktree, or start the ongoing implementation goal. No tracked files, evidence files or private agent instructions were edited. No report file was written.

All invoked commands terminated; no ongoing tool session remains. The recorded failed probe PID `5528` was absent at the final process check. No remote computation started and no process was signalled.

The candidate remains frozen at the checked HEAD and digest. This approval closes **design-review v3 F1 only at design level** and grants no later-stage authority.