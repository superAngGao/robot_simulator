Initiative: Q54 / P12.4 / C4 input and intersection precision  
Stage: Fresh independent read-only follow-up code review  
Author: Codex independent reviewer (`/root`)  
Version: v2  
Date: 2026-10-10  
Status: accepted for bounded repair-code scope, with comments  
Reviewer session: `01a1235e-95ec-7a82-8b54-7044ed85995b`  
Related Files: input-precision design-request/design-review v4; gap-review v1; candidate repair sources and tests  
Owner Summary: F1/F2/F3 reproduce on the base and are repaired in the inspected candidate. Fresh CPU verification passed. Required GPU/native closure remains pending.

**Verdict: Approve with comments — repair code and completed CPU evidence only.**

No substantive new code finding remains in this bounded review. This does **not** close the original findings’ required native acceptance, approve the complete precision amendment, or approve C4.

I did not author the candidate or use an implementation conversation. I fully read `AGENTS.md`, `collab/README.md`, `collab/REVIEW_PRINCIPLES.md`, the approved v4 design/request review, and the complete original report from session `01a1234e-14f7-7e00-891f-4e4bb8f5eb28`.

**Identity and repository state**

| Item | Observed result |
|---|---|
| Local repository | `C:/Users/AG/新建文件夹/c4_input_precision` |
| Base and observed merge-base | `ea15e152d0a40674ba2b711e3d481dae32a403b6` |
| Initial and final candidate HEAD | `5e79749009f47795325b4995790c98dffec2b051` |
| Local branch | `implementation/c4-input-precision` |
| Local initial/final status | Clean; working-tree and index diffs empty |
| Remote host | `tileops-dev`, primary host |
| Remote repository | `/data2/shared/ang.gao/robot_sim_input_precision_wip` |
| Remote initial/final HEAD | Same exact candidate |
| Remote branch/status | Detached HEAD; clean |
| GitHub repository identity | User-supplied `superAngGao/robot_simulator`; not independently queried |

Local `origin` points to `C:/Users/AG/新建文件夹/robot_sim_review_workflow`; remote `origin` points to `/data2/shared/ang.gao/c2d2-code-v1-full.bundle`.

The base-to-candidate history contains the archived original report and repair commit. The delta comprises ten files: that report, three implementation sources, source policy/catalog updates, and four test files.

The implementation/test whitespace check passed. The complete delta’s whitespace check reports eight trailing-space lines in the verbatim archived report’s Markdown header; these are not implementation defects.

**F1 — CUDA triangle-update ABI validation**

Disposition: **repair verified; native transaction closure pending.**

Location: `optics/cuda_scene.py:629`.

The indexed argument builder puts `x_world_R`, `x_world_r`, and `env_idx` at positions 6, 7, and 8. The validator now checks the scalar at position 8.

I executed the actual builder, validator, and matching-function bodies extracted separately from both revisions, using identity-preserving placeholder arrays:

| Revision | Builder-to-validator result |
|---|---|
| Base `ea15e152…` | `ValueError: CUDA triangle scene-update ABI binding changed` |
| Candidate `5e797490…` | Accepted |

The candidate regression tests additionally passed:

- Substitution at every one of the 15 argument positions.
- Wrong environment value.
- Equal-valued `False` and `0.0`, which must not satisfy the exact Python-integer requirement.

Tensor identity checking remains intact. The plane validator remains at its correct scalar position, 8. The change does not alter dispatch arguments or weaken validation.

This establishes the Python ABI repair. It does not establish a real CUDA R1 scene-update → LBVH-rebuild → rendering transaction.

**F2 — Grazing-hit normal orientation**

Disposition: **repair verified by source, independent host execution, and Warp CPU JIT; GPU closure pending.**

Locations:

- `optics/warp_native_render.py:938`
- `optics/_cuda_direct_light_source.py:400`

Both repaired helpers determine facing from the normalized binary64 world-edge cross product, then apply that orientation to the existing float32 shading normal. They retain the accepted output layout and add no normal buffer or ABI argument.

I independently replayed the original counterexample:

```python
vertices = [
    [0., 0., 0.],
    [-0.123456789, 1., 0.],
    [-0.3, 0., 1.],
]
origin = [-0.26367883497322825, 1.3257985665584018,
          0.333333333012683]
direction = [0.12252657197322826, -0.9924652332250684,
             3.2065034296667426e-10]
max_distance = 2.
```

The probe used the unchanged CPU helper, actual Warp function bodies, and separately compiled base/candidate CUDA helper source as host C++.

| Measurement, original direction/winding | Base | Candidate |
|---|---:|---:|
| Hit distance, both helper implementations | `0.9999999882523075` | Same |
| Returned normal orientation | Incorrect, positive | Correct, negative |
| Normal L2 error against CPU | `1.99999998080983` | `2.2406403432706915e-08` |

Candidate normal:

```text
[-0.9511989951133728, -0.11743197590112686, -0.2853597104549408]
```

All four combinations of ray reversal and winding reversal reproduced the base failure and passed on the candidate, in both the Warp-body and CUDA-host probes.

Fresh real Warp CPU JIT tests also passed for:

- Brute-force primary traversal.
- Device-scene scan.
- Device-scene traversal with AABB.
- Device-BVH traversal.
- Directional lighting with and without shadows, including RGB/intensity and the surface-offset case.

The compiled tests compare normals with the CPU result rounded to the existing float32 output format. CUDA host compilation is additional arithmetic evidence, not CUDA-device execution.

**F3 — Parallel-slab policy**

Disposition: **repair verified by independent reproductions and Warp CPU primary/shadow execution; GPU closure pending.**

Location: `optics/warp_native_render.py:801`.

Warp now compares a parallel-axis origin directly with the widened bounds. The nearest-hit tie epsilon no longer expands the slab. Nearest-hit comparison and physical-distance checks remain separate and unchanged.

I independently replayed:

```python
vertices = [
    [0., -1., -1.],
    [0.,  3., -1.],
    [0., -1.,  3.],
]
origin = [-5e-13, 0., -1.]
direction = [5e-13, 0., 1.]
max_distance = 2.
```

The actual outward-bounds helper produced `[0,-1,-1]` through `[0,3,3]`.

| Predicate | Base | Candidate |
|---|---|---|
| Original CPU triangle | Hit, `t=1` | Unchanged |
| Original CPU AABB | Miss | Unchanged |
| Warp AABB body | Hit | Miss |
| CUDA AABB, compiled as host C++ | Miss | Miss |

This preserves the original **CPU BVH policy**, including its distinction from an unaccelerated triangle predicate.

Candidate tests passed for outside origins, exact-boundary inclusion, both signs and zero parallel-axis direction, and physical limits immediately below, equal to, and above the intersection distance.

I also ran nine additional real Warp CPU primary/shadow cases using:

```text
origin.x    ∈ {-5e-13, 0, +5e-13}
direction.x ∈ {-5e-13, 0, +5e-13}
```

All nine passed. Exact-boundary origins with nonzero small directions hit in both traversals; outside origins were rejected. Zero determinant remained a miss.

**Source identities and inspection coverage**

I independently checked all **87 catalog entries** against candidate Git blobs and LF-normalized local worktree bytes. Remote tests additionally validated the actual POSIX checkout without newline normalization.

Verified SHA256 identities:

| Source | SHA256 |
|---|---|
| Design-request v4 | `df1035b0213fe0c48110439222c014f1a81af86f50996254d88634e520ecb9df` |
| Design-review v4 | `be13a105ddc8ff40c3e12a09986004a69b80706d901c66a436a5c7a0007018a5` |
| `optics/warp_native_render.py` | `e9fa34a9f4d730041c0e6b03950f891c0e2f2414c9225aa9c53babfcf92a4627` |
| `optics/_cuda_direct_light_source.py` | `07fa617a6f6f01b3a2145837b88978e7c4d563010c1a27d91bbcb85eab66d16b` |
| `optics/cuda_scene.py` | `9f9e13ff9331835f8f8c8d7ff1db9fe13919b72ffb69fd31338f9b11ff968637` |
| `optical_worker/native_source_policy.py` | `2b20b2f4286c034437cf3ecbf62eaa16a89abd6d1e1574f586cad9dd0563c115` |
| `optical_worker/data/native_sources/v17.json` | `ed6fd305fb17d7f035677dae728ebcec35ca4ecde7668e2ae6b996b610b277a2` |

The renderer source pin is updated, while the pinned snapshot-function identity remains unchanged. Recipe tests passed with the changed CUDA renderer-source digest. Renderer contraction remains disabled in Warp and CUDA.

Inspection covered ABI construction/validation, geometric-normal derivation, primary/shadow helper callers, slab and physical-limit behavior, tests’ use of actual implementation functions, and source-policy/catalog consistency.

Git reports no changes to `optics/execution.py`, `optics/scene.py`, or `physics/spatial.py` from original oracle revision `812ca29030a45d802addf10ca5811a4354ce0eb9` to this candidate.

**Executed verification**

All Python executions used `-B`; pytest cache writing was disabled. Every remote test used `CUDA_VISIBLE_DEVICES=''` and inspected candidate HEAD/status before execution. No dependencies were installed.

Local environment: Python 3.14; MSYS2 UCRT64 `g++`. Remote environment: Python `3.12.13`, NumPy `2.5.1`, Warp `1.12.0`.

The initial local command was:

```text
C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
  --basetemp=<local-scratch>/local_pytest_1
  tests/unit/optics/test_native_intersection_precision.py
  tests/unit/optics/test_cuda_scene_native.py
  tests/unit/optics/test_cuda_extension_recipes.py
  --tb=short
```

Result: **99 passed, 73 failed, 1 error in 15.63s**.

The failures/error originated from the host compiler fixture: MinGW’s linker could not open the absolute output path containing Chinese characters. Direct compiler reproduction confirmed the path-encoding failure.

I then ran the unchanged numerical/ABI tests through an external scratch launcher that adapts only the compiler invocation to use relative filenames and its scratch directory as `cwd`. It changes no source, compiler flags, assertions, or numerical behavior:

```text
C:/Python314/python.exe -B <local-scratch>/run_local_tests.py
```

Result: **168 passed in 5.22s**. This reproduces the author’s local count with the stated toolchain-path adaptation.

The independent base/candidate probe was:

```text
C:/Python314/python.exe -B <local-scratch>/reproduce.py
```

Result: **all reproduction and source-catalog assertions passed**.

Remote execution used `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -`, with a stdin launcher invoking `pytest.main`. The first invocation selected:

```text
-q -s -p no:cacheprovider
--basetemp=/tmp/precision_gap_review_v2_90dhfrlt/pytest
tests/unit/optics/test_native_intersection_precision.py
tests/unit/optics/test_cuda_scene_native.py
tests/unit/optics/test_double_query_kernels.py
tests/unit/optics/test_indexed_scene_kernels.py
```

Result: **221 passed in 13.11s**.

Warp used a fresh reviewer-specific cache and reported actual CPU compilation of the query and indexed-scene modules. Its device listing contained only CPU. The hidden-CUDA initialization diagnostic did not prevent CPU execution.

The second remote invocation selected:

```text
-q -p no:cacheprovider
--basetemp=/tmp/precision_gap_review_v2_90dhfrlt/identity_pytest
tests/unit/optics/test_native_sources.py
tests/unit/optics/test_native_source_policy.py
tests/unit/optics/test_cuda_extension_recipes.py
tests/unit/optics/test_source_geometry_precision.py
tests/unit/optics/test_native_scalar_precision.py
```

Result: **242 passed, 2 skipped in 26.47s**. The skips were the real-Go2 source-materialization checks requiring an asset-path configuration.

The additional nine Warp primary/shadow cases reused only the CPU kernels compiled during this review.

The author’s supplied remote log reports **221 passed in 10.77s**. Its SHA256 is:

```text
0f26831696ef838b016fb9c22ddc37b4495c5c29245cf61d32cbac66946676bc
```

That historical log was inspected, not treated as fresh execution. The independent 221-test run above verifies the claim against the inspected candidate. Local and remote counts overlap and must not be added as unique coverage.

**C1 — Non-blocking for this bounded approval: native closure remains required**

The original report’s native proof obligations remain binding:

1. **F1:** a real triangle-containing CUDA R1 transaction through scene update, LBVH rebuild, and rendering.
2. **F2:** fresh GPU execution in both lanes, both ray directions and windings, coverage of all six retained compatibility executors, and directional-light RGB/intensity/shadow-side evidence.
3. **F3:** fresh Warp/CUDA GPU primary and shadow traversal of the counterexample, exact-boundary inclusion, and strict physical-limit regressions.
4. Fresh CUDA artifacts and execution provenance tied to the repaired sources, followed by independent review of the native evidence.

These are mandatory before full native acceptance of the original findings. They are explicitly outside this repair-code/CPU-evidence approval.

No GPU numerical rerun, CUDA-device execution, complete Go2 reconstruction, full R1/profile/frame/failure matrix, amendment acceptance, or whole-C4 review was performed. Known unfinished matrices are not new code findings.

**Process hygiene and final state**

Local probes and generated host binaries reside outside the candidate in:

```text
C:/Users/AG/新建文件夹/c4_assets/precision_gap_fix_review_v2_scratch
```

Remote CPU test temporaries and the isolated Warp cache reside in `/tmp/precision_gap_review_v2_90dhfrlt`.

No implementation or tracked file was edited; no report was written into the repository. No agents, GPU jobs, installations, process signals, worktree switches, or deployment-job interactions occurred. All invoked commands completed.

Both inspected candidate checkouts remain clean at `5e79749009f47795325b4995790c98dffec2b051`.