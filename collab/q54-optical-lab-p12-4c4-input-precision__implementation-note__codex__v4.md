Initiative: Q54 / P12.4 / C4 input and intersection precision
Stage: Repair implementation and native regression evidence
Author: Codex
Version: v4
Date: 2026-10-10
Status: implemented for the bounded repair; full amendment incomplete
Related Files: input-precision design-request/design-review v4; gap-review v1/v2; implementation-note v3; systematic-refactor gap-review v1
Owner Summary: Repair the three independently reproduced precision defects directly in the existing implementations. Fresh native Go2 960x640 R0 runs pass all ten CPU-reference channels in both lanes. Dynamic/sentinel native closure and final graph integration remain unfinished.

## Exact repair and independent scope

Original candidate `ea15e152d0a40674ba2b711e3d481dae32a403b6` was inspected by
fresh session `01a1234e-14f7-7e00-891f-4e4bb8f5eb28`. Its complete Request changes
report is archived as gap-review v1, without editing its conclusions. Saved
report SHA256: `da9a18144c411d0919439366d9e1dd000eb6ded64bd875c6de5e9bb62ec7281b`.

Repair commit `5e79749009f47795325b4995790c98dffec2b051`:

- F1: validate the indexed CUDA triangle-update environment scalar at argument
  8. All other arguments still require the original tensor identity.
- F2: decide triangle normal facing from the existing normalized binary64
  world-edge cross product, then orient the retained float32 shading normal.
  Warp and CUDA add no buffer, argument, shading precision or backend wrapper.
- F3: remove Warp's extra nearest-hit epsilon from parallel AABB slab bounds,
  matching the original CPU and CUDA policy. Tie and physical-limit rules stay
  separate.

Source catalog v17 has 87 entries and SHA256
`ed6fd305fb17d7f035677dae728ebcec35ca4ecde7668e2ae6b996b610b277a2`.
The snapshot source pin and CUDA recipe-source test pin reflect the repair;
the callable/array ABI and catalog schema are unchanged.

Fresh independent session `01a1235e-95ec-7a82-8b54-7044ed85995b` reviewed that
exact clean revision locally and on the primary server. Its complete report
is gap-review v2, saved-byte SHA256
`e9328f57b5cebc6d83869c98394819752cfb0bc986d8978e1bf899a2b4494425`.
Verdict: **Approve with comments, repair code and completed CPU evidence only**.
The reviewer independently reproduced all three base failures and candidate
repairs. It explicitly did not close the required GPU/native findings or the
whole amendment. Built-in fresh-agent creation again returned thread limit;
both reviews used new CLI sessions without inherited author conversation.

## Verification and original failures

Local Python3.14 numerical/ABI command, with MSYS2 UCRT64 compiler on PATH:

```text
python -B -m pytest -q -p no:cacheprovider
  tests/unit/optics/test_cuda_scene_native.py
  tests/unit/optics/test_native_intersection_precision.py
```

168 passed in10.72s. Tests include the exact grazing counterexample, both
windings/directions, parallel/outside/exact-boundary slabs, physical limits,
every indexed ABI argument replacement, and equal-valued noninteger indices.

An initial source/recipe command before committing reported224 passed,
4 skipped,3 failed. Two checks intentionally read HEAD's old renderer against
the newly refreshed worktree pin; the third exposed the recipe test's stale
CUDA source digest. After committing the coherent source and correcting that
test pin, the two source checks plus recipe module passed:7 passed in6.76s.
This is not a rerun claim for the entire231-case invocation.

Remote `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`3.12.13, Warp1.12.0,
CUDA hidden, at clean5e79749 ran the two files above plus
`test_double_query_kernels.py` and `test_indexed_scene_kernels.py`:
**221 passed in10.77s**. These execute actual Warp CPU JIT kernels, including
all four primary variants, directional RGB/intensity with/without shadows,
and primary/shadow traversal; CUDA helpers compile as host C++.
The independent reviewer separately obtained221 passes with a fresh Warp CPU
cache and242 passes/2 asset-configuration skips in source/identity checks.
Counts overlap and are not cumulative unique coverage. Its initial Windows
compiler failure and scratch-path adaptation are preserved in its report.

## Fresh deployment and GPU R0 regression

All native work used the unchanged existing deployment and supervised
warmup/readback drivers. No new runtime control layer or timing path was added.
Both remote and local precision candidate remained clean at5e79749 throughout
the builds and runs. The CPU oracle remains original
`812ca29030a45d802addf10ca5811a4354ce0eb9`.

Deployment: `/data2/shared/ang.gao/c4-precision-gap-cuda-artifacts-v1`.
Manifest SHA256:
`8451b31e7794ee66547b52667734601e005b618383867a928b42a9769e53e980`.
Composed environment: `c4-precision-gap-cuda-environment-v1.json`, SHA256
`000fe3bece3850327e3930630804fd5d5f5a9585172415dc7e0f34ec18ceb0e2`.
Existing compilation/base-platform profiles and installed dependencies were
unchanged. All three build units
`robot-sim-cuda-ae7c5050a6ba491f9717bed247a00d13-{0,1,2}.service` exited0,
reported success, and their actual cgroup paths were absent after completion.
Exact commands, environment and per-recipe records are in the archive.

Each lane used device1, GPU UUID `GPU-3a3112ff-837c-3b87-38d1-cb203d656589`,
driver595.71.05, real Go2 assets, all ten setup warmups, and the existing
`runtime_render_warmup_geometry` case. New output directories:

- `c4-precision-gap-cuda-960-geometry-v1`
- `c4-precision-gap-warp-960-geometry-v1`

Every pixel of frame10, 614400 pixels per lane, was compared with the retained
original CPU960 frame after verifying all CPU chunk and native channel hashes.
All ten channels have **zero failed pixels** under the unchanged tolerances;
hit/instance/material IDs are exact. Native rays remain f64. Maximum RGB
absolute error is3.7175754741269884e-7; intensity3.992438959432576e-7.
The two lanes' RGB/intensity bytes match. This does not exercise the synthetic
grazing/parallel counterexamples on GPU and is not R1 evidence or timing.
The repaired revision was not rerun at1920 in this checkpoint; v3's original
four-run evidence remains attributed to its own source revision.

Comparison SHA256 values:

- CUDA: `3605812bd444613f8b9ac210a6ed4b4456910daee030335321c54d4e63dd177d`
- Warp: `48f17febb6ead8f2cd95f264bb017a3913f654e845b014d01636723f672dee08`

Archive `c4-precision-gap-native-evidence-v1.tar.gz`, retained under
`/data2/shared/ang.gao/` and local `c4_assets/`:53398857 bytes,187 unique members,
SHA256 `e138c83e460da4e4e772c8fe8ee0ab53480b13dd4203b53373b51e3c887e69bd`.
It contains both complete raw runs/readbacks, comparisons, deployment including
binaries/build records, CPU/build logs, launch/comparison/archive scripts, and
process closure. The archive script's first attempt compared systemd's string
exit status with an integer and stopped before creating an archive; the final
script checks the actual schema's `'0'` and `'exited'` fields.

All ten owned parent/build/pytest/worker/supervisor PIDs are absent:
3929638,3929644,3929640,3929649,3936429,3936550,3936549,3943743,3943885,3943884.
Closure record SHA256:
`47e7c3b2c7df352ba4aba47da37aaced37bb0f2dee5c17b609495b692aaab8cc`.
Only the pre-existing unrelated MPS server2238410 remained in the GPU process
listing; it was untouched. Both reviewer sessions and all author test/runtime
handles completed.

## Integration/refactor worktree and required continuation

`c4_final_graph`, branch`implementation/c4-final-graph`, contains the precision
repair (cherry-pickfce2b4b) and both full review reports. Separately committed
f7eee4b extends the existing LF checkout rule from sequencev1 to every sequence
version. A fresh Windows checkout exposed CRLF in v2, causing52 fixture errors
before graph code ran; restoring exact Git LF bytes and the durable attribute
rule gave97 passes across render preparation/warmup support/CUDA arena checks.

Current uncommitted integration work is **unfinished and not native-admitted**:
the CUDA C3 adapter moves to its production owner with explicit compatibility
aliases; camera sample and transform replay selection become distinct in the
existing arena; setup/final frame binding shares one helper; warmup bookkeeping
uses named fields instead of positional lists. It reuses existing storage,
transactions and arena control. The final consumer, source closure, both-lane
final graph and guarded barrier are still to be completed together.

The named-record warmup tests passed68 cases, including copied authority,
transaction/evidence, failed leaves and missing DISARMED acknowledgement. An
intermediate cleanup accidentally removed the still-used output-buffer local;
62 fixture errors plus a focused reproduction exposed it, and it was restored
before the passing run. CUDA selector/alias tests passed12 cases after fixing
test-call argument/name errors. Historical extraction tests passed32 after
explicitly recognizing v4's indexed validation and SpatialTransform changes;
unchanged-body and exact compatibility-alias checks remain. Four changed
production modules pass the AST policy. These are worktree-level component
checks, not final source-catalog or full integration acceptance.

Required next work preserves the original scope: real triangle-containing
CUDA R1 update/rebuild/render, both-lane GPU counterexamples and all six
compatibility executors, remaining full R1/profile/frame/failure matrix and
fresh native/full-amendment review; then coherent final R0 readiness/origin,
both-lane C3/arena plans, seal-once guarded barrier and post-DISARMED commit,
R1 integration, full C2f lifecycle matrix, C4c3/c4 and C4d/B3, systematic whole-C4
refactor and independent whole review. No timing result, backend selection,
product activation or later gate is issued. C4 remains active and incomplete.
