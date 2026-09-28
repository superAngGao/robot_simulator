# Frozen Go2 asset recovery on the authorized fallback

Initiative: q54-optical-lab-c4-go2-asset-recovery
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-28
Status: implemented
Related Files: benchmarks/optical/assets.py; benchmarks/optical/workload_contract.py
Owner Summary: Exact frozen Go2 asset bytes and both scene/fixture contracts reproduced without changing any gate.

The primary 183 transport remained unreachable. Using the owner's local-relay
instruction, a new isolated local checkout fetched official
`https://github.com/google-deepmind/mujoco_menagerie.git` at full revision
`da76818e269b82289eba39808e2fb91d679d6994`, sparse-checking out `unitree_go2`.
No packages were installed and no pre-existing asset checkout was overwritten.

The frozen aggregate corresponds to CRLF text checkout bytes. A normal LF
checkout has aggregate
`01109aab17841e2b4eb48c1baeab1a05a90d1fde038f9fa0899295da072b6040`,
which is correctly rejected by the frozen contract. Reproducing the original
Git `core.autocrlf=true` text checkout convention (without changing binary
files or Git blobs) yielded the exact required 25-file aggregate:
`479dcbdb52c64ca86243d6b1d3b7167bb7d3820af1e60e355168b3c17ddf3ed0`.
This is a reconstruction of frozen bytes, not a gate change or an alternate
accepted digest. The checkout's tracked-model Git diff is empty.

The checkout, including its Git provenance and per-file manifest, was relayed
as a tar archive to 101 and extracted into a new isolated root:
`/data3/ang.gao/c4_asset_frozen/out/external/mujoco_menagerie/unitree_go2`.
An earlier LF diagnostic copy remains untouched under c4_asset_validation.

Using unmodified production/benchmark materializers from clean candidate
266fc821 (`robot_sim_c4_recording_review2`) and the robot_sim environment,
`preflight_case_asset(..., repo_root=Path('/data3/ang.gao/c4_asset_frozen'))`
passed. Both canonical materializers then independently reproduced:

| Case | Scene SHA-256 | Fixed-fixture SHA-256 |
| --- | --- | --- |
| go2_static_960x640_v1 | 745f01f3d526681c3a350086d55211f92d4a1e5f5b22bcbbc9c43f7dd18f072f | fce32d13427918022c4eaee45485ed6fbfaf5e03b13d1390a0216e138e483182 |
| go2_static_1920x1080_v1 | 57e86366429979a2460a0ceb01f00749c7e4be7449bbc3acc1577ed5ce94b923 | 8354993fb682d5a13a2dff62d1d7521afd7d3d46d20b876e15732b4c38415db0 |

Reproduction script: `/data3/ang.gao/verify_go2.py`; local counterpart
`C:/Users/AG/新建文件夹/c4_assets/verify_go2.py`. It hashes all tracked model
files, invokes the formal asset preflight, then builds each packed scene and
fixed fixture; each formal materializer checks its frozen digest internally.
Archive: `/data3/ang.gao/c4-assets-go2-crlf.tar.gz`. No rendered image, GPU
benchmark, performance sample, baseline or result record was generated.

This evidence removes the missing-asset prerequisite for later scoped tests;
it does not approve C4c3, activate a3b, or reconcile the unavailable original
183 Git ancestry. Future tests must still preflight the actual local asset
copy, not trust this narrative.
