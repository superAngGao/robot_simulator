# C4c2a final functional evidence

Initiative: q54-optical-lab-p12-4c4c2a-native-contract-extraction
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-29
Status: accepted
Related Files: C4c2a implementation-note/design-request/review v1
Owner Summary: Archive final observations for independently approved
67b401878bb92371590885af7835be36430d3009. No implementation or gate change.

## Revisions and environment

Executed code/test revision: `0dcbf2a790adf3d695b9501e0ff01be564c8c06a`.
Reviewed revision: `67b401878bb92371590885af7835be36430d3009`; its only delta
from the executable revision is the implementation note. The note recorded
then-pending checks honestly; their final results are recorded here, not
retroactively inserted into the reviewed candidate.

Author clean runtime tree: `/data3/ang.gao/robot_sim_c4_c2a_impl_0dcbf2a`.
Independent clean tree: `/data3/ang.gao/robot_sim_c4_c2a_review_67b4018`.
Primary183 SSH timed out; owner-authorized fallback101 was used. No existing
dirty checkout was reset or overwritten and no primary-history reconciliation
is claimed. Runtime: robot_sim Python3.12.13, Torch2.9.1+cu128, Warp1.12.0,
NVCC12.8.93, CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim. Three real
devices are visible; target1/ambient0 tests use actual hardware.

Runtime invocation prefix:

```text
/data3/ang.gao/anaconda3/bin/conda run --no-capture-output -n robot_sim env CUDA_HOME=/data3/ang.gao/anaconda3/envs/robot_sim python
```

## Author results

All test groups ran with `-m pytest -q --tb=short` from the author runtime tree.
The implementation note records the first two groups' complete path lists.

| Group | Result | Log under `/data3/ang.gao/` |
| --- | --- | --- |
| Unit benchmark and affected optical contract/protocol suites | 413 passed,21.22s | c4-c2a-0dcbf2a-cpu.log |
| Main realGPU native/arena/scene/BVH/render/runtime suites | 198 passed,82.31s | c4-c2a-0dcbf2a-gpu.log |
| test_optical_closed_worker_setup.py + test_cuda_optical_timeline.py | 19 passed,6.94s | c4-c2a-0dcbf2a-setup-timeline.log |
| Complete tests/gpu/test_warp_sacrificial_render.py | 135 passed,847.81s | c4-c2a-0dcbf2a-supervised.log |

Total: **765 author server tests passed**, no skips. Functional groups
overlapped; elapsed times are not performance evidence. Existing permitted
compiler caches were not purged. These are new executions, not old pass logs.

Log SHA-256 values, in table order:

```text
07112fa514c09098de6bb06e383a95327b1c4881780074346bfdb9c517d6aac5
5f0aa299d5544bd660136cc2480eeedc36126c3b1a96c49914ad6cb82fe61b96
f7d9e668774315ca1f74610cdc20848c0973f5938f6d6f44cde8a4b345f16d39
2e92c0667b52cf05858592fd3376bb76f9b18952e0fb56b04c51ebcd273371c4
```

All four logs were copied to `C:/Users/AG/新建文件夹/c4_assets/` with matching
local/remote hashes. Reviewer separately read final summaries and recomputed
remote hashes; author executions are not counted as independent reruns.

The all-five-content probe ran separately and exited0 against the exact
recovered Go2 assets, without substitution or gate edits:

```text
python /data3/ang.gao/c4-c2a-content-probe.py /data3/ang.gao/robot_sim_c4_c2a_impl_0dcbf2a /data3/ang.gao/c4_asset_frozen
probe SHA256: 8723902ebfd082d02cc401fb148e6ae5ced598b7545fb6df17440153eb1e4d13
```

Every frozen case/scene and applicable fixture/replay digest matched. Probe
stdout is preserved in the task tool record; reviewer independently reran it
against67b4018. This legacy Go2 materializer check is not a claim that the
future closed worker may import examples or benchmarks.

Local32 extraction tests, Ruff check and formatcheck of all22 changed Python
files and diffcheck passed. Local Python3.14 has no Torch/Warp; earlier broad
local environment failures and corrected draft tests are disclosed in the
implementation note, not silently counted as passes.

## Independent verdict and limits

Fresh `/root/c4_c2a_contract_code_review_v1` approved exact67b4018 after
**412 independently executed tests**, no skips, all-five content reproduction,
full AST moves, alias/registry/context identity, real storage/authority tests,
source admission and gates checks. Full unchanged report is archived alongside
this record; SHA256:
`0bf44395d8e3f791a481f4e0b9039b2e841be65dd42a18c19cfbb69778d0aabd`.

Approval is only for the bounded extraction. C4c2 runtime integration and its
separate design, C4c3/c4, C4d/B3, product GPU preparation, a3b/result/baseline,
host payload/shipping and backend selection remain closed. No push, merge,
installation, default change, new performance threshold or benchmark result
was authorized or performed. This evidence-only archive does not alter code,
review verdict, design requirements or activation.
