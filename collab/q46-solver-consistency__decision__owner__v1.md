Initiative: q46-solver-consistency
Stage: decision
Author: owner
Version: v1
Date: 2026-04-20
Status: accepted
Related Files:
- collab/q46-solver-consistency__proposal__claude__v1.md
- collab/q46-solver-consistency__challenge__codex__v1.md
Owner Summary: 按 proposal 范围推进（Q46 dim 3 + dim 6），采纳 Codex 的断言策略修改。Part A 比较物理派生量，Part B 改为语义级一致性断言。Q46 dim 6 不关闭。

---

## Chosen Direction

**Part A — Cross-Solver Consistency（Q46 dim 3）**

新增 `TestCrossSolverConsistency` 测试类，使用 3-robot 9-body mixed-shape fixture，
对 `jacobi_pgs_ms`、`colored_pgs`、`admm` 三个稳定 solver 各跑 50 步。

断言策略（采纳 Codex 建议）：
- 比较 base body COM height（每个 robot 的 link0 z 坐标）
- 比较 per-robot final height range（最高/最低 body 的 z 范围）
- 比较 `max|qdot|`（速度范数上界）
- 不对完整 `q` 做宽松一刀切比较；如需比较 q，quaternion 分量单独处理
- 排除 `jacobi_pgs_si`（已知在该场景 step 1 发散）

**Part B — CPU vs GPU Box-Ground 差异（Q46 dim 6）**

更新 `test_b5_d6d7_cpugpu_multienv.py` 的断言策略：
- 语义级一致性：ground-contact body set 一致（CPU 和 GPU 检测到接触的 body 集合相同）
- 法线方向一致（atol=1e-3）
- 代表性深度在容差内（每个 body 取最大深度，atol=1e-3）
- GPU 允许比 CPU 产生更多 box-ground contacts，不写死精确数量
- docstring 明确记录差异来源：GPU 顶点枚举多点 manifold vs CPU GJK/EPA 单点

## Accepted Constraints

- CPU box-ground 不改为多点（独立 thread，本轮 out of scope）
- 不覆盖 `jacobi_pgs_si` 的一致性（已知发散，加入无意义）
- 测试设计为"基线框架"，后续复杂场景可复用同一形态

## Deferred Items

- Q46 dim 1/2（RL 训练验证）：Phase 3
- Q46 dim 4（Colored PGS 性能优化）：独立 thread
- Q46 dim 5（Mass splitting 收敛精度）：独立 thread
- CPU box-ground 多点化：独立 thread

## Out of Scope

- 任何 solver 性能改动
- CPU narrowphase 架构变更
- Q47/Q48 相关内容

## Acceptance Conditions

1. `TestCrossSolverConsistency` 新增测试通过，3 个 solver 在 50 步内物理派生量一致
2. `test_b5_d6d7_cpugpu_multienv.py` 断言改为语义级，CPU/GPU 差异在 docstring 中明确记录
3. Q46 更新：dim 3 标注"新增定量基线"，dim 6 标注"差异显式记录，manifold 尚未统一"，两者均不关闭
