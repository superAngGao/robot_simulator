Initiative: q46-solver-consistency
Stage: challenge
Author: codex
Version: v1
Date: 2026-04-20
Status: draft
Related Files:
- collab/q46-solver-consistency__proposal__claude__v1.md
- OPEN_QUESTIONS.md
- tests/gpu/solvers/test_solver_backends.py
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
Owner Summary: 方向基本正确，建议继续做 Q46 的 solver consistency 与 CPU/GPU 差异澄清；但不要把当前 box-ground 接触点个数实现细节固化成长期契约。推荐保留范围、收紧断言设计、把“差异已知但未统一”写清楚。

## Keep

- 保持本轮只做 Q46 的维度 3 和维度 6，这个切口合理，范围也收得住。
- 保持对 `jacobi_pgs_ms`、`colored_pgs`、`admm` 三个稳定 solver 做复杂场景一致性测试，排除已知在该场景下 step 1 发散的 `jacobi_pgs_si` 是合理的。
- 保持“不做 CPU box-ground 多点修复”的边界。这个确实是独立 thread，不适合混入本轮。

## Change

- 不建议把 CPU/GPU box-ground 差异收敛为“CPU 每个 box 1 个接触点、GPU 每个 box 4 个接触点”的精确断言。当前 GPU 实现是“最多 4 个最深顶点”，而不是稳定的长期平台契约；未来 CPU 多点化、GPU manifold 选择策略调整、或 tilted/edge cases 都可能改变确切个数。
- 建议把维度 6 的测试目标改成“语义级一致性 + 已知差异显式记录”：
  - CPU 和 GPU 的 ground-contact body set 应一致
  - 法线方向应一致
  - 每 body 的代表性深度应在容差内
  - GPU 允许比 CPU 产生更多 box-ground contacts，且原因在 docstring / comments 中明确记录
- 对维度 3 的 solver consistency，建议不要直接对完整 `q` 做宽松 `atol=0.05` 一刀切比较。更稳妥的做法是优先比较更有物理意义、也更抗表示噪声的量：
  - base position / body COM proxy
  - per-robot final height range
  - velocity norm 或 `max|qdot|`
  - 如需比较完整 `q`，应注意 free joint quaternion 分量的表示问题
- 建议在 `OPEN_QUESTIONS.md` 对 Q46 的更新使用“dim 3 partially addressed / dim 6 clarified but not resolved”这类表述，而不是把 dim 6 写成已解决。因为 proposal 自己也承认 CPU 多点仍未做。

## Risks

- `atol=0.05` 对 0.01s 仿真可能过宽，可能把真实 solver 差异吞掉；但如果直接收得太紧，又可能因为表示和浮点噪声导致脆弱测试。这里需要选“物理量 + 适中容差”，而不是仅靠统一的大 `q` 容差。
- 如果把 CPU/GPU 差异测试绑定到“精确接触点数”，后续实现改进时测试会先阻碍正确演进，而不是保护真实行为。
- 这轮如果同时想做“文档化差异”和“判定差异无需修复”，容易过早冻结设计判断。Q46 item 6 目前更像“已知差异待后续统一策略”，不是最终设计结论。

## Future Compatibility

- 未来 soft / fluid / cloth 接入后，“接触点个数”更不适合作为跨后端、跨子系统的一等契约；更可持续的契约应是接触语义、法线、代表性深度、稳定性与可观测力。
- 如果这轮把 CPU/GPU 差异表述为“允许不同 manifold cardinality，但共享同一物理语义”，后面无论 CPU 补多点、GPU 调整 manifold 生成，测试和文档都更容易演进。
- Solver consistency 这轮最好建立成“基线测试框架”，而不是只为 Q46 做一次性 patch。后续复杂机器人、长时间轨迹、甚至未来软体/耦合 solver 都可以复用同一测试形态。

## Recommendation

- 继续按这个 initiative 推进，但修改 Part B 的断言策略：不要写死 CPU=1 / GPU=4 的精确数量关系，改成语义级一致性断言 + 明确注释当前差异来源。
- Part A 建议优先比较物理意义更强的派生量，再决定是否补完整 `q` 级比较；如果保留完整 `q` 比较，请单独解释 quaternion 处理策略。
- Q46 更新建议写成“dim 3 新增定量基线；dim 6 差异被显式记录，但 CPU/GPU manifold 尚未统一”，不要把维度 6 提前关掉。
