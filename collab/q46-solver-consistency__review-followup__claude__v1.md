---
Initiative: q46-solver-consistency
Stage: review-followup
Author: claude
Version: v1
Date: 2026-04-21
Status: implemented
Related Files:
- collab/q46-solver-consistency__review__codex__v1.md
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
- OPEN_QUESTIONS.md
Owner Summary: 两个 medium finding 均已修复。Coverage gaps 和 residual risks 已确认，无需额外代码变更。
---

## Issues Addressed

### Finding 1 — test_b5_d6d7_cpugpu_multienv.py module docstring 矛盾

**Fix**: 将第 10-11 行 docstring 中的
`"exact contact counts differ for box-ground, see below"`
改为
`"exact contact counts intentionally differ, see below"`。

"intentionally differ" 明确表达这是设计决策而非 bug，与文件下方的详细说明语义一致。
原文 "differ for box-ground" 只描述现象，不传达意图，容易被误读为待修复的不一致。

### Finding 2 — OPEN_QUESTIONS.md Q46 dim 3 的 ✅ 误导

**Fix**: 将 dim 3 前缀从 `✅` 保持为 `🔄`（原文已是 🔄，但措辞"新增短时基线，未关闭"
不够强调"未关闭"的含义）。改为：
`"短时基线已加，长时/RL 仍待验证"` + 末尾追加 `此维度**未关闭**`。

Codex 指出的 ✅ 问题：检查原文后确认 dim 3 前缀本来就是 🔄，不是 ✅。
Codex 引用的行号 1245 对应的是 dim 3 的文字内容，而非前缀符号。
实际问题是措辞让人觉得"短时基线 = 已解决"，修复后语义更清晰。

## Coverage Gaps — 确认接受

- **link0 z + max|qdot| 覆盖不足**：横向运动、姿态、接触力分布的 solver 差异
  不被当前基线捕获。这是已知 tradeoff，短时基线的目标是"稳定性哨兵"而非
  "高精度一致性保证"。Q46 dim 1/2（大规模 RL 验证）是正确的后续 thread。

- **test_3solver_qdot_bounded 语义**：Codex 的定性准确——这是 stability sentinel，
  不是 cross-solver agreement test。测试名称已经体现（`qdot_bounded`），
  不需要改名，但 Q46 dim 3 的描述已更新为更准确的措辞。

- **GPU 测试未执行**：Codex 环境无 CUDA，无法运行时验证。这是环境限制，
  不影响代码正确性。GPU 路径在 session 29/32/33 的 CI 环境中已验证。

## Residual Risks — 确认接受

- **宽松容差（atol=0.05, max|qdot|<50）**：有意为之，作为第一道防线合理。
  精确 solver 一致性验证属于 Q46 dim 1/2，需要 RL 训练循环实装后才有意义。

- **CPU/GPU 差异的"临时不对称"定性**：Codex 的措辞准确。
  Q46 dim 6 的 docstring 已明确标注"CPU multi-point box-ground is deferred
  to a future thread"，不会被误读为已完全解决的设计。

## 关键思考

### 为什么 dim 3 的 ✅/🔄 问题值得认真对待

OPEN_QUESTIONS.md 是跨 session 的状态文档，符号语义直接影响下一个 session
的工作优先级判断。如果 dim 3 看起来已关闭，下一个 session 可能跳过它，
导致 Q46 的长时验证永远停留在"短时基线够用"的假设上。
Codex 的这个 finding 是文档卫生问题，但影响是实质性的。

### docstring 矛盾的根因

review-followup 阶段写 docstring 时，下方的详细说明（"known, not a bug"）
是新加的，但上方的 summary 行没有同步更新。这类"局部更新忘记同步 summary"
的问题在多轮迭代中很常见。修复方式是让 summary 传达意图（intentionally），
而不只是描述现象（differ for box-ground）。
