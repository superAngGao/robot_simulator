---
Initiative: q46-dim6-cpu-multipoint
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-21
Status: implemented
Commit: e22241c
Related Files:
- collab/q46-dim6-cpu-multipoint__proposal__claude__v1.md
- collab/q46-dim6-cpu-multipoint__challenge__codex__v1.md
- OPEN_QUESTIONS.md (Q46 dim 6, lines 1252-1259)
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
---

## Summary

按 Codex challenge 建议，实施 Q46 dim 6 的"部分澄清"版本：

1. **OPEN_QUESTIONS.md Q46 dim 6**：更新为"部分澄清"状态，不关闭。
2. **test_b5_d6d7_cpugpu_multienv.py**：
   - module docstring 已更新（session 34 已经完成）
   - `test_cpu_gpu_sorted_ground_contact_match`：加入 body 1 的 per-body count 显式断言

原 proposal 的"关闭 dim 6 + 总 count 等号"方案被 Codex 否决，采用 Codex 推荐的"语义级契约 + body 精确断言"方案。

## 实际变更内容

### 1. OPEN_QUESTIONS.md Q46 dim 6（行 1252-1259）

旧文本（已过时）：
```
6. 🔄 **CPU vs GPU 一致性**（2026-04-20 差异已显式记录，manifold 尚未统一）：
   GPU 顶点枚举返回最多 4 个 box-ground 接触点；CPU GJK/EPA 返回 1 个。
   两者在接触 body 集合、法线方向、per-body 最大深度上一致。
   差异来源已在 test_b5_d6d7_cpugpu_multienv.py docstring 中明确记录。
   CPU 多点化为独立后续 thread。
```

新文本：
```
6. 🔄 **CPU vs GPU 一致性**（2026-04-21 部分澄清，契约未完全统一）：
   ~~CPU GJK/EPA 返回 1 个~~ — 此描述已过时（commit `35ca490`，session 33）。
   CPU `ground_contact_query()` 对 Box/ConvexHull 已走 `contact_vertices()` 顶点枚举，
   flat box 实测返回 4 个接触点，与 GPU 行为对称。
   **残余差异**：CPU 返回所有穿透顶点（无数量上限），GPU cap 到 4 个最深点。
   对 `BoxShape`（恰好 8 顶点）平地场景 count 一致；对任意凸包（>4 底面顶点）
   仍可能 count mismatch。语义级契约（body set / normal / per-body max depth）已对齐。
   count-level parity 留待专项 box-ground CPU/GPU parity test 正式固化。
```

### 2. test_cpu_gpu_sorted_ground_contact_match 新增断言

在 per-body max depth 比较之后，加入：

```python
# 6. Body 1 (A.link1) has BoxShape at origin_xyz=[0,0,-0.04], body z=0.05 →
#    box bottom at z=0.01 → 4 bottom vertices penetrate ground by 0.01m.
#    Both CPU (all penetrating vertices) and GPU (cap-4 deepest) return 4 for a flat box.
body1_cpu = [c for c in cpu_sorted if c.body_i == 1]
body1_gpu = [c for c in gpu_sorted if c.body_i == 1]
assert len(body1_cpu) == 4, (
    f"Body 1 CPU ground contacts: expected 4 (flat box), got {len(body1_cpu)}"
)
assert len(body1_gpu) == 4, (
    f"Body 1 GPU ground contacts: expected 4 (flat box), got {len(body1_gpu)}"
)
```

同时更新末尾注释：
```python
# Contact point XY is not compared: both CPU and GPU use vertex enumeration
# (same algorithm), but float32 vs float64 causes sub-mm position differences.
# Physically meaningful quantities (body_i, normal, depth) are compared above.
```

## 关键思考

### 为什么采用 body 精确断言而不是总 count 等号

Codex challenge 的核心论点：`assert len(cpu_ground) == len(gpu_ground)` 会把
mixed-shape scene 的总数等号当成"dim 6 已完全解决"的证据，但这个等号是
"fixture 恰好满足"的结果，而不是"契约已统一"的结果。

body 精确断言的优势：
- 语义清晰：明确说这个 flat box body 应该是 4 个点，而不是让总数等号隐含这一事实
- 对无关 shape 变化不敏感：改 C 机器人的 sphere 数量不会让这个断言失败
- 退化检测精确：box-ground parity 退化时这个断言会直接失败，而总数等号可能被其他 shape 的变化掩盖

### CPU all-vertices vs GPU cap-4 的实际影响范围

对当前 fixture（BoxShape 底面恰好 4 个顶点）：无差异。
对 ConvexHullShape 底面 >4 顶点的场景：GPU 会比 CPU 少返回接触点。
这个差异在当前仓库的 contact 路径中不会导致物理错误，因为：
- PGS solver 用的是接触点的力，而不是接触点的数量
- 但多点 manifold 比单点有更好的旋转稳定性，所以 GPU cap-4 在极端 hull 场景下
  可能比 CPU all-vertices 稍差

这个问题留给专项 box-ground parity test 正式固化，而不是在 mixed-shape 集成测试里隐式断言。

### 为什么不直接关闭 Q46 dim 6

Codex 的分层定义建议是正确的：
- dim 6 的"语义级契约"已对齐（body set / normal / per-body max depth）
- dim 6 的"count-level parity"只对 BoxShape 平地场景成立，对一般 ConvexHull 不成立
- 因此 dim 6 应保持 🔄，直到有专项 box-ground CPU/GPU parity test 正式固化 count 契约

这避免了"已关闭"状态让后续 session 误以为 CPU/GPU ground manifold 已全面统一。

## 覆盖情况

- Q46 dim 6 文档：✅ 已更新为"部分澄清"
- body 1 flat box 4-point 断言：✅ 已加入（CPU 和 GPU 分别断言）
- 总 count 等号：❌ 未加（Codex 建议不加，已采纳）
- ConvexHull >4 顶点场景的 count mismatch：🔄 留待专项测试

## 残余风险

- body 1 的显式 4-point 断言假设 fixture 不变（A.link1 始终是 flat box at z=0.05）。
  如果 fixture 重构，这个断言需要同步更新。这是可接受的 tradeoff（断言更精确但更脆）。
- GPU 测试在无 CUDA 环境跳过，body 1 GPU count 断言未被 CI 持续验证（已知限制）。
