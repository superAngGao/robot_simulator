---
Initiative: q46-dim6-cpu-multipoint
Stage: implementation-note
Author: claude
Version: v1
Date: 2026-04-21
Status: implemented
Related Files:
- collab/q46-dim6-cpu-multipoint__proposal__claude__v1.md
- collab/q46-dim6-cpu-multipoint__challenge__codex__v1.md
- collab/q46-dim6-cpu-multipoint__review__codex__v1.md
- OPEN_QUESTIONS.md (Q46 dim 6)
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
Owner Summary: 按 Codex challenge 建议，实施 Q46 dim 6 的"部分澄清"版本。旧的"CPU 单点"描述已从测试和文档中移除；body 1 flat-box 的 per-body count parity 已显式固化为 4；Q46 dim 6 保持 🔄，不提前关闭。

## Open Questions Addressed

- **Q46 dim 6 — CPU vs GPU 一致性**：`部分澄清` open →
  - 旧描述"CPU GJK/EPA 返回 1 个"已确认过时（session 33 commit `35ca490` 已实现多点路径）
  - 新描述：CPU 走 `contact_vertices()` 顶点枚举，all-vertices；GPU cap 到 4 个最深点
  - 对 `BoxShape` 平地场景 count 一致，对 ConvexHull >4 底面顶点可能 mismatch
  - 状态保持 🔄，待专项 box-ground CPU/GPU parity test 正式固化 count 契约后关闭

## REFLECTIONS.md / PROGRESS.md Impact

不需要更新。此次变更是文档修正和测试断言加强，未引入新功能或架构变化。

## What Changed

1. **OPEN_QUESTIONS.md Q46 dim 6**（行 1252-1259）：
   - 删除过时的"CPU GJK/EPA 返回 1 个"描述
   - 改为"部分澄清"措辞，记录 CPU all-vertices vs GPU cap-4 的残余差异
   - 明确"对 BoxShape 平地场景 count 一致；对任意凸包仍可能 mismatch"

2. **tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py**：
   - module docstring（行 18-26）：已更新（session 34 初期已完成）
   - `test_cpu_gpu_sorted_ground_contact_match`：
     - 删除"3. GPU may have more per body"注释（已过时）
     - 在 per-body max depth 比较之后加入 body 1 的显式 4-point count 断言
     - 末尾注释更新为准确描述"float32 vs float64 差异"而非"不同算法"

## Files Touched

- `OPEN_QUESTIONS.md` — Q46 dim 6 文字更新
- `tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py` — 测试断言加强
- `collab/q46-dim6-cpu-multipoint__proposal__claude__v1.md` — 新增
- `collab/q46-dim6-cpu-multipoint__challenge__codex__v1.md` — 新增（Codex 写）
- `collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md` — 本文件

## Tests Added / Updated

**Updated**: `test_cpu_gpu_sorted_ground_contact_match`（`test_b5_d6d7_cpugpu_multienv.py`）

新增 step 6 断言：
```python
# 6. Body 1 (A.link1) flat BoxShape → 4 ground contacts on both CPU and GPU.
body1_cpu = [c for c in cpu_sorted if c.body_i == 1]
body1_gpu = [c for c in gpu_sorted if c.body_i == 1]
assert len(body1_cpu) == 4, ...
assert len(body1_gpu) == 4, ...
```

此断言与 proposal 中的"总 count 等号"方案不同——它只精确断言 body 1（fixture
中唯一 flat BoxShape），不依赖整个 mixed-shape scene 的总数，对无关 shape 变化不敏感。

## Known Limitations

- body 1 的 4-point 断言是 fixture-sensitive 的：若 A.link1 的 shape composition、
  姿态或初始高度被重构，此断言需同步更新。
- GPU 测试在无 CUDA 环境跳过，body 1 GPU count 断言未被 CI 持续验证（已知限制）。
- ConvexHull >4 底面顶点的 count mismatch 仍只在文档层说明，无专项回归测试。
- `assert len(cpu_ground) == len(gpu_ground)` 总数等号断言未加（采纳 Codex 建议）。

## Commit

实现 commit: `e22241c` — docs: q46 dim6 partial-clarify — CPU multi-point vertex enumeration, residual cap-4 vs all-vertices difference noted
Note commit: `ab35b16` — docs: q46 dim6 implementation-note — codex challenge accepted, body1 count assertion added

## 关键思考

### 为什么采用 body 精确断言而不是总 count 等号

Codex challenge 的核心论点：`assert len(cpu_ground) == len(gpu_ground)` 会把
mixed-shape scene 的总数等号当成"dim 6 已完全解决"的证据，但这个等号是
"fixture 恰好满足"的结果，而不是"契约已统一"的结果。

body 精确断言的优势：
- 语义清晰：明确说这个 flat box body 应该是 4 个点，而不是让总数等号隐含这一事实
- 对无关 shape 变化不敏感：改 C 机器人的 sphere 数量不会让这个断言失败
- 退化检测精确：box-ground parity 退化时这个断言直接失败，而总数等号可能被其他 shape 变化掩盖

### CPU all-vertices vs GPU cap-4 的实际影响范围

对当前 fixture（BoxShape 底面恰好 4 个顶点）：无差异。
对 ConvexHullShape 底面 >4 顶点的场景：GPU 会比 CPU 少返回接触点。
这个差异在当前仓库的 contact 路径中不会导致物理错误，因为 PGS solver
用的是接触点的力，而不是接触点的数量。但多点 manifold 比单点有更好的旋转稳定性，
所以 GPU cap-4 在极端 hull 场景下可能比 CPU all-vertices 稍差。
