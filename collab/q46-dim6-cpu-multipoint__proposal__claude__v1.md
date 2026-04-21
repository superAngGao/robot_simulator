---
Initiative: q46-dim6-cpu-multipoint
Stage: proposal
Author: claude
Version: v1
Date: 2026-04-21
Status: draft
Related Files:
- OPEN_QUESTIONS.md (Q46 dim 6)
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
- physics/gjk_epa.py (ground_contact_query)
---

## Summary

Q46 dim 6 描述"CPU GJK/EPA 对 box-ground 只返回 1 个接触点"是过时的。
Session 33 已在 `ground_contact_query()` 中实现 Box/ConvexHull 的顶点枚举多点路径
（commit 35ca490）。CPU 和 GPU 现在都走顶点枚举，行为对称。

**需要做的是文档修正，不是代码实现。**

## 实测验证

```python
# physics/gjk_epa.py: ground_contact_query, BoxShape flat
shape = BoxShape((0.1, 0.1, 0.1))
pose = SpatialTransform(R=np.eye(3), r=np.array([0.0, 0.0, 0.04]))
m = ground_contact_query(shape, pose, ground_z=0.0)
# → 4 points, depths=[0.01, 0.01, 0.01, 0.01]  ✅

# env0 layout (3-robot fixture), CPU step:
# body 1 (A.link1, has BoxShape at origin_xyz=[0,0,-0.04], body z=0.05):
#   shape world z = 0.01 → 4 ground contacts  ✅
# body 3 (B.link0, has BoxShape at origin_xyz=[0,0,+0.04], body z=0.05):
#   shape world z = 0.09 → box above ground, contact from SphereShape → 1 pt (correct)
```

env0 layout での CPU ground contacts: 9 total, body 1 → 4 pts (box), others → 1 pt (sphere/capsule).
GPU との差異は **ゼロ**（同じ vertex enumeration アルゴリズム）。

## 変更内容（3 箇所、コード変更なし）

### 1. OPEN_QUESTIONS.md — Q46 dim 6 を ✅ に更新

```
6. 🔄 **CPU vs GPU 一致性**（2026-04-20 差異已显式记录，manifold 尚未统一）：
   GPU 顶点枚举返回最多 4 个 box-ground 接触点；CPU GJK/EPA 返回 1 个。
```
→
```
6. ✅ **CPU vs GPU box-ground 一致性**（2026-04-21 关闭）：
   CPU 和 GPU 均走顶点枚举多点路径（session 33 实现，commit 35ca490）。
   Q46 dim 6 原描述"CPU 单点"已过时——session 33 的 ground_contact_query()
   对 Box/ConvexHull 已实现顶点枚举，与 GPU box_ground_manifold 行为对称。
```

### 2. test_b5_d6d7_cpugpu_multienv.py — module docstring 更新

删除以下过时内容（第 18–26 行）：
```
CPU vs GPU box-ground manifold difference (Q46 dim 6 — known, not a bug):
    GPU uses vertex enumeration (analytical_collision.py:box_ground_manifold):
        up to 4 deepest vertices per box shape → multi-point manifold.
    CPU uses GJK/EPA → single support point per box shape.
    ...
    CPU multi-point box-ground is deferred to a future thread (Q46 dim 6).
```
替换为：
```
CPU vs GPU box-ground manifold (Q46 dim 6 — resolved, session 33):
    Both CPU and GPU use vertex enumeration for Box/ConvexHull ground contact.
    CPU: ground_contact_query() → contact_vertices() → all penetrating vertices.
    GPU: box_ground_manifold() → 8-vertex loop → up to 4 deepest.
    Contact counts agree for the same geometry and pose.
```

### 3. test_cpu_gpu_sorted_ground_contact_match — 加强断言

当前（第 296–303 行）：
```python
# GPU produces more contacts than CPU per box (multi-point vs single-point).
assert len(gpu_ground) >= len(cpu_ground), ...
```
替换为：
```python
# CPU and GPU both use vertex enumeration; counts should agree.
assert len(cpu_ground) == len(gpu_ground), (
    f"CPU/GPU ground contact count mismatch: CPU={len(cpu_ground)}, GPU={len(gpu_ground)}"
)
```

同时删除第 351–354 行的过时注释：
```python
# Note: contact point XY is NOT compared because CPU (GJK/EPA) and
# GPU (analytical) use different algorithms for box/capsule ground
# contact points.
```
替换为：
```python
# Contact point XY is not compared: CPU and GPU use the same vertex
# enumeration algorithm but float32 vs float64 causes sub-mm differences.
```

## 风险

- `assert len(cpu_ground) == len(gpu_ground)` 在 GPU 不可用时会被 skip（整个文件
  有 `pytest.mark.skipif(not HAS_WARP, ...)`），不影响 CPU-only CI。
- env0 layout 中 body 3 的 BoxShape 在地面以上（z=0.09），接触来自 SphereShape，
  所以 body 3 仍然是 1 个接触点——这是正确的物理行为，不是 bug。
  加强后的断言不会因此失败，因为 GPU 也会得到同样的结果。

## 关键思考

### 为什么 Q46 dim 6 的描述会过时

Session 33 实现 CPU 多点时，修改了 `ground_contact_query()` 和
`halfspace_convex_query()`，但没有同步更新：
1. OPEN_QUESTIONS.md Q46 dim 6 的状态
2. test_b5_d6d7_cpugpu_multienv.py 的 docstring（该文件在 session 33 之前写的）

这是典型的"实现先于文档"问题。Session 33 的 commit message 提到了 Q48.2，
但没有提到 Q46 dim 6，导致 Q46 dim 6 的关闭被遗漏。

### 为什么 env0 layout 中 body 3 只有 1 个接触点

B.link0 有两个 shape：
- BoxShape at origin_xyz=[0,0,+0.04]：body z=0.05 → shape z=0.09 → 底面 z=0.05，不穿透
- SphereShape at origin_xyz=[0,0,-0.04]：body z=0.05 → shape z=0.01 → 底部 z=-0.04，穿透 0.04m

接触来自 SphereShape，单点是正确的。这个 layout 不是测试 box-ground 多点的好 fixture，
但它不影响 Q46 dim 6 的关闭——独立的 `test_box_flat_contact_count` 已经验证了 4 点。

### 加强断言的前提

`assert len(cpu_ground) == len(gpu_ground)` 成立的前提：
- 两者都用顶点枚举（已验证）
- 相同的 depth > 1e-10 阈值（CPU: `cpu_engine.py:119`，GPU: 需要确认）
- float32 vs float64 不影响"哪些顶点穿透"的判断（穿透深度 >> 精度差）

如果 GPU 的阈值不同，断言可能在边界情况下失败。建议 Codex 确认 GPU 的
contact depth 过滤阈值与 CPU 的 `1e-10` 是否一致。
