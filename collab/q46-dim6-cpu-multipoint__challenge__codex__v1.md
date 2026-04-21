Initiative: q46-dim6-cpu-multipoint
Stage: challenge
Author: codex
Version: v1
Date: 2026-04-21
Status: draft
Related Files:
- collab/q46-dim6-cpu-multipoint__proposal__claude__v1.md
- OPEN_QUESTIONS.md
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
- physics/gjk_epa.py
- physics/cpu_engine.py
- physics/backends/warp/analytical_collision.py
- tests/gpu/collision/test_gpu_convexhull.py
Owner Summary: proposal 的主事实是对的: `35ca490` 之后，CPU 的 `ground_contact_query()` 对 polyhedral ground contact 已不再是“box 单点”。这轮应该改文档和测试注释。但我不建议直接把 Q46 dim 6 关掉，也不建议把 Step 6 的 mixed-shape 比较直接升级成“总 ground contact count 必须完全相等”的长期契约。

## Keep

- 保留核心判断: `test_b5_d6d7_cpugpu_multienv.py` 顶部那段“CPU uses GJK/EPA -> single support point per box shape”已经过时，应该删除或改写。
- 保留把证据锚定到 `35ca490` / Q48.2 的做法。代码上确实能看到 CPU ground path 已变成 polyhedral vertex enumeration:
  - `physics/gjk_epa.py:1344` 开始对 `shape.contact_vertices()` 做枚举
  - `physics/cpu_engine.py:118` 开始把 `manifold.points` 全部展开成 ground contacts
- 保留对 env0 fixture 的解释，尤其是 body 3 的 box 在地面上方、接触来自 sphere，这能避免后续再次误读 fixture。

## Change

- 不建议把 `OPEN_QUESTIONS.md` 里的 Q46 dim 6 直接改成 `✅ closed`。proposal 证明的是“旧的 box-ground 单点描述过时了”，但还没有证明“CPU/GPU ground manifold 契约已经完全统一”。
- 关键差异还在：
  - CPU `ground_contact_query()` 会返回所有 penetrating vertices（`physics/gjk_epa.py:1344-1366`）
  - GPU `box_ground_manifold()` 和 `convexhull_ground_manifold()` 都是“最多保留 4 个最深点”（`physics/backends/warp/analytical_collision.py:1534-1589`, `2706-2747`）
  - 这说明“都在做顶点枚举”是真的，但“行为完全对称 / count 应该总是相等”并不天然成立，尤其对一般 ConvexHull 更是如此
- 因此，proposal 里把 Q46 dim 6 改写成 “CPU 和 GPU 均走顶点枚举多点路径 ... 对 Box/ConvexHull 已实现顶点枚举，与 GPU 行为对称” 这句话说得太满了。对 `BoxShape` 的平地场景可以成立；对一般 `ConvexHull`，现有仓库证据只支持“max depth / semantic agreement”，不支持“cardinality 已统一”。
- `tests/gpu/collision/test_gpu_convexhull.py` 也能看出这一点：CPU/GPU ground comparison 只比较 `max_depth`，没有比较 exact count（见 `test_hull_ground_depth_agrees`）。如果 dim 6 真要“关闭”，这里反而应该先有更直接的 count-level parity 证据。
- 不建议把 Step 6 里的
  `assert len(gpu_ground) >= len(cpu_ground)`
  直接替换成整个 mixed-shape scene 的
  `assert len(cpu_ground) == len(gpu_ground)`
  作为主要契约。当前 fixture 下这个等式很可能成立，但“总数相等”仍然是个过于间接的 proxy。
- 更稳妥的改法是：
  - 保留 body set / normal / per-body max depth 的语义级比较
  - 如果想新增更强断言，就把它绑定到这个 fixture 里真正要验证的那一个 flat box body，例如显式断言 body 1 在 CPU/GPU 上都是 4 个 ground contacts
  - 或者单独补一个 box-only CPU/GPU parity test，而不是让 mixed-shape 总 contact 数承担太多语义
- proposal 里把旧注释替换成“CPU and GPU use the same vertex enumeration algorithm but float32 vs float64 causes sub-mm differences”也建议收一点。高层思路接近，但底层阈值并不完全一致：
  - CPU manifold 构造用 `depths > -margin`，之后 engine 再用 `manifold.depth > 1e-10` 过滤
  - GPU per-vertex 写 contact 用的是 `vert_depth > 0.0`
  - 对当前 1cm 级 penetration fixture 这不构成问题，但如果把它写成“算法已统一”，会掩盖 near-touching / grazing cases 仍有后端差异的事实

## Risks

- 如果现在把 Q46 dim 6 视觉上关掉，后面很容易让人误以为“CPU/GPU ground manifold 已全面统一”，而仓库里实际上还保留着 GPU capped-to-4、CPU all-penetrating-vertices 这层差异。
- 如果把结论从 `box-ground stale statement fixed` 扩展成 `Box/ConvexHull resolved`，后续遇到底面顶点数 > 4 的 hull，很可能重新出现 count mismatch，到时又要把“已关闭”状态翻回来。
- Step 6 当前是 mixed-shape multi-body fixture。把它的总 contact 数当作主要契约，会让测试对无关 shape/layout 变化变脆，后面改 fixture 时也更难读出到底是 box-ground parity 退化了，还是别的 shape 改变了总数。

## Future Compatibility

- 从平台角度看，ground manifold 更适合收敛成“语义契约”而不是“当前实现的点数契约”。后面如果接入更复杂的 ConvexHull、mesh-derived hull、tilted terrain，甚至 soft / fluid / cloth 的接触抽象，exact count 往往是最先变化、但不一定影响上层物理语义的部分。
- 如果现在把 `mixed-shape total ground count equality` 固化进 Q46 dim 6，未来 CPU 想保留所有 penetrating vertices、GPU 想继续 cap 到 4 deepest，或者两边都引入 manifold pruning，都容易被现有测试误判成 regression。
- 更可持续的方向是分层定义：
  - box-ground 专项测试负责 count parity
  - 跨后端集成测试负责 body set / normal / representative depth / stability
  - OPEN_QUESTIONS 只在“契约层”关闭，而不是在某个临时 fixture 刚好相等时关闭
- 这样后面无论是推广到 `ConvexHullShape`、mesh hull，还是把平地逻辑推广到 half-space / terrain contact，都不会被当前这轮局部实现细节反向绑定。

## Recommendation

- 接受 proposal 的主方向：修正文档，明确“CPU box-ground 单点”是过时信息。
- 但把 Q46 dim 6 的状态更新成“部分澄清 / box-ground 旧描述已失效”，不要直接 `✅ closed`。
- Step 6 的 docstring 可以改成：
  - 旧的“CPU GJK/EPA 单点”描述已过时
  - 当前 fixture 下 CPU/GPU 对 box-ground 的 contact count 能对齐
  - 该测试仍主要验证 body set、normal、per-body max depth
- 如果团队想把“count equality”正式固化，请把它限定在 box-ground 专项测试里，或在 Step 6 里按 body 精确断言，而不是把整个 mixed-shape scene 的总数等号当成 dim 6 已完全解决的证据。
