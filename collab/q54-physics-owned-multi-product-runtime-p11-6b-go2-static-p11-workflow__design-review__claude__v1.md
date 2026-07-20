Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: design-review
Author: Claude
Version: v1
Date: 2026-07-20
Status: review_complete
Related Files: tools/optical_pipeline_lab/go2_backend.py, tools/optical_pipeline_lab/menagerie_static_runner.py, tools/optical_pipeline_lab/preset_workflows.py, tools/optical_pipeline_lab/preset_runtime.py, tools/optical_pipeline_lab/preset_products.py, tools/optical_pipeline_lab/product_workflow.py, tools/optical_pipeline_lab/runner.py, tools/optical_pipeline_lab/frame_products.py, examples/optical_lab/README.md, tests/unit/optics/test_optical_pipeline_lab.py
Parent Document: collab/q54-physics-owned-multi-product-runtime-p11-6b-go2-static-p11-workflow__design-request__codex__v1.md

# Q54 P11.6b Design Review: Critical Architecture Issues

## Summary

The P11.6b plan to delete `go2_backend.py` through a static P11 workflow is **directionally correct** but has **four critical architecture issues** that must be resolved before implementation. The plan correctly identifies the deletion order (implement workflow first, then delete shim) and test strategy, but the proposed internal design introduces:

1. Runtime protocol mismatch (physics vs static frame providers)
2. Unnecessary 200-line product duplication
3. Runtime dispatch complexity in product specs
4. Inconsistent API surface (different parameter requirements per preset type)

These issues will compound when Phase 3 multi-physics support arrives. Recommend pausing P11.6b-2 implementation until architecture is corrected.

## Critical Issues

### Issue 1: Runtime Protocol Mismatch (P0)

**Problem**: `StaticAssetLabRuntime.step_tick()` returns `SimulationFrameTick` with a static `published_frame`, but `PhysicsVideoFrameProduct` assumes the frame provider needs a real physics `published_frame`.

**Evidence** (from current code):

```python
# Proposed StaticAssetLabRuntime.step_tick() (design doc line 214-226)
def step_tick(self, frame_index: int, *, env_idx: int = 0) -> SimulationFrameTick:
    return SimulationFrameTick(
        ...
        published_frame=self.base_frame,  # ⚠️ Same base_frame every tick
        ...
    )

# PhysicsVideoFrameProduct.consume() (runner.py:415-419)
with self.frame_provider.begin_frame(
    tick.frame_index,
    env_idx=tick.env_idx,
    published_frame=tick.published_frame,  # ⚠️ Passed to physics provider
) as frame_context:
```

**Root cause**: The design says static should use `static_frame_context_provider(runtime.pipeline)` which does NOT need `published_frame` (line 295), but `PhysicsVideoFrameProduct` hardcodes passing it.

**Impact**:
- Protocol incompatibility between physics and static frame providers
- `SimulationFrameTick` protocol becomes ambiguous (can `published_frame` be None?)
- Static provider must either ignore the parameter or fail

**Why this matters**: The `FrameProduct` protocol is generic, but `PhysicsVideoFrameProduct`'s implementation is physics-specific. This violates the abstraction needed for multi-runtime support.

### Issue 2: Product Duplication vs Abstraction Trap (P0)

**Problem**: The plan proposes duplicating `PhysicsVideoFrameProduct` → `StaticVideoFrameProduct` (lines 273-303) to handle static scenes, citing "clear lifecycle boundary." But video product core logic is identical:

- `begin_run()` / `end_run()` — initialize/flush delivery
- `consume(tick)` — render + submit + record CSV
- **Only difference**: frame provider invocation

**Evidence**: `PhysicsVideoFrameProduct` is ~80 lines (runner.py:397-476), but references:
- `_render_video_frame()` — delegates to `video_loop.build_video_render_plan()`
- `_submit_video()` — delivery facade submission
- `_record_delivered_video()` — CSV row writing via `video_loop.record_delivered_video_frame()`

All of these are **generic** and work for any frame context.

**Impact**:
- Duplicating 200+ lines of video logic (including render plan, delivery, CSV)
- Future maintenance burden: any video product bug fix must be applied twice
- Violates DRY principle for no architectural benefit

**Why this matters**: Phase 3 will add deformable/fluid runtimes. Duplicating video product for each runtime type is not sustainable.

### Issue 3: Factory Dispatch Complexity (P1)

**Problem**: The plan adds runtime dispatch to `VideoProductSpec.build()` (lines 308-314):

```python
if is_physics_published_frame_source(context.config.frame_source):
    return build_physics_video_frame_product(...)
if context.config.frame_source is FrameSourceKind.STATIC_ASSET_BUILDER:
    return build_static_video_frame_product(...)
raise ValueError(...)
```

**Root cause**: Product spec becomes a "dynamic dispatcher" instead of a configuration dataclass.

**Impact**:
- `VideoProductSpec` now has runtime logic, not just configuration
- Violates P11 design: preset registration should associate the correct product spec factory directly
- Every new runtime type (deformable, fluid) requires modifying the central `build()` dispatch

**Why this matters**: This makes `VideoProductSpec` stateful and breaks the declarative registration pattern.

### Issue 4: ArtifactOutput Parameter Confusion (P1)

**Problem**: The plan requires `ArtifactOutput` to carry `model_dir/model_xml` for static runtime construction (lines 241-263). This creates inconsistent API:

```python
# Physics preset: output is optional, only for artifact paths
run_optical_lab_preset(preset="physics_body_triangle_video_smoke", frames=120, products=("video",))

# Static preset: output is required, carries scene config + artifact paths
run_optical_lab_preset(preset="go2_video_ordered_static", frames=120, products=("video",),
    output=ArtifactOutput(root=..., model_dir=..., model_xml=...))
```

**Root cause**: `ArtifactOutput` has dual responsibilities:
- For physics: output paths only
- For static: output paths + scene construction parameters

**Impact**:
- API inconsistency across preset types
- Violates "preset" semantics: Go2 Menagerie's `model_dir/model_xml` should be **part of the preset definition**, not runtime parameters
- Users must know preset implementation details to call the API correctly

**Why this matters**: This leaks abstraction. A "preset" should encapsulate all scene configuration. Users should only specify outputs and frame count.

## Recommended Architecture Corrections

### Solution A: Frame Provider Abstraction Layer (Recommended for Issue 1, 2)

**Do not duplicate video product.** Instead, unify the frame provider protocol:

```python
# Unified protocol
class FrameContextProvider(Protocol):
    def begin_frame(self, frame_index: int, *, env_idx: int, **kwargs) -> FrameContext:
        """Static provider ignores kwargs; physics provider uses published_frame."""

# Physics provider
class PhysicsFrameContextProvider:
    def begin_frame(self, frame_index: int, *, env_idx: int, published_frame, **kwargs):
        # Use published_frame for physics-backed frame context

# Static provider
class StaticFrameContextProvider:
    def begin_frame(self, frame_index: int, *, env_idx: int, **kwargs):
        # Ignore published_frame, use static pipeline.session
```

**Changes**:
- Rename `PhysicsVideoFrameProduct` → `VideoFrameProduct` (generic)
- Pass `published_frame` as `**tick_kwargs` so static can ignore it
- Frame provider becomes the injection point for runtime differences

**Benefits**:
- No 200-line duplication
- Video product logic stays unified
- Follows dependency inversion principle
- Scales to deformable/fluid runtimes (just add new provider types)

**Implementation location**: `tools/optical_pipeline_lab/frame_providers.py` (new module)

### Solution B: Preset Carries Scene Configuration (Recommended for Issue 4)

**Do not pass `model_dir/model_xml` in `ArtifactOutput`.** Instead, define them in the preset registry:

```python
# In preset registry
_STATIC_SCENE_CONFIGS = {
    "go2_video_ordered_static": {
        "scene_preset": "go2_menagerie_static",
        "model_dir": "out/external/mujoco_menagerie/unitree_go2",
        "model_xml": "go2.xml",
    },
}

# User API becomes consistent
run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video",),
    out=Path("runs/p11/go2"),  # Only output path, just like physics presets
)

# Advanced users can override via runtime_kwargs
run_optical_lab_preset(
    preset="go2_video_ordered_static",
    frames=120,
    products=("video",),
    out=Path("runs/p11/go2"),
    runtime_kwargs={"model_dir": "custom/path"},  # Optional override
)
```

**Benefits**:
- API consistency: all presets only require `out` parameter
- Scene config becomes part of preset definition (matches "preset" semantics)
- Advanced overrides available through explicit `runtime_kwargs`

**Implementation location**: `tools/optical_pipeline_lab/preset_runtime.py` (extend registry)

### Solution C: VideoProductSpec Carries Provider Factory (Recommended for Issue 3)

**Do not dispatch in `build()`.** Instead, let spec carry the correct provider factory:

```python
@dataclass(frozen=True)
class VideoProductSpec:
    build_video_camera: Callable
    synchronize_event: Callable
    pack_rgb8: Callable
    frame_provider_factory: Callable  # ← New field

# Physics preset
def create_physics_body_triangle_video_product_spec() -> VideoProductSpec:
    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
        frame_provider_factory=lambda runtime: physics_frame_context_provider(runtime),
    )

# Static preset
def create_go2_video_ordered_static_product_spec() -> VideoProductSpec:
    return VideoProductSpec(
        build_video_camera=build_lab_video_camera,
        synchronize_event=synchronize_ready_event,
        pack_rgb8=pack_video_rgb8,
        frame_provider_factory=lambda runtime: static_frame_context_provider(runtime.pipeline),
    )
```

**Changes**:
- Remove `if is_physics_published_frame_source()` dispatch from `build()`
- Each preset explicitly declares its provider in the spec
- `VideoProductSpec.build()` becomes pure assembly: `return VideoFrameProduct(..., frame_provider=self.frame_provider_factory(runtime))`

**Benefits**:
- Spec remains declarative, not imperative
- No central dispatch point to maintain
- Clear ownership: preset → spec → provider

**Implementation location**: `tools/optical_pipeline_lab/product_specs.py` (add field)

## Implementation Priority

**Must resolve before P11.6b-2 implementation** (in order):

1. **Frame Provider Protocol** (Issue 1, 2) — P0
   - Create `frame_providers.py` with unified protocol
   - Extract physics/static provider implementations
   - Generalize `VideoFrameProduct` to accept provider factory

2. **Preset Scene Configuration** (Issue 4) — P1
   - Move `model_dir/model_xml` from runtime args to preset registry
   - Unify `run_optical_lab_preset()` API surface

3. **Product Spec Provider Injection** (Issue 3) — P1
   - Add `frame_provider_factory` to `VideoProductSpec`
   - Remove dispatch logic from `build()`

4. **Documentation Update** — P1
   - Update design doc with corrected architecture
   - Document frame provider protocol contract

## What the Original Plan Got Right

These aspects should be preserved:

1. ✅ **Deletion order**: Implement P11 workflow first, then delete shim
2. ✅ **Test strategy**: Unit tests + real GPU smoke
3. ✅ **Keep `menagerie_static_runner.py`** as legacy CLI module
4. ✅ **`ProductRunResult` alias**: Low-risk rename, introduce now
5. ✅ **Implementation slices**: 4-step plan is reasonable once design is corrected

## Risk Assessment

**If P11.6b proceeds with current design**:

- **Short-term**: 200 lines of duplicated video product code
- **Medium-term**: Every video product bug requires dual fixes
- **Long-term**: Phase 3 multi-physics support requires N×200 line duplication or costly refactor

**If architecture is corrected first**:

- **Short-term**: 1-2 day design iteration delay
- **Medium-term**: Clean abstraction supports Phase 3 without refactor
- **Long-term**: Video product logic remains unified across all runtime types

## Recommendation

**Pause P11.6b-2 implementation** until architecture issues are resolved.

**Proposed revised sequence**:

1. **P11.6b-design-v2**: Update design doc with corrected architecture (Solutions A, B, C)
2. **P11.6b-provider**: Implement frame provider abstraction layer
3. **P11.6b-preset-config**: Move scene config to preset registry
4. **P11.6b-static-workflow**: Implement static P11 workflow (original P11.6b-2)
5. **P11.6b-delete-shim**: Delete `go2_backend.py` (original P11.6b-3)
6. **P11.6b-smoke**: Real Go2 render smoke (original P11.6b-4)

**Estimated additional timeline**: +2 days for design iteration and provider abstraction.

**Avoided technical debt**: ~500 lines of duplicate code + brittle dispatch logic.

## Review Questions Answered

From the original design doc:

1. **Q1: Should the public API add `output_options={...}`, or require `output=ArtifactOutput(...)` for advanced static options?**
   - **Answer**: Neither. Use Solution B (preset carries scene config). API should be consistent: `out=Path(...)` for all presets. Advanced overrides via `runtime_kwargs`.

2. **Q2: Should `ProductRunResult` be introduced now as the generic alias?**
   - **Answer**: Yes. This is low-risk and correct. Introduce the alias now.

3. **Q3: Should static runtime construction live in `preset_runtime.py`, or in a new `static_runtime.py`?**
   - **Answer**: Create new `frame_providers.py` for the abstraction layer. Runtime construction stays in `preset_runtime.py`, but delegates to provider factories.

4. **Q4: Is duplicating a small `StaticVideoFrameProduct` acceptable?**
   - **Answer**: No. Use Solution A (frame provider abstraction). Duplication is not small (~200 lines including referenced helpers) and does not scale.

5. **Q5: Should the real Go2 smoke use `readback_payload='none'` for speed, or keep RGB readback?**
   - **Answer**: Keep RGB readback for end-to-end delivery verification. Speed is not the concern for a smoke test.

## 关键思考

### 为什么 "clear lifecycle boundary" 不足以证明代码复制的合理性

Design doc 第 301 行说："This is small duplication but gives a clear lifecycle boundary."

这是一个**常见的过度设计陷阱**：

1. **"清晰边界"不等于"独立实现"**：
   - 清晰边界应该通过**抽象协议**实现（如 frame provider protocol）
   - 不是通过复制 200 行代码实现

2. **"小复制"会指数增长**：
   - Phase 3 会有 deformable/fluid runtimes
   - 如果每个 runtime 复制一份 video product → 4 份 × 200 行 = 800 行重复

3. **Video product 的核心职责是通用的**：
   - Render → Delivery → CSV recording
   - 这些逻辑与 runtime 类型无关
   - 唯一的差异是"如何获取 frame context" → 这正是 provider 抽象的职责

**教训**：当发现自己在说"这是小复制"时，应该先问：
- 复制的代码是否有**不同的变化原因**？
- 如果未来增加相似场景，复制会线性增长吗？
- 差异能否通过**策略注入**解决？

本案例中，答案是：video product 的变化原因是统一的（video delivery 逻辑），差异可以通过 provider 注入解决，所以复制是错误的。

### 为什么 preset 应该携带场景配置而不是 runtime 参数

当前设计让 `model_dir/model_xml` 成为 `ArtifactOutput` 参数，这违反了"preset"的语义：

**Preset 的本质是什么**：
- 一组**预先配置好的**场景、渲染、输出参数
- 用户只需选择 preset 名称 + 运行参数（frames, device）
- 不需要了解 preset 的内部实现细节

**如果场景配置是 runtime 参数**：
- 用户必须知道 Go2 preset 需要 MuJoCo Menagerie 路径
- 用户必须知道路径的正确值（`out/external/mujoco_menagerie/unitree_go2`）
- 这些是**实现细节**，不应该暴露给 preset 用户

**类比**：
- 错误设计：`run_preset("go2_static", model_dir=..., model_xml=...)`
- 正确设计：`run_preset("go2_static")` — preset 已经知道自己的场景在哪
- 高级覆盖：`run_preset("go2_static", runtime_kwargs={"model_dir": "custom/"})` — 显式覆盖

这也是为什么物理 preset 不需要传递物理引擎参数 — preset 已经封装了这些细节。

### 为什么 ProductSpec 不应该做运行时分发

当前设计在 `VideoProductSpec.build()` 里加入：

```python
if is_physics_published_frame_source(context.config.frame_source):
    return build_physics_video_frame_product(...)
if context.config.frame_source is FrameSourceKind.STATIC_ASSET_BUILDER:
    return build_static_video_frame_product(...)
```

这违反了**开闭原则**（Open-Closed Principle）：

1. **每增加一个 runtime 类型，就要修改 `build()` 方法**
   - Phase 3 加 deformable → 修改 `build()`
   - Phase 4 加 fluid → 修改 `build()`
   - 中心分发点成为维护瓶颈

2. **Spec 从配置变成了逻辑**
   - 原本 `VideoProductSpec` 是纯数据（camera builder, pack_rgb8）
   - 现在它包含运行时决策逻辑
   - 违反了 data vs behavior 的分离

3. **Preset 注册失去了意义**
   - 如果 `build()` 根据 frame source 分发，那为什么还要注册不同的 preset？
   - Preset 应该直接关联正确的 spec factory，不是注册一个"会自己判断"的 spec

**正确设计**：
- Physics preset 注册 → `VideoProductSpec(frame_provider_factory=physics_provider)`
- Static preset 注册 → `VideoProductSpec(frame_provider_factory=static_provider)`
- `build()` 只做组装：`VideoFrameProduct(provider=self.frame_provider_factory(runtime))`

这样 `VideoProductSpec` 保持纯数据，扩展性通过注册新 preset 实现，不需要修改中心代码。

## Next Steps

1. **Codex decision**: Approve revised architecture or propose alternatives
2. **If approved**: Claude updates design doc (P11.6b-design-v2) with Solutions A, B, C
3. **Implementation**: Follow revised 6-step sequence (not original 4-step)

---

**End of review.**
