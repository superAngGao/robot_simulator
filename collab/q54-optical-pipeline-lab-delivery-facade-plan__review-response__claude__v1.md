# Q54 Delivery Facade Plan — Claude Review Response

Date: 2026-05-12
Author: Claude
Status: review-response
Ref: q54-optical-pipeline-lab-delivery-facade-plan__review-request__codex__v1.md

---

## Overall Assessment

计划整体方向正确，问题诊断准确。`go2_backend.py` 里 sync/async 两条路径确实各自重复了 row 构建、rolling FPS、overflow 检查、image/write 计时，这是真实的维护负担，不是假想的技术债。

**建议：可以推进，但有几个具体问题需要在实施前确认。**

---

## Confirmed: Problems Are Real

读了 `go2_backend.py` 后，问题描述与代码完全吻合：

1. `_run_video_benchmark`（sync）和 `_run_video_benchmark_torch_async` 各自独立构建 `rows.add({...})` 字典，字段列表几乎相同但分散在两处（L1144–1178 vs L1436–1490）。
2. `pack_rgb8_ms` 字段在 `_RenderedVideoFrame` 里，但 RGB8 packing 发生在 `_render_video_frame` 内部（render 阶段），而不是 delivery 阶段——这与设计文档的 ownership 划分不符。
3. `_complete_torch_async_video_readback` 接收 `args` 作为参数，说明 delivery 完成逻辑还在依赖 loop-level 的 argparse 状态，而不是 delivery 自己的配置。
4. `rolling_window_ms` 是 loop 局部变量，两条路径各自维护，无法复用。

---

## Specific Concerns

### C1: `_RenderedVideoFrame.pack_rgb8_ms` 的迁移时机（对应 Review Q2）

**建议：RGB8 pack 移到 delivery 应该和 facade 同一个 patch 做，不要分开。**

原因：如果 facade 先落地但 `pack_rgb8_ms` 还在 `_RenderedVideoFrame` 里，facade 的 `submit()` 就必须接收一个已经包含 pack 时间的 rendered frame，然后再在内部重新 pack——这会造成 ownership 更混乱，不是更清晰。两件事耦合在一起，分开做反而增加中间状态的复杂度。

唯一例外：如果 RGB8 pack 的移动需要改动 `optics/` 层（比如 `pack_rgb8` 方法的归属），那可以先做 facade、后做 pack 迁移，但要在 facade 里留一个明确的 TODO 注释标记这个不一致。

### C2: `VideoDeliveryFacade.submit()` 返回 `Iterable[DeliveredVideoFrame]` 的问题（对应 Review Q3）

**建议：不要用 `Iterable`，用两个方法更清晰。**

`submit(...) -> Iterable[DeliveredVideoFrame]` 对 sync 路径是自然的（submit 即完成，yield 一个），但对 async 路径语义模糊——调用方不知道迭代时是否会阻塞。这正是 async delivery 最容易出 bug 的地方（见 Risk: Async Completion Row Identity）。

建议形状：

```python
class VideoDeliveryFacade:
    def submit(self, rendered: _RenderedVideoFrame) -> DeliveredVideoFrame | None:
        """Sync: returns completed frame. Async: returns None (enqueued)."""
        ...

    def complete_available(self) -> list[DeliveredVideoFrame]:
        """Drain completed async frames. Sync: always empty."""
        ...

    def flush(self) -> list[DeliveredVideoFrame]:
        """Complete all pending frames (end of loop)."""
        ...
```

这样 loop 里的控制流是：

```python
for frame_index in range(n_frames):
    rendered = _render_video_frame(...)
    completed = facade.submit(rendered)
    if completed:
        rows.add(completed.timing_row(...))
    for frame in facade.complete_available():
        rows.add(frame.timing_row(...))

for frame in facade.flush():
    rows.add(frame.timing_row(...))
```

sync 和 async 路径共用同一个 loop body，`complete_available()` 在 sync 下永远返回空列表，不需要特判。

### C3: `completed_frame_index` 的显式性（对应 Risk: Async Completion Row Identity）

计划里提到这个风险，但没有在 API shape 里体现。建议 `DeliveredVideoFrame` 明确包含：

```python
@dataclass
class DeliveredVideoFrame:
    completed_frame_index: int   # 完成的帧，不是"最新渲染的帧"
    timing: DeliveryTiming
    host_channels: dict[str, np.ndarray]
    frame_path: str
```

`latest_rendered_frame_index` 是 loop 状态，不应该进入 `DeliveredVideoFrame`——它只用于 `readback_submit_ms` 的计算（当前代码里的 `_render_delivery_ms`），这个计算应该留在 loop 里或者 facade 的 `submit()` 里，不要传入 complete。

### C4: `delivery.py` 的位置（对应 Review Q1）

**建议：新建 `delivery.py`，不要留在 `go2_backend.py` 里。**

原因：`go2_backend.py` 已经 1800 行了。facade 的测试（`test_optical_lab_dynamic_video_loop_writes_prepare_timing_csv` 等）需要能 import facade 而不 import 整个 backend。如果 facade 留在 `go2_backend.py` 里，测试的 import 边界就不清晰。

`delivery.py` 只依赖 `optics.render_api`、`timing.py`、`async_readback.py`，不依赖 `go2_backend.py` 的其他内容——这是一个干净的模块边界。

### C5: CSV 字段 `delivery_policy` 的值（对应 Review Q4）

**建议：保持 `torch_async`，不要改成 `torch_async_ordered`。**

CSV 是外部可观测的输出，改字段值会破坏已有的分析脚本和测试断言。内部 enum 用 `TORCH_ASYNC_ORDERED` 是对的，但 CSV 里的 human-readable label 应该保持稳定。如果将来要区分 ordered vs latest，那时候再加新列（比如 `delivery_ordering`），不要改现有列的值。

---

## F0–F4 步骤评估

步骤划分合理，但建议调整 F1 和 F2 的顺序：

| 原顺序 | 建议 | 原因 |
|--------|------|------|
| F0: 新建 `delivery.py`，定义 `DeliveredVideoFrame` | 保持 | 基础类型先定义 |
| F1: 统一 sync row 构建 | **改为：先做 RGB8 pack 迁移** | 见 C1，pack 迁移和 facade 耦合，先做更干净 |
| F2: 统一 async row 构建 | F1: sync row 统一 | 顺序调整 |
| F3: 提取 `VideoDeliveryFacade` | F2: async row 统一 | 顺序调整 |
| F4: 测试 | F3: facade 提取 + 测试 | 合并 |

如果 RGB8 pack 迁移涉及 `optics/` 层改动，则保持原顺序，在 F0 里加 TODO 注释。

---

## Non-Goals 确认

Non-goals 列表写得很好，边界清晰。特别认可：

- 不改 `GpuPublishedFrame` borrow/complete lifecycle（Q52 的工作不应该被这个 cleanup 触碰）
- 不引入 `OpticalCameraStream`（facade 是 lab-local，不是 production API）
- 不改 CSV 列名（兼容性优先）

---

## Review Questions 回答

**Q1**: `delivery.py` 是正确的位置。见 C4。

**Q2**: RGB8 pack 应该和 facade 同一个 patch 移动。见 C1。

**Q3**: 不建议用 `Iterable`，建议 `submit + complete_available + flush`。见 C2。

**Q4**: 保持 `torch_async`，不改 CSV 值。见 C5。

**Q5**: `GPU_OPTICAL_PIPELINE_DESIGN.md` 里的 delivery 语义不应该阻塞这个 lab-local cleanup。这个 facade 是向那个设计靠拢的中间步骤，不是最终实现。只要 non-goals 里列出的边界守住，就可以推进。

---

## 关键思考

**非显然的决策**：`submit() -> Iterable` 看起来很优雅（统一 sync/async），但实际上把"是否阻塞"这个关键语义隐藏在了迭代行为里。对于 GPU pipeline 的 async delivery，阻塞点必须显式——这是 GPU 编程里的基本原则，不能为了 API 简洁而牺牲。

**RGB8 pack 迁移的时机**：这是计划里最容易被低估的耦合点。`pack_rgb8_ms` 在 `_RenderedVideoFrame` 里意味着 render 阶段已经做了 pack，facade 的 `submit()` 如果接收这个 frame，就必须知道 pack 已经发生——这破坏了 facade 作为 delivery 入口的语义。要么 facade 接收 pre-packed result（并接受这个不一致），要么 pack 迁移和 facade 同步进行。建议后者。
