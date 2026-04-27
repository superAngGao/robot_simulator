Initiative: render-physics-pipeline
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-04-25
Status: day-end-summary
Related Files: collab/render-physics-pipeline__proposal__codex__v1.md, OPEN_QUESTIONS.md#Q52, physics/publish.py, physics/engine.py, physics/cpu_engine.py, physics/gpu_engine.py, physics/telemetry.py, rendering/debug_exporter.py, rendering/scene_builder.py, rendering/published_frame_renderer.py, tests/unit/physics/test_publish.py, tests/unit/physics/test_cpu_publish_runtime.py, tests/unit/physics/test_telemetry_snapshot.py, tests/unit/rendering/test_debug_exporter.py, tests/unit/rendering/test_published_frame_bridge.py, tests/integration/test_published_frame_render_backend_integration.py
Owner Summary: 记录 2026-04-25 基于 2026-04-24 proposal/Q52 的第一阶段代码落地结果，供 Claude 做实现级 review。重点不是重复架构哲学，而是说明：哪些控制平面类型已经变成代码、CPU/GPU engine 已经接上了哪些 publish/runtime 接口、哪些行为已有测试覆盖、当前还缺哪些关键实现。

## Day-End Summary (2026-04-26)

今天这轮收敛后，phase-1 consumer integration 可以视为完成了一个可提交、可审查的阶段版本。

### 已完成

- `physics/publish.py`
  - 已形成稳定的 publish/control-plane 骨架：
    - `PublishPolicy`
    - `PublishPlan`
    - `BorrowedFrameLease`
    - `SnapshotHandle`
    - `SlotReclaimer`
    - `GpuPublishedFrame` / `CpuPublishedFrame`
- `CpuEngine` / `GpuEngine`
  - 已接上统一的 published-frame surface
  - `frame_id` 语义、`skip` 语义、`on_ring_full` 语义已明确
- Debug / render / telemetry 三类真实 consumer 已落地
  - `PublishedFrame -> host debug snapshot`
  - `PublishedFrame -> RenderScene`
  - `PublishedFrame -> RenderScene -> MatplotlibBackend`
  - `PublishedFrame -> TelemetrySnapshot`
- review 收尾项已吸收
  - `LeaseExpiredError`
  - `SlotReclaimedError`
  - `SnapshotHandle.is_ready/frame_id`
  - CPU latest-only 限制文档化
  - GPU contact fallback 注释化
  - telemetry CPU/GPU 字段不对称文档化

### 当前测试状态

本阶段关键链路目前已验证：

- `29 passed`
- compileall 通过

覆盖范围包括：

- publish/control-plane 单测
- CPU publish runtime 单测
- debug exporter 单测
- published-frame -> RenderScene bridge 单测
- published-frame -> backend 端到端集成测试
- published-frame -> telemetry snapshot 单测

### 对应提交

- `c0d208f` — `Add published-frame debug export and render pipeline`
- `5b8e1d6` — `Add published-frame telemetry snapshot bridge`

### 当前边界

现在已经明确做到了：

- 不让 consumer 偷读 engine-private scratch
- 不把 telemetry schema 提前写死到 `RenderScene`
- CPU path 保持 reference runtime，不引入 ring
- GPU path 仍停留在 phase-1 synchronous publish / snapshot bridge

仍未开始的下一阶段内容：

- async host staging / export queue
- retained / realtime render view
- 更正式的 sensor path
- typed published slot / block dataclass

### 明天最自然的起点

如果继续实现，最合适的下一步不是再扩 phase-1，而是开始收敛 phase-2 的入口设计，二选一：

1. `sensor path`
   - 在不污染 `RenderScene` 的前提下，定义更正式的 sensor-facing view
2. `async host staging`
   - 把 `SnapshotHandle` 从同步桥升级成真正的 queue/event 模型

如果先做设计审查，这份文档加上：

- `collab/render-physics-pipeline__proposal__codex__v1.md`
- `OPEN_QUESTIONS.md#Q51`
- `OPEN_QUESTIONS.md#Q52`

就已经足够支撑明天继续讨论。

2026-04-25 review follow-up:

- 已吸收 Claude 第一轮评审中的三条低成本修改：
  - `PublishPolicy.publish_every_n_steps`
  - `LeaseExpiredError`
  - `SnapshotHandle`
- 已吸收 Claude 第二轮评审中的两条接口加固：
  - `PublishPolicy.on_ring_full`
  - `SlotReclaimedError` + `GpuPublishedFrame` reclaim guard
- 已吸收 Claude 第三轮评审中的两个低成本收尾：
  - 明确 `frame_id` 在 skipped publish 步骤中仍按 physics step 单调递增
  - `SnapshotHandle.is_ready` / `SnapshotHandle.frame_id`
- 另外两条建议当前处理方式是：
  - `PublishedSlotMeta.invalidated` 已加入最小标记位，但 typed slot/block dataclass 仍留待 phase-2
  - CPU path 继续保持“语义 reference，无 ring”定位

---

## Reference Baseline

本说明不是独立 proposal，而是建立在两份上游材料之上：

1. 昨天的设计与 review 共识：
   - `collab/render-physics-pipeline__proposal__codex__v1.md`
   - 尤其参考其中的：
     - `Layered Interface Draft`
     - `Published Ring Sizing And Consumer Modes`
     - `borrow / snapshot`
     - `Ack / Reclaim`
     - `Ring Pressure Behavior Matrix`
     - `Review Resolution Snapshot`

2. 今天挂入正式 backlog 的实施条目：
   - `OPEN_QUESTIONS.md#Q52`

这份文档只汇报“第一阶段已经真正变成代码的部分”，并请求 Claude 从实现边界、接口一致性、未来集成可行性角度 review。

---

## What Is Implemented

### 1. New control-plane module: `physics/publish.py`

新增模块：

- `physics/publish.py`

当前已落地类型：

- `ViewPolicy`
- `PublishPolicy`
- `PublishPlan`
- `PublishedFrameCore`
- `CpuPublishedFrame`
- `GpuPublishedFrame`
- `BorrowedFrameLease`
- `HostSnapshotSpec`
- `DeviceSnapshotSpec`
- `SnapshotHandle`
- `ConsumerState`
- `AckPolicy`
- `PublishedSlotMeta`
- `LeaseExpiredError`
- `SlotReclaimedError`
- `RingPressureStats`
- `SlotReclaimer`

这些类型已经把 proposal 里的几条核心语义变成代码：

1. QoS 与 access mode 正交：
   - `best_effort` / `lossless`
   - `borrow` / `snapshot`

2. `PublishPolicy -> PublishPlan` 的调度边界：
   - `PublishPlan.from_policy(frame_id, policy)`
   - `publish_every_n_steps`
   - `do_rigid_block_write`
   - `do_telemetry_block_write`
   - `frame_id` 采用 physics-step timeline 语义：即使某一步跳过 materialize，后续真正 publish 的 frame 也会带着更大的 `frame_id`

3. `BorrowedFrameLease` 是 context-managed ephemeral lease：
   - 支持 `with ... as frame:`
   - 离开作用域后失效
   - 支持 release callback
   - 失效后抛 `LeaseExpiredError`
   - 当前通过 proxy 方式代理属性/索引访问，离开作用域后的再次读取会尽早暴露

4. `SnapshotHandle`
   - phase-1 中同步持有结果
   - `.result()` 立即返回
   - `is_ready=True`
   - `frame_id` 已挂到 handle 上
   - 为 phase-2 异步 staging queue 保留前向兼容接口

5. `AckPolicy.default_for(...)`：
   - `best_effort -> none`
   - `lossless + borrow -> on_borrow_complete`
   - `lossless + snapshot -> on_snapshot_staged`

6. `SlotReclaimer`：
   - `min_lossless_acked_frame_id()`
   - `reclaimable(slot)`
   - `ring_pressure_stats(target_slot)`

当前 blocker attribution 语义是：

- 只有 `acked_frame_id < target_slot.frame_id` 的启用中 `lossless` consumer 会被视为 blocker

7. `PublishPolicy.on_ring_full`
   - `raise`：保持 phase-1 默认行为，遇到 lossless pin 直接报错
   - `skip`：跳过本次 publish，不 materialize 新 slot，但 physics step 仍完成
   - `block`：当前 phase-1 显式 `NotImplementedError`，留待 async staging / queue 完整落地后再支持

8. `GpuPublishedFrame` reclaim guard
   - `slot_meta.invalidated=True` 后，再通过 `frame.q_wp / frame.qdot_wp / ...` 访问 payload 会抛 `SlotReclaimedError`
   - 这是 phase-1 针对 GPU slot aliasing 风险的最小运行时保护

---

### 2. Physics package export surface updated

更新：

- `physics/__init__.py`

上述 publish/control-plane 类型已全部导出，后续上层模块可以直接从 `physics` 导入。

---

### 3. `PhysicsEngine` surface contract extended

更新：

- `physics/engine.py`

在抽象基类上增加了 publish/runtime 相关方法：

- `set_publish_policy(...)`
- `latest_published_frame()`
- `register_consumer(...)`
- `unregister_consumer(...)`
- `borrow_latest_frame(...)`
- `snapshot_frame_to_host(...)`

其中 `snapshot_frame_to_host(...)` 现在约定返回：

- `SnapshotHandle`

当前这些默认仍是 `NotImplementedError`，CPU/GPU 路径各自 override。

这一步的目标是：

- 先统一 engine surface
- 避免上层未来直接分叉成“CPU 一套、GPU 一套”的 publish API

---

## CPU Path Status

更新：

- `physics/cpu_engine.py`

### 已实现

1. `step()` 现在会自动更新 `latest_published_frame()`

也就是说，CPU path 已经和 GPU path 统一到：

- plain `step()` 之后就有 published frame

而不是必须额外记住调用 `step_and_publish()`。

2. `step_and_publish()` 仍保留

但现在本质上只是：

- 调 `step()`
- 返回 `latest_published_frame()`

3. 支持 publish/runtime 控制接口：

- `set_publish_policy(...)`
- `register_consumer(...)`
- `unregister_consumer(...)`
- `latest_published_frame()`
- `borrow_latest_frame(...)`
- `snapshot_frame_to_host(...)`

4. 现在也支持：

- `PublishPolicy.publish_every_n_steps`

因此 plain `step()` 不一定每次都 materialize 新 frame；是否真正 publish 由 policy 控制。

这里当前已经明确采用：

- `frame_id` 按 physics step 单调递增
- 即使某一步跳过 publish，下一次真正 materialize 的 frame 也会表现为 `frame_id` 跳跃
- 例如 `publish_every_n_steps=2` 时，materialized frame 可能表现为 `0, 2, 4, ...`

### 当前 `CpuPublishedFrame` 内容

每步发布的 CPU frame 目前包含：

- `frame_id / sim_time / step_index`
- `q / qdot`
- `X_world / v_bodies`
- `contact_count / contacts`
- `telemetry`（当前即 `force_state`）

### 当前 ack 行为

1. `lossless + borrow`

- `borrow_latest_frame()` 返回 `BorrowedFrameLease`
- lease 退出时，根据 `AckPolicy.default_for(...)` 自动推进 `acked_frame_id`

2. `lossless + snapshot`

- `snapshot_frame_to_host(...)` 返回 `SnapshotHandle`
- `.result()` 生成 host-owned snapshot
- staged handle 建立完成后推进 `acked_frame_id`

当前 CPU path 没有 ring，也没有 reclaim 压力；它主要承担的是：

- 同语义 reference implementation
- 无 CUDA 环境下的可运行行为验证

---

## GPU Path Status

更新：

- `physics/gpu_engine.py`

### 已实现：phase-1 synchronous publish core

目前 GPU path 已接上的不是最终异步版本，而是**最小同步 publish 版本**。

### 当前 engine 内状态

`GpuEngine.__init__` 现在会初始化：

- `PublishPolicy`
- `publish_ring_size = 3`
- `PublishedSlotMeta[3]`
- published slot buffers
- `ConsumerState` list
- `SlotReclaimer`
- `latest_published_frame`

### 当前 published ring / slot 内容

每个 slot 目前固定分配：

- `q`
- `qdot`
- `x_world_R`
- `x_world_r`
- `v_bodies`
- `contact_count`
- `qacc_smooth`
- `qacc_total`
- `force_sensor`

对于 dense `RigidBlock`：

- 默认不写
- 只有 `PublishPlan.do_rigid_block_write=True` 时才会：
  - 懒分配 contact arrays
  - 写入 `contact_bi/bj/active/depth/normal/point`

### 当前 publish 时机

GPU 侧：

- `step()` 结束后自动 `_publish_after_step(...)`
- `step_n()` 每个 substep 结束后也会 publish

当前 `_publish_after_step(dt)` 做的事：

1. `PublishPlan.from_policy(...)`
2. 选下一 slot
3. 根据 `PublishPolicy.on_ring_full` 处理 lossless backpressure：
   - `raise` -> `RuntimeError`
   - `skip` -> 跳过本次 publish
   - `block` -> 当前显式 `NotImplementedError`
4. 把 core buffers 从 mutable scratch 拷到 published slot
5. 按 plan 可选写 telemetry / rigid block
6. 更新 `PublishedSlotMeta`
7. 生成 `GpuPublishedFrame`
8. 更新 `latest_published_frame`

这意味着：

- 现在已经有正式的 frame boundary
- 还没有真正的 async stream/event/staging queue 模型

### 当前 GPU runtime surface

已支持：

- `set_publish_policy(...)`
- `register_consumer(...)`
- `unregister_consumer(...)`
- `latest_published_frame()`
- `ring_pressure_stats()`
- `borrow_latest_frame(...)`
- `snapshot_frame_to_host(...)`

### 当前 GPU ack 行为

1. `borrow_latest_frame(consumer_id)`

- 返回 `BorrowedFrameLease[GpuPublishedFrame]`
- 会更新 `latest_seen_frame_id`
- 对 `lossless + borrow`，lease release 时推进 `acked_frame_id`

2. `snapshot_frame_to_host(consumer_id, frame_id, spec)`

- 当前是**同步** host snapshot bridge
- 通过 `.numpy().copy()` 从 published slot 拿到 host-owned数据
- 通过 `SnapshotHandle` 返回
- `SnapshotHandle.is_ready` 在 phase-1 中恒为 `True`
- `SnapshotHandle.frame_id` 与被 snapshot 的 published frame 对齐
- 对 `lossless + snapshot`，在 snapshot 完成后推进 `acked_frame_id`

这里刻意只做了“语义对齐的 phase-1 版本”，还没有实现：

- async host staging queue
- copy completion event plumbing
- multi-frame ring traversal runtime
- consumer-side background worker

---

## Tests Added

### 1. `tests/unit/physics/test_publish.py`

覆盖：

- `PublishPlan.from_policy(...)`
- `BorrowedFrameLease` 生命周期与 release callback
- `LeaseExpiredError`
- `SlotReclaimedError`
- `SnapshotHandle`
- `PublishPolicy.publish_every_n_steps`
- `PublishPolicy.on_ring_full`
- skipped publish 场景下的 `frame_id` 跳跃语义
- `AckPolicy.default_for(...)`
- `SlotReclaimer` 行为
- `ring_pressure_stats(...)` blocker attribution

### 2. `tests/unit/physics/test_cpu_publish_runtime.py`

覆盖：

- `CpuEngine.step_and_publish()`
- `CpuEngine.step()` 自动更新 latest published frame
- `CpuEngine` respects `publish_every_n_steps`
- `lossless + borrow` 在 context exit 时 ack
- `lossless + snapshot` 在 snapshot 完成时 ack

### 实际跑过的验证

```bash
PYTHONPATH=. pytest -q tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py
python -m compileall physics/engine.py physics/cpu_engine.py physics/gpu_engine.py physics/publish.py
```

当前结果：

- `20 passed`
- compileall 通过

---

## First Consumer Landed

新增：

- `rendering/debug_exporter.py`
- `tests/unit/rendering/test_debug_exporter.py`

当前 `DebugExporter` 是第一个真实的 published-frame consumer，职责刻意保持很小：

- 通过统一 engine surface 注册一个 `host_export + snapshot` consumer
- 从 `latest_published_frame()` 取最新 frame id
- 通过 `snapshot_frame_to_host(...)` 拿到 host-owned snapshot
- 输出：
  - JSON
  - JSONL
  - CSV

当前它不直接构建 `RenderScene`，原因是这一阶段先验证：

- consumer 是否真的只依赖统一 publish contract
- `HostSnapshotSpec` / `SnapshotHandle` 的调用体验是否顺手
- CPU / GPU 路径是否都能以同样方式被消费

### 当前覆盖

`tests/unit/rendering/test_debug_exporter.py` 目前覆盖：

- 基于真实 `CpuEngine` 的 JSON / CSV 导出
- 基于 mock engine 的 surface-contract 验证
  - `DebugExporter` 不偷读 engine 私有状态
  - 只依赖：
    - `register_consumer`
    - `latest_published_frame`
    - `snapshot_frame_to_host`
    - `unregister_consumer`

### 更新后的验证结果

```bash
PYTHONPATH=. pytest -q tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py tests/unit/rendering/test_debug_exporter.py
python -m compileall rendering/debug_exporter.py rendering/__init__.py tests/unit/rendering/test_debug_exporter.py
```

当前结果：

- `22 passed`
- compileall 通过

---

## Second Consumer Landed: `PublishedFrame -> RenderScene`

更新：

- `rendering/scene_builder.py`
- `rendering/__init__.py`
- `tests/unit/rendering/test_published_frame_bridge.py`

新增公开 bridge：

- `build_render_scene_from_published_frame(engine, frame=None, env_idx=0, include_contacts=True)`

当前策略是：

1. CPU published frame
   - 直接消费 `CpuPublishedFrame.X_world` / `contacts`
   - 不重跑 FK
   - 不回退到 engine 内部状态

2. GPU published frame
   - 直接从 published slot buffer 读取 `x_world_R_wp / x_world_r_wp`
   - 重建 `SpatialTransform` 列表后交给现有 `build_render_scene(...)`
   - 若该 frame 已 materialize dense contact block，则直接从 published contact buffers 组 contacts
   - 若 dense contact block 未 materialize，则暂时 fallback 到 `engine.query_contacts(env_idx)`

这意味着当前已经有第二个真实 consumer 路径：

`PublishedFrame -> RenderScene`

但它仍然是 debug-oriented bridge，不是未来 realtime renderer 的最终接口。

### 当前覆盖

`tests/unit/rendering/test_published_frame_bridge.py` 覆盖：

- 真实 `CpuEngine` published frame -> `RenderScene`
- mock `GpuPublishedFrame` published buffers -> `RenderScene`
- GPU frame 在未 materialize dense contact block 时回退到 `engine.query_contacts(env_idx)`

### 更新后的验证结果

```bash
PYTHONPATH=. pytest -q tests/unit/rendering/test_debug_exporter.py tests/unit/rendering/test_published_frame_bridge.py tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py
python -m compileall rendering/scene_builder.py rendering/__init__.py tests/unit/rendering/test_published_frame_bridge.py
```

当前结果：

- `25 passed`
- compileall 通过

---

## End-to-End Backend Path Landed

新增：

- `rendering/published_frame_renderer.py`
- `tests/integration/test_published_frame_render_backend_integration.py`

新增薄 helper：

- `render_published_frame(...)`
- `render_latest_published_frame(...)`

它们的职责非常克制：

- 从 engine 取 published frame
- 通过 `build_render_scene_from_published_frame(...)` 构造 `RenderScene`
- 直接调用 backend 的 `render_frame(scene, timestamp, env_index)`

也就是说，这条链路现在已经能完整跑通：

`PhysicsEngine.step()`
`-> latest_published_frame()`
`-> RenderScene`
`-> MatplotlibBackend.render_frame(...)`

### 当前覆盖

`tests/integration/test_published_frame_render_backend_integration.py` 覆盖：

- 真实 `CpuEngine`
- 真实 `MatplotlibBackend`
- published frame -> render helper -> GIF 文件输出

这意味着本阶段已经不只是“有 publish contract + 有 consumer”，而是已经有一条真实的 backend end-to-end 路径。

### 更新后的验证结果

```bash
PYTHONPATH=. pytest -q tests/unit/rendering/test_debug_exporter.py tests/unit/rendering/test_published_frame_bridge.py tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py tests/integration/test_published_frame_render_backend_integration.py
python -m compileall rendering/published_frame_renderer.py rendering/__init__.py tests/integration/test_published_frame_render_backend_integration.py
```

当前结果：

- `26 passed`
- compileall 通过

---

## Third Consumer Landed: `PublishedFrame -> TelemetrySnapshot`

新增：

- `physics/telemetry.py`
- `tests/unit/physics/test_telemetry_snapshot.py`

新增 host-side telemetry bridge：

- `TelemetrySnapshot`
- `build_telemetry_snapshot_from_published_frame(engine, frame=None, env_idx=0)`

当前策略刻意保持保守：

1. 它只消费已经在 published-frame contract 里的 telemetry
   - CPU：
     - `ForceState`
   - GPU：
     - `telemetry_ref["qacc_smooth_wp"]`
     - `telemetry_ref["qacc_total_wp"]`
     - `telemetry_ref["force_sensor_wp"]`

2. 它不反向读取 engine-private scratch

3. 它不把 telemetry schema 写死到 `RenderScene`
   - 也就是说，这一步是一个独立 telemetry consumer
   - 不是对 Q51 / `RenderScene.sensor_data` 的提前定案

### 当前覆盖

`tests/unit/physics/test_telemetry_snapshot.py` 覆盖：

- 真实 `CpuEngine` published frame -> `TelemetrySnapshot`
- mock `GpuPublishedFrame` published telemetry buffers -> `TelemetrySnapshot`

### 更新后的验证结果

```bash
PYTHONPATH=. pytest -q tests/unit/physics/test_telemetry_snapshot.py tests/unit/physics/test_publish.py tests/unit/physics/test_cpu_publish_runtime.py tests/unit/rendering/test_debug_exporter.py tests/unit/rendering/test_published_frame_bridge.py tests/integration/test_published_frame_render_backend_integration.py
python -m compileall physics/telemetry.py physics/__init__.py tests/unit/physics/test_telemetry_snapshot.py
```

当前结果：

- `28 passed`
- compileall 通过

---

## Current Gaps / Known Limits

这版实现仍然是 phase-1，不应被误解为“publish pipeline 已 fully implemented”。

### 1. GPU snapshot 仍是同步 bridge

现在 `GpuEngine.snapshot_frame_to_host(...)` 的 staging 语义只是：

- host-owned copy 已经生成

但还没有真正的：

- `HostExportQueue`
- async copy stream
- copy completion event
- queue-pressure backpressure
- typed `SnapshotHandle` backend

### 2. GPU reclaim runtime 还只是最小化版本

当前已实现：

- next-slot reclaim check
- `lossless` blocker attribution

未实现：

- 多 frame 历史读取 API
- target-frame borrow/snapshot beyond latest frame
- 真正的 async consumer runtime
- typed `GpuPublishedSlotBuffers` / block dataclass

### 3. CPU/GPU 高层 API 仍未完全镜像

目前共享的是核心 surface 和主要语义，但还没有完全对齐：

- GPU 有 ring/reclaim/pressure
- CPU 当前只是 reference runtime，没有 ring

### 4. 尚未接到完整 rendering / sensor 管线

这次已经开始让：

- debug export
  通过 `DebugExporter` 消费 `PublishedFrame`
- debug render
  通过 `build_render_scene_from_published_frame(...)` 消费 `PublishedFrame`
- backend render
  通过 `render_latest_published_frame(...)` 走到 `MatplotlibBackend`
- telemetry consumer
  通过 `build_telemetry_snapshot_from_published_frame(...)` 消费 `PublishedFrame`

但还没有开始真正让：

- sensor path
- retained / realtime render view
- async host export queue

消费 `PublishedFrame`。

这一步仍然在后面。

### 5. `PublishedSlotMeta.invalidated` + `SlotReclaimedError` 目前是最小安全层

按照 Claude 的建议，当前已加入：

- `PublishedSlotMeta.invalidated`
- `GpuPublishedFrame` 的 payload access guard
- `SlotReclaimedError`

但它们目前仍只是 reclaim/重用时的最小运行时防护，不构成完整的 stale-slot / alias-safe runtime 方案。

也就是说：

- 这是 phase-1 的 TODO 锚点
- 不是最终 typed-slot/runtime-safety 方案

---

## Specific Review Questions For Claude

下面这些点最值得 challenge。

### 1. `publish_every_n_steps` 这个“逃生口”形状是否合理

当前按照 Claude 建议，新增了：

- `PublishPolicy.publish_every_n_steps: int = 1`

语义是：

- `step()` 仍维持统一 publish contract
- 但是否真正 materialize 新 frame，由 policy 决定

请 review：

- 这个逃生口是否足够干净
- 还是应该进一步区分“更新 frame_id”与“真正写 slot”的语义层级

### 2. `BorrowedFrameLease` 的当前保护是否够用

当前实现：

- context-manager + release callback
- 失效后 `get()/__enter__` 抛 `LeaseExpiredError`

请 review：

- 这是否已经足够作为 phase-1 保护
- 是否还需要更激进的 stale-frame runtime guard

### 3. `SnapshotHandle` 的相位划分是否合理

当前实现：

- phase-1：同步 `.result()`
- phase-2 预期：异步 staging queue，仍保留 `.result()`

请 review：

- 这个前向兼容接口是否足够
- 还是应该现在就把 completion/event/status 一并纳入 handle

### 4. GPU published slot 当前用 `dict[str, object | None]` 存 buffer

优点：

- 很快接通最小版本

缺点：

- 不够类型化
- 未来可能要演化成 block dataclass

请 review：

- 是否现在就该抽成 `GpuPublishedSlotBuffers/CoreBlock/RigidBlock/TelemetryBlock`
- 还是这个过渡结构在 phase-1 可以接受

### 5. CPU reference path 的当前粒度是否合适

现在 CPU path 已经能端到端验证：

- `borrow`
- `snapshot`
- `ack`

但它没有 ring。

请 review：

- 这是否是合理的 reference-level 对齐
- 还是说 CPU 也应该更早显式模拟 ring/reclaim

---

## Suggested Next Step

如果 Claude 认为当前 phase-1 方向没有大问题，我建议下一步二选一：

1. 继续沿 GPU path 往前推：
   - 真正引入 async host staging / queue
   - 完善 slot reclaim runtime

2. 开始把上层 consumer 接到 published frame：
   - 先做 debug/export path
   - 再逐步接 `RenderScene` / sensor path

我个人当前倾向先做 2，因为它能更快暴露 publish contract 是否真的适合上层消费。
