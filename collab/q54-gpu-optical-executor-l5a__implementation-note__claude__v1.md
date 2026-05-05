Initiative: q54-gpu-optical-executor
Stage: l5a-implementation-note
Author: claude
Version: v1
Date: 2026-05-03
Status: complete
Related Files: optics/device.py, optics/warp_execution.py, optics/__init__.py, tests/unit/optics/test_device_optical.py, tests/gpu/test_optical_warp_executor.py, MANIFEST.md
Owner Summary: Q54 L5A is implemented. The first GPU optical path uses Warp brute-force first-hit kernels from a host `OpticalSceneSnapshot`, returns `OpticalComputeResult(location="device")`, and stages back through `optics/device.py`. CPU and GPU parity tests cover plane, triangle, source-order tie-break, role filtering, and camera postprocess.

# Q54 L5A GPU Optical Executor Implementation Note

## 1. 实现摘要

本次实现完成 Q54 L5A：第一版 in-repo GPU optical executor。

实现路径：

```text
host OpticalSceneSnapshot
  + OpticalRaySensorSpec
  -> build_host_optical_primitive_workload(...)
  -> Warp arrays
  -> GpuBruteForceOpticalExecutor
  -> OpticalComputeResult(location="device")
  -> stage_optical_compute_result_to_host(...)
```

新增模块：

- `optics/device.py`
- `optics/warp_execution.py`
- `tests/unit/optics/test_device_optical.py`
- `tests/gpu/test_optical_warp_executor.py`

## 2. Device Result Contract

L5A 保持单一 result 类型：

```text
OpticalComputeResult(location="device")
```

device channels：

```text
hit_mask              int32[num_rays]
range_m               float32[num_rays]
position_world        float32[num_rays, 3]
normal_world          float32[num_rays, 3]
numeric_instance_id   int32[num_rays]
```

host staging 后恢复 canonical host dtypes：

```text
hit_mask              bool
range_m               float64
position_world        float64
normal_world          float64
numeric_instance_id   int64
```

## 3. 关键思考

### 3.1 为什么 L5A 从 host snapshot 开始

L5A 没有直接接 `GpuPublishedFrame`。这是刻意的。

GPU optical 的复杂度有两类：

1. ray/primitive intersection kernel 是否正确；
2. Q52 device-consumer lifecycle 是否正确。

如果第一步就把这两者合在一起，任何失败都会很难定位。当前实现先让 host
`OpticalSceneSnapshot` 打包成 device primitive workload，再验证 GPU result schema 和
CPU parity。这样 L5B 可以专注于 `borrow_device_frame(...)` /
`complete_device_consumer(...)` 的生命周期，不需要同时怀疑 kernel 数学。

### 3.2 为什么 role filtering 在 host 做

device kernel 不处理 Python 字符串，也不处理 arbitrary role set。

L5A 的目标是 correctness-first，所以在 host 侧按 `spec.sensor_role` 过滤 instance，
只上传当前 query 可见的 primitives。这让 kernel 只负责求交，避免引入 role bitmask
设计。role bitmask 应等到 L5C 的长生命周期 device scene buffers / GPU BVH 阶段再做。

### 3.3 为什么 source-order key 打包成 int64

CPU tie-break 使用：

```text
(instance_index, primitive_index_within_instance)
```

Warp kernel 里写二元 lexicographic compare 容易出错。实现采用 Claude review 建议：

```text
key = instance_index * 2**32 + primitive_index_within_instance
```

这样 GPU 上只需要：

```text
key < best_key
```

同时仍保持和 CPU 的全局 source-order parity。packer 会拒绝超过
`MAX_PRIMITIVES_PER_INSTANCE` 的 primitive index。

### 3.4 为什么 device 用 float32，host staging 升回 float64

Warp first-hit kernel 使用 float32 device buffers，适合机器人仿真常见的 <~1000m 场景。
host contract 仍然保持 float64/int64，避免破坏 CPU executor 和 downstream consumer 的
schema。

这不是说 future GPU optical 永远只能 float32。若未来出现大尺度场景，候选方案是：

- float64 device buffers；
- local-origin rebasing；
- camera-relative coordinates。

### 3.5 调试过程

实际调试中先确认 base shell 没有 Warp，而项目 conda 环境
`env_tilelang_20260119` 有 Warp 1.12.0 和 H200 CUDA 设备。因此 GPU tests 需要通过：

```text
conda run -n env_tilelang_20260119 python -m pytest ...
```

实现时先让 CPU-only tests 覆盖 packing/staging，然后在 conda env 中跑真实 Warp parity。
这避免了 CPU-only 环境 import Warp 失败的问题，也保证 kernel 语法和 device dtype 真实可用。

## 4. 测试覆盖

CPU-only：

```text
tests/unit/optics/test_device_optical.py
```

覆盖：

- packed source-order key；
- role filtering；
- workload shape/dtype；
- device result staging normalization；
- host-result rejection。

GPU：

```text
tests/gpu/test_optical_warp_executor.py
```

覆盖：

- plane hit + miss parity；
- triangle hit + miss parity；
- source-order tie-break parity；
- role filtering parity；
- staged device result + camera postprocess。

## 5. 验证结果

```text
PYTHONPATH=. pytest tests/unit/optics tests/unit/sensing -q
90 passed

conda run -n env_tilelang_20260119 python -m pytest tests/gpu/test_optical_warp_executor.py -q
5 passed

ruff check optics tests/unit/optics/test_device_optical.py tests/gpu/test_optical_warp_executor.py tests/unit/sensing
All checks passed

git diff --check
passed
```

## 6. Deferred

L5A 不包含：

- `GpuPublishedFrame` borrow/complete；
- Q52 device-consumer lifecycle；
- `ready_event` propagation；
- device result pooling；
- device scene cache；
- GPU BVH；
- GPU direct-light / shadow rays。

这些属于 L5B+。
