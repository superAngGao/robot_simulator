Initiative: q50-render-backend
Stage: implementation-note
Author: claude
Version: v2
Date: 2026-04-23
Status: implemented
Related Files: collab/q50-render-backend__implementation-note__claude__v1.md
Owner Summary: RerunBackend 修复了与 rerun-sdk 0.31 的 API 不兼容问题，4 个测试全部通过（含真实 .rrd 文件写入）。变更极小（3 行），无架构影响。

## Open Questions Addressed

- **Q50 — 渲染层架构完善**: Step 2 (RerunBackend) 现在可用。之前 v1 实现基于 rerun-sdk 0.16 API，安装 0.31 后测试失败。本次修复使 Step 2 真正完成。

## REFLECTIONS.md / PROGRESS.md Impact

不需要更新。本次是 API 适配修复，不改变架构或功能范围。

## What Changed

`rendering/backends/rerun_backend.py` 中两处 API 调用更新：

1. `rr.set_time_seconds("sim_time", timestamp)` → `rr.set_time("sim_time", timestamp=timestamp)`
   - `set_time_seconds` 在 rerun 0.21+ 已移除，统一为 `set_time(timeline, *, timestamp=...)`

2. `rr.connect()` → `rr.connect_grpc()`
   - `rr.connect()` 在 rerun 0.21+ 已移除，gRPC 连接改为 `rr.connect_grpc()`

3. docstring 中版本号 `0.16` → `0.31`

## Files Touched

- `rendering/backends/rerun_backend.py` — 3 行修改

## Tests Added / Updated

无新增测试。现有 4 个测试在 rerun-sdk 0.31 下全部通过：

```
tests/rendering/test_rerun_backend.py::TestRerunBackend::test_set_output_saves_rrd        PASSED
tests/rendering/test_rerun_backend.py::TestRerunBackend::test_all_supported_shapes_do_not_raise  PASSED
tests/rendering/test_rerun_backend.py::TestRerunBackend::test_convex_hull_uses_precomputed_faces PASSED
tests/rendering/test_rerun_backend.py::TestRerunBackend::test_contacts_render_as_arrows   PASSED
```

全量 fast suite：794 passed, 1 skipped（无回归）。

验证环境：`env_tilelang_20260119` conda env，Python 3.12，rerun-sdk 0.31.3。

## Known Limitations

v1 implementation-note 中记录的 coverage gaps 仍然有效，特别是：

- `RerunBackend` 不渲染 terrain（`scene.terrain` 未消费）
- `rr.Cylinders3D` / `rr.Capsules3D` 在 rerun < 0.16 不可用（现在要求 0.21+）
- GPU engine 集成测试（真实 GpuEngine + RerunBackend 端到端）延后到 Q51

## 关键思考

### Non-obvious technical decisions

rerun-sdk 的 API 在 0.16→0.31 之间有两处 breaking change：
- `set_time_seconds` / `set_time_nanos` 等便捷函数被统一为 `set_time(timeline, *, sequence/duration/timestamp)`
- `connect()` 被 `connect_grpc()` 取代（rerun 现在支持多种传输协议，gRPC 是默认）

这两处变化在 rerun changelog 中有记录，但不在 0.16 的 docstring 里。发现方式：运行测试 → `AttributeError: module 'rerun' has no attribute 'set_time_seconds'` → `dir(rr)` 查找替代 API。

### Debugging difficulties

无。错误信息直接，修复路径明确。

## Commit

`4b69fe2 fix: update RerunBackend for rerun-sdk 0.31 API`
