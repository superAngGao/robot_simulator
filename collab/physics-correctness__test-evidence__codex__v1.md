Initiative: physics-correctness
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-12
Status: accepted
Related Files: physics-correctness__review__codex__v1.md; physics-correctness__implementation-note__codex__v1.md; physics-correctness-evidence/
Owner Summary: 记录冻结候选的最终验证与独立审阅，范围为 CPU PGS 惯性修复、解析验收与有界设计审计；不扩大为整体物理正确性或真实材料认证。

## 版本与证据边界

基线 `f771e612beaeea126e26926738097512cd2d809c`；最终代码审阅候选
`70841b9c509a51c70ca0fe6b1f094597e29026ce`。
本地 `robot_physics_correctness` 与服务器 `/data2/shared/ang.gao/robot_sim_physics_correctness`
在审阅期间均为该精确 HEAD，工作树干净。服务器最初未提交的实验副本已完整保存在
`physics-audit-pre-sync-20261012` stash，随后通过 bundle fast-forward 同步；未丢弃原始实验。
原服务器 `robot_simulator` 的 main 未改，未发布或合并到 GitHub。

后续记录提交仅添加审阅原文与执行证据，不改动候选代码、测试、设计方程或验收标准。
批准范围与非阻塞意见以相邻独立 review 原文为准。

## 作者执行

服务器隔离解释器 `/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python`，
`OPENBLAS_NUM_THREADS=1`，仓库根目录：

```sh
python -m pytest tests/reference/analytical tests/reference/pinocchio tests/integration/contact_pipeline tests/unit/collision tests/integration/test_cpu_engine_shapes.py tests/validation/rigid_body/test_physics_metrics.py tests/validation/rigid_body/test_contact_vs_mujoco.py::TestQuadrupedDropPgsSi::test_pgs_si_vs_admm_settling_height -q -rs
```

**432 passed in 114.97s**，无依赖跳过。日志：`physics-correctness-evidence/final-candidate-tests.log`。
该命令不包含全部慢速轨迹测试。此前更广泛的中间候选运行是 439 passed / 1 failed，
唯一失败为已推翻的 v2 slop 终态假设，原始日志保留。不能把该历史失败运行改称全通过。

全部 13 个改动/新增的生产与验证 Python 文件通过 Ruff lint 与 format check；
生产/测试差异的 `git diff --check` 通过。独立审阅发现全差异检查会报告保存的两份失败日志
中的原始行尾空格；为保留原始证据未改写日志，不将其算作生产代码检查失败。
原先 35 个真实 GPU 项与 1 个 GPU/CPU 四足项属于基线证据；
本轮生产 GPU 代码未改，独立 v3 设计审阅与最终代码审阅均另行实际执行了 H200 契约诊断。

## 独立执行与意见处置

独立代码审阅在精确候选运行三组共 **433 项通过**：419 个核心/参考/解析参考项、13 个
CPU 形状/修正落地项、1 个完整两秒双摆能量收敛项。不是与作者 432 项互不重叠的测试集合。
原始 20 反例在干净基线重新全部失败，在候选重新全部通过。另以世界坐标几何有限差分验证
12 个斜向、多根、关节耦合与 condim1/3/4/6 场景的 W 和非零广义冲量；具体命令、结果、
设备和限制见审阅原文。

非阻塞后续：把独立审阅中更强的斜向有限差分反例纳入后续正式回归，覆盖非零动量与力矩变化。
本次原样保留独立脚本 `physics-correctness-evidence/physics_review_fd.py` 与审阅记录，
不在冻结候选之后悄悄改测试。ADMM/GPU/几何/材料等明确缺口
仍按设计审计保留，不以已通过的 CPU 修复验收关闭。

## 数值解释与产物

`physics-correctness-evidence/corrected-reference-interpretation.json` 使用候选的解析 helper，
重新解释原始 v2 sweep 的 PGS 测量；包含源文件 SHA256 和计算代码版本。
这是**旧测量的后处理，不是新仿真**，原 `v2-oracle-metrics.json` 未改。
dt=.4/.2/.1ms 的尾高相对首次离散触地参考误差分别约 -3.069/-3.561/-9.601 nm；
默认 dt 下 120 次迭代约 -4.527 nm。它们只描述此对称 fixture 的数值残差，不表示现实接触精度。
v3 设计审阅还独立重跑了这四条轨迹，并验证未放宽的 1 μm / 1e-5 m/s 阈值。

双摆每步 CSV 与能量曲线 PNG 保留在本地
`C:/Users/AG/新建文件夹/physics_correctness_evidence/physics_correctness_audit_final/`，
服务器原始副本在 `/tmp/physics_correctness_audit_final/`。完整模型/测量表、真实足地标定方案和
未完成事项见实施记录。CUDA 渲染归档任务未恢复。
