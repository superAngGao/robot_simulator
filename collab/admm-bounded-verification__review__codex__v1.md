Initiative: admm-bounded-verification
Stage: review
Author: codex
Version: v1
Date: 2026-10-14
Status: accepted
Related Files: collab/admm-bounded-verification__analysis__codex__v1.md; collab/admm-method-evidence/admm_bounded_verification.py; collab/admm-method-evidence/admm_bounded_verification_20261014.json; collab/admm-method-evidence/admm_bounded_verification_20261014.log; PROGRESS.md
Owner Summary: Approve with comments，仅批准当前有界诊断与建议顺序。真实 H200 独立重跑复现全部归档数值，另以独立解析/KKT 探针核实关键反例和参考解。保留一个非阻断范数说明；不批准生产修复、完整下一阶段设计、材料标定或完整刚体可信度。

## 身份、范围和冻结版本

- Reviewer session: `/root/admm_bounded_analysis_review_v1`，独立 Codex 审阅会话；只收到审阅任务包及后续范围/冻结状态消息，没有参与候选编写，没有修改候选。这是上下文和职责隔离，不声称不同供应商或独立安全边界。
- Review kind: analysis / diagnostic evidence。
- 范围：ADMM 单/双接触及球落地/斜碰撞的有界诊断、证据质量和建议修正顺序。不是生产实现审阅，也不是完整接触模型或下一阶段设计审阅。
- 按 handoff 授权，完整读取 `C:/Users/AG/新建文件夹/c4_worker_completion/AGENTS.md`、`collab/README.md`、`collab/REVIEW_PRINCIPLES.md`（UTF-8）作为现行协议。
- 候选仓库：`C:/Users/AG/新建文件夹/robot_physics_correctness`。
- Base: `5d80e87ee9514a72e7007c0f1552cc30d212333f`。
- Frozen HEAD: `3d63e9bb2df45e3db1afc3bdc2c52d5e2ba74025`。
- 本地分支：`physics/correctness-audit-20261012`。审阅前后 `git status --porcelain=v1` 均为空。
- 分析文档 SHA256: `90851D66FCD52443D6E739EC1851FB6AC02F8FEF980B293B2F78EFE361F346C3`。
- 诊断脚本 SHA256: `0869838E90F0960E52C91905875ABFAA7AC94CDB6901DF3AFF857F23BB9362E9`。
- Base → HEAD 仅 5 个文件：分析文档、诊断脚本、JSON、log 和 PROGRESS 新段，4094 行新增；没有生产代码或正式测试变更。`git diff --check base HEAD` 通过。

实际运行在 `tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness`，分支同名，HEAD 为 `171a6fc2e71910f22bcf079681700dc3742bc968`。运行前后唯一 untracked 是上传的诊断脚本，其 SHA256 与候选相同。该服务器没有本地 base 对象，因此服务器上的直接 `git diff 171a6fc... 5d80e87...` 返回 `fatal: bad object`；转而在拥有两个对象的本地仓库执行该 diff，确认差异仅旧 `rigid-admm-projection__proposal__codex__v1.md`，89 行新增。此为对象可用性限制，不是代码或数值执行失败。

读取了前驱 `admm-method-audit__analysis__codex__v2.md`，没有把其结论或旧 projection 草案当成本轮批准证据。旧草案仍暂停且未批准。

## Verdict

**Approve with comments。没有阻断本轮诊断分析的实质发现。**

候选有效区分了：错误投影产生的错误固定点、固定迭代预算不足、GPU R 来源不一致、瞬时凸模型的一致性，以及真实接触物理与时间离散的可信度。声明的局限与实际采样范围一致。建议先修复声明模型的算子一致性，再单独修正 GPU R、设计求解质量和缓存策略，最后验证物理定律与离散误差，得到现有证据支持。

这仅确认诊断分析和作为建议的先后顺序可被采用，不授权实施建议，不批准旧 proposal，也不证明修正后的轨迹会更符合真实材料。

## 发现与非阻断备注

### C1 — Low / non-blocking：误差界必须说明 stationarity 的范数

- 位置：分析文档第 79–81 行附近的强凸性误差界说明；诊断脚本 `cone_conditions`，尤其第 90 行。
- 触发：将 JSON 的 `stationarity` 直接解释为全向量欧氏驻点缺陷，并代入 `||f-f*||_2 <= defect/lambda_min(Q)`。脚本实际记录各接触 block 缺陷范数的最大值。对两个 block，即使 Q=I、每个 block 的缺陷都是 1，全向量缺陷也可以是 sqrt(2)。
- 后果：在将来用该字段认证欧氏力误差时，可能漏掉至多 sqrt(接触数) 的因子。当前分析没有据此计算或批准一个数值力误差界，因此不影响本轮主要诊断、表格误差或参考解真实性。
- 现有检查为何不处理：oracle 检查的是各 block 的可行性/驻点条件、双初值一致性；报告的力误差直接与参考解比较，没有调用误差界函数。所有原始记录仍可解释和复用。
- 建议/处置：保留原始记录为历史诊断，在后续设计或数值认证中明确 `stationarity` 是 max-block 值；欧氏误差界使用拼接驻点残差的二范数，或保守使用 `sqrt(n_contacts) * max_block_defect / lambda_min(Q)`。同时保留“可行点”的前提，不能把近似可行性容差直接当作严格可行证明。
- 后续证明：若把误差界变成正式 gate，增加独立的多 block 范数换算检查并明确可行性处理。这不是本轮需要扩大实施的任务，也不要求重写历史 JSON。

除上述表述澄清，没有发现错误的核心数值、把凸模型正确性等同真实物理的声明，或本轮范围内未关闭的必要证据缺口。

## 检查的真实调用链与参考解

1. CPU 冻结路径：`ADMMQPSolver.solve` 实际构造接触框架、J、CRBA H、A、R 和参考加速度；Trace 仅记录分解输入、投影调用次数及返回力。冻结用例独立构造 `[direction, arm cross direction]` 和球惯量，检查 J/Q/c。Trace 对 c 的重建与本次固定 d=.9 的生产公式一致；不能把此固定参数探针泛化为整个 sigmoid 的验证。
2. CPU 轨迹：`CpuEngine.step → _detect_contacts → StepPipeline.step → AccelLevelAdapter.solve → ADMMQPSolver.solve → ABA(tau + J^T f) → semi-implicit integration`。接触查询对应当前步输入状态，与诊断用旧 q 计算的深度一致；参考力不反馈到轨迹。质量固定 1 kg，累计世界线冲量与速度/重力平衡的单位成立。
3. GPU：`GpuEngine` 实际调用 `batched_build_W_joint_space`、`batched_admm_solve`，输出 lambdas 供冲量应用。构造 W 后，`crba_kernels.py` 的 W_diag 法向含 cfm、切向含 PGS 柔顺正则化；`admm_kernels.py` 再从 W_diag 计算 R，但主矩阵保留原始 W。候选所说的 R 来源差异与代码相符。诊断使用该次真实 Q/rhs 构造 oracle，确实隔离了投影/迭代误差与预期 R 的差异。
4. CPU `_project_all` 与 GPU `_cone_project_all` 均在标量 rho ADMM 的 s-update 中固定非负法向后裁切切向；不是欧氏锥投影。CPU 返回 f、GPU 返回 s 的差异已在诊断解释范围内，不能仅凭残差或可行性宣称最优。
5. 独立检查 SLSQP 参考问题的目标、约束与法锥符号；双初值一致性不是唯一凭据。独立解析计算进一步核实双法向解为每接触 `18/43`，双摩擦解为每接触 `(162/181,162/181,0)`，其梯度满足正确 KKT。Q 均正定。单接触对角 Q 的解析公式在实际这些样本的分区与单位下成立；另逐个验证所有 8 个 GPU 参考解的法锥 KKT。

## 新执行与实际结果

以下是本会话新执行，不复用作者的旧运行作为新执行证明。Warp 正常复用编译缓存，数值求解重新运行。

服务器实际命令：

```sh
cd /data2/shared/ang.gao/robot_sim_physics_correctness
git rev-parse HEAD
git status --short
sha256sum collab/admm-method-evidence/admm_bounded_verification.py
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 \
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python \
collab/admm-method-evidence/admm_bounded_verification.py --gpu \
--output /tmp/admm_bounded_independent_review_v1.json
```

Exit 0，输出 `frozen_cases=24, trajectories=18, gpu_cases=8`。实际 Warp 1.12.0，CUDA Toolkit 12.9 / Driver 13.2，设备 NVIDIA H200 (140 GiB, sm_90)。JSON 环境为 Python 3.12.13、NumPy 2.5.1、SciPy 1.18.0。没有 mock GPU。

将新 JSON 下载到候选外 `C:/Users/AG/新建文件夹/admm_bounded_independent_review_v1.json`；本地用 PowerShell `Get-Content -Raw` 与作者归档作 `-ceq` 比较，返回 True。新输出 SHA256 为 `DFA930EBAB58570AB5B774BB091DB4C3B053DC50A91F77C057E455749B68BABA`，服务器与下载文件一致。

另编写独立 scratch 探针 `C:/Users/AG/新建文件夹/admm_bounded_review_probe_v1.py`，上传到 `/tmp/admm_bounded_review_probe_v1.py`。它不导入候选诊断脚本；直接调用生产 CPU projector，并以独立矩阵构造和解析/KKT 检查重新核实关键结果：

```sh
cd /data2/shared/ang.gao/robot_sim_physics_correctness
OPENBLAS_NUM_THREADS=1 \
/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python \
/tmp/admm_bounded_review_probe_v1.py
```

Exit 0，观察到：

- 生产投影输入 `(1,8,0)` 返回 `(1,1,0)`；真正投影 `(4.5,4.5,0)`。平方距离分别 49 和 24.5；投影变分不等式出现正违反量 24.5，直接证明算子不正确。
- 双法向解析参考 `0.4186046511627907`，Q 最小特征值 `2.3888888888888884`，梯度约零。
- 双摩擦解析参考每接触 `(0.8950276243093923,0.8950276243093923,0)`，Q 最小特征值 `0.3888888888888886`；每接触梯度 `(1.138121546961326,-1.138121546961325,0)` 满足锥边界 KKT。
- 18 条轨迹的最大线冲量平衡误差 `6.8833827526759706e-15`；vx<=1 的最大冻结步力误差 `9.00257646208047e-12`；漏检穿透步总数 0。这支持候选限定于这些采样轨迹的声明，不形成通用检测保证。
- GPU 相对冲量误差重新核算如下，所有参考解独立 KKT 检查通过：

| Mass / vx | cap 30 | cap 1000 |
| --- | --- | --- |
| 1 kg / 0 | 2.318724113980753e-8 | 2.318724113980753e-8 |
| 50 kg / 0 | 0.5171767444506755 | 1.7839782249538537e-6 |
| 1 kg / 1 | 0.6760033641764681 | 0.6760033537861823 |
| 50 kg / 1 | 0.8435673967011871 | 0.6760049900929894 |

这些数值直接支持“增加迭代可改善所采样纯法向重物体响应，却不能消除已知摩擦投影错误”。脚本退出 0 仅表示诊断和 oracle 自检完成，绝不是生产准确性通过。

本会话也有几次 Windows rg 通配路径或假定文件名查找失败，随后定位到实际 `physics/constraint_solvers.py`、`physics/step_pipeline.py`、`physics/backends/warp/crba_kernels.py` 完成检查；这些是只读工具路径错误，没有影响数值运行或候选文件。

## 限制、残留风险和仍关闭的范围

- 没有修改生产代码；没有运行修正后的 solver，也没有把参考力反馈到轨迹。所以本报告不证明任何修复完成，不声称修复后穿透或反弹改善。
- 没有运行全仓库回归、性能基准、默认 warmstart/adaptive-rho 验证、广泛质量/尺度 sweep 或材料实验。本轮都是有意冻结的参数和小样本。
- CPU 双接触点是合成代数 fixture，不是可实现的单球平地双接触几何；GPU 仅单球首接触，不是完整 GPU 碰撞轨迹验收。
- 3 个步长和现有穿透数字不能证明时间收敛，也不能单独把穿透归因为 ADMM。冲量平衡只能证明计算力被一致应用。
- 角向 condim4/6、一般接触检测、缓存身份和 frame 映射、有限预算输出策略、终止状态、GPU R 修复、物理模型选择、材料/几何标定、完整刚体可信度及 GPU/tensor-core 性能均保持未批准。
- 建议第 3/4 步仍需单独具体设计与独立审阅。本轮不要求它们被扩写为实施规格；建议顺序本身不选择最终摩擦定律，不打开生产实施或发布门禁。
- C1 作为非阻断后续澄清保留。任何新的设计、代码或 operative 文档变更仍须按协议审阅其准确版本；仅原样归档本报告可使用 evidence-only 记录例外。

报告与 scratch 全部位于候选仓库外。候选审阅后仍 clean，服务器保留原先唯一 untracked 诊断脚本；没有覆盖作者 JSON/log。
