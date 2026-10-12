Initiative: physics-correctness
Stage: request
Author: owner (request transcribed by Codex)
Version: v1
Date: 2026-10-12
Status: draft
Related Files: OPEN_QUESTIONS.md; REFLECTIONS.md; PLAN.md; PROGRESS.md; AGENTS.md; collab/README.md; collab/REVIEW_PRINCIPLES.md
Owner Summary: Return to the simulator's physics correctness. Audit weak tests and establish quantitative evidence before claiming contact fidelity. The following prompt records the owner's direction and the proposed first work sequence; it does not assert that physics validation has already passed.

## 可直接复制到新 session 的 prompt

我们继续开发 superAngGao/robot_simulator。本 session 的主线切换为物理正确性与碰撞可信度，不继续 CUDA/Warp 渲染替代或测速实验。旧实验已保存到 GitHub 分支 `archive/cuda-render-experiment-2026-10-12`；它是未完成的 WIP，不能当成已批准、已验收或已有性能结论的实现。

我最担心的是测试只证明“不崩、不发散”，却给出“物理正确”的印象。例如，现有双摆能量测试允许首尾能量相对漂移 50%，PGS 与 ADMM 落地高度比较允许 10 cm。这些是必须审计的具体证据，不是让你直接把断言改成更小的数字。我要查清实际误差、阈值依据、模型差异和真实实现缺陷。

### 先恢复工作位置与事实

1. 阅读适用的 `AGENTS.md`、`collab/README.md`、`collab/REVIEW_PRINCIPLES.md`。按现行协议，在设计/实现检查点进行独立审阅；控制范围，不另建庞大验收平台。若基线没有这些文件，可从上述归档分支读取，不默认把旧渲染工作流整套迁入物理实现。
2. 主服务器是 SSH `tileops-dev`，仓库 `/data2/shared/ang.gao/robot_simulator`，解释器 `/home/ang.gao/miniforge3/envs/robot_sim/bin/python`。截至 2026-10-12，该目录 clean main 为 `f771e612beaeea126e26926738097512cd2d809c`。GitHub main 当时仍是较旧的 `a20d75ecae624de3be054916142386477f0a352c`。
3. 开始时重新核对分支、提交和工作树，在服务器最新已核实的主线基础上创建独立物理分支/worktree。不要重置或覆盖既有目录、未提交修改，不要直接从过时的 GitHub main 推断项目现状。
4. 归档前已用 Git tree 验证：服务器 main 与实验 HEAD `157a3948d5f4f97dffe831a1f2bcc39ca57ef207` 的 physics、robot、rendering、sensing、rl_env、domain_rand 和 simulator.py 一致。physics tree 为 `1bf3c590a9644a7375c97d61d6bd8479f63fe0c7`。渲染实验没有需要先合并才能开始物理工作的核心改动。
5. 本地归档目录 `C:/Users/AG/新建文件夹/c4_worker_completion`。归档说明见 `collab/cuda-render-experiment-archive__implementation-note__codex__v1.md`。已有 290 项通过仅为旧光学 CPU 回归证据，不是物理测试或真实 GPU 验收。

### 先读物理设计，再解释测试结果

按下面顺序做有针对性的阅读；用章节标题定位，不依赖可能变化的行号：

1. `PLAN.md`：项目目标、分层架构、刚体动力学和 GPU 后端路线；结合 `PROGRESS.md` 核对实际进展。它们用于恢复全局上下文，旧勾选状态不代表当前实现状态。
2. `REFLECTIONS.md` 的“力系统重构：ForceState + DynamicsCache + StepPipeline”：理解普通力聚合 → 无约束加速度 → 接触约束求解 → 积分的因果关系，以及 DynamicsCache 和 ABA/CRBA 的分工；对照 `physics/step_pipeline.py`、`physics/cpu_engine.py`、`physics/gpu_engine.py` 的真实调用路径。
3. 接触建模与求解决策：阅读 `REFLECTIONS.md` 中“设计决策：per-row R + 摩擦 warmstart 归零”“CPU 多接触冲量：GS vs Jacobi 的根本差异”，以及 `docs/mujoco_style_solver.md`。结合当前 `physics/solvers/` 和 GPU kernels，核对约束/力/冲量的语义、摩擦、柔顺性、穿透修正和积分处理，区分设计选择与实现错误。
4. `OPEN_QUESTIONS.md`：重点读 Q18 接触能力差距、Q25 摩擦引发假角速度、Q30 **“CPU ADMM vs MuJoCo 稳态穿透深度差 0.86mm”**、Q46 大规模验证与 CPU/GPU 差异、Q48 碰撞检测缺口及各自后续更新。文档存在重复 Q 编号，按标题定位；上述 Q30 不是 GPU Multi-Shape Data Layout。历史上将误差解释为建模差异的结论也需要核实，不能自动豁免验证。
5. 接口边界：读 Q31 的训练环境定位，理解哪些物理量将进入训练；涉及对外观测时再查 Q52 的物理状态发布合同。光学设计文档只在确需解释跨模块接口时查阅，本轮不需要重走渲染实验设计。

先形成简短的“设计意图—当前实现—现有验证—未验证假设”对照。文档与代码冲突时，分别指出预期行为和实际行为，查后续决策与提交记录，并用最小实验裁定；不能认为代码存在就一定正确，也不能直接按旧文档重写。这里是定向恢复上下文，不要求先完成全仓文档重构才能开始数值验证。

### 项目边界

- 物理拥有状态和步进；CPU 用于精度对照和调试，GpuEngine/Warp 用于批量仿真。
- 渲染/传感器消费物理状态。GPU 训练环境在物理外组织动作、观测、奖励、终止与重置。
- 状态型训练无需等待光学 CUDA 替代完成。当前 Manager-based GPU RLEnv 尚未实现，CPU Env 中 reward/termination 仍有占位；本轮先建立物理可信度，不直接扩大成完整训练框架。
- 阅读 OPEN_QUESTIONS 中 Q31、Q46、Q48、Q30 及有关接触/恢复系数的后续记录。旧 PLAN 和 Q 条目存在过期状态，必须以当前代码和可复现实验核对，不能把未勾选项或历史结论直接当事实。

### 本轮先做什么

先做有边界的物理验证审计，并实际运行代表性测试，输出“场景—模型/求解器/设备—物理量—参考依据—原阈值—实际误差—结论”的表。明确区分解析解验证、独立引擎对照、CPU/GPU 一致性、数值稳定性和真实世界标定。

优先检查：

- `tests/validation/rigid_body/test_dynamics_vs_mujoco.py` 的 `test_energy_conservation`：当前仅比较首尾，阈值 50%。检查能量归一化是否受势能零点影响、全过程最大漂移、时间步长收敛和积分器适用范围。不要把有限步长能量误差一概判成动力学 bug，也不能称宽松稳定性检查为精度认证。
- `tests/validation/rigid_body/test_contact_vs_mujoco.py` 的 `test_pgs_si_vs_admm_settling_height`：当前阈值 10 cm。核对初态、接触几何、slop/ERP/compliance 参数和两种模型是否真的可直接比较。CPU/MuJoCo 5 mm、GPU/CPU 2 mm 也只是特定落地终态测试的阈值，不是实测物理精度结论。
- `tests/reference/analytical/`、`tests/reference/pinocchio/`、`tests/integration/contact_pipeline/`、`tests/unit/collision/` 及对应 GPU 测试：检查断言是否覆盖真实入口、是否能识别错误法向/力矩/摩擦、是否存在过宽容差或被跳过的关键覆盖。
- Q48 中 box-cylinder / box-hull 的 GJK distance 提前终止与增大穿透量 workaround；Q46 中 CPU/GPU 凸包接触点数量差异。分别评估几何检测与接触响应，不混为一谈。

从简单场景建立定量基线：自由落体/摆、静态承重、斜面滑动/滚动、简单碰撞/落地，再到机器人足地接触。合理使用解析解、力/冲量平衡、适用条件下的动量/能量规律以及步长和求解迭代收敛。先报告实测曲线/误差，再依据目标、解析误差阶或测量精度确定容差；不能仅凭“比 50% 小”拍一个新门槛。

发现缺陷后，先保留可复现的最小反例，再修复共同根因，补针对性回归，独立审阅。统一同类物理逻辑，避免在各测试或各后端叠加 hotfix。环境缺依赖导致 skip、代码失败和数值模型差异必须分开报告。不要用 mock 代替真实 GPU 物理验证，也不要靠修改 golden、放宽容差或避开浅接触来关闭问题。

### 怎样谈“和现实一致”

目前没有已核实的真实材料/足端接触标定证据。与 MuJoCo 一致不能等同于与现实一致；CPU/GPU 一致也可能共享错误。数值测试通过后，设计一个小型足地接触标定方案：静态承重、拖动/斜面、足垫落地、单腿受控动作。测量力、滑移、冲量、接触时长等，用不同工况分别标定和验证。没有真实数据时明确保留这一缺口，不虚构标定结果，不将随机化当成准确性证明。硬件或数据确实成为必要条件时，说明所需最小输入，同时继续可独立完成的数值工作。

先给我现有物理可信度的清晰画像、最重要的实际问题和一个范围小但有定量结果的首轮修复。持续推进这一主线，不把某个测试子集通过或一套基础设施建好当作“物理正确性已完成”；也不要恢复旧光学目标来替代当前任务。
