Initiative: admm-method-audit
Stage: review
Author: codex
Version: v1
Date: 2026-10-12
Status: accepted
Related Files: collab/admm-method-audit__analysis__codex__v1.md; collab/admm-method-evidence/
Owner Summary: Approve with comments，仅批准冻结版本的分析和诊断证据。核心推导、源码分类及 CPU/H200 反例独立复现成立；两条非阻断表述意见保留。未批准新模型、实施方案、求解器正确性或性能能力。

Reviewer session: /root/admm_method_analysis_review（fresh independent review；未参与候选编写）
Verdict: Approve with comments

## 审阅对象与隔离

- Base: 98f4e7dda5cdf41ee34625d7c5f0f00dcc180516
- HEAD: a6838abd2e2638cc01956ae69e05b2d1ab0bd40b
- 本地候选: C:/Users/AG/新建文件夹/robot_physics_correctness
- 远端候选: tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness
- 分析文件本地 SHA256: A2773E8B242B5AFBB6A9AB87751006829CE21AD3E540D25E70B08D6094DC643C
- 两端 HEAD 均独立核实；开始和结束 git status --porcelain 均为空。实际 diff 仅新增 7 个分析/诊断文件，1780 行，无生产代码或正式测试变更。
- 阅读了 c4_worker_completion/AGENTS.md、collab/README.md、collab/REVIEW_PRINCIPLES.md。协议明确授权新独立审阅。本报告和独立执行产物均存候选之外，未改候选、原服务器 main 或其他代理私有上下文。

## 结论及批准边界

无阻断发现、无本轮必须补齐的运行证据缺口。批准的是当前分析对方法、模型、实现偏差和有限预算的分类，以及所附局部证据能够支持的结论。特别认可：R 是目标模型的一部分，rho 是求解参数；正确子问题和数值收敛不能由运行次数替代；单步凸优化不证明时间推进稳定或材料真实性。

本批准不认可当前 CPU/GPU ADMM 已正确求解全部声明问题，也不批准第 6 节为可直接实施的完整设计。不开放投影、R、warmstart、默认参数、输出策略的生产修改或模型替换，不构成完整仿真、性能、真实材料、部署/发布批准。已存在且被报告揭示的生产问题不因分析获批而消失。

## 非阻断意见

C1 — Low，数学表述精度，非阻断。
位置：分析文件第 46 行，“正确对应是 Qp=dt*c”。
触发：锥边界激活（滑动摩擦或零法向接触）时，此等式不是完整最优性条件。例如报告自身的 Q=diag(1,3.5,3.5)/.9、c=[1,8,0]、f*=[1.8,1.8,0] 给出 Qf*−c=[1,-1,0]，不为零。
影响：单独摘引这行，可能把受约束问题误读为无约束线性系统。后面的 split 和 KKT 讨论已经正确，实测 GPU 标量例约束不活跃，因此不影响本轮核心结论或数值。
方向：今后引用时写为 dt>0、K 为锥时 min_{p∈K} .5 pᵀQp−dt cᵀp；一般 KKT 是 0∈Qp−dt c+N_K(p)。Qp=dt*c 仅用于法锥项为零的情况。
验证：保留上述锥边界反例并检查受约束目标的力/冲量变量替换。无需为本轮分析新增生产测试。
处理边界：本意见随审阅记录保留，可在用户解释和下一份设计中明确；不要求改动冻结候选来获得本次批准。

C2 — Low，诊断字段清晰度，非阻断。
位置：admm_method_probe.py 第 53 行及其 JSON，fixed_rho_theoretical_relative_error。
触发：.1kg、1kg 固定 rho 实测分别在 8、25 次提前退出，但该字段统一计算收缩因子的 50 次幂；自适应条目也带同一固定 rho 对照值。
影响：读原始 JSON 的人可能把该值当作“实际执行迭代数下的理论误差”。正文使用的 50kg/1000kg 固定 rho 场景均确实执行 50 次且吻合，不影响正文表格。
方向：后续探针可将字段命名为 fixed_rho_50_step_reference，并另记固定 rho 的 actual-iteration 对照；保留作者现有数据来源，不改写历史输出。
验证：用 .1kg 的实际 iterations=8 检查 (1/(1+1/(mass*.9)))**8 与实测误差一致到舍入精度即可。
处理边界：作为证据解释的非阻断后续项保留，不要求修改冻结候选。

## 独立源码与数学核对

1. 检查 physics/solvers/admm_qp.py 全部生产路径：J/H 到 A、逐行 R、a_ref、rhoI 分裂、两个残差、内部 rho/u 缩放、跨 solve 缓存、无接触早退、返回 f 和分解保护。报告与代码一致。凸性陈述按正常物理参数域（H 正定、0<d≤1、非负摩擦）理解，未将缺失参数校验当作本轮新实现范围。
2. 检查 physics/constraint_solvers.py 的 AccelLevelAdapter 无接触早退，以及 physics/step_pipeline.py 对 H 的预构造与末端 semi-implicit Euler。求解器内确有重复动力学工作；单步解的唯一性不推出离散时间系统稳定。
3. 检查 physics/gpu_engine.py 实际 launch、physics/backends/warp/crba_kernels.py 的 batched_build_W_joint_space（第 655 行起）、physics/backends/warp/admm_kernels.py 全路径。确认实际传入 W_diag：法向 W_nn+cfm，切向 W_tt+ratio*|W_tt|；ADMM 再乘自身 ratio 构造 R。这里是 R 来源污染，不能理解成所有方向简单多加一遍相同 R。
4. GPU 默认 rho=1、30 次、单线程/环境标量循环、输出投影 s，CPU 输出 f，均为源码事实。未把这些源码事实写成性能测量或 tensor-core 能力。CPU np.linalg.solve 没有显式使用三角结构；无加速倍数结论。
5. 独立推导标量收缩：内部正力解且 u=0 时 f_next=(c+rho*f)/(Q+rho)，故 e_next=rho/(Q+rho)*e。R/A 固定来自固定 d 的孤立方向，而非一般矩阵除法。重质量的主要局部机制是 Q/rho 变小。
6. 独立小脚本调用生产投影：mu=1、输入 [1,2,0]，生产 [1,1,0] 距离平方为 1；欧氏最近点 [1.5,1.5,0] 距离平方为 .5。独立检查报告的受约束最优点 [1.8,1.8,0] 的梯度 [1,-1,0] 与乘子 1 满足边界 KKT。证明的对象是错误 z 子问题，不是一般 ADMM 不收敛。
7. 核实 warmstart 的 u 尺度与接触顺序问题只足以证明缓存语义/有限预算退化。报告明确没有声称任意初值破坏正确 ADMM 的渐近收敛。诊断 deep-copy 对照没有改变生产算法。
8. condim4 零角向行确实产生零特征值；rhoI 的子问题可逆与原目标严格凸不同。对角条件数反例亦成立。未把存在零行等同于所有正常 condim3 情形失去唯一解。

## 独立执行与结果

环境：tileops-dev；解释器 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python；OPENBLAS_NUM_THREADS=1；仅 CUDA_VISIBLE_DEVICES=3，运行日志确认为 NVIDIA H200，Warp 1.12.0、CUDA Toolkit 12.9、Driver 13.2。允许 Warp 编译缓存；数值均为 reviewer 本轮新执行。

实际主命令（在远端候选 cwd）：

```sh
git rev-parse HEAD
git status --short
OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python collab/admm-method-evidence/admm_method_probe.py > /tmp/admm_method_independent_review_cpu.json
OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python collab/admm-method-evidence/admm_regularization_recheck.py > /tmp/admm_method_independent_review_regularization.json
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python collab/admm-method-evidence/admm_gpu_iteration_probe.py > /tmp/admm_method_independent_review_gpu.log
cmp collab/admm-method-evidence/admm_method_probe.json /tmp/admm_method_independent_review_cpu.json
cmp collab/admm-method-evidence/admm_regularization_recheck.json /tmp/admm_method_independent_review_regularization.json
git status --porcelain
```

三组均退出 0；退出 0 只说明诊断执行完成，不代表被诊断的生产契约通过。两份 CPU JSON 与作者证据 cmp 字节一致。GPU 八条数值记录与作者记录一致（编译缓存加载计时自然不同）。独立 math 脚本位于候选之外 /tmp/admm_method_independent_math.py，调用同一解释器并输出 /tmp/admm_method_independent_review_math.json。

- 50kg 固定 rho，50 次：30.004994877248503 N，正确 45 N，相对误差 .3332223360611444；自适应 13 次/13 分解得到 44.999999819249076 N。
- 1000kg 固定 rho，50 次：48.610220375069964 N，对照 900 N；自适应 18 次得到 899.9999998884455 N。
- 分离方向 1/5 次返回 -.47368421052631576 / -.0002912315839384287 N；默认预算在第 8 次退出，返回 -1.435219954929871e-9 N。正文没有把这一微小默认误差夸大为同量级的常态吸引力。
- 重复零力问题 final rho=512 → 1：真实对偶约 -1 被解释为 -.001953125；原缓存重复需 10 次，换算副本需 1 次。
- 顺序交换的一次迭代结果 [5.1631578947368215,4.736842105262947]，重排副本 [8.999999999999787,.8999999999999789]。
- condim4 的 Q 特征值为 [0,1.1111111111111112,1.1111111111111112,1.1111111111111112]。
- 原文 36 组含 R 摩擦反例本轮由 reviewer 重新执行。作者准确标为前轮证据，未声称它是作者本轮新运行；本审阅的新执行与其完全一致。
- H200 50kg：30 次相对冲量误差 .5171803378880017，300 次 .0013698347524869476；1000kg：.9672311961162434 / .7166481838147742。参考使用实际 W+R 和 rhs，确实隔离了有限迭代而非套用 CPU 理想 R。

独立输出 SHA256：

- CPU: b6dae8d2aa447360037731a3f8fcb99332b633e20dbd865e5f0c16862f319823
- regularization: b75f8116b97a405ff9ef7aca9fb439291c4823448a9c6dcc18dfd445acf3073a
- GPU log: fafa7b502b5879052ddd695a3fa2e11ece7af4242ca6dad944068c47156e16b6

这些独立输出也已复制到 C:/Users/AG/新建文件夹/admm_method_independent_review_{cpu.json,regularization.json,gpu.log,math.json}。只有正常 Python/Warp 缓存副作用，无候选跟踪文件或未跟踪工作文件变化。

## 原始参考资料复核

实际浏览核对而非仅信作者引用：

- [Boyd 等 §3.2–3.4](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)：闭、proper、凸目标和鞍点假设；残差与停止条件；自适应 rho 的限制及 u 重缩放。报告使用这些结论的边界恰当。
- [Wohlberg §III–IV](https://arxiv.org/pdf/1704.06209)：原始残差平衡不具一般缩放不变性，支持报告的局限性分类。
- [Todorov §III.E、V](https://www.roboti.us/lab/papers/TodorovICRA14.pdf)：带度量的锥投影、对角正则化下限、摩擦方向调整；报告没有把本项目逐行 R 冒充完整论文实现。
- [MuJoCo 3.3.7 computation/parameters](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#parameters)：插值解释依赖对角假设，通常不精确成立，支持报告对耦合结构的限定。
- [NumPy solve 官方说明](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)：确认 gesv 路径；只用于实现分类，没有推断未测的耗时。

## 限制与过程错误记录

未运行全仓库回归、完整机器人轨迹、真实材料标定、性能基准、所有参数域收敛或 factor-floor 触发统计；这些不属于只增分析和证据的本轮必要门槛。GPU R 与投影实现的判定结合源码、当前数学反例及明确标记的历史背景，不声称本轮新增完整 GPU 摩擦回归。

最初读取 REVIEW_PRINCIPLES 的根目录路径失败，按 AGENTS 指定的 collab 路径正确补读；搜索猜测的 solver_adapter 文件名失败后定位到 constraint_solvers.py；一次远端 cmp 从登录 cwd 执行导致文件找不到，切到候选 cwd 重跑后两项通过；Stanford 无 www 的 PDF 深链工具错误后用其 web.stanford.edu 官方镜像完成章节核对。均为 reviewer 定位/工具问题，不是生产失败或未完成验证。

本报告记录原冻结内容的批准。保留本报告与独立输出的证据专用提交可依协议豁免递归重审；若修改分析结论、设计或生产实现，应按实际变更范围另行独立审阅。
