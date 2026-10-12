Initiative: physics-correctness
Stage: design-audit
Author: codex
Version: v1
Date: 2026-10-12
Status: in_review
Related Files: REFLECTIONS.md; OPEN_QUESTIONS.md; physics/step_pipeline.py; physics/constraint_solvers.py; physics/solvers/; physics/backends/warp/; tests/validation/rigid_body/audit_solver_contracts.py
Owner Summary: 两阶段动力学的架构合理，但相互作用模型与数值求解算法尚未清楚分开；数处现有代码不符合所声明的方程。首轮修复 CPU PGS 惯性，其余问题保留可复现反例。

## 范围与判断方式

用户追加要求：梳理物理相互作用求解设计，判断设计是否合理、代码是否符合设计。
本记录以服务器 main f771e61 的物理代码为基线；本分支的唯一生产行为修复是 CPU
PGS 使用一致的广义惯性，含保留的 ImplicitContactStep 速度入口。
以下区分设计选择、确认实现缺陷、尚未实现的能力与待验证假设；不会把静态检查
或局部测试通过写成完整仿真可信度。真实材料标定仍缺失。

## 应有的物理分工与当前路径

```text
状态 q,v + 质量/惯量/关节 + 外部输入
  ├─ 几何：形状、姿态、配对、距离/穿透、法向、接触点
  └─ 平滑动力学：被动力+驱动力+外力 -> tau_smooth -> a_smooth
          ↓
相互作用模型：单边接触、摩擦、柔顺、恢复/耗散的具体方程
          ↓  接触映射 J，约束空间惯性 A=J H^-1 J^T
数值求解：PGS / Jacobi / colored GS / ADMM，产生冲量 p 或力 f
          ↓  v+=v+dt*a_smooth+H^-1 J^T p，或 a+=a_smooth+H^-1 J^T f
积分：q+=integrate(q,v+,dt)，公布该步状态与接触解
```

CPU 真实路径：CpuEngine 检测接触 -> StepPipeline -> AccelLevelAdapter 或
VelocityLevelAdapter -> tree.integrate_q。无接触时 ABA，有接触时构建 CRBA 缓存；
ADMM adapter 仍重复 ABA，ADMM solver 仍重复 CRBA，故“全链路只算一次”的设计
目标尚未完全落地，但这项重复本身不是已确认的数值错误。

GPU 真实路径：GpuEngine._step 的 CRBA/RNEA -> predicted velocity ->
batched_detect_multishape -> batched_contact_jacobian ->
batched_build_W_joint_space -> solver dispatch -> batched_apply_contact_impulse
-> batched_constraint_integrate。GPU 使用广义惯性，主要数组为 float32；CPU float64。
当前 GPU 每接触固定 3 行，CPU 低层 PGS 支持 1/3/4/6 行，两个引擎的检测入口均
实际产生 3 行接触。不能把低层参数字段当作端到端能力。

## 1. 设计是否合理

**合理的骨架**：几何与响应分开、平滑力与约束阶段分开、利用 J/J^T 保持速度与
虚功对偶、用完整 H 处理关节耦合、物理状态由物理层拥有。这些值得保留，无需
另造一个平台。CPU 作参考调试、GPU 作批量求解也合理，但“一致”必须先限定同一
物理模型、参数、接触流形与积分方法。

**尚不充分的模型设计**：

1. `solver` 选项同时切换了算法和物理模型。PGS 是速度层硬法向+CFM、Baumgarte、
   摩擦行柔顺；ADMM 声明是加速度层 per-row R 的软接触 QP。二者不同，不应只以
   “PGS vs ADMM 算法一致性”命名验收。先写清模型方程，再比较同模型的算法收敛。
2. 摩擦模型不同：PGS 两个切向分量独立 box bound，允许对角线合力达到 sqrt(2)*mu*N；
   ADMM 使用圆盘截断。方形摩擦近似可以是有意的取舍，但不能宣称等价的各向同性
   Coulomb 摩擦，也不能只测单切向场景后推断任意滑动方向。
3. ERP 名称同时承载无量纲比例（<=1）和 1/s 时间常数倒数（>1）。两套量纲在
   1 的边界切换，并非连续参数模型。保留现行为兼容，但后续应明确命名/单位，
   不能拿 CPU .8 与 GPU 10 的默认值直接比较。slop 是容许区间，不是目标深度。
4. solimp/solref 是该接触模型的数值/响应参数，不自动等价于真实杨氏模量、阻抗
   或恢复系数。旧 Q30 的“穿透更小所以物理更优”不能成立；较小几何误差不是
   更准确的力-位移定律。单接触 A 抵消推导也不能直接外推到耦合多接触。
5. REFLECTIONS session 20 的 bulk constitutive / interface law / geometry 分离，
   以及先只做刚体 per-shape InterfaceMaterial 的边界合理；代码还没有该接口。
   当前 terrain/global mu 与硬编码 body-body mu 无法表达不同足垫/地面材料对。
   暂不需要把软体、流体类壳子都建出来，也不能把该架构记为已实现。
6. 恢复/耗散必须按所选模型定义。PGS 可用法向速度反转，ADMM 通过软接触动力学
   产生反弹，两者不是无条件互换的 e 与阻尼比。历史阻抗叙述已有后续反思；
   本轮未复核其外部论文或证明材料参数映射，只保留其为未验证研究方向。

## 2. 代码是否符合设计

| 层/约定 | 实际实现 | 判断与证据 |
|---|---|---|
| 广义惯性求解与施加必须一致 | 基线 CPU PGS 用独立体 W，adapter 却用 H^-1 施加 | **确认缺陷，首轮修复**；1 kg 固定连接体改变质量分配，碰后速度从 -0.95 到 -0.05 m/s；正确均为 0 |
| ADMM 解声明的凸锥 QP | `_project_all` 先固定法向 max(0,z_n)，仅缩切向 | **确认违背所声明 QP**，见下方三变量反例；CPU 与 GPU 共享问题，法向静态测试不可见 |
| PGS 摩擦 R 是柔顺项 | CPU 把 R 加进 W；GPU 仅放入 W_diag 作更新分母，residual 仍用原 W | **确认不一致**；真实 CUDA 迭代收敛到硬摩擦解，增加迭代不能使它变成 CPU 软模型 |
| ADMM per-row R 取物理 A_ii | GPU 用已含 PGS 摩擦正则化的 W_diag 再乘 ADMM ratio | **确认重复计入**；同一接触 R_t/A_tt=.0531632，CPU 模型应为 .0526316 |
| 地形输入由模型传入 | GPU 拒绝非 FlatTerrain（明确能力边界）；from_merged 未传 FlatTerrain 高度 | 有意的非平地限制合理；**非零平地高度丢失是实现缺陷**，诊断脚本检查请求 .5 m 的高度与实际 contact count |
| 几何检测应覆盖浅穿透 | GJK distance 用最后 simplex 顶点作距离，支持点列表只截尾；box-hull 浅穿透漏检 | **确认几何缺陷**，不是响应参数可补救；原 3*margin workaround 绕开反例 |
| CPU/GPU 流形等价 | CPU 凸包地面枚举全部穿透顶点；GPU 最多 4 点 | **设计/契约缺口**；法向和最大深度相同不保证支撑力矩、柔顺或负载分配相同 |
| condim/restitution 能力 | CPU PGS 低层有角向行/恢复；CpuEngine 建 contacts 固定 condim3 且 e 默认0；CPU ADMM 角向 J 行为0，GPU固定3 | **未完成的接线/能力**，不能仅因字段存在就宣称支持 |
| 积分适用范围 | 半隐式 Euler，非严格能量守恒 | **模型与算法适用性**；双摆误差随 dt 一阶下降，不是由 50% 门槛认证的精度 |

### ADMM QP 反例：可行不等于最优

现有声明：min 0.5*f^T Q*f - rhs^T f，满足 fn>=0、||ft||<=mu*fn。
取 Q=(10/9)I、rhs=[1,2,0]、mu=1。对应低层真实 solver 的单自由体、
接触在体原点、depth=0、默认 d=.9 和给定无约束加速度。
解析最优解为 [1.35,1.35,0]，目标值 -2.025；现实现返回 [.9,.9,0]，
目标值 -1.8。单独把 z=[1,2,0] 投影到该锥，欧氏投影应 [1.5,1.5,0]，
实现却 [1,1,0]。增加迭代没有解决这个模型/算法不匹配。

这并不要求立刻无讨论地替换投影：需要明确要的是文档中的关联凸锥 QP，还是
固定法向后求 Coulomb 切向响应的其他模型；两者滑动时法向耦合不同。选择前
不得继续沿用“严格求解已声明 QP”的保证。本轮保留反例，不擅自改变接触模型。

### GPU 正则化反例：预条件与物理柔顺不能混为一谈

实际 CUDA kernels 输入 A=diag(1,3.5,3.5)、v_free=[-2,2,0]、d=.99、mu=1。
CPU 软模型应 p_t=-2*.99/3.5=-.565714286；GPU Jacobi 200 次得到 -.571428537，
colored GS 得到 -.571428597，与硬模型 -2/3.5=-.571428571 相同。
原因是分母的改变只影响迭代步长，固定点仍满足 A*p+v_free=0，而不是 (A+R)*p+v_free=0。
Mass splitting 的分母放大可以是数值松弛，但不能拿同一个 W_diag 兼任物理 R，
再在另一路 ADMM 里把它当未经正则化的 A 对角。

## 新验收应检验什么

1. 几何：分离/相切/浅穿透/深穿透的距离、法向符号、点位置；对称交换与旋转；
   流形数量/分布单独检查，不能让求解器稳定性掩盖漏检。
2. 接触算子：B*v 与体速度一致，J^T 与虚功一致，求解/施加使用同一 H；碰撞冲量
   与线/角动量平衡；测试偏心、转动关节、多根与固定连接，不能只有单球直落。
3. 求解器：先固定模型方程，再检查残差、可行性和必要的最优性/KKT；改变迭代数
   能收敛到同一模型解，不能以“落地高度差小”代替这些条件。
4. 轨迹：零点不敏感能量、全程峰值、等物理时长 dt 收敛、碰撞时刻量化误差。
   解析参考、独立引擎、CPU/GPU 一致性、稳定性、真实标定是不同等级的证据。
5. 材料与训练：至少需要负载、速度、入射角不同的真实力/位移/冲量/滑移数据；
   用 held-out 工况验证，不能用随机化或与 MuJoCo 一致代替。

## 本轮边界与后续顺序

本轮完成 CPU 广义惯性修复与有依据的能量/落地测试，保留上面其他缺陷及其
诊断脚本。下一轮优先冻结摩擦/柔顺模型，修复 ADMM 优化契约与 GPU R 数据流，
再处理浅接触 GJK 和流形策略；非零 FlatTerrain 参数丢失可独立修复。
材料接口实施以足端/地面实际需求为边界，不恢复旧渲染主线，也不扩成完整多物理场。
