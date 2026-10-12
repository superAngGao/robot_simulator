Initiative: physics-correctness
Stage: implementation-note
Author: codex
Version: v1
Date: 2026-10-12
Status: in_review
Related Files: physics-correctness__design-audit__codex__v1.md; physics-correctness__proposal__codex__v2.md; physics-correctness__proposal__codex__v3.md; physics-correctness-evidence/
Owner Summary: 首轮修复 CPU PGS 关节惯性不一致；建立能量收敛、离散触地与接触模型的量化证据。整体相互作用设计审计确认 ADMM/GPU/浅接触等重要缺口，不能宣称物理正确性完成。

## 工作位置与范围

2026-10-12 核实服务器主线 clean main=f771e612beaeea126e26926738097512cd2d809c；
GitHub main=a20d75ecae624de3be054916142386477f0a352c。
physics tree=1bf3c590a9644a7375c97d61d6bd8479f63fe0c7。
分支 physics/correctness-audit-20261012；服务器 worktree
/data2/shared/ang.gao/robot_sim_physics_correctness；本地镜像 robot_physics_correctness。
未改动原服务器 main，未合并/发布，未继续归档的 CUDA 渲染实验。

生产改动只涉及 CPU PGS 接触惯性及旧入口的公共路径复用。
VelocityLevelAdapter 用 body-frame B_i 与缓存 H 的 Cholesky 因子构建
Jg 和 W=Jg H^-1 Jg^T，求解/施加采用相同惯性。PGS 和 PGS-SI 共用一套
正则化/投影/迭代；ImplicitContactStep 速度分支也调用该 adapter。
直接独立刚体 API 保留；新 context 成对提供，维度、有限性及下三角因子先验证。

## 测量表

| 场景 | 模型/求解器/设备 | 物理量与参考 | 原阈值 | 实测 | 结论 |
|---|---|---|---|---|---|
| 双摆 2s | CPU float64 / 半隐式 Euler | 全程 max abs(E-E0)，相对 E0-E_hanging | 首尾 / abs(E0) <50% | dt=.4/.2/.1ms：.273532/.142552/.072710J；5.2805/2.7520/1.4037% | 一阶收敛；旧分母随势能零点改变；不是能量精度认证 |
| 双摆随机状态30组 | CPU ABA / MuJoCo3.15.0 | 同状态加速度 | 新诊断 | max差3.979e-13 | 独立引擎局部动力学一致，支持积分误差解释 |
| 四足自由落体 .1s | CpuEngine | 解析 z=z0-gt²/2；Euler误差=-g*T*dt/2 | 既有5mm级 | dt1/.5ms：-.4905/-.24525mm，和离散公式差<1e-12m | 可解释的一阶离散误差 |
| 固定连接体撞地 | CPU PGS与PGS-SI，M=1kg，g=ERP=CFM=0 | 非弹性冲量应停止vn=-1m/s | 原落地终态测试未覆盖 | 基线子质量.05/.5/.95kg给vn=-.95/-.50/-.05；修复后50项针对性回归通过 | 确认实现缺陷；不能称模型差异 |
| 四足从 .45m 下落 | CPU ADMM / PGS | 软接触平衡 / Euler首次离散触地停止 | 两者高度差<100mm | 基线z=.4197491641/.4149798895m，差4.769275mm；修复PGS z=.4199280300m，与ADMM差.178866mm | 两种模型本来不同；分别验证其方程 |
| 同上 ADMM 与 MuJoCo | CPU ADMM / MuJoCo | 尾部 .1s 平均高度 | 差<5mm | .4197491641 vs .4188928876m；差.856277mm | 复现历史Q30数值；不据此宣称真实物理更优 |
| 同上 GPU 与 CPU | Warp/H200 ADMM / CPU ADMM | 尾部平均高度 | 差<2mm | .4197500050 vs .4197491641m；差.840905微米；GPU max abs(vz)=2.3184e-5m/s | 单场景正常向一致；无法发现摩擦R差异 |
| 无摩擦斜面 | CPU/30度/0.2s | v=-g*sin(theta)*T | 原近无摩擦15% | mu=0精确对照误差约2e-15相对量级 | 解析一致；原mu=.01不能严格称无摩擦 |
| 滑动斜面 | CPU/30度/mu=.1/0.2s | v=-g*(sin(theta)-mu*cos(theta))*T | 20% | dt1/.5ms相对误差1.047e-9/5.237e-10 | 此解析滑动场景已量化 |
| 滚动球斜面 | CPU/20度/mu=1/0.5s | a=(5/7)g*sin(theta) | 15% | dt1/.5ms相对误差.03957%/.002133% | 此condim3无滑滚动场景已量化；不证明滚动阻力 |
| box-hull浅穿透 | CPU GJK margin | 解析轴向重叠 .01/.1/1mm | workaround增至3mm | 三种浅穿透均漏检；box-cylinder在1mm漏检 | 几何真实缺陷，保留反例；本轮未修复 |

## 新验收与无效假设的修正记录

能量测试改用真实 CpuEngine，包含t=0/2s，全程峰值；检查势能平移1000J不改变
归一化误差；dt减半比在[1.8,2.2]。该10%带宽围绕一阶理论比2，实测1.919/1.961。
它是固定fixture的收敛回归，不是通用能量误差上限；绝对误差继续报告。
MuJoCo依赖只在调用参考引擎时触发跳过，不再使CPU-only能量/接触测试一起跳过。
CpuEngine导入异常不再被宽泛except包装成依赖缺失。

接触验收用两个模型各自的参考，不以100mm差值通过：ADMM四耦合法向行平衡
R_ii*(Mg/4)=k*d*depth；PGS落地参考为离散自由落体的首次穿透高度。
默认dt=.2ms时n=391，z=.4199280336m。高度容差1微米，尾段range1微米，
速度1e-5m/s；另用可分辨CFM验证ERP两分支的单位与单/四足负载冲量。

**保留失败证据**：v2初稿把稳态深度slop+CFM*F*dt²/ERP当成1s落地终态，
导致修复后PGS错误地偏离该参考4.928mm。实际slop内可先承重，CFM仅产生纳米级
蠕变，故该初稿测试的预期错误；没有改求解器或放宽阈值去满足它。v3按解析触地
时刻修订，旧run的metrics和失败日志原样保留，不把旧invalid oracle改写成已通过。

迭代/步长测量：PGS 60->120次默认dt尾高变化约0.97nm；dt=.4/.2/.1ms尾高
.4196972993/.4199280300/.4199663811m，分别跟随离散首次触地位置。
ADMM 50->100次尾高无可见变化，三步长均在.4197491641m附近（约1e-14m范围）。
四冗余接触的低CFM精确负载分配测试需要较高迭代上限：1000次尚有约2.5e-8rad/s
残差，10000次上限、1e-14停止精度满足1e-10验收；不把默认60次宣称为普遍收敛。

解析摩擦测试也改为验证当前软模型：d=.99时预期切向残速=.02m/s，
而非用.03m/s容差近似硬模型零速度；法向CFM冲量与切向W/d均直接验证解析值。

## 环境与执行证据

原robot_sim：Python3.12、NumPy2.5.1、SciPy1.18.0、Warp1.12.0、Torch2.9.1+cu128、
pytest9.1.1；缺MuJoCo与Pinocchio。单独创建 system-site-packages venv：
/data2/shared/ang.gao/robot_sim_physics_audit_env，安装MuJoCo3.15.0、pin4.1.0。
原环境未安装/升级这些包。物理GPU2在忙；验证使用CUDA_VISIBLE_DEVICES=3的H200。
Warp缓存允许复用，但结果来自本轮实际执行，没有mock设备。

运行入口（仓库根目录）：
```sh
OPENBLAS_NUM_THREADS=1 $PY -m pytest tests/reference/analytical tests/reference/pinocchio tests/integration/contact_pipeline tests/unit/collision tests/integration/test_cpu_engine_shapes.py tests/validation/rigid_body/test_dynamics_vs_mujoco.py tests/validation/rigid_body/test_contact_vs_mujoco.py -q -rs
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 $PY -m tests.validation.rigid_body.audit_physics --output /tmp/physics_audit --sweep --gpu
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 $PY -m tests.validation.rigid_body.audit_solver_contracts
```
PY为上述isolated环境的bin/python。audit脚本输出诊断，包括明确未解决的反例，
不把进程退出0当作其中所有物理断言成立。

基线：356 CPU项通过，5 Pinocchio依赖跳过；目标25项全部因MuJoCo跳过。
补依赖后原目标25+Pinocchio5=30项通过（619.74s）。新旧对照20反例在基线全失败，
修复后通过。真实CUDA碰撞/求解器35项通过（127.87s），原四足GPU/CPU高度测试
1项通过（51.70s）。最终候选验证与独立code review记录在后续证据文件。

## 未完成与下一步

整体设计判断见design-audit：ADMM非欧氏锥投影、GPU柔顺项只作迭代预条件、
GPU ADMM二次正则化、非零平地高度丢失、GJK浅接触漏检、流形策略、材料接线
与condim能力仍未闭合。没有把这些变更混入首轮CPU修复，也没有用skip/大穿透/
宽容差把反例关闭。

真实足地标定小方案：先记录足垫/地面样本、几何、质量、重力、传感器单位与
同步时间；静态加载测力-位移和迟滞，恒速拖动/倾斜测滑移起点与动摩擦，足垫
不同落高测冲量/接触时长/反弹，最后单腿受控加载测力矩与支撑力。
校准与验证使用不同负载、速度、角度；最小输入是时间同步的力、位姿/位移与
已知质量/几何及材料配对。没有这些数据，本轮仅建立数值可信度证据，不声称
sim-to-real接触已验证。
