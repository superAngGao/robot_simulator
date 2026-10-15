Initiative: admm-method-audit
Stage: review
Author: codex
Version: v2
Date: 2026-10-12
Status: accepted
Related Files: collab/admm-method-audit__analysis__codex__v1.md; collab/admm-method-audit__analysis__codex__v2.md; collab/admm-method-audit__review__codex__v1.md; collab/admm-method-evidence/
Owner Summary: Approve，仅批准本次冻结分析/证据增量。独立推导并复现确认 C1 的受约束冲量目标与法锥条件正确，C2 对固定 50 次参考字段的解释与源码及新执行一致。未批准实施方案、求解器正确性、新物理模型或生产变更。

Reviewer session: /root/admm_method_analysis_review_v2（fresh independent bounded delta review；未参与候选编写）
Verdict: Approve

## 对象、隔离与实际增量

- Base: a6838abd2e2638cc01956ae69e05b2d1ab0bd40b
- Frozen HEAD: 0c0d39f590c2729e00d908780df3c9ef9970eb19
- HEAD tree: 05e7c36a85ed4fa287ffb4504ac3fe2af5c2a178
- Branch: physics/correctness-audit-20261012
- Local: C:/Users/AG/新建文件夹/robot_physics_correctness
- Remote: tileops-dev:/data2/shared/ang.gao/robot_sim_physics_correctness
- Analysis v2 local SHA256: F0CF805FF51B52A55D4D6E6D030162BA9525F3C504B04A3458DED66A5134D47C
- 两端 branch、HEAD、tree 均独立核实相同；两端开始/结束的 git status --porcelain 均为空。候选未被修改。
- 实际 base→HEAD 只新增分析 v2（191 行）及前次审阅记录（110 行）。无生产、测试或 probe 变更。
- 另外执行 v1→v2 的语义 diff：仅版本号、修订说明、完整受约束冲量目标/KKT，以及 JSON 字段解释。其余分析及历史数字未改动。
- 前次完整报告与归档文件 SHA256 均为 4328CF174429FF871603F894CE29F9E41F0354B4AAC91C22EA12373D88635538，确认逐字节保留。

已完整阅读 c4_worker_completion/AGENTS.md、collab/README.md、collab/REVIEW_PRINCIPLES.md。报告、独立脚本及结果均在候选外；未改其他代理私有文件或服务器其他 checkout。

## 发现与前次意见处置

无新增实质发现，无阻断项，无本轮必要证据缺口。前次 C1、C2 均在分析解释层面关闭；原始脚本/JSON 保留原状，符合不改写历史证据的边界。

C1（原 Low，非阻断）— 已关闭。

位置：analysis v2 第 49–52 行。设 F(f)=.5 fᵀQf−cᵀf，p=dt f。dt>0 且 K 是锥时，f∈K 当且仅当 p∈K；将 F(p/dt) 乘以正数 dt²，不改变最小化解，得到 G(p)=.5 pᵀQp−dt cᵀp。对闭凸锥采用 N_K(p)={n: nᵀ(z−p)≤0, 对所有 z∈K}，凸目标最优性条件为 0∈Qp−dt c+N_K(p)。因此新表述完整，Qp=dt c 确实仅在可选法锥项为零时成立。

独立边界反例：Q=diag(1,3.5,3.5)/.9、c=[1,8,0]、mu=1、f*=[1.8,1.8,0]。可行点位于锥边界；Qf*−c=[1,−1,0] 非零，外法向 n=[−1,1,0] 属于 N_K(f*)，两者相加为零。Q 正定使该 KKT 点成为唯一最优点。对 dt=.0002、.001、.3、2，p*=dt f*，dt n 仍是相应边界法向；数值核对冲量 KKT 残差最大 2.01e−16，目标缩放差最大 1.12e−16。该例直接覆盖原等式遗漏约束的触发条件。

同时检查 CPU 实际 c=a_ref−J a_u，以及 GPU admm_kernels.py 第 270–276 行的法向/切向 rhs。源码确为 dt*a_ref−v_free+v_current，新文继续明确 v_free−v_current=dt*J*a_smooth 的条件，没有把条件换算误说成当前 CPU/GPU 所有参数已一致。

C2（原 Low，非阻断）— 已关闭。

位置：analysis v2 第 138–142 行；原 probe 第 53 行。字段表达式明确为 (1/(1+1/(mass*.9)))**50，位于 adaptive=False/True 共用路径。独立新执行生产 probe 后，与原 JSON cmp 字节相同。固定 rho 实际迭代数 .1kg/1kg/50kg/1000kg 分别为 8/25/50/50；自适应分别为 5/8/13/18。新解释准确区分实际迭代和固定 50 次参照。

再次独立由 f_next=(c+rho*f)/(Q+rho) 推出冷启动、正力且投影不活跃时，相对误差为 alpha^k，alpha=rho/(Q+rho)。用固定 rho 的实际次数核对：

| mass | actual iterations | alpha^k | measured relative error |
|---:|---:|---:|---:|
| .1 | 8 | 2.160369772038167e−9 | 2.160369722620853e−9 |
| 1 | 25 | 7.712989003113406e−9 | 7.712988889549719e−9 |
| 50 | 50 | .3332223360611466 | .3332223360611444 |
| 1000 | 50 | .9459886440277049 | .9459886440277 |

最大绝对差 4.89e−15，符合浮点舍入。独立脚本还逐条核实所有八个质量/自适应条目中的该字段均等于固定 rho 的 alpha^50。正文已有的 50kg/1000kg 满预算数字及公式对照仍成立。

## 实际执行与证据

环境：tileops-dev；/data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python；OPENBLAS_NUM_THREADS=1。此次只新执行 CPU probe 和小型数学校验，没有新 GPU 数值执行。

主要命令（远端命令在候选 cwd，数学脚本除外）：

```sh
git branch --show-current
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git status --porcelain
git diff --name-status a6838abd2e2638cc01956ae69e05b2d1ab0bd40b HEAD
git diff --no-index -- collab/admm-method-audit__analysis__codex__v1.md collab/admm-method-audit__analysis__codex__v2.md
OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python collab/admm-method-evidence/admm_method_probe.py > /tmp/admm_method_analysis_review_v2_cpu.json
cmp collab/admm-method-evidence/admm_method_probe.json /tmp/admm_method_analysis_review_v2_cpu.json
OPENBLAS_NUM_THREADS=1 /data2/shared/ang.gao/robot_sim_physics_audit_env/bin/python /tmp/admm_method_analysis_review_v2_math.py > /tmp/admm_method_analysis_review_v2_math.json
sha256sum /tmp/admm_method_analysis_review_v2_cpu.json /tmp/admm_method_analysis_review_v2_math.json
git status --porcelain
```

读取了生产 CPU solver、GPU rhs 区段、原始 probe/JSON、前次完整报告，核实差异边界；Get-FileHash 核实分析摘要及归档报告一致性。probe、cmp、数学 assert 均退出 0。probe 的退出 0 仅代表诊断完成，不意味着它揭示的生产缺陷已修复。

独立结果 SHA256：

- CPU: b6dae8d2aa447360037731a3f8fcb99332b633e20dbd865e5f0c16862f319823
- Math: 4094b3517e5bdb2be2fe01b6446776a312174dab3a01fdfae39ac8297a9940a7

脚本及两份结果保存在 /tmp/admm_method_analysis_review_v2_math.py、/tmp/admm_method_analysis_review_v2_{cpu,math}.json，并已保留于 C:/Users/AG/新建文件夹/ 下同名文件。仅可能有正常 Python 缓存副作用，无候选跟踪或未跟踪文件变化。

工具/定位问题：最初按根目录寻找 REVIEW_PRINCIPLES.md 失败后，立即按协议补读 collab 路径；一次 rg 文件枚举命令误带模式参数，随后正确枚举；PowerShell 对未加引号的 HEAD^{tree} 进行了错误解释，改为单引号后成功核实；最初猜测前次报告外部文件名失败，经 rg 定位 admm_method_analysis_review_v1.md 后成功比对。均属 reviewer 工具调用问题，不是生产失败，且未留下未完成的检查。

## 批准范围与限制

批准仅绑定 HEAD 0c0d39f590c2729e00d908780df3c9ef9970eb19 的分析 v2 语义增量及前次报告忠实归档。本轮是在前次分析审阅基础上对 C1/C2 的新独立增量审阅；不是重新执行整个历史审计。

未重跑无关全仓库测试、36 组摩擦实验、H200 probe、机器人时间轨迹、性能基准或真实材料验证；未重新浏览未变动的论文/官方文档引用。此次文字增量不改变那些数值/源码/参考结论，没有触发重跑它们的必要性。其已有证明范围和限制仍按前次报告保留，不冒称本轮新验证。

不批准第 6 节为可直接实施的完整设计，不批准生产投影/R/warmstart/默认参数/输出策略修复，不批准新模型、求解器整体正确性、一般收敛、性能、部署或发布。已知生产缺陷仍存在。后续实施仍须对应范围的设计及实现独立审阅；忠实归档本报告的证据专用提交可按协议豁免递归审阅。
