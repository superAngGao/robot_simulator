Initiative: Q54 / P12.4 / C4 scope convergence
Stage: proposal, based on current source inspection
Author: Codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: runtime-integration design v3; compilation-admission design v3;
  remaining-setup gap-scan v1; systematic-refactor implementation-note v1;
  compiler-process test-evidence v1 and interrupted review v3
Owner Summary: 用户质疑当前复杂度。先收敛运行路径与编译边界，保留正确性、
  公平计时和生命周期要求；停止继续细分编译基础设施。本提案没有改变现有授权。

## 1. 核对结果

本地及主机隔离工作树的实际代码均为92a8ac646ac182f2dac4dc4eef8a3dd0b331f1af，
核对时干净。compiler_process_preflight_v3 已结束，只有部分静态检查，没有结论、
新 ABI 探针或进程测试。其完整报告另行原样归档。先前0497573的唯一实际尝试
在首个 supervisor 的 API 检查失败，尚未启动 worker；原始失败及进程退出记录保留。
C2d3 的既有整体验收不撤销；整个 C4 仍是未完成的持续目标。

当前真正影响完整运行的缺口：

- optical_worker/worker.py::_reject_gpu_preparation 仍拒绝产品 PREPARE。
- optics 中还没有计划的 warp_timeline.py / warp_arena.py；C2e 物理执行图未交付。
- 计划中的 SetupWarmupExecutionAuthority、WarmupCompletionEvidence、
  FinalReadinessOrigin、PostReclaimIdleEvidence 尚无同名实现。既有 input upload
  与 kernel/factory 部件不等于完整十次预热、最终就绪、提交及回收流程已接通。
- native_setup.py 的旧 seal_and_quiesce_* 仍有直接同步及旧 abort 组合；
  完整集成须采用已批准的 seal → 受监督 barrier → DISARMED → quiescence 顺序。
- cuda_scene.py::prepare、cuda_lbvh.py::prepare/工作区查询、
  cuda_direct_light.py::prepare 仍会通过 _load_* 进入 load_inline。
  LBVH/direct-light 的模块顶层也导入 cpp_extension；只替换一个加载入口不够。

因此，完成当前15个进程用例并不能直接得到速度比较。此前把它描述成即将测量
的主要障碍不够准确。C2e/f、后续 C4 图与生命周期工作仍然存在。

## 2. 必须保留的产品要求

同一 immutable logical source、各后端独立 native allocation、真实 Go2 和冻结场景；
数值/图像正确性；明确 target/stream 和实际完成；十次真实预热；一次覆盖 seal 与
最终 barrier；失败状态不可恢复、资源安全回收或隔离退出。保留 R0/R1/R2/S 的不同
计时边界，编译/准备成本对双方对称处理，不混入 BVH 收益，不改性能门槛或退出条件。
完整 C4 与最后的单后端选择/删除仍是不同的验收范围。

这些要求不需要以不断增加第三方 Python 内部对象审计作为默认实现方式。
parent runtime-integration v3 §1/4 已明确：依赖是冻结且可信的，不是敌对 Python 沙箱。
固定源码/产物来源和真实资源生命周期仍需证明，但应优先利用现有部署及加载边界。

## 3. 建议的收敛方向：部署编译，worker 加载固定产物

remaining-setup gap-scan v1 的 G1 已明确列出这个可选方向。建议用一次有界设计替换
当前 worker 内 JIT 方案，而不是同时维护两条路径。它仍须完成独立设计审阅后才能实施。

| 边界 | 保留/复用 | 拟移除或迁移 |
| --- | --- | --- |
| 部署构建 | 三份现有 CUDA_EXTENSION_RECIPES 和唯一 kernel 源、固定工具链/ABI | 构建及编译器子进程责任集中到部署任务；不放入渲染 worker |
| worker 加载 | 已有固定文件清单、ExtensionFileLoader 托管、实际模块/entry 绑定和 guardian | 移除 cpp_extension/setuptools 初始化、shim、alias/resource/metadata 专用执行链 |
| 工厂准备 | 现有 scene/LBVH/render 实现、工作区和 out-style ABI | 工厂及工作区查询统一消费本 session 已加载模块；删除隐式 JIT 和缺失产物时回退编译 |
| 运行与计时 | native 输入、slot/event、预热、最终就绪、提交/完成/回收及监督 | 不把源码扫描、构建、导入审计塞进计时路径；不新增独立 comparator 产品实现 |

具体删减候选是 worker 的 cuda_compilation.py 执行链、setuptools_admission.py、
native_bootstrap.py 中相应 continuation/hook/初始化接线，以及不再需要的当前版本
编译事件重放/清单规则和测试。复用代码先迁移到唯一责任方；历史证据与历史记录解析
的保留要求单独核对。此处是删除面，不声称这些文件可以整文件直接删除。

compiler_supervision.py/compiler_job.py 的编译器树责任可从 worker 边界移出；
部署任务本身仍须有可靠的退出与残留处理。预编译不会让编译器进程凭空消失，
也不能把未证明的部署清理包装成简化成果。是否复用现有 owner，取决于部署方案；
本提案不新增另一套通用进程框架。

## 4. 一次替换设计必须解决的范围

固定产物清单需绑定三份 recipe 的源/参数、工具链及 Python/Torch/CUDA/架构兼容性、
构建输出和依赖文件；worker 只接受该部署清单的现存字节，缺失/变更直接失败。
加载仍在原 target 的受监督 setup 中，保留部分 create/exec 返回对象与失败资源。
部署报告只证明来源，不冒充 worker 实际模块、entry、设备或预热证据。

设计必须明确环境清单、source continuation、编译完成 readiness 的替代来源和版本
迁移；接通三个工厂及 LBVH 工作区查询；禁止新旧 worker JIT 双轨及自动回退。
Warp 的 wp.load_module/record_cmd 属于另一个实际编译模型，不能把 CUDA 的 .so
方案直接套给它。其已要求的 source/cache 来源、ModuleExec 持有和预热仍保留。
不在此处把构建时间估计成性能收益，也不承诺改方案会立即比完成现有 WIP 更快。

## 5. 完整路径与验收顺序

1. 一次确定构建/加载替换和旧机制删除面，完成独立设计审阅。
2. 交付实际固定产物加载及工厂绑定，连同必要失败路径做一次完整代码/执行预检。
   编译器、依赖和 ABI 不再各自成为持续扩展的产品子系统；保留实际必要验证。
3. 完成既定 C2e 的 Warp timeline/arena，并接上双方 provisional 图、十次真实预热、
   新 final readiness、最终图、seal/barrier。按 C2f 做真实场景正确性及失败生命周期。
4. 按既有阶段完成 C4c3/c4、C4d/B3、联合回收和系统重构，取得各正式审阅与 C4 验收。
   C4c3 中已独立批准的 acceleration interop 不当成通用 optical 后端桥删除。
5. 测量启用仍遵守冻结的后续 gate；先报告实测能力，再做正式速度比较与后端选择。
   本提案既不授权提前出 benchmark，也不把 C4 宣称为最终后端迁移完成。

结构性删减纳入用户已要求的系统 refactor，既要减少状态和特殊接线，也要验证行为。
审阅集中在真实架构/执行检查点；相关原始反例作为同一回归集保留，不为每个小修复
再建立一个长期子项目。没有实际故障或缺失契约，不增加防护层。

本轮只完成现状核对与可审阅的收敛提案。没有删除已验证代码、放宽既有契约、
运行新进程矩阵或 GPU。下一步是把上述替换范围落实成精确设计，而非继续当前
pidfd 预检链或把此提案本身当作 C4 完成。
