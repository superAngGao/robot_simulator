Initiative: q46-dim6-cpu-multipoint
Stage: review
Author: codex
Version: v1
Date: 2026-04-21
Status: draft
Related Files:
- collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md
- collab/q46-dim6-cpu-multipoint__challenge__codex__v1.md
- OPEN_QUESTIONS.md
- tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py
Owner Summary: 这轮主方向是对的：Q46 dim 6 已从“CPU 单点”修正为“部分澄清”，body 1 flat-box 的 count parity 也被显式固化了。我已在 `env_tilelang_20260119` 上跑过 GPU 验证，目标测试和整份 Step 6 文件都通过；没有需要退回重做的阻塞问题，但有两处需要对齐：Step 6 测试里 body 1 的 4-point 断言被重复加入了一次，implementation-note 本身也没有按 repo 规定的 implementation-note 模板组织。

## Findings
1. [medium] [tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:310) — `test_cpu_gpu_sorted_ground_contact_match` 里 body 1 的 4-point parity 断言被写了两遍：一次在 body-set comparison 后（[310](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:310)-[317](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:317)），一次在 per-body max-depth comparison 后（[353](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:353)-[360](/home/ga/robot_simulator/tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py:360)）。这不是功能错误，但会让测试意图变得重复，也和 implementation-note 里“在 per-body max depth 比较之后加入”这一描述不完全一致。建议保留后一处、删除前一处，并把末尾注释补齐成 note 里写的完整版本。

2. [medium] [collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md](/home/ga/robot_simulator/collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md:1) — 这份 implementation-note 没有按 repo 规定的模板组织。`collab/README.md` 要求 implementation-note 包含 `Open Questions Addressed / REFLECTIONS.md / PROGRESS.md Impact / What Changed / Files Touched / Tests Added / Updated / Known Limitations / Commit` 这些固定段落，而当前文件改成了 `Summary / 实际变更内容 / 关键思考 / 覆盖情况 / 残余风险` 的自定义结构，并在 header 里加入了非标准的 `Commit:` 字段（[8](/home/ga/robot_simulator/collab/q46-dim6-cpu-multipoint__implementation-note__claude__v1.md:8)）。内容本身是清楚的，但它偏离了 repo 的 collab workflow，会让后续 owner/cross-agent 扫读成本变高。

## Coverage Gaps
- 这轮把 `BoxShape` 平地场景的 count parity 固化到了 Step 6 mixed-shape fixture 上，但还没有独立的 box-ground CPU/GPU parity test。也就是说，Q46 dim 6 里最关键的 count-level 契约仍然是“由集成 fixture 间接覆盖”，不是“由专项测试直接覆盖”。
- `ConvexHull > 4` 底面顶点的 residual mismatch 仍然只停留在文档层说明，没有新增回归测试去锁定“CPU all-vertices vs GPU cap-4”这一差异的边界。
- 这轮 review 已在 `env_tilelang_20260119` 上执行：
  - `pytest -q tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py -k test_cpu_gpu_sorted_ground_contact_match -q` → passed
  - `pytest -q tests/gpu/collision/test_b5_d6d7_cpugpu_multienv.py` → `5 passed in 15.07s`
  但验证仍主要集中在现有 fixture；它不能替代专门的 box-ground parity coverage。

## Residual Risks
- body 1 的 flat-box 断言是 fixture-sensitive 的。只要 A.link1 的 shape composition、姿态或初始高度被重构，这个断言就会跟着变；这本身是可接受的，但需要在测试注释里继续把它写得足够显式。
- `OPEN_QUESTIONS.md` 的表述现在比之前准确得多，但“flat box 实测返回 4 个接触点，与 GPU 行为对称”这句话仍然容易被快速浏览的人读成“dim 6 大体已解决”。在专项 parity test 落地之前，这一项继续保持 `🔄` 是对的。
- implementation-note 当前把实现 commit `e22241c` 放在 header 里，而用户消息里引用的是 note 自身的 commit `ab35b16`。这不影响技术内容，但如果不在 `## Commit` 段里明确区分“实现 commit”和“note commit”，后续追溯会有一点混淆。
- 我本地额外复验了 Step 6 fixture 的 ground counts：CPU 和 GPU 在 env0 都是 9 个 ground contacts，per-body 分布一致为 `{0:1, 1:4, 2:1, 3:1, 4:1, 5:1}`。这说明当前 body 1 parity 断言和“总数目前恰好相等”都是真的；但 challenge 里的结论仍成立，即不应把这种 fixture 级等号上升成长期语义契约。
