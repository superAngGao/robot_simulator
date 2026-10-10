Initiative: Q54 / P12.4 / C4c2
Stage: Bounded repair and native evidence
Author: Codex
Version: v1
Date: 2026-10-10
Status: implemented
Related Files: native-state-reader design-request v1 and code-preflight-review v3; runtime-integration design-request v3; terminal-target test-evidence v1; warp-exit-fault test-code-preflight-review v1
Owner Summary: The corrected foreign-stream runner reached a real ownership-order regression. Candidate cf7f5a6 moves the existing returned-stream adoption before reader validation. No new mechanism or weaker assertion is introduced. Fresh independent preflight approves this repair; all four original native foreign/unknown cases now pass. The full17-case reader/input regression is running. Separately, the approved actual Warp exit-not-restored test passed.

## Corrected harness and real regression

The old reader matrix's profile-selection bug was fixed only in the external
runner: pytest_collection_modifyitems binds current pinned PROFILES and
GO2_PROFILES on each actual collected item.module, verifies its function globals
and the exact four selected lane/fault pairs. Original test bodies/assertions,
production sources, inventory pins and 120000ms deadlines stay unchanged.
Collection-only verification selected exactly four tests. Runner SHA256:
d221278460e7510053e4883f4575e8361710fbd5f71972099d843e6a588911bb.

At exact22b1db0ee619f05e30f03bd5e339f03e883e97a5, campaign
`/data2/shared/ang.gao/c4-reader-22b1db0-foreign-scope-v2` ran from
2026-10-10T10:50:26.463943Z to10:50:44.903861Z. First case failed:
runtime_foreign_stream_return / Warp. Three others did not execute due to -x.
The real helper created/returned a device0 stream under target1 setup guard3.
Production retained and journaled the return, then reader.stream_returned
rejected its target cache prerequisite before the existing device-owner
adoption. Terminal custody was safely full_quarantine, unknown=true, exit74,
no emergency sync or release, but possible_dirty_device_ordinals remained[1].
The unchanged assertion that known returned device0 is recorded failed.
This is a bookkeeping/adoption-order regression, not premature GPU release.

All owned PIDs2821013,2821020,2821025,2821086,2821087 are absent in scoped
OS/GPU checks. Unrelated GPU processes were preserved. Observer captured one
live identity chain, zero errors. Complete raw failed attempt is preserved:

- c4-reader-22b1db0-foreign-scope-v2-evidence-v1.tar.gz:33files,
  2,226,370raw bytes; SHA256
  f57b304eaee6af6f73a04a3038bd9e832754112cfc758eccae884da50f9b7df7.
- Inventory SHA256680fb3e9dbce8e816f391e31bc9f830585ff7f6ee09456323ec70b636f011606.

## Small repair, exact candidate and proof

Base91e11eab275f6434754b9449200bf5cb7bf45655;
candidatecf7f5a6d9ff9b3971144410a7c3ad156a806646c.
Both initialization branches now run their existing _hold_runtime_owner stream
adoption before reader.stream_returned. Immediate source retention and
returned_owner journaling still precede metadata access and adoption. The
existing guardian records foreign owner scope then fails closed before any
reader continuation. A target stream is adopted before reader/cache/query
failure. Metadata/adoption errors still use the original retained graph and
unknown/quarantine path. No new getter, guard, ownership table, admission rule,
native ABI, release or successful observation is introduced.

Four CPU regression cases execute actual initialize_native_runtime with real
guard/controller/guardian and modeled runtime leaves. Both lanes are covered
for foreign returned ordinal0 and target ordinal1 with after-query failure.
Before the repair all four failed at the intended assertions: missing0 or
stream not adopted at the reader failure. An earlier test-draft import typo
failed before execution and was corrected before this meaningful reproduction.
After repair, binding/reader observer/integration suites passed125 with no
skips in11.16s, using retained pinned Warp source for its optional CPU fixture.
These are not real-device acceptance of the candidate.

All91 prospective committed-LF sources pass the unchanged AST rules. Only the
bootstrap source/class pins and resulting catalog digests refresh:
bootstrap7822eba259d80919cb5791e884d1084b4c099ba3fa5bf78fe3c6e38cc962546c;
catalog0433890aa9f5720001e91426c8bd37bb9061370338f56861a22f852dce5297cc.
Ruff comparison: bootstrap76 existing diagnostics both revisions; binding2
before/1after, no new diagnostics. This is not a clean full lint result.
Whitespace check passes. Full local source/bootstrap/policy/custody regression
completed317passed,13skipped in371.07s. The13 Windows skips (real Go2 input,
actual POSIX release bytes/entry and pinned CPython3.12 generated stdlib)
were separately run on primary183's exact frozen cf7f5a6 checkout:
13passed/no skips in37.41s. Remote evidence prefix
`/data2/shared/ang.gao/c4-stream-order-cpu-cf7f5a6-v1`; runner2853810 and
pytest2853817 both absent in scoped OS/GPU closure, clean tracked checkout.
Local pytest34544 exited0; it and its last observed child36624 are absent.
Local log: `c4_assets/stream_adoption_order_source_regression_v1.txt`.
Remote launch/exit/closure/stdout/stderr were copied into local c4_assets.

Frozen independent checkout: `C:/Users/AG/新建文件夹/c4_stream_adoption_order_review`.
Fresh built-in reviewer creation again returned agent thread limit reached;
the owner-authorized equivalent new CLI session was started without author
conversation. Session01a12574-b4a1-7d92-b575-8308db14ce63, CLI37468,
launcher47368, started2026-10-10T10:56:07Z, task handle38556.
Request SHA256fcdc80ac8386d290a9a5c7f8a9431b8ef98bab01847e56b1ff393cf502599c08.
Fresh review returned Approve for bounded native validation of the exact
four-file delta. It independently passed211 CPU cases with no skips, reproduced
the old order defect, checked parser adjacency and all91 LF sources, and
replayed the preserved original real failure. Full report is archived verbatim
as stream-adoption-order code-preflight-reviewv1; working-file SHA256
745068d126713b352c49e5c36a9278d55d0641db77547d2d43c2eaa7e706fb8e.
Tool38556 exited0; CLI37468 and parent47368 are absent. Frozen checkout remains
clean including ignored files. This is preflight, not whole hardware acceptance.

After confirming earlier native rounds remain closed, started exactcf7f5a6's
four-case native rerun at2026-10-10T11:07:06.017857Z. Remote campaign
`/data2/shared/ang.gao/c4-reader-cf7f5a6-foreign-scope-v1`, wrapper2868772,
driver2868786, tool10186. Runner SHA256
de6f3cc93f73180f6c2e2c8f677d159328f215323efb1b61e895cb179f1f6211.
Same four cases, original assertions and exact current profiles. This round
ended2026-10-10T11:10:08.064714Z:4passed,83deselected in181.69s. Both returned
foreign-stream cases record dirty[0,1]; both pre-return unknown cases retain
unknown scope. All four workers exited74 with expected whole-graph quarantine,
zero emergency sync and no ordinary release; every original assertion passed.
Four real identity chains were observed, with no observer errors. All11 owned
OS/GPU identities are absent:2868772,2868786,2868791,2868852,2868854,2869783,
2869786,2872054,2872055,2872841,2872847. Original faulty attempts are preserved.

Archive `c4-reader-cf7f5a6-foreign-scope-v1-evidence-v1.tar.gz` contains96raw
files,29,000,102bytes, plus inventory. Archive SHA256
47f16a1092d84f4f0e9aae8b84d0020e0dbaca92993c4c2083dbc1d696a28af6;
inventory SHA2568971d7f8556edfc4fc84e25340c1060ef2674a7b89b2da5bc769f32cdbcee600.
Copied to local c4_assets. This is bounded author-observed native success;
whole-reader independent hardware acceptance remains outstanding.

After this terminal/closure check, complete17-case reader/input regression
started2026-10-10T11:11:43.042205Z at
`/data2/shared/ang.gao/c4-reader-cf7f5a6-native-scope-v1`, same exact candidate.
Wrapper2877617, driver2877625, tool22881. Original cases/assertions are unchanged;
the same corrected actual-module profile hook and external identity recorder
are used. No complete outcome is claimed while it runs. Runner SHA256 is
ab65b9aae54e929e3a3086a6d291f73517bda1b2a141bae8f93e0ec176b579ea.

## Actual Warp exit-not-restored evidence

Exact approvedfc23b4e38d816d5108b61363a5372710db6a2937, primary isolated
checkout `/data2/shared/ang.gao/robot_sim_worker_fc23b4e`.
Campaign `/data2/shared/ang.gao/c4-reader-fc23b4e-warp-exit-v1` started
10:52:23.679125Z, ended10:52:50.641186Z. Collection selected exactly one case;
the original assertions passed:1passed,16deselected in26.61s.
Runner SHA256404623fb03e6c7b51a17db9efb551766bb1be368308a7aa3a0c1733d6afcfce6.

The real original Warp scope entered. Controlled omission of its native pop
left current CUDA context98343657594864 equal to retained target_context and
ordinal1, while Python default_device was restored to0. Only primary1 was
active. Exactly one raw readback_scope/after observation preserved this state.
Existing assertions require original guard5, failed full quarantine, no new
normal ARM/COMPLETE, terminal ARM/sync, release, or input-validation receipt.
Worker2829805 exited74; supervisor2829804 observed/reaped it; wrapper2829681,
driver2829698 and pytest2829703 also exited. All five are absent in scoped
OS/GPU checks. No owned native job remains from either round.

- Archive c4-reader-fc23b4e-warp-exit-v1-evidence-v1.tar.gz:35files,
  4,752,167raw bytes; SHA256
  7172c810480f3cfbab5848ef4908597cad38e08ba8fcda1ee0f3825a945c07aa.
- Inventory SHA2564608d283eeba064d4d96c5608f22a970e23a1179137ef12e6bf62f91612978b5.

Both archives are copied to local c4_assets. This is author-observed actual
fault evidence, not independent whole-reader hardware approval or evidence
for the unexecuted ordering repair. The fault models omitted exit restoration;
it does not claim an actual driver malfunction.

## Remaining full objective

The four original foreign/unknown cases and the complete reader/input regression
have now passed on the exact frozen repair; completed evidence follows below.
Obtain the complete reader hardware assessment. Final-worker physical/numerical proof, whole fixed-work
and preallocation refactor, remaining worker faults/close/F3, C2f and subsequent
C4 stages remain unfinished. No speed comparison or later gate opens. This
bounded fix is not the goal's completion, and existing per-event history
copy/replay complexity still requires systematic consolidation.

## Completed current-revision native regression and framed probe

The17-case campaign above ended2026-10-10T11:34:35.192660Z, exit0:
17passed/no skips in1371.78s. All eight normal combinations passed (both lanes,
asymmetric/dynamic/Go2 960/Go2 1920), as did six getter/after-return/after-append
faults, missing Warp outer scope, CUDA C1 copy and actual Warp exit-not-restored.
All17 external live process identity chains were captured with zero observer
errors. The recorded37 owned PIDs (wrapper/driver/pytest plus34 workers and
supervisors) are absent from OS and GPU inventories. The stale local observation
SSH46448 (parent5164, creation2026-10-10T19:11:41.6893550+08:00) was closed only
after remote terminal exit and full scoped closure, matching exact command and
creation identity; no remote native process was killed or restarted.

Archive c4-reader-cf7f5a6-native-scope-v1-evidence-v1.tar.gz:378raw files,
185829259bytes plus inventory. SHA256
5e54d56e64de3e65c09c73df0f47a9ecfbfd6e9e37a41fc3ae41f003336fd652;
inventory b2e7f21eae437b9317d888e83dffc0143aa6dc39d77c4ecb0a28c3e7dad1b91b.
Downloaded local c4_assets copy matches the remote archive hash.

After that closure, currentcf7f5a6 actual framed worker probe ran at
/data2/shared/ang.gao/c4-worker-cf7f5a6d9f-probe from11:37:30.418783Z to
11:41:54.329307Z. Both original lanes passed two asymmetric FULL/shadow samples
with resolve/release/reclaim and normal close, exit0. The unchanged approved
driver is run_worker_matrix_0be9876.py, SHA256
89ef276136c74efd830d8c4b476d62893e8d67148317f598d808dd2d93375ca3;
its revision argument selects exactcf7f5a6, not the older name in the script.
Wrapper2968488, driver2968498, Warp supervisor2968884/worker2968889 and CUDA
supervisor2973774/worker2973792 are all absent from OS/GPU inventories.
Tool82063 returned exit0. This probe retains its two-sample scope; it is not
the missing final-output CPU oracle, full worker-fault or constant-work proof.

Archive c4-worker-cf7f5a6d9f-probe-evidence-v1.tar.gz:93raw files,
13829929bytes plus inventory. SHA256
700412e637e00bda645431ba6bd20067b15f8f8095ccf600efdc318c6c936a7c;
inventory f0a94c77744d34e6f32f909e59d8e3d0b4aa542cf5267dcf70f390d576f73964.
Local downloaded archive hash matches. No wall-clock duration here is a
renderer timing or backend speed comparison. Independent full reader acceptance
and all remaining C2f/C4 work are still open.
