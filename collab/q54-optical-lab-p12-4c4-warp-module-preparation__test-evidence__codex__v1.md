# Warp module preparation: author runs and first code review

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: author native evidence and code-review repair, not acceptance
Author: Codex
Version: v1
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation implementation-note v2; code-review v1; design-request v2
Owner Summary: Both real Go2 resolutions compiled all three existing Warp modules
  and reused the original executables before terminal disposal. Independent code
  review found one ABI/entry snapshot defect; repaired with regressions. The
  revised candidate still needs fresh review and native verification. C4 remains
  incomplete and no performance or optical-output claim is made.

## Executed native evidence

Primary host tileops-dev; isolated /data2/shared/ang.gao/robot_sim_warp_modules_wip.
Python /home/ang.gao/miniforge3/envs/robot_sim/bin/python, Warp1.12, H200 target1,
ambient0. Original CUDA worktree stayed unchanged at66240e4. The author used
the existing run_case and captured worker/supervisor, with scratch launch script
/data2/shared/ang.gao/run_warp_module_author.py. Each run selected the committed
183_warp_go2_author_v4 profile with trusted expected digest
02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.

| Directory under /data2/shared/ang.gao | Code | Case | Actual result |
| --- | --- | --- | --- |
| c4-warp-module-author-v1 | 9a3e4728d310d7dbac21fd3d4eecf983b5b53700 | Go2 960x640 | worker74; guarded import inventory rejected the omitted four renderer structs; runner/tool88571 exit1 |
| c4-warp-module-author-v2 | 4400cfd3adc1eee09c5d1f71f2dc711f0d68a189 | Go2 960x640 | worker0, runner/tool22800 exit0; 11 normal completions,12 arms |
| c4-warp-module-author-v3 | 4400cfd3adc1eee09c5d1f71f2dc711f0d68a189 | Go2 1920x1080 | worker0, runner/tool16778 exit0; 11 normal completions,12 arms |

Both positives loaded all three actual modules, then the same worker loaded
each again with unchanged actual ModuleExec identities. Cache directories were
/tmp/robot-sim-warp-mha7wy_q and /tmp/robot-sim-warp-9bd9xq3e, respectively:
distinct newly created mode0700 roots. Post-exit checks matched all nine recorded
file sizes/hashes for each run, including generated .cu, metadata and CUBIN.
readelf reported ELF64/NVIDIA CUDA and flags0x5a055a for all three binaries;
each generated CUDA file includes builtin.h. No optical kernels were launched.

The original observer proves selected device/stream inside each load, but these
first two positives do not themselves record the outer post-ScopedDevice state.
Do not claim their input-phase observations prove compilation restoration.
The revised support driver now separately records before/after outer compile
scope and asserts restored ambient0/current-context-none without reusing the
input-phase custody schema; that revised observation is not yet executed here.

The archive script observed all nine parent/supervisor/worker PIDs absent:
2674645,2674785,2674787;2685623,2685873,2685901;2737991,2738154,2738162.
All named native tool sessions are terminal. No ordinary atexit occurred in
the workers. Exact caches and failure history were retained; nothing was deleted.

Remote and local copies of c4-warp-module-author-evidence-v1.tar.gz have SHA256
04d2919f56d436f4d5387b8e98dc03f445daa851726673e33e8eef5313b06462.
Remote path /data2/shared/ang.gao/c4-warp-module-author-evidence-v1.tar.gz;
local C:/Users/AG/新建文件夹/c4_assets/c4-warp-module-author-evidence-v1.tar.gz.
It contains all three raw runs, both successful recorded cache artifact sets,
the profile and author launch/inspection/archive scripts. Adjacent JSON summary
records exact terminal statuses. This archive proves execution at4400cfd only.

## Independent review and repair

Fresh /root/warp_module_code_v1 reviewed base0356108 to exact clean4400cfd.
Its full report is archived unchanged in code-review v1, SHA256
3a3f3c9cfbdf80b00549513de6a22c28a7199a2acc633cc4ddef6b7546e02af2.
Verdict Request changes, one blocking P2 F1. Independent CPU regression result:
152 passed,9 skipped in275.10s, session31478 exit0. Reviewer read both terminal
native evidence sets and checked compiler flags/header closure; it did not run
a new native campaign or approve native acceptance.

F1 reproduced in-place array dtype/ndim and kernel entry-hash mutation passing
actual later module-set validation. Stored arg.type values aliased mutable
annotations; cached module hashes did not change. Four scratch checks passed
as counterexamples/control, not as proof of correctness.

Repair retains original owners and freezes Warp's own immutable type codes,
array dtype/dimension values, recursively described struct fields and scalar
storage type. Kernel key/hash/mangled entry are captured after legitimate
module hashing initializes the entry, before native loading. The same snapshot
is checked after loading and before later admission. No additional renderer,
guardian, configuration state or generic attestation system is introduced.

Author tests after repair:23 module-preparation tests passed in5.10s, covering
the original mutations, nested struct/array values, unchanged repeated admission,
and the prior lifecycle/partial-publication cases. The unchanged reviewer repro
is preserved at c4_assets/test_review_warp_abi_alias.py. A separate adaptation,
c4_assets/test_author_warp_abi_repair.py, additionally executes pinned Warp
get_type_code and changes expected outcomes to rejection:4 passed in2.46s.
These are CPU metadata checks, not native ABI acceptance. The fixed native driver
now has dtype/ndim/entry-hash fault cases against real compiled kernels; their
execution and fresh independent re-review are pending at this record cut.

## Remaining obligations

Complete the revised native positives/restoration checks and mandatory cache/
configuration/source/ABI/record negatives, original-guard load/publication/
adoption failure and hang, guarded import timing, actual-IDLE entry/issuance
failure disposition and fixed-CUDA/input regressions. Then continue the parent
timeline/arena, one-slot graph, ten actual warmups, new final readiness, final
recording/seal/barrier, rendering correctness/lifecycle matrices, C4c3/c4,
C4d/B3 and systematic whole-C4 refactor with fresh independent whole review.
Product PREPARE, benchmarks, results/baselines and backend selection stay closed.
