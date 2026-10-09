# Warp preparation: bounded code approval and actual IDLE failures

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: evidence-only record of approved code and author native cases
Author: Codex
Version: v3
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation code-review v4; implementation-note v4;
  test-evidence v2; design-request v2
Owner Summary: Fresh independent review approves bounded code at beedab6.
  Actual unknown-scope issuance quarantines without an invented guard; healthy
  IDLE entry failure performs one bounded sync and release. Native acceptance
  and all parent rendering/C4 work remain incomplete.

Approved exact code: beedab6143e702a1a50ef0e2686fd0138e194376.
Fresh independent session /root/warp_module_code_v4 received no implementation
conversation. Its full report is archived unchanged as code-review v4, SHA256
fa110c54ef0b0f2d753a45cdd3be1e361f95e355db51ca7bfb4713cdb58fecc8.
Verdict Approve for bounded module-preparation code, no substantive findings.
It independently closed F1/F2/F3 with12 original repro checks (4.77s) and365
repository tests (143.81s, tool47254 exit0), plus bounded replay of native v9.
It did not approve complete native acceptance or whole C4.

Both author cases below use the approved exact code in clean detached remote
/data2/shared/ang.gao/robot_sim_warp_modules_wip on tileops-dev, real Go2 960x640,
Warp1.12/H200 target1/ambient0. Interpreter is
/home/ang.gao/miniforge3/envs/robot_sim/bin/python (Python3.12.13); this corrects
test-evidencev2's concatenated interpreter/version typography. The same selected
v4 environment and original1000ms ACK/120000ms operation guards apply.

Command prefix: python -I -B /data2/shared/ang.gao/run_warp_module_author.py;
arguments are the exact output directory, go2_static_960x640_v1 and fault below.
Neither case emits an optical image or timing/result record.

| Raw directory under /data2/shared/ang.gao | Fault | Terminal result |
| --- | --- | --- |
| c4-warp-module-author-v9 | runtime_warp_modules_unknown_issuance | tool22167 exit0; worker3032750 exit74; supervisor3032749; parent3032698 |
| c4-warp-module-author-v10 | runtime_warp_modules_entry_raise | tool93138 exit0; worker3038685 exit74; supervisor3038684; parent3038633 |

v9 calls the actual issuer after all three real compiles and final DISARMED.
Exact error: ValueError("Warp module issuance requires healthy IDLE target scope").
Worker: IDLE/no reference/no failed guard,10 original completions,zero sync,
full quarantine/no release authorization. Supervisor:10 arms/10 completions,
EXITED/complete,worker74,no timeout/kill. The author reran the production custody
validator on the local archive, exit0; the reviewer independently did likewise.

v10 invalidates only the input report after asserting otherwise healthy IDLE
target scope and original valid input. Production rejection is SourceBootstrapError
("Warp modules require the original uploaded and validated input"). The existing
runner validates6 original completions/7 arms,one emergency sync,completed release,
aborted_synced_exit disposition,zero fabricated failed normal guards,worker74
and terminal supervisor exit. The nonzero worker code reports the expected
preparation error; runner exit0 reports that the negative assertions passed.
Both runtime-custody selection/attachment and guarded artifact replay pass.
No ordinary worker atexit occurred.

Exact remote/local archives match:

- c4-warp-module-author-v9.tar.gz:
  add2066ba7d3e62e9bd870134f3cd7966164f70aa87d2dae582410999f137188.
- c4-warp-module-author-v10.tar.gz:
  3a2922c2d248a88c020419aa9fd46c10615693aaee92fe3ac0359f36a6b87231.

Remote base /data2/shared/ang.gao; local base C:/Users/AG/新建文件夹/c4_assets.
Each archive contains its complete raw output directory. Before archiving and
starting another run, exact parent/supervisor/worker PIDs were checked absent
in /proc. A read-only ps check for v9 returned no rows (exit1, consistent with
absence); the subsequent explicit absence checks passed. Post-v10 nvidia-smi
listed only unrelated MPS server2238410, which was not touched. All named author
and reviewer sessions are terminal at this cut; no caches or evidence removed.

Limits: held_compilation_records asserts identities before fault injection;
it alone does not prove every root after arbitrary failure. Successful compile
outer snapshots do not prove load/adoption failure restoration. The remaining
native cases must establish those properties with actual failure observations.
Healthy issuance, incomplete entry/issuance, updated cold positives and the
remaining source/cache/artifact/load/hang cases still require execution and
independent native acceptance. Continue the approved one-slot graph/timeline/
arena, real warmups/final readiness/barrier, CPU-oracle rendering/lifecycle,
C4c3/c4,C4d/B3, systematic refactor and fresh whole review. Product PREPARE,
performance/results/baselines and backend selection remain closed.
