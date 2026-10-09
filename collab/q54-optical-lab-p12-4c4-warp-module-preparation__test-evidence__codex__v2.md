# Warp module preparation: revised native observations

Initiative: Q54 / P12.4 / C4 remaining render integration
Stage: bounded author evidence, not integrated native acceptance
Author: Codex
Version: v2
Date: 2026-10-10
Status: draft
Related Files: warp-module-preparation design-request v2; implementation-note v3;
  code-review v2/v3; test-evidence v1
Owner Summary: Revised real Go2 compilation preserves ambient context, actual
  compiled ABI/entry mutations reject, and the repaired unknown-scope entry
  preserves IDLE history without synchronization. All named processes exited.
  The separate aggregate-issuance defect and complete acceptance remain open.

## Executed cases

Primary tileops-dev, isolated /data2/shared/ang.gao/robot_sim_warp_modules_wip,
Python /home/ang.gao/miniforge3/envs/robot_sim/bin/python3.12.13, Warp1.12,
H200 target1/ambient0. Existing supervisor runner and 120000ms operation guard,
1000ms ACK; no revised timeout, new compiler manager or dependency installation.
All cases use real go2_static_960x640_v1 and the selected v4 environment whose
trusted digest is02429134207218e87ca1305e6c89cc7b34948cda41499f1ac7399ac961ffbeeb.

Command pattern:
python -I -B /data2/shared/ang.gao/run_warp_module_author.py
/data2/shared/ang.gao/c4-warp-module-author-vN go2_static_960x640_v1 FAULT

| N | Revision | FAULT | Worker/supervisor/parent | Observed result |
| --- | --- | --- | --- | --- |
| 4 | 0194cb2442c52300b06fd33f7042ed291e723af3 | runtime_warp_modules_success | 2803385/2803383/2803295 | worker0,11 completions/12 arms,one terminal sync |
| 5 | 0194cb2 | runtime_warp_modules_abi_dtype | 2812987/2812979/2812923 | worker74,10 completions/11 arms,zero sync |
| 6 | 0194cb2 | runtime_warp_modules_abi_ndim | 2855443/2855440/2855212 | worker74,10 completions/11 arms,zero sync |
| 7 | 0194cb2 | runtime_warp_modules_entry_hash | 2913574/2913570/2912530 | worker74,10 completions/11 arms,zero sync |
| 8 | f83edafa8c30d448127d230a5fb670c89da00194 | runtime_warp_modules_unknown_entry | 2988790/2988789/2988720 | worker74,6 completions/6 arms,zero sync |

All five runner calls exited0, respectively tool87788/26149/31144/35549/49925.
For negative cases exit0 means the real failure was observed and matched the
acceptance assertions, not successful preparation. Each supervisor reported
EXITED/complete with no timeout/kill. Runtime-custody attachment and existing
guarded artifact replay passed. No worker performed ordinary atexit.

Case4 records three compile outer snapshots, each before==after, ambient default0
and current-context-none. Its six inner load/reuse calls bind target1/held stream,
and all three actual ModuleExec identities remain unchanged on reuse. Its fresh
cache /tmp/robot-sim-warp-cko762m3 differs from prior workers; post-exit inspection
verifies all9 recorded file sizes/SHA256, private0700, ELF64 NVIDIA CUBINs and
generated includes. This proves successful restoration, not failure restoration.

Cases5..7 mutate actual compiled kernel annotation dtype/dimension or entry hash,
then call the actual production validator. All raise exactly
ValueError("Warp executable or kernel ABI changed") before module reuse.
The original active guard fails; no additional ARM or sync occurs.

Case8 retains an otherwise valid real input report, records unknown scope while
IDLE and calls production prepare_warp_modules. Actual error is SourceBootstrapError
("Warp module preparation requires healthy IDLE target scope"). Worker summary:
guard_state=IDLE,guard_ref=null,guarded_segment_failed=false,unknown_device_scope=true,
disposition=multi_or_unknown_device_quarantine_exit,custody=full_quarantine,
release_status=not_authorized. There are no guard_failed events. This closes the
native original F2 entry case only; post-final-DISARMED issuance is distinct.

## Raw archive and process hygiene

Remote /data2/shared/ang.gao/c4-warp-module-author-evidence-v2.tar.gz and local
C:/Users/AG/新建文件夹/c4_assets/c4-warp-module-author-evidence-v2.tar.gz have matching
SHA256 f5dc01cec70068c944429d2f68856ef332ad52525bf7828cf3d82a0c1a9d5877.
Adjacent JSON records the exact cases, all15 parent/supervisor/worker PIDs absent,
and post-run nvidia-smi process records. Unrelated jobs remain untouched. The
archive contains all5 raw directories, case4 recorded cache artifacts, selected
environment and exact author launch/inspection/archive scripts. No cache deleted.
All named author jobs are terminal; remote tree remains clean atf83edaf.

The first read-only remote absence check had a PowerShell quoting error and
did not execute the intended remote Git command. A corrected check confirmed
the prior v7 PIDs absent before fast-forwarding the isolated checkout. Two local
archive-inspection attempts failed on quoting/Unicode stdin; the successful
relative-path stdlib read independently printed case8's actual failure/summary.
These were inspection errors, not native retries or overwritten execution runs.

Complete current-revision positives, source/cache/options/artifact/record faults,
native load/publication/adoption/hang, ARMED-before-import, healthy/incomplete/
unknown IDLE entry and issuance, failure restoration and retained partial-root
checks remain distinct obligations. Continue the approved graph/timeline/arena,
warmups/final readiness, rendering/lifecycle, C4c3/c4,C4d/B3, systematic refactor
and fresh whole review. Product PREPARE, performance/results/baselines and
backend selection stay closed. No optical output or speed comparison is claimed.
