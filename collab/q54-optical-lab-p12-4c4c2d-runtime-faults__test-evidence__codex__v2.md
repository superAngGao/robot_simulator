# Actual native fault continuation and preserved fixture failures

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v2
Date: 2026-10-08
Status: implemented (bounded evidence, not whole-stage approval)
Related Files: runtime-custody test-evidence v1 and implementation-note v2;
native-scope-aggregate gap/design reviews v1/v2
Owner Summary: Preserve actual fault attempts, their test-fixture defects and
scoped cleanup evidence. Native ambient completion remains separate. C4 is
active and no later product gate is opened.

## Actual executions and identities

The first48-case attempt at8c794db stopped after10 passes on a missed Warp
backend exec injection. The original implementation-note v2 records its raw
archive and failure. The corrected actual SourceLoader interception at4435593
then passed all4 selected Warp/CUDA before/after backend-exec cases, with70
deselected, in77.58s. Four live external worker/supervisor/pytest identity chains
were captured; all owned OS identities and GPU PIDs were absent afterward.
Archive c2d3-4435593-backend-exec-faults-v1.tar.gz:422179bytes, SHA256
5918fffb767b38fe6247c342ecff00c171bf80593230a5f478ae9557af9f340e.

The remaining70-case run at clean bcc1bac4a6fa0ae9cfc758e85bf0a7c77fdca455
ended **1failed,40passed,4deselected in1920.97s**. It stopped at
runtime_terminal_cleanup_hang-warp. Assigning release_whole_graph on the actual
slotted guardian instance raised AttributeError during fixture installation;
the fault marker was absent and the test correctly failed. This is not an
executed cleanup hang and does not validate cleanup behavior.

The run captured41 distinct live OS identity chains, no observer errors and no
outer timeout. The raw attempt retains all outputs, partial/current aggregates,
trusted inputs, selections and external process identities. Subsequent exact
PID/start checks found all83 distinct worker/supervisor/pytest identities absent;
the runner was also absent and no owned PID appeared in nvidia-smi. Only owned
identities were checked; no process was signaled. The stale local SSH session
was closed after remote completion and those absence checks.

Archive c2d3-bcc1bac-runtime-fault-matrix-v2.tar.gz:10154215bytes, SHA256
35a45ffa9711fb0ae2c57796191d40366bcf0e0a76fb0b5376c8453a8cc9adcf.
Runner run_c2d3_runtime_fault_matrix_v2.py SHA256
fe6b6c4cd9e37f131887aa8ab8fd419df2697bdaeda7599d62ba2f061ad58085.
Both archives are retained under /data2/shared/ang.gao and local sibling
c4_assets, with local hashes verified after transfer.

Fresh CPU-only replay of all41 retained bcc records additionally passed durable
selection checks against each available full terminal trace, existing parser
checks and worker-prefix attachment; backend imports were empty. Killed-worker
prefixes remain prefixes. This includes the failed fixture's early record as
valid descriptive custody, not as a passed intended fault. Script/log:
c4_assets/replay_c2d3_fault_records_v1.py and
c2d3-bcc1bac-fault-record-replay-v1.log (log SHA256
3d4e25ac726f46993349fcccba4d9852e1afa9f9efce15f8a870c19f9761d336).

## Repairs and completed continuation

14accac054605459966fe37af51fd7258efc073d changes only tests: intercept the
guardian class method with an exact-instance check; bind available terminal
selection counts to the full actual trace; check exact ordinal/UUID/driver rows.
Local parser/manifest53passed1.60s;85 fault cases collected. New11 scope/
publication negative cases were already present at3617ab5 and remain distinct
from positive native-scope proof.

Remote was advanced by verified bundle and detach, without reset, to clean
71566e0ffd51283071f56903bfd414291348f8b1. Its production remains unchanged9b5702b.
The41 remaining fixed negative cases passed under the original deadlines:
**41passed,44deselected in2516.08s**, in
c2d3-71566e0-runtime-fault-matrix-v3. Runner SHA256
c2db3da6fab291ac1b9684f0aa5a985548e697153309f10f88ba76ac687c4894.
All41 live OS chains were captured without observer errors or outer timeout.
All83 distinct owned worker/supervisor/pytest PID/start identities were absent
afterward; the runner was absent and no owned GPU PID remained. Archive
c2d3-71566e0-runtime-fault-matrix-v3.tar.gz:7819940bytes, SHA256
7e8998a5bd1a1fdbcdecd45d056417761000a3dd513aad67e57147f64d550163,
retained on both hosts with matching hash.

Real cleanup hangs now hit guardian.release_whole_graph under emergency guard6;
missing-exit hangs occur after actual terminal evidence. The original watchdog
kills/reaps these workers. All ACK/EOF and six-phase parent-pipe EOF cases pass
their actual protocol/exit checks. Initialization wrong-ACK/write-EOF and the
two Go2 native-global admitted-then-raise cases select unchanged prior validated
cuts (six total); the other35 select current records, including explicit killed
prefixes. Both-lane real foreign-return and unknown-pre-return cases quarantine;
CUDA C1 stream-only copy, module-replacement and interpreter-alias faults also
pass. Original records and raw failure tails remain in the archive.

Together with the preceding4 backend and40 native passes this covers85 distinct
fixed fault cases at the explicitly stated test revisions, all with production
unchanged9b5702b. It is not independent code or whole-C2d3 acceptance. The new
scope schema/readback WIP is local only and cannot be attributed to these runs.

## CPU supervisor regression history

At4435593 the CPU driver/custody/source-process suite produced1failed33passed
in41.28s: the source fixture still expected12 captured modules instead of the
approved16. Its audit-fault trigger had the same stale value. Updating only
those fixture counts atbcc1bac produced **65passed172.65s**. Both original logs
and artifacts remain. Later scoped checks found all recorded worker/supervisor
PIDs absent (34 identity files in the failed attempt,65 in the repair run), with
no owned GPU PID. These older CPU identity files lack external live start ticks;
that stronger proof is not invented retrospectively. Original test assertions
still check actual child exit/reap.

Combined archive c2d3-supervisor-cpu-regressions-v1.tar.gz:3888430bytes, SHA256
2bb891e10ab557feb4a98459eaf2d5ee97e9f82c318fd33127cf5d88eeb7d373,
retained remotely and locally with matching hash. This is CPU custody regression,
not actual native ambient or final preparation acceptance.
