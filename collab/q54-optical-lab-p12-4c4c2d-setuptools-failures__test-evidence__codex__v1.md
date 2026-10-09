# Actual CPU dependency positive and failure matrix

Initiative: Q54 / P12.4 / C4c2 / CUDA G1
Stage: actual bounded execution evidence
Author: codex
Version: v1
Date: 2026-10-09
Status: draft
Related Files: setuptools-failures review/request v1/v2; accepted setuptools-admission design v2; setuptools-cpu-preflight test-evidence v2
Owner Summary: The approved exact campaign completed all33 cases: one real installed-dependency positive and32 intended denials. All33 terminal records replay through the production parser, all67 recorded campaign/parent/child PIDs are absent, and no backend/compiler/device execution occurred. This is dependency CPU evidence, not compilation or C4 acceptance.

## Revisions and independent review

Executed source/test candidate: `427c90ae6b69360b5cd16178e027677bf3ae09a0`.
Target isolated checkout remained clean at that revision before and after every
case. Local report-only archive commit: `9e83746` (resolve through Git history).

Original candidate44a57da was not executed. Fresh independent preflight v1
requested changes after reproducing two driver defects: expecting healthy
RELEASED cancellation for failed guard6, and injecting partial alias failures
after the protected operation returned. Its unchanged report SHA256 is
`9d7892c6bce0cf9ac83dbce1482c5f125135945facd4dceff05a9616a2f02a90`.
The original probe, reviewer runner and logs remain preserved with hashes in it.

The repaired candidate changes test code only. Fresh reviewer
`/root/setuptools_failures_preflight_v2` approved the exact finite execution
scope; unchanged complete report SHA256:
`f1aa407bc01034eb91f73d21779ee699306595f0a2cd7735d878902443b75497`.
Independent target source-only verification passed37 in6.61s, including adapted
original counterexamples and1,791 frozen file checks. Author target regression
passed36 in6.80s; author runner849585 and reviewer857306 were verified absent.
Those CPU fixtures remain separate from the actual campaign below.

## Executed campaign

Primary host tileops-dev, Python3.12.13, isolated/no-site/no-bytecode:

```
/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -S -B /data2/shared/ang.gao/run_setuptools_failure_matrix_427c90a.py
```

The exact independently inspected orchestration SHA256 was
`988e1372bf407d4b0fe2e387351f26455caa266fdca9036d08646af6b9570705`.
It invokes the fixed parent command once per case; complete argv arrays and
individual waits are in campaign.json. Fresh mode0700 campaign directory:

```
/data2/shared/ang.gao/setuptools-failures-427c90ae6b69360b5cd16178e027677bf3ae09a0-v1
```

All33 parent invocations and children exited0; no timeout occurred. Negative
child exit0 means the test proved its intended rejection, not successful
initialization. No failure.json exists, and every child stdout/stderr is empty.
The positive reached RESTORED using actual installed setuptools83.0.0 and
build_ext, real same-object alias, one closed resource read and one metadata
read, with source/admission/retention checks. It used healthy host cancellation.

Each negative reached its fixed injection and recorded the intended first error,
owner/session/admission poison and three subsequent real denials. All32 have
exactly one failed initialization action and no later success action. Each uses
the existing failed-guard quarantine: quarantined_process_exit,
TRANSFERRED_TO_QUARANTINE, not_authorized release and full_quarantine. Actual
initialization/compile-root owners remain held until hard process exit. Zero
device operations, leaf entries, emergency synchronization or terminal arm were
observed in those failed guards. No production cleanup rule was weakened.

| Coverage | Cases | Actual result |
| --- | --- | --- |
| Positive actual installed loading | 1 | RESTORED, original hooks and same alias verified |
| Reserved dependency preloads | 4 | Rejected before dependency entry |
| Extra/copied/vendor-first paths | 3 | Exact topology checks reject |
| Repeated begin, early/repeated/foreign-thread shim | 4 | Order/context/thread checks reject |
| Foreign shim, function code, wrapper closure | 3 | Actual hook identity checks reject |
| Alias spec/closure/module/loader, missing alias | 5 | Actual alias checks reject |
| Unlisted child after actual loading | 1 | Production finder rejects |
| Resource traversal, caught denial, changed captured bytes | 3 | Denial stays poisoned even when caught |
| Metadata path and enumeration misuse | 2 | Actual version-query/read boundaries reject |
| Partial installation before/after assignment | 2 | Original error retained, each attempted hook restored once |
| Partial alias create/exec after actual delegate | 2 | Original error retained before shim unwind; no corresponding success action |
| Restoration before/after assignment | 2 | Failed restoration never becomes success; no retry |
| Late publication after poison | 1 | Actual source retained, no successful publication action |

Partial_create records alias_spec but no alias_created. Partial_exec records
alias_created but no alias_executed. Both retain the actual returned source.
Restoration-before deliberately leaves its failed binding; the complete graph
remains quarantined and the process exits. Restoring a binding in the
restoration-after case does not clear failure. The foreign test thread joined.

## Replay, hygiene and raw artifacts

A separate read-only inspector loaded all33 raw terminal payloads through
production GuardedLifecycleEvidence.from_payload and required unchanged replay
output. It checked positive/version/resource facts, negative action order and
first-error patterns, all parent/child identity joins, absence of failure files,
and no backend/COMPLETE/native/readiness claims. This is parser verification of
actual CPU observations, not full compilation-record validation.

Campaign orchestrator871388, all33 parents and33 children (67 unique PIDs total)
were absent in /proc after completion. Inspector899951 then exited and was also
checked absent. Full IDs and raw outcomes are in the inspection record. Per-case
wall durations sum459.87259153835475s; this is validation time, not a renderer
speed measurement. No new GPU/compiler process was launched. Exact group cleanup
and direct-child waits are inherited from the reviewed CPU driver; this does not
prove compiler subreaper ownership or exhaustive host process accounting.

Raw archive, both target /data2/shared/ang.gao and local c4_assets:
`setuptools-failures-427c90a-matrix-v1.tar.gz`, SHA256
`21ecb21bebbef6c1e4518c98664544cdec4e524bd1c1ee7af7d847cc6409590c`.
It contains the complete fresh campaign, orchestration and inspection scripts,
raw console log and inspection result. Local/target digests agree.

Inspection JSON: `setuptools-failures-427c90a-inspection-v1.json`, SHA256
`84b2e394cb9348789474ef822720b6304b89ba4651299944f2dfd57721501cee`.
Raw campaign.json SHA256:
`24b4683b5178fd7a5638dfb6704afc06e7e834ed8c4f096c622622c1c42cf330`.

## Remaining scope

This consumes the finite preflight approval; no changed-code retry is implied.
The production installed loader and initialization failure paths ran; outer
input/device/toolchain context remains explicit CPU fixtures. There is still no
actual cpp_extension COMPLETE, CUDA compilation, native observer, compiled entry,
readiness, timing or full-C4 claim. The actual compilation-record joins, dedicated
compiler-tree OS/subreaper cases, exact compiler/native preflight, three cold
builds, process/device/failure matrices and fresh whole-checkpoint review remain.
The owner's systematic whole-C4 refactor still follows functional/real validation
and precedes final C4 acceptance. C4 remains the active objective.
