# Exact-candidate complete fault matrix and actual-record mutation failure

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v3
Date: 2026-10-09
Status: implemented (evidence only; whole-stage approval outstanding)
Related Files: runtime-faults test-evidence v2; native-scope-aggregate
test-evidence v1; runtime-adoption design v2; runtime-custody implementation-plan v3
Owner Summary: The complete85 fault rerun passes at exact e585956. All observed
owned processes have exited. The actual-record mutation suite exposes one
invalid test expectation; its original failure remains archived. Both-lane
terminal-sync exception/hang coverage remains a separate required gap.

## Complete fixed native fault run

Clean detached execution revision e585956eb29e3a33483d8cb0c7444f893b390ac4,
/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip, primary tileops-dev,
/home/ang.gao/miniforge3/envs/robot_sim/bin/python. External runner
run_c2d3_runtime_fault_matrix_v4.py SHA256
9a84e404e2def345c44dbeaadf92b42706b15e3021bdc0ff5a76dba9899d15cc.

```text
ROBOT_SIM_TEST_C2D3_FAULTS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -B -m pytest tests/gpu/test_native_runtime_fault_matrix.py -x -q -s
  -p no:cacheprovider --basetemp /data2/shared/ang.gao/c2d3-e585956-runtime-fault-matrix-v4
85 passed in 4488.89s (1:14:48)
```

No subset filter, skip, observer error or outer timeout. The fixed cases cover
both-lane native import/init/upload/readback, returned-owner and storage audit
failures; original ACK/DISARMED/EOF handling; cleanup and missing-exit hangs;
real foreign/unknown device scope; Go2 native-global publication; and CUDA C1,
module-replacement and interpreter-alias failures. This suite deliberately
excludes its driver's initial three selectors (success, terminal sync raise,
terminal sync hang). Its85 passes do not prove those missing terminal cases.

Runner1598724/pytest1598730 and85 worker/supervisor chains were captured by the
external procfs observer while live. After the interrupted chat resumed, fresh
ps checks showed both runner and pytest absent. The archive procedure verified
all85 raw/external identity joins and checked all172 unique owned PID references:
none had a /proc entry or appeared in the successful GPU compute PID query.
No author signal, broad kill, GPU reset, deadline change or dependency mutation.
Tests used the existing supervised watchdog/reaper for intended hang cases.

Archive contains all raw per-case records, sidecars, requests, inputs, artifacts,
diagnostics, external identities and original stdout/stderr/summary plus the
final scoped residual result. Retained remotely under /data2/shared/ang.gao
and in local sibling c4_assets; downloaded SHA256 was verified:

```text
c2d3-e585956-runtime-fault-matrix-v4.tar.gz
bytes: 18390205
SHA256: df9f92da1bfe8dcf4095dd598381c22f4b0974f1e982b89af97ef56eed83b359
```

## Actual v2 record mutation run: preserve the failure

The existing tests/unit/optics/test_runtime_custody_actual_records.py was run
against all8 normal actual v2 executions from native-scope-v1. External runner
run_actual_record_mutations_v4.py fixes exact clean e585956 and compares SHA256
of every original per-case file before/after. All original files were unchanged.
It uses CPU parsing/mutations only; no new native execution or worker child.

Result **1failed,451passed,36skipped in732.43s**. Skips are12 non-Go2 case
comparisons,4 absent native-global occurrences,4 absent module replacements,
4 absent interpreter aliases and12 cases without zero-size arrays. These are
explicit inapplicable mutations, not successful native coverage.

The failure is the [strides-value9] mutation on the CUDA dynamic replay record
test_actual_native_scope_under5. It replaces the first active array's strides
with [1,1], recomputes the descriptive storage digest, and expects ValueError.
For the actual single-row shape this changes only a singleton-axis stride;
the array still has a contiguous layout. The approved implementation-plan v3
requires contiguous progression for nondegenerate dimensions, not a unique
stride for singleton axes. The unchanged production parser implements that
rule. Parsed records are descriptive and do not mint live storage authority.
This is the author's diagnosis to be independently checked, not a review verdict.

The bounded repair will mutate an actual dimension greater than one and change
its observed stride, retaining the recomputed digest and rejection assertion.
It does not tighten the production schema or pretend arbitrary changes to
self-consistent descriptive evidence can prove an original physical allocation.
The complete old failure is preserved; no pass is claimed for the repair yet.

Runner1601687/pytest1601692 are both absent from OS and GPU PID queries. Archive:

```text
c2d3-e585956-actual-record-mutations-v4.tar.gz
bytes: 7635
SHA256: 2b6b073073c9ab82e2af4669459c70f2eda3db3c4f45b1bba79b35ca8a7a019d
```

Both hosts retain matching copies. Summary records the exact directories,
command and before/after raw file hashes; logs include the failing assertion.
Archive script archive_c2d3_completed_v4.py is retained outside the candidate.

## Scope

These facts complement the16 native-scope and637 CPU passes; counts are not
whole-stage acceptance. Required both-lane terminal tests, final evidence audit
and fresh independent whole-C2d3 review still follow. C2d3/C2d/C4 are unaccepted;
C2e/f,C4c3/c4,C4d/B3,product GPU preparation,a3b,result/baseline,host,R2,shipping
remain closed. No optical performance result, selection or deletion is issued.
