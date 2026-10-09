# Repaired-candidate actual input and native-scope validation

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v2
Date: 2026-10-09
Status: implemented
Related Files: whole-review-repairs code-preflight-review/test-evidence v1;
native-scope-aggregate test-evidence v1; runtime-adoption whole-code-review v1
Owner Summary: All16 native-scope cases pass on repaired fc86182. This supplies
the repaired real positive readiness/disposal evidence required by F1. Full85
fault and12 terminal reruns plus fresh whole review still precede acceptance.

Clean detached remote fc8618268dd8f3fb10e29e32626219500751c41a at
/data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip, primary tileops-dev.
Interpreter /home/ang.gao/miniforge3/envs/robot_sim/bin/python. Exact reviewed
runner ../c4_assets/run_c2d3_native_scope_v2.py SHA256
f05b8f4d03b276637692d541f63bc6abe557d9487639b100b9d271c2777c35e8.
Invoked via UTF-8 stdin over SSH to python -I -B -; its unchanged command is
python -B -m pytest tests/gpu/test_native_scope_runtime.py -x -q -s
-p no:cacheprovider --basetemp /data2/shared/ang.gao/c2d3-fc86182-native-scope-v2,
with ROBOT_SIM_TEST_NATIVE_SCOPE=1 and plugin autoload disabled.

Actual result:16passed1183.04s, exit0; no skips, author timeout or observer error.
Eight positives are both lanes x asymmetric/replay/Go2 960x640/Go2 1920x1080.
They perform actual input upload and ready-event wait, registered storage and
supervised D2H validation, native scope observations and terminal disposal.
All eight return0. Eight intended observer/scope failures return74 with the
existing UNKNOWN/no-extra-sync/no-release assertions and raw diagnostics.
No rendering, final sample readiness, warmup, performance or product dispatch
is exercised. The real foreign-ready-event fault is not newly injected here;
the required metadata counterexamples remain CPU lifecycle tests and independent
repros, while these native positives test the changed real metadata path.

Archive c2d3-fc86182-native-scope-v2.tar.gz is7823610bytes, SHA256
d872f90f699fea8a1ea3b6a3aecc2c6a481bf34d4936b9a4de732cbaaef23130.
It is retained under /data2/shared/ang.gao and downloaded/hash-verified under
../c4_assets. Original per-case request, exact launch/source/environment facts,
GPU mapping, custody/native records, input validation, raw supervisor output,
diagnostics, pytest stdout/stderr and summary are retained. All16 same-run
external worker/supervisor/parent identity chains exist. Residual check finds
all34 owned PID references absent from /proc and GPU query(rc0). Direct ps also
confirms runner3293204/pytest3293209 gone. No other-user process was touched.

The Windows SSH connection remained open after remote completion. After raw
summary/archive and exact remote absence checks, only its positively identified
local ssh.exe436 (parent21356, command containing this runner) was stopped.
The local tool session therefore reports1; the independently retained remote
pytest/run result is0. Earlier CPU mutation connection had the same cleanup:
local ssh.exe14096/parent39924, after runner3242890/pytest3242918 were absent.
Neither action signals a live remote test or replaces the original supervisor.

After this round's residual check, the exact preflight-approved full85 fault
runner starts on unchanged fc86182: runner3342425/pytest3342430, output
c2d3-fc86182-runtime-fault-matrix-v5. It is running at this evidence cut; no
outcome is claimed. The12 terminal rerun follows with the same reviewed source.
C2d3 and C4 remain incomplete; all later product/performance/selection gates
remain closed as specified in the independent preflight report.
