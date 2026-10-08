# Native-scope candidate CPU regression

Initiative: Q54 / P12.4 / C4c2 / C2d3
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: implemented — CPU evidence only
Related Files: native-scope-aggregate implementation-note v1;
native-scope-aggregate code-preflight-review v1; native-scope-cache design-request v1
Owner Summary: Exact b0f22fb passed501 CPU tests on the primary pinned Python,
including real Go2 materialization, process protocols and synthetic codec
mutations. This does not close the two independent preflight findings or
authorize actual native-scope observations.

## Exact execution

Primary tileops-dev, /data2/shared/ang.gao/robot_sim_c2d3_runtime_custody_wip,
clean detached b0f22fb4f6eda395f72ae34f37044ed918254184 before and after.
Original binding_wip stays separately preserved. Python is
/home/ang.gao/miniforge3/envs/robot_sim/bin/python, 3.12.13.

External runner run_c2d3_native_scope_cpu_v1.py uses the prior procfs observer,
generalized only to the actual three CPU child scripts and their -S choice.
It neither imports a backend nor performs GPU operations. It launches:

python -B -m pytest tests/unit/optics/test_{native_runtime_driver_process,
guarded_custody_process,native_source_custody_process,guarded_runtime_binding,
native_source_policy,native_sources,native_bootstrap,guarded_native_readback,
runtime_custody,native_manifest,native_scope_observer,runtime_native_scope_codec}.py
-q -ra -p no:cacheprovider --basetemp /data2/shared/ang.gao/c2d3-b0f22fb-cpu-regression-v1

The brace list above abbreviates twelve distinct explicit arguments retained
verbatim in the runner. Environment: PYTHONDONTWRITEBYTECODE=1,
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, ROBOT_SIM_TEST_GO2_MODEL_DIR points to the
existing c2d2_frozen_assets/mujoco_menagerie/unitree_go2;
ROBOT_SIM_RUNTIME_CUSTODY_ARCHIVE points to the exact historical9b5702b archive,
SHAeb08d4ed62f4271596a7b8a1e982f6cd7c2b7f3ab8ec7488319c6bb9aeab5510.
The runner rejects unexpected device-remapping/loading ambient variables.

Result: **501 passed, 1 skipped in331.06s**, pytest exit0, no outer timeout,
no procfs observer errors. The single test_native_bootstrap.py:261 skip is the
unsupported-platform behavior covered on the local Windows interpreter.
The real Go2 CPU case and all32 POSIX synthetic-native-row codec cases ran.
Source/cold process fixtures remain CPU doubles, not real runtime/GPU proof.

## Identity, residuals and raw artifact

Runner PID1280468; pytest PID1280475. All65 process cases produced identity.json.
The external observer captured43 live worker/supervisor/parent ancestry chains,
with executable, command, boot ID and start ticks. The other22 cases ended too
quickly to capture live chains: their PID/PPID files are descriptive and do not
acquire retrospective start-identity proof. No requirement was silently changed
to call all65 externally observed. These are CPU process regressions; future
actual native cases retain their mandatory complete live-identity requirement.

After runner/pytest completed, scoped residual_check.json records all132 unique
owned PID references absent from /proc and none among the nvidia-smi compute
PID query. The query succeeded. No process was killed by the external runner,
no broad process cleanup or GPU reset occurred, and no other user's GPU was
modified. This is round-specific absence evidence, not a general no-hang claim.

Archive c2d3-b0f22fb-cpu-regression-v1.tar.gz,36,446,610bytes, SHA256
a07578733848d00daccbafd9381916e76ae7e02ce11b01c485079c61b0d247b2.
It retains the complete pytest basetemp, stdout/stderr, run summary,43 external
identity files,65 child identity files and residual check. Both remote archive
and downloaded local c4_assets copy independently match this digest.

## Review and authorization boundary

Independent code/native preflight report for the same candidate is archived
unchanged at f630f81, SHAb18527c80bc1daa951879de99ed39aeed4c8942685635e5e39e48a7daf3adeca.
Its Request changes identifies F1 missing Warp cache proof and F2 incomplete
CUDA ordered getter diagnostics. These501 passes do not refute either finding.
Native-scope-cache design supplement at083688e is under fresh independent review.
No new native-scope GPU case has run. All16 new cases, required affected faults
and whole C2d3 review remain. C2d3/C4 are unaccepted and downstream gates closed.
