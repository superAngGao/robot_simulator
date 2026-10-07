# C2d3 singleton regression and first captured Go2 attempt

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v6
Date: 2026-10-08
Status: implemented
Related Files: absent-resolution design/review v1; Go2 deployment-table review v1; native_bootstrap.py; tests/gpu/test_native_runtime_cold_warp.py
Owner Summary: Narrow literal-table review passed. Current synthetic cold GPU regression passes, but actual captured Go2 still fails closed on a generated interpreter module. This is not C2d3 acceptance or a performance result.

## Candidate and independent scope

Code baseline 4f82a87d225f5d5bbb08d5dddb8bfcef855b2983; test/data/evidence
candidate 89e89c15e1bcbca2ec1171d6814a9d4e9f478b83. The latter adds the
unchanged independently reviewed Go2 table/report and fixes a stale test's
policy-v7 expectation to the actual v8. No production code changes between them.
Both local and isolated primary183 checkouts are clean at 89e89c1.

Independent /root/c4_c2d3_go2_table_scope_review_v1: Approve with comments
for literal table only. Report SHA e9930be397c5bd2cecf09887f0e2a2a95461c8a8335bfda92c8f2f0988681988.
Table raw SHA 747e680530d2a02b8de372f7f0c660840b22b4cd3a0e375656a84e82a0898d60;
canonical SHA d2b1edc48c5a28a4ef71237fdf16d92d089f02f3fe7926c858d3e0c651146d3b.
Scope is permission for the next guarded author attempt, not captured closure
or whole C2d3 implementation approval. Its plugin/native-custody and both-lane
proof obligations remain open. No runtime-derived permission was appended.

## Regression observations

Local Python3.14: test_native_source_policy.py, 167 passed in8.86s.
Primary183 robot_sim Python3.12.13: same test, 167 passed in1.44s.
The previous 4f82a87 seven-file run was 433 passed,2 skipped,1 failed; the sole
failure was the stale version assertion. This is not relabeled a passing run.

At89e89c1, freshly reran the seven-file CPU set on primary183:
test_runtime_environment, test_runtime_import_admission, test_native_bootstrap,
test_runtime_import_cold, test_runtime_import_failure_cold,
test_native_source_policy and test_native_sources (all under tests/unit/optics).
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, timeout240s, python -B -m pytest -q
-p no:cacheprovider. Result434 passed,2 skipped in161.70s, exit0.
Parent3552565/timeout3552564 started02:29:47+08. The two unconfigured Go2
environment skips are not actual Go2 acceptance.

C0 regression: tests/unit/benchmarks/test_optical_closed_worker.py,
tests/unit/benchmarks/test_closed_worker_protocol.py and
tests/gpu/test_optical_closed_worker_setup.py; same interpreter, timeout120s,
plugin autoload disabled, -q -rs -p no:cacheprovider. Result30 passed in12.99s,
exit0, no skips. C0 protocol/dispatcher remains unchanged.

Actual current GPU command, primary183/H200 target1:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ROBOT_SIM_TEST_C2D3_COLD_WARP=1 timeout 240s /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest -q -s -p no:cacheprovider tests/gpu/test_native_runtime_cold_warp.py --basetemp=/data2/shared/ang.gao/c2d3-89e89c1-cold-warp-proof-v1
```

6 passed in164.66s, tool exit0, not the outer timeout. Parent3548057,
timeout3548056, approximately02:26:37–02:29:22+08. Each actual child used
-I -S -B and the production captured importer/controller/upload/readback.
Normal exits0, sync exceptions74, intentionally hung terminal syncs-9 under
the original30s supervisor deadline. All child processes were observed reaped.

Artifact base `/data2/shared/ang.gao/c2d3-89e89c1-cold-warp-proof-v1/`,
each under `test_actual_cold_canonical_warN/custody_artifact.json`:

| N | supervisor/worker | Case/fault | SHA-256 |
| --- | --- | --- | --- |
| 0 | 3548137/3548138 | asymmetric/success | 628ea2141d4099e107bb80e0271418666b974a146f5d841514074d430aa6cce7 |
| 1 | 3548719/3548720 | dynamic/success | 6f4e84787fa9e0e7753f3e2610b932f9510eafaa4d140d2846ef1262cdb11935 |
| 2 | 3549133/3549134 | asymmetric/sync-raise | d53f1cd7eecd5c953458b48598ca34dfa72e0f9055c4f7cc66fb08c63880b8f4 |
| 3 | 3549598/3549604 | dynamic/sync-raise | d6e750a188ac04b62b5e7c2780b63221a8a96e55cf2793b6cab59cf0db233671 |
| 4 | 3550062/3550064 | asymmetric/sync-hang | dbde61101c8ebd4346ceeac0e3dc821c8f2dbb62e029b270ca219d9d99d315b5 |
| 5 | 3551182/3551186 | dynamic/sync-hang | fe908bb3dad1a4e10a507f50122817564eeecc9dd69b859b69226e1b118fb43a |

Killed-worker artifacts contain a prefix with null disposition; only supervisor
exit/reap proves termination. Do not invent successful cleanup or final worker
evidence for killed workers. These six tests do not cover the required whole
Go2/CUDA/setup-fault/dual-device matrix.

## First actual captured Go2 attempt: failure, not acceptance

Outside-candidate fixed author invoker:
`/data2/shared/ang.gao/run_c2d3_go2_guarded_author_v1.py`,
SHA95f652b05e6381887d84a5091d0f8a01e8b210852ef60251ee51d30d372b8d78.
It invokes the existing repository fixed driver/supervisor, not an alternative
materializer/importer/uploader. Canonical case go2_static_960x640_v1; exact
reviewed table; actual frozen asset unitree_go2. The author invoker imports only
CPU contract/stdlib, no NumPy/Warp/Torch or benchmarks. It makes no product call.

Predeclared operation watchdog120000ms and ACK1000ms. The operation budget was
chosen before the attempt from observed40–46s ordinary CPU materialization,
not extended in flight and not a performance gate. No timeout occurred.

Start02:29:28+08; parent3552120, supervisor3552131, worker3552132. The worker
failed in canonical_materialization and exited74; supervisor observed/reaped it,
complete=true, state=EXITED, deadline_expired=false, kill_requested=false.
Worker full_quarantine, release not authorized, zero emergency sync and terminal
ARM, normal_complete_count2. No upload/readback, final readiness or sample result
was reached. Parent assertion exit1 correctly retains the unsuccessful attempt.
No ordinary_atexit artifact. Both worker/supervisor absent from /proc on recheck.

Evidence directory `/data2/shared/ang.gao/c2d3-89e89c1-go2-960-guarded-author-v1`:

| File | SHA-256 |
| --- | --- |
| request.json | d75019d6b8888c7424ca671ed4a83b63d852a3dcea5082ba2a7b47c77e5191b1 |
| custody_artifact.json | d7db839cfd17bb651ac176d1f022fbeff925bc75689f6538db24a87b84772b7a |
| external_failure.json | 2401dde57398cd803f049b688769e203131b10f69d9835d0c3a2488ac20abf99 |
| failure.json | 2e85fc9cecc90e479051e0647ab417a952068fc8d74c67ad8fd74cf722fc5696 |
| supervisor_stdout.json | c47ddd3a2681edd659d6e086ff4dda70c0e15173bebe7428cb9cc757da4b20c8 |

Last external event614 rejects `_sysconfigdata__linux_x86_64-linux-gnu` while
scipy._lib._testutils executes; outer SystemExit says the preview loader
requires trimesh. Actual installed _testutils.py:44 calls
sysconfig.get_config_var('HOST_GNU_TYPE'). Read-only isolated CPU inspection:
the generated module is under this interpreter's lib/python3.12 but absent from
sys.stdlib_module_names. Its71715 bytes hash
07c6cf395105faefff17ea4a888e00858ca3da7e2f8f9266cb7953c772085683.
Installed sysconfig.py hash
e9a916680e27278b130d8c4add824d55b952bef284bee354083ca9b436d5103e.
Its name selection supports _PYTHON_SYSCONFIGDATA_NAME; current override is absent.
No dependency installation or table expansion was attempted. Fresh independent
gap scan /root/c4_c2d3_sysconfig_gap_review_v1 confirmed the bounded P2
generated-stdlib delegation/design gap and requested a supplement. Report SHA
20c318f64a79a7034949448438ec31b6b7af96807f71b65bd1c6bebf25808d91;
it is not whole C2d3 code review. In addition to site-root restrictions, the
existing external module grammar rejects this hyphenated generated name.

## Safety and remaining work

Primary183 only. Original robot_simulator worktree untouched. No GPU reset or
signals to others' work. MPS2238410/GPU0 and user232071 python3/GPU2 are unrelated;
their changing memory use is not an author leak. GPU1 is the test target.
At02:33:57+08, ps for ang.gao showed no Python/pytest/supervisor; all author
test sessions had terminal exits. Remote git show --check and clean-tree checks
passed. This is scoped round-end hygiene, not proof the historical101 hang is
diagnosed or that no other user has active work.

Whole C2d3 still requires real captured Go2 success, complete CUDA environment,
both-lane failure/hang/dual-device matrix and versioned runtime-custody schema,
then independent full code review. All C2e/f, C4c3/c4, C4d/B3, product GPU
preparation, a3b, result/baseline, host-payload/shipping gates remain closed.
C4 remains incomplete. Current author test results are not review approval.
