# C2d3 creator-self/late-custody WIP: current-code regression evidence

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v4
Date: 2026-10-08
Status: in_review
Related Files: native-self-alias design-request/review v1; runtime-adoption implementation-note v1; tests/unit/optics/test_runtime_import_failure_cold.py; tests/gpu/test_native_runtime_cold_warp.py
Owner Summary: The bounded alias design was independently approved, then implemented as WIP with late-failure custody repair. Current-code targeted CPU, cold Warp GPU, C0 and real Go2 CPU regressions passed. No residual test/worker process remained on183. This is author evidence, not whole C2d3 implementation approval or a substitute for the missing real Go2/CUDA admission matrix.

## Revision, host and test boundaries

- Design approved at `29077305468e2cd1147aef4c80b502548629f174`; full independent
  report archived at149edd4 without text changes. Design scope only.
- WIP code `55817c08da2ec19a251a09c036d9b6d4cace02dd`.
- Follow-up `25fee981c24c2dcd98ff96339378371d135edd2f` changes tests/evidence only:
  fixes the cold test's stale v2 expected hash and adds a CPU packaging regression.
  All production source bytes/catalog are identical to55817c0.
- Primary183 alias tileops-dev, hd03-gpu2-0031, robot_sim CPython3.12.13,
  NumPy2.5.1, Torch2.9.1+cu128, Warp1.12.0. Eight NVIDIA H200 devices were
  observed again at01:20:21+08:00. No101 fallback was used.
- Isolated remote `/data2/shared/ang.gao/robot_sim_c2d3_binding_wip` was clean at
  the exact test revision throughout each batch. Never changed revision during
  a batch. Original `/data2/shared/ang.gao/robot_simulator` remains clean at
  `f771e612beaeea126e26926738097512cd2d809c` and was not modified.
- Transfer bundles were SHA-256 matched before fast-forwarding the isolated tree:
  `c2d3_self_alias_55817c0.bundle`:
  `472a63ed75a39d0643f23c1e41c907dfd698118c6d5d62f9d3edb78d49eae7cf`;
  `c2d3_v3_fixture_pin_25fee98.bundle`:
  `ad295210531e679a39c4150deb71997d56356b5e4375024ed713c1305cf35d11`.

New schema v3/policy v7/catalog v6 and their author-local positive/negative
results are in implementation-note v1. Local seven-file regression was354 passed,
8 skipped in160.16s, not whole-repository coverage. Before the fix, the actual
cold session/controller CPU fixture reproduced the late-owner failure for both
lane labels; its promoted regression subsequently passed both. Fake native
delegates test mechanics only; no real SciPy alias acceptance is claimed yet.

## Commands and actual outcomes

All remote commands used bounded SSH transport options (BatchMode=yes,
ConnectTimeout=8, ServerAliveInterval=5, ServerAliveCountMax=2). Working directory
is the isolated repository; `python` below means the exact executable
`/home/ang.gao/miniforge3/envs/robot_sim/bin/python`.
The outer timeout was an operational failsafe, never a passing test criterion.
Every command below returned exit0; no outer timeout fired.

1. At55817c0, start01:11:58:

```text
timeout 240s python -m pytest tests/unit/optics/test_runtime_environment.py tests/unit/optics/test_runtime_import_admission.py tests/unit/optics/test_runtime_import_cold.py tests/unit/optics/test_runtime_import_failure_cold.py tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_runtime_driver_process.py -q -rs -p no:cacheprovider
```

Result: **360 passed, 2 skipped in119.11s**. Only two Go2 cases skipped because
their explicit asset variable was absent; separately rerun below. Real POSIX
source-byte and supervisor descriptor cases ran, unlike the local Windows batch.
Observed timeout PID3456779 / pytest3456781; both later absent.

2. At25fee98, explicit schema/pin regression:

```text
timeout 60s python -m pytest tests/unit/optics/test_runtime_environment.py -q -p no:cacheprovider
```

Result: **47 passed in1.41s**. This overlaps the previous suite; do not sum totals
as unique tests. It checks the new test-only follow-up and actual POSIX contract.

3. At25fee98, start01:15:25, fresh previously nonexistent artifact directory:

```text
ROBOT_SIM_TEST_C2D3_COLD_WARP=1 timeout 240s python -m pytest tests/gpu/test_native_runtime_cold_warp.py -q -s --basetemp=/data2/shared/ang.gao/c2d3-25fee98-cold-warp-proof-v1 -p no:cacheprovider
```

Result: **6 passed in162.24s**, no skips. Timeout3461011, pytest3461013.
Each case uses a new captured-source worker and actual supervisor. Canonical
asymmetric42-array and dynamic27-array upload/readback identities are checked,
including actual replay inputs. No final optical readiness or sample result.

| Fault / canonical case | Supervisor PID | Worker PID | Observed worker exit |
| --- | ---: | ---: | ---: |
| success / asymmetric | 3461082 | 3461083 | 0 |
| success / dynamic | 3461441 | 3461443 | 0 |
| terminal-sync raise / asymmetric | 3461860 | 3461862 | 74 |
| terminal-sync raise / dynamic | 3462227 | 3462232 | 74 |
| terminal-sync hang / asymmetric | 3462591 | 3462592 | -9 |
| terminal-sync hang / dynamic | 3463376 | 3463377 | -9 |

The last two are deliberate test-only hangs after real setup/readback, with
parent alive and the real30s supervisor deadline. Tests validate kill/reap and
absence of invented completion/release. They do not reproduce or prove a fix
for the earlier historical101 hardware/process hang. No normal atexit cleanup
ran. The matching supervision artifacts were validated by the production schema.

Artifact SHA-256 values, suffix directory under the fresh base above:

| test_actual_cold_canonical_war suffix | custody_artifact.json SHA-256 |
| --- | --- |
| 0 | 8bca5849ac2a91ea9ae0198ce4b52d0b8b48e0b8dc199046d297e1df585eda5a |
| 1 | 83c2372b1dc311d08003284932540e97d3309afeb6f70b4783e7af4454ea285d |
| 2 | 6b7ec2b94fcc8256da056e3ac418e45d1a4bfe923e562213103177b9ff4aa279 |
| 3 | ba449b753c7e3911c6de9263a59c6507ab711b7aa9816d6c513681740f4d7184 |
| 4 | c7f60bfa77838f93ffa8db44e7a9b9090e7df2e7cc51d78b15ff34c4660031db |
| 5 | c9ee9814856afa0c8a7476789f0898d9611af50fcc431c08b79ccb09d092cfa9 |

4. At25fee98, start01:18:43:

```text
timeout 120s python -m pytest tests/unit/benchmarks/test_optical_closed_worker.py tests/unit/benchmarks/test_closed_worker_protocol.py tests/gpu/test_optical_closed_worker_setup.py -q -rs -p no:cacheprovider
```

Result: **30 passed in13.23s**, no skips. Actual C0 regressions, not permission
to open C0 product GPU preparation. Exact transient parent/child PIDs were not
sampled for this short batch; completion was awaited and a subsequent01:19:13
account process check showed no Python/test/worker process. No fabricated PID
ledger or claim of independently enumerating every short-lived child.

5. At25fee98, start01:19:15:

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 timeout 120s python -m pytest tests/unit/optics/test_native_sources.py -k real_go2 -q -rs -p no:cacheprovider
```

Result: **2 passed,39 deselected in46.32s**, no skips. Timeout3465724 /
pytest3465726. Both lane labels each exercise both real Go2 resolutions and
repository import ledger through the existing canonical CPU source builder.
This is not the new complete external-runtime admission proof, not a GPU lane
test and not a surrogate asset. The original assets were not edited or reset.

## Hygiene, interpretation and next obligations

At01:18:23, all14 recorded GPU batch parent/supervisor/worker PIDs were absent;
NVML listed only pre-existing MPS2238410 at60MiB. At01:20:21, ang.gao account
processes were only current SSH/read-only readers and login services; no Python,
pytest, worker or supervisor remained. NVML again listed only that unchanged MPS.
Remote git status was clean and git show --check passed. No broad kill, GPU
reset, package install, original checkout change or material deletion occurred.
Only the test supervisor intentionally killed its own two injected-hang workers.

This evidence closes no whole-stage gate. Still required: complete reviewed
Go2/CUDA Python/native environments; actual four self-aliases through production
captured admission; both-lane/all-canonical upload/readback; remaining real
multi-device/import/setup fault matrix; versioned runtime-custody records/replay;
then fresh independent whole C2d3 code review. No CPU subset approval is requested.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, R2,
host-payload and shipping remain closed. C4 is not complete.
