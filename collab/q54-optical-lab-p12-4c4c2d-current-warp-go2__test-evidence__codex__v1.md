Initiative: q54-optical-lab-p12-4c4c2d-current-warp-go2
Stage: test-evidence
Author: codex, implementation coordinator
Version: v1
Date: 2026-10-08
Status: bounded author evidence; whole C2d3 and C4 unaccepted
Related Files: runtime-adoption design v2; native-global gap-review v4;
cuda-normal-workloads and captured-ambient-and-terminal test-evidence v1
Owner Summary: Both real Warp Go2 cases now pass on the same production
a3d750b as current CUDA results. All four normal input cases on both lanes
match independent CPU references. Complete fault/device matrix and aggregate
schema remain required; this does not compare optical performance.

Production/tests-support stayed clean detached
a3d750bbaf79d760a6727e56146d277e1cb3b770 on
tileops-dev:/data2/shared/ang.gao/robot_sim_c2d3_binding_wip. No remote candidate
edit or fallback. Local schema review candidate890fe75 remained frozen while
these external runs exercised existing approved production mechanics.

Fixed invokers derive from the already reviewed normal Go2 invokers, changing
only the lane, exact current tracked Warp Go2 profile and exclusive artifact
destination. No observer runs inside the worker and no native path is added.
The unchanged repository child/supervisor, actual canonical builders/real Go2
assets and shared Warp upload/readback execute. Original ACK1000ms and per-
operation120000ms bounds remain. Outer parent wait grants no deadline renewal.
The fixed wrapper pins invoker bytes and records/rejects all12 native launch
variables previously enforced by the CUDA normal wrapper, without sanitization.

Commands, each prefixed by pinned robot_sim python -I -B:
`/data2/shared/ang.gao/launch_c2d3_warp_go2_fixed_v1.py go2_960` and `go2_1080`.
Profile: tests/data/runtime_environments/183_warp_go2_author_v3.json,
schema runtime-environment.v5, canonical
88d295b90bc4047b2f2cd4c3d0e48c33ead1a1050919b47d526b4576159146e8,
raw e9386330bdb8e8110b615899dbe8ebcc995d7daeb0a3c4aa66e40d5bd137ce10.
The actual production parser and verify_files passed before each launch.
CPython3.12.13; pinned Warp1.12.0/NumPy2.5.1 deployment; H200 driver595.71.05.
Real asset directory /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2.
GPU0 UUID GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274;
target1 UUID GPU-3a3112ff-837c-3b87-38d1-cb203d656589. No visibility remapping.

## Real results and process identity

| Case | SSH / invoker / supervisor / worker | Live start ticks (invoker / supervisor / worker) |
| --- | --- | --- |
| Go2 960x640 | 614337 / 614346 / 614383 / 614384 | 230830408 / 230830497 / 230830504 |
| Go2 1920x1080 | 620687 / 620688 / 620730 / 620737 | 230842149 / 230842238 / 230842245 |

Read-only external identity observers615388/615381 and622116/622114 captured
boot ID, actual executable, exact child/supervisor commands and parent ancestry,
then rechecked worker start ticks. Both workers exited0; supervisor observed and
reaped each. Scoped final ps included all named owned invokers, workers,
supervisors and observers, and returned header only for each run.

Both cases:42 nonempty arrays,70,214,100 source bytes,137 entered leaves:
42allocation,42H2D,42D2H,2each event create/record/wait and5runtime entries.
7ARM/6COMPLETE; one target terminal sync succeeds, guardian table RELEASED;
no timeout/kill/UNKNOWN/foreign dirty device, no failure.json or ordinary atexit.
Release describes that table, not proof that every native cache/destructor ran.
No execution readiness, sample, optical render or performance result is issued.

The six ARM-to-COMPLETE intervals (seconds) were
960:0.548,2.030,64.544,6.000,7.162,5.154;
1080:0.553,2.018,63.905,5.539,7.063,5.122.
These include source/dependency/custody validation and are watchdog evidence,
not fair benchmark timings and not a CUDA-versus-Warp performance gate.

Each actual D2H result matches its fresh CPU canonical manifest and all42 names.
Manifest960:ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12;
manifest1080:4785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e.
Both content8e79f7061cf08e8778fd4fca57aac7ad6936ddf3e9ef67f0f6c074faf69c3c18,
topology3ef81f6fc01bfda47983cb77ddb7523d45b8b0291010956f400652ccb7561e69.
The CPU reference was materialized independently; no child hash was used as the
expected source. Identical descriptive report bytes across lanes are expected
because those reports contain source/readback facts, not process identities.
They do not demonstrate shared allocation or reuse of an earlier GPU run.

Existing production custody/native-global/replacement/interpreter-alias parsers
accept the actual component prefixes. Each run has4678 external events:
925 each file resolution/create-start/created/exec-start/completed,29 absent
probes,6 creator publications,2 namespaces,2 disabled denials and2 fallback
confirmations,12 creator-self absence checkpoints. Three real native-global
observations validate through their pinned observer relationships. This is
component replay only; no aggregate record is synthesized from these diagnostics.

GPU residual snapshots after960/1080 respectively showed pre-existing
MPS2238410/60MiB on0 and unowned619846/python/768MiB or627195/python/768MiB on1.
No listed residual PID belonged to these attempts. No other-user process was
signalled and no GPU reset, dependency install or asset simplification occurred.

## Retained provenance

Artifacts live both in sibling local c4_assets and remote /data2/shared/ang.gao,
directories c2d3-a3d750b-warp-go2-author-v1 (960) and-v2 (1080). Complete outputs,
request, identity, phase prefixes, worker and supervisor custody are retained.

- `c2d3-a3d750b-warp-go2-author-v1/request.json`: `ab305f066e5e4d91c7cf0ac9fa62299fbac2b7fe5231362d33aac7816dd57363`.
- `c2d3-a3d750b-warp-go2-author-v1/external_process_identity.json`: `3740db778fac00e8d185309b7141d3adf366aaa17d4cd46983599bbced2731d5`.
- `c2d3-a3d750b-warp-go2-author-v1/worker.json`: `1c9f52c157fc0a5e5059815d9755252b99ecab72fb55d47d4a2e1464291a127d`.
- `c2d3-a3d750b-warp-go2-author-v1/custody_artifact.json`: `a6a607e9362ecfedd19d7efff9b1c06bd6525bd8a502353604378b6b4be1ff1e`.
- `c2d3-a3d750b-warp-go2-author-v1/input_validation_report.json`: `f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20`.
- `c2d3-a3d750b-warp-go2-author-v1/supervisor_stdout.json`: `45f9b07ae36d44dade48401a07ee0fc2e722388d971e8535a20c1fd16aac15ed`.
- `c2d3-a3d750b-warp-go2-author-v1/author_component_verification.json`: `74c5c3b2d9d5e3dce97bcaad03e6d209300e89d4cbaaa991cf99eb55a719d3f1`.
- `c2d3-a3d750b-warp-go2-author-v2/request.json`: `3da3056a0d4ad497e6c753dac5d8bb37aafee00aa7463f42501420f6202eece3`.
- `c2d3-a3d750b-warp-go2-author-v2/external_process_identity.json`: `6fee661f8280b45f282053dfb7f46edae35a1d7f9d1714546552c443b16acf4d`.
- `c2d3-a3d750b-warp-go2-author-v2/worker.json`: `9f31e4d24f4a5133cd63e048ab0f2fa46cb4b20685d70a081fadb22cc0bc4ecb`.
- `c2d3-a3d750b-warp-go2-author-v2/custody_artifact.json`: `897002edfb42d81467b64fc3b82ba503f9ff4aba40cf7c16e2cbc339217abe37`.
- `c2d3-a3d750b-warp-go2-author-v2/input_validation_report.json`: `7f34a3f42ccf973f8516b75182696e380608d23d6b4910c08bf07126afaed8f5`.
- `c2d3-a3d750b-warp-go2-author-v2/supervisor_stdout.json`: `faee8e37c55f701a206bba941e849d6398d1bb29a079bb4646c73402247a9aa1`.
- `c2d3-a3d750b-warp-go2-author-v2/author_component_verification.json`: `be1e5fdea09d410edbcfa33d8f6e293f0345cf607cb85bdcd95ad0fddc21ca75`.
- `run_c2d3_warp_go2_author_v1.py`: `22acd19504a99b901a88de8c20b08a16f04624f0fe281b74d26c4c52bbde7140`.
- `run_c2d3_warp_go2_author_v2.py`: `1908ce3c570e6696d061972fb80ff2aae494954655166e1bfb882bb9ee37ec74`.
- `launch_c2d3_warp_go2_fixed_v1.py`: `57aaa7ed96312f6ff53caa8596cac130f841ba52b4045a1c1abfbbcf4fc694a6`.
- `c2d3_warp_go2_process_identity_v1.py`: `4d7e1dc91020639c1b92549c7176d6204e329912f14d2ec2f716f255b80e7499`.
- `check_c2d3_warp_go2_v1.py`: `be9f3a7d54f732b1378202128cb95f6ccf83b9474b58c24ad150f70429ec85f7`.
- `check_c2d3_both_lane_normal_inputs_v1.py`: `08af72f685cd15281953848024eda3c8a2a72b800010d6d1f3b60c36fab31c24`.
- `c2d3-a3d750b-both-lane-normal-input-comparison-v1.json`: `579545ab226e2a8c5c4c0956bea73bc2e536ea4aa7aa12964f618f97532cd3b2`.

The last checker compares all eight actual normal readback reports (both lanes
x asymmetric/replay/two-Go2) with the four independent CPU manifests; all match.
It performs no GPU operation and is not the mandatory aggregate parser.
One local hash listing raced an SCP started in parallel with creation of the
author checker output; the GPU artifact was intact. The missing checker file
was fetched after completion and hashed; no GPU rerun or evidence fabrication.

Full both-lane setup-fault/hang/ACK/EOF/foreign/UNKNOWN/dual-GPU/cleanup/missing-
exit coverage, formal versioned runtime-custody and independent whole C2d3
review remain. Normal input equivalence does not certify those failure paths.
All later product, readiness, result, performance, host/R2/shipping gates remain
closed; C4 is active. Temporary paired input verification follows P12.4e cleanup.
