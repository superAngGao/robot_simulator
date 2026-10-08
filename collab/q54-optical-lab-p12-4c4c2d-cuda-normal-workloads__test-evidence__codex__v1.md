Initiative: q54-optical-lab-p12-4c4c2d-cuda-normal-workloads
Stage: actual author test evidence
Author: codex
Version: v1
Date: 2026-10-08
Status: bounded actual results; C2d3 not accepted
Related Files: torch-upload-workloads gap-review v1; cuda-asym-upload test-evidence v1;
runtime-binding-fixture code-review v1; runtime-adoption design v2; CUDA ambient v1
Owner Summary: CUDA canonical replay and both real Go2 inputs now complete actual
captured allocation/upload/ready wait/D2H validation and supervised terminal exit.
Together with the prior asymmetric run, all four normal CUDA workloads have
bounded input results. Full ambient, failure/hang/dual-GPU matrix, versioned
aggregate custody and whole C2d3 review remain required.

## Revision, approval and exact launch

All three actual runs used clean productiona3d750bbaf79d760a6727e56146d277e1cb3b770
at tileops-dev /data2/shared/ang.gao/robot_sim_c2d3_binding_wip, pinned robot_sim
CPython3.12.13,Torch2.9.1+cu128,H200,driver595.71.05. Local later commits contain
evidence/profile and a separately reviewed CPU test-fixture repair; no production
or tests/support change was deployed during these runs.

Fresh /root/torch_upload_workloads_gap_v1 approved only these three fixed normal
attempts. Report SHA0b7361db53054e11951fcd746b78fef239189dd60ce31f475a1327916823313b
is archived unchanged. C1 environment enforcement and C2 release meaning remain:
the exact wrapper checks script bytes and rejects12 native environment variables,
then execves with the same snapshot; guardian RELEASED clears its owner table,
while other references/native caches may remain until hard process exit.
C3 fixture failure was subsequently independently repaired atdcece18435e7719e87114ce8d95fa2e17ccb8519:
the new reviewer reproduced58pass/2fail, then60pass, with no skipped original cases.
Report SHAbb8110d1c0121e34cfbdca7aabeeeb9c235eee5486c7627243fb1870a63c9524.
C4's missing asymmetric worker start ticks remain a historical gap; all three
new runs recorded live worker/supervisor/parent start ticks and ancestry externally.

Commands use `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B
/data2/shared/ang.gao/launch_c2d3_cuda_upload_fixed_v1.py` with fixed selectors
replay,go2_960,go2_1080 respectively, via SSH primary. Assertions enabled.
Wrapper SHAf5bbb7dd6fce0b788537dd623aa4666076409d19031d34a91d5373f3a958bbd6.
Invoker v2SHA160715958d666103d4b969b26046cd3c481448101ae222d4cfcba7a3fbbf783a;
v3SHAb6bf6741b2c41c864d904e13055687f38ddbbc603498878b5513573123b8f511;
v4SHA60de9fba981c2f1007b9d80e66ac9961322bd8b612ce0a9a6efb1d8d61e823ae.
Unchanged production child/supervisor, fresh-I-S-B worker, target1,
ACK1000ms/op120000ms per segment, no renewal or relaxed threshold.
Launch stdout is retained locally in c4_assets/c2d3-cuda-upload-author-v2.log,
v3.log,v4.log. It records all12 enforced settings null and actual parent/supervisor
proc stat. Neither warmup nor visibility remapping was used.

Replay profile canonicalb4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47;
bothGo2 profile79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919.
Real Go2 model path /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2;
production asset preflight and canonical materializers were used without substitutes.

## Actual inputs, supervision and content

Artifacts are retained under local c4_assets and remote /data2/shared/ang.gao,
basename c2d3-a3d750b-cuda-upload-author-v2/v3/v4. Each has actual input report,
phase observations, worker/supervisor trace, final parsed custody artifact and
external_process_identity.json. Fresh CPU reference manifests, generated before
these attempts through the sole canonical builders, are retained in
c2d3-a3d750b-fresh-cpu-reference-v1. Existing validators and independent expected
array names/manifest/content/topology comparison passed after each run via
check_c2d3_cuda_upload_v1.py; this checker is not the final aggregate schema.

| Attempt | Actual arrays | Input bytes | Entered allocation/H2D/D2H leaves | Worker |
|---|---:|---:|---|---|
| v2 dynamic replay | 27 | 5936 | 28/27/27 | 481951,exit0/reaped |
| v3 Go2 960 | 42 | 70214100 | 43/42/42 | 489525,exit0/reaped |
| v4 Go2 1080 | 42 | 70214100 | 43/42/42 | 507967,exit0/reaped |

All three:7ARM/6normalCOMPLETE,2each event-create/record/wait,3runtime import/init
entries; one sole supervised target sync succeeds, guardian table RELEASED and
hard process exit observed/reaped. No deadline expiry/kill or ordinaryatexit.
Declared/checked returned target scope[1],UNKNOWNfalse. These are no new raw/logical/
context observations. Counts are repository leaf entries; replay has7empty arrays
whose native copy can return early, whereas all42Go2 arrays are nonempty.
Replay uploads120frames of canonical transforms; it does not render120samples.
Measured sample count0 and final_execution_readiness_issuedfalse throughout.
Both Go2 input payloads have identical content/topology but case-bound manifests
differ. This is input validation, not rendering correctness/performance or B3 delivery.

All3841 replay external events and7963 events for each Go2 replay as component
facts. Go2 has1573file resolutions/1572completed modules,48publications,2namespaces,
36absence events (35distinct rules),2disabled adapter denials/2confirmed fallbacks,
and12native self-alias absence checkpoints. Three native-global pending/admitted
relations, seven replacements and interpreter alias pass their production parsers.
Triton remains resolution-only. Observations never add permission.

## Guard durations and live process identities

Times below are supervisor ARM-to-normal-COMPLETE intervals including audit and
custody overhead, not performance measurements. All remain below each original
120s bound; no deadline was renewed.

| Attempt | Definitions | Environment | Canonical | Runtime | Upload | Validation |
|---|---:|---:|---:|---:|---:|---:|
| v2 | 0.585s | 8.537s | 5.793s | 34.135s | 11.505s | 10.162s |
| v3 | 0.604s | 9.423s | 74.946s | 68.792s | 17.075s | 13.939s |
| v4 | 0.597s | 9.034s | 75.845s | 67.935s | 17.241s | 14.227s |

Read-only c2d3_owned_process_identity_v1.py observed live /proc stat/command/exe,
boot ID and clock tick rate, verified exact worker-parent-supervisor chain and
the pinned Python executable, then rechecked the worker start tick against PID
reuse. It invoked no GPU APIs and changed no guard state. The record is descriptive.

| Attempt | Parent / supervisor / worker | Start ticks in that order |
|---|---|---|
| v2 | 481190 / 481950 / 481951 | 230556299 / 230556716 / 230556723 |
| v3 | 489382 / 489524 / 489525 | 230570389 / 230570846 / 230570853 |
| v4 | 507450 / 507966 / 507967 | 230605269 / 230605706 / 230605712 |

After each attempt scoped ps for all owned worker/supervisor/parent/SSH identities
returned header only; CPU observation helpers also exited and were absent.
v2 residual NVML showed only preexistingMPS2238410/60MiB on device0.
v3 showed that MPS plus unowned Python505927-505930 on GPUs4-7 (772-780MiB,
executable path under /home/jieneng.yu); no signal or modification.
v4 showed that MPS plus transient unowned522257/usr/bin/python/714MiB on device1;
the later scoped ps522257 returned no row before user/parent could be established.
Do not call v3/v4 snapshots empty or classify these transient users without facts.
None of these PIDs is the recorded owned worker. No GPU reset, broad kill or
dependency installation occurred. No owned GPU process remained unresolved.

## Raw file SHA256

Attempt v2:

| File | SHA256 |
|---|---|
| input_validation_report.json | 816006fb9fd52057fd32fa330d46d9980f447408daa05297c33f1cc8940c1bfb |
| worker.json | 0238b5eecb6a232b96d7683a86a7c91eab6dc06290dd4a7deadaa20f80919e91 |
| custody_artifact.json | fb35f4a52dc27f76cdeb929b57d2d146d0b657b49e6b6cec82c9c78c3ca75c27 |
| supervisor_stdout.json | 07a38d4c17522637b8efc634d284e1122dd4ce6e06263e298c9a5a9395eb92d7 |
| external_process_identity.json | aa45b46ffb41005d0469b68348f1f2361e23d7f70ed7fa1603c002467cd99766 |
| author_component_verification.json | 5337430814b9ce9e77d733182b7b8ba3327eda4a05224d14bee3828050d63608 |

Attempt v3:

| File | SHA256 |
|---|---|
| input_validation_report.json | f76ba232ba326d720de31334fdab6608a265449b379a15adca1e5303e9eb9e20 |
| worker.json | e32db1fb74a300216bcba4f4b1dd926e3276cfeeb8b48437b52950bd27b8afc6 |
| custody_artifact.json | f18a834b1519688be73603ff512bea825f158f42e586722edafd9a41ed86bcde |
| supervisor_stdout.json | ca20e4424f0516f8133ed46d5c0c1a6443266333be63e74da2dd2c7477e1524c |
| external_process_identity.json | d9905d4a3a38fb40dd5abab8ee3a9668967f7ad179d25fe8f59cd6becfc4b6f3 |
| author_component_verification.json | 3540e6fca6c930929826c55df68ae832bc4a832db535fe90565bf2a0f1acbe75 |

Attempt v4:

| File | SHA256 |
|---|---|
| input_validation_report.json | 7f34a3f42ccf973f8516b75182696e380608d23d6b4910c08bf07126afaed8f5 |
| worker.json | 1a984402c3f41c235c4da3f085e1cc152e351b528691caa110b39b7dfb6b47cb |
| custody_artifact.json | 14b2d4fc681dcfe81883eb3567dd60ed6fd31a5a3104bbe28fcddb816d59d7ea |
| supervisor_stdout.json | fb03c208388e4035a162d5864e39e578e5505017e16318b2dc724757e00af417 |
| external_process_identity.json | f8e13057e1e1c0ea5b30a9fb3f13b17fa4ffcf35053d17a93b8d7e7ed11550d4 |
| author_component_verification.json | c0e50f4d4fe7ef86c718da8b09504c612b58128e1ff4a645c4c8a43983a99b87 |

## Remaining scope

These are actual normal CUDA input results, not whole-stage acceptance. Full
captured logical/raw/stream/context restoration and real foreign/UNKNOWN cases,
both-lane failure/hang/ACK/EOF/terminal matrix and regression closure, versioned
runtime-custody aggregate cross-binding and whole independent review remain.
The old asymmetric start-time gap is retained. Warp historical results do not
become fresh current-revision matrix results by this record. Product GPU prep,
C2e/f,C4c3/c4,C4d/B3,a3b,result/baseline,host,R2/shipping remain closed; C4 active.
