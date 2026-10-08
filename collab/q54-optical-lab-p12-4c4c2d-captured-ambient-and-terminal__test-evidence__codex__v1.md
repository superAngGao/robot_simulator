Initiative: q54-optical-lab-p12-4c4c2d-captured-ambient-and-terminal
Stage: test-evidence
Author: codex, implementation coordinator
Version: v1
Date: 2026-10-08
Status: bounded author evidence; C2d3 and C4 remain unaccepted
Related Files: captured-cuda-ambient gap-review v1; runtime-adoption design v2;
cuda-normal-workloads test-evidence v1; tests/gpu/test_native_runtime_cold_warp.py
Owner Summary: The reviewed positive captured CUDA ambient attempt passed its
actual upload/readback and all267 observation-to-guard/leaf checks. The unchanged
six-case Warp normal/terminal matrix passed with live OS identity evidence after
repairing an external observer's pytest-symlink bug. These are partial required
runtime proofs, not complete fault coverage, aggregate-schema or whole acceptance.

## Exact candidates and scope

Actual production and tests/support on primary183 remained clean detached
`a3d750bbaf79d760a6727e56146d277e1cb3b770`. Local ambient review bound clean
`68fdb5084363eb70416580fd31816a0a69871ac6`; its unchanged report/request were
archived in `1fd7b6c`. Production bytes are identical between these revisions.
No fallback host, package change, remapping, broad signal, GPU reset or
other-user process modification occurred. robot_sim CPython3.12.13 was used.
Target1 UUID: GPU-3a3112ff-837c-3b87-38d1-cb203d656589; device0 UUID:
GPU-a4b256c9-c56a-0ef5-19fa-482196cc3274. Driver595.71.05, eight H200s.

## Captured CUDA positive ambient result

Command: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B
/data2/shared/ang.gao/launch_c2d3_captured_cuda_ambient_fixed_v1.py asym`.
Exact four scripts, frozen profile and bounded preflight approval are in the
unchanged independent report (SHA42219938af86e2068aa29da743b5213e543949f1bcc6f6d62fc6d201ef67f278).
Actual wrapper recorded all12 enforced native environment variables absent.
ACK1000ms and original operation120000ms were not renewed.

Artifacts are retained locally in sibling c4_assets and remotely under
`/data2/shared/ang.gao/c2d3-a3d750b-captured-cuda-ambient-author-v1`.
SSHparent580802, invoker580830, supervisor581047, worker581048.
Live procfs start ticks: invoker230756785, supervisor230757215, worker230757221.
External observer582677/582670 recorded exact executable, command, boot ID and
ancestry, rechecking worker start to detect PID reuse. The first read-only
observer v1 rejected an extra proc_stat identity field before writing; v2
corrected its exact schema and captured the live chain. Both scripts remain.

The child observed pristine uninitialized CUDA before the actual first Stream.
After binding adoption and before/after every actual helper, all267 records show
logical Torch0, raw CUDA1, primary-context presence only on1, and current target
raw stream0. This observes the restored target current stream, not the owned
nondefault upload stream. Original source-audited helpers execute once each.

The author checker verified exact ordered labels, original guard role/sequence,
the referenced lifecycle event, leaf descriptor kind and strictly increasing
timestamps: binding adoption in setup3;42 allocations,42H2D and3event helpers in
setup4;42D2H and3event helpers in setup5; terminal before/after in emergency6.
All132 distinct helper leaves are covered, with paired before/after on the same
leaf event. The sole additional setup4/5 leaf is the original upload-scope entry.
Upload175 and validation265 snapshots are exact prefixes of the final267.
No new ARM, COMPLETE, device setter or synchronization was added by observers.

42 arrays were actually read back and match the fresh independent CPU canonical
manifest f65056b303775f4d13ad1a48660467cbae06a9d9dbe09fc5596a341ba5af6728,
content1771f0ce65a929075d542032c54a95e4c777de361e904a377c6f2a197303a32d,
topology6a6136c2e6991470f5fd6823af9fbcf87faddf71a7d574cfa42ce198a4e81966.
Existing custody/native-global/replacement/interpreter-alias parsers passed.
3841 external events include759resolutions,758completed file imports and42
creator publications. Unexecuted Triton resolution is not a completed import.

7ARM/6normalCOMPLETE;136 entered leaves;43allocation,42H2D,42D2H,2each event
create/record/wait and3runtime entries. Exactly one target terminal sync succeeded;
guardian table RELEASED; worker0 and observed supervisor reap. Zero-size arrays
mean these helper counts are not a physical native memcpy count. No sample or
final execution readiness was issued. RELEASED does not mean all native owners,
caches or CUDA allocations were destroyed; session/observer roots survive until
the one-shot os._exit boundary. Restored observations do not exclude transient
native effects; the pinned dependency exception-safety/source obligation remains.

Read-only checker v1 passed label/guard/descriptor checks but used incorrect
prefix filenames and raised FileNotFoundError. v2 corrected only those filenames
and all checks passed. This is an author verification-tool failure, not a failed
GPU attempt. Both versions and their actual outputs remain available.

Scoped final ps for580802,580830,581047,581048,582677,582670 returned header only.
GPU residual snapshot had pre-existing MPS2238410/60MiB on0 and unowned
584677/usr/bin/python/714MiB on1; none was an owned worker. They were untouched.

## Existing Warp normal/terminal matrix

Command: `/home/ang.gao/miniforge3/envs/robot_sim/bin/python -I -B
/data2/shared/ang.gao/run_c2d3_existing_warp_matrix_v2.py`.
The external CPU runner invokes the unchanged opt-in test file with explicit
ROBOT_SIM_TEST_C2D3_COLD_WARP=1, no plugin autoload/cache writes, and an exclusive
nonexistent --basetemp. Original ACK1000ms/operation30000ms remain unchanged.
The observer only reads procfs; no worker injection, replacement upload or GPU API.

First runner v1: pytest6passed160.12s, but author runner exited1 because it treated
pytest's current symlink as another case and failed a command/path check.
observed_cases0; no live worker-start claim. Preserve the complete v1 directory,
stdout/stderr and author_run_summary.json. Actual identities printed by pytest
were checked absent afterward:552084,552085,552090,552180,552181,553179,553180,
554664,554665,556222,556227,557105,557106,559417,559419. GPU residual showed
MPS2238410 and unowned578756/usr/bin/python/908MiB on1, untouched.

Runner v2 changes only its exclusive directory and skips symlink directories.
pytest6passed160.06s, author exit0, six live identity records, no observer errors
or outer timeout. SSH585294, runner585295, pytest585300. Each actual child
completed canonical materialization, upload and content-validated D2H first.
The real tests compare42/27 arrays against fresh canonical builders in the
parent; no copied child hash acts as the expected result.

| Case | Fault | Supervisor | Worker | Exit | Terminal result |
| --- | --- | --- | --- | --- | --- |
| asymmetric_synthetic_7x3_v1 | success | 585394 | 585399 | 0 | aborted_synced_exit |
| synthetic_dynamic_rigid_960x640_v1 | success | 586275 | 586276 | 0 | aborted_synced_exit |
| asymmetric_synthetic_7x3_v1 | sync_raise | 586845 | 586853 | 74 | quarantined_process_exit |
| synthetic_dynamic_rigid_960x640_v1 | sync_raise | 587431 | 587433 | 74 | quarantined_process_exit |
| asymmetric_synthetic_7x3_v1 | sync_hang | 588943 | 588944 | -9 | killed; retained preterminal prefix only |
| synthetic_dynamic_rigid_960x640_v1 | sync_hang | 591168 | 591169 | -9 | killed; retained preterminal prefix only |

Normal cases have one successful terminal sync and table release. Fixed raise
cases have one failed callback attempt, full quarantine and no release/retry.
Fixed hang cases stall before native synchronize; the original emergency
watchdog expires, kills/reaps worker(-9), with no worker terminal JSON. Their
retained worker prefix must not be relabeled as completed sync or release.
These fault injections prove supervision behavior, not a real CUDA/Warp driver
error or kernel hang. All six have7ARM/6COMPLETE and zero measured submissions.

Final scoped ps for585294,585295,585300,585399,586276,586853,587433,588944,
591169,585394,586275,586845,587431,588943,591168 returned header only. GPU
snapshot: MPS2238410/60MiB on0 and unowned593717/python/608MiB on1; untouched.

## Artifact digests

All hashes below identify retained raw bytes, not authority or execution proof.

- `c2d3-a3d750b-captured-cuda-ambient-author-v1/request.json`: `cc5f9f28640ab006054848d2991701c103a9832cdb3e03bf40815556e019e796`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/identity.json`: `f33aa1e31484043562172f9299c276faf8c1d5c1446ca9a891074735ee3b6bbc`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/external_process_identity.json`: `af7017ead32bdf63548c7d01b7f2c1e618a732ac0d57771608e248246bc483d3`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/captured_stream_before.json`: `53cb1af1869a3d3568a11aefdda5123e3bfe30fa5efd52f527e73f6161512917`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/captured_stream_after.json`: `ef617fce3d0e93f9ae52fae95ed037680c57b61e73dcc99231b45d568701c980`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/captured_ambient_scope.json`: `2c5e2fb8ea200908291a19a637ca16c5324dcc49afaf519e5ba8137109d11a26`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/input_validation_report.json`: `334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/custody_artifact.json`: `28e0838cce100b413fb5f6dc7d6d9d905d5724a5585ddf7f8dc36012cb8eb333`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/worker.json`: `57b3db8390d1e673291dc82aa1e07504df4f481d656107c48a8610baae5301ab`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/author_ambient_verification.json`: `e5cd0aec6380253d5c60080a70449cfae6cfd636abd7bdd38332cff5e59a5811`.
- `c2d3-a3d750b-captured-cuda-ambient-author-v1/author_component_verification.json`: `65757bd925908a9851a543648988491441ef68e2b81d1cea591c9a1944be74ca`.
- `c2d3-a3d750b-existing-warp-matrix-v2/author_run_summary.json`: `00089326884b837fe97d582c8fdc93465afee279372de3fd2d3d6211537ffbaa`.
- `c2d3-a3d750b-existing-warp-matrix-v2/pytest_stdout.txt`: `aecfb72c48553211f369ecab4a52458c365a35e861cc8e15ca9b25e00b7cbd19`.
- `c2d3-a3d750b-existing-warp-matrix-v2/pytest_stderr.txt`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `warp-v2/test_actual_cold_canonical_war0/external_process_identity.json`: `f5303103ac719175bbbe9e4a0a1ab906096a219f739d0f9d13af186599a353af`.
- `warp-v2/test_actual_cold_canonical_war0/custody_artifact.json`: `e0f08f6c796511412167751798f9fd6664c48abcbc41963b6f4f117d394b374e`.
- `warp-v2/test_actual_cold_canonical_war0/input_validation_report.json`: `334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760`.
- `warp-v2/test_actual_cold_canonical_war1/external_process_identity.json`: `1b8b5ba3c7d2fdbca171d970c3e051dc991deb69b809c19c155ae915e7e40c1d`.
- `warp-v2/test_actual_cold_canonical_war1/custody_artifact.json`: `6cb7417ce74458ed2f329ccb4d9d0fdf939004974c2aaa7d3073f0d5258fe727`.
- `warp-v2/test_actual_cold_canonical_war1/input_validation_report.json`: `816006fb9fd52057fd32fa330d46d9980f447408daa05297c33f1cc8940c1bfb`.
- `warp-v2/test_actual_cold_canonical_war2/external_process_identity.json`: `33e145d1a4b30baddeec64fb5e11609fb329cf3968f4be6abf95d2f15b362a3c`.
- `warp-v2/test_actual_cold_canonical_war2/custody_artifact.json`: `69454aa15c592ca9e51879da842daa0cbae5977433792e4cb6560649e98130fa`.
- `warp-v2/test_actual_cold_canonical_war2/input_validation_report.json`: `334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760`.
- `warp-v2/test_actual_cold_canonical_war3/external_process_identity.json`: `e2631cc033fa73ef1c4c59fd114cb9295aaa8fbf7484f694f174e1fb577c4824`.
- `warp-v2/test_actual_cold_canonical_war3/custody_artifact.json`: `f6f73755280cf9161d08d85b02ea5e9bd6223061043b15a6df8a7fd36ef6b0ff`.
- `warp-v2/test_actual_cold_canonical_war3/input_validation_report.json`: `816006fb9fd52057fd32fa330d46d9980f447408daa05297c33f1cc8940c1bfb`.
- `warp-v2/test_actual_cold_canonical_war4/external_process_identity.json`: `aa6f3df281174a40c8a5009683aed6480c03d96d08cc28d66b2db76882f36e2b`.
- `warp-v2/test_actual_cold_canonical_war4/custody_artifact.json`: `9556fcf50ad2de30410e3ca2c2e73c84b8138a9e2d5bb18b15a484cc2e695e2d`.
- `warp-v2/test_actual_cold_canonical_war4/input_validation_report.json`: `334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760`.
- `warp-v2/test_actual_cold_canonical_war5/external_process_identity.json`: `f20f5c0de7d38703e0ca3d9271e4df75271edcd7c6f2028f85b61b4c29601d19`.
- `warp-v2/test_actual_cold_canonical_war5/custody_artifact.json`: `82d002f774fb036b7cf49ac91bac6bb9460fc8a3497e1bcbb069711663eee714`.
- `warp-v2/test_actual_cold_canonical_war5/input_validation_report.json`: `816006fb9fd52057fd32fa330d46d9980f447408daa05297c33f1cc8940c1bfb`.

## Still required

Captured C1 negative, real foreign/UNKNOWN device cases, both lanes' actual
import/init/allocation/copy/event/owner/audit failures and hangs, ACK/EOF/cleanup/
missing-exit matrix, current Warp real Go2 coverage, formal versioned aggregate
runtime-custody implementation and fresh whole-C2d3 review remain required.
The historical missing asym worker start remains historical, not repaired by
this new worker. This evidence makes no performance or backend-choice claim.
C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host,
R2 and shipping remain CLOSED. Continue toward C4 under the staged approvals.
