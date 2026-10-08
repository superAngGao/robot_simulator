# C2d3 CUDA canonical replay and real Go2 bounded launch review v1

Fresh independent code/gap review, not a new design or whole acceptance.
Repository C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design clean HEAD
fd89a7ea7fec6032e9756db8d0602aa304393a0b, base a30a761da355ca4cd884b202428e29d8f5418285.
Production code unchanged at a3d750bbaf79d760a6727e56146d277e1cb3b770, also clean
on tileops-dev /data2/shared/ang.gao/robot_sim_c2d3_binding_wip.
Read full AGENTS.md,collab/README.md,collab/REVIEW_PRINCIPLES.md first.

Question: do the existing approved native helpers and exact new launch variants
support the next actual captured CUDA dynamic replay and both real Go2 attempts?
Independently inspect changed workload extents/dtypes/layouts, materialization
order, dependency descriptor, indexed allocation/copy/event/readback and terminal
scope/lifetime. No GPU run by reviewer; CPU/source/file/replay checks permitted.
No candidate edits, dependency mutation, broad signals or other-user interference.

Frozen outside-repo candidate scripts under C:/Users/AG/新建文件夹/c4_assets:
run_c2d3_cuda_upload_author_v2.py SHA160715958d666103d4b969b26046cd3c481448101ae222d4cfcba7a3fbbf783a
run_c2d3_cuda_upload_author_v3.py SHAb6bf6741b2c41c864d904e13055687f38ddbbc603498878b5513573123b8f511
run_c2d3_cuda_upload_author_v4.py SHA60de9fba981c2f1007b9d80e66ac9961322bd8b612ce0a9a6efb1d8d61e823ae
launch_c2d3_cuda_upload_fixed_v1.py SHAf5bbb7dd6fce0b788537dd623aa4666076409d19031d34a91d5373f3a958bbd6.
All staged identically under /data2/shared/ang.gao, not yet executed.
Variants replay/go2_960/go2_1080 differ from approved asym only in fixed case,
exclusive output directory, session string, and Go2 profile/model path as needed.
Use pinned Python with assertions: python -I -B launch_c2d3_cuda_upload_fixed_v1.py <variant>.
Unchanged production child/supervisor, CUDA target1, ACK1000/op120000ms per original
segment, no renewals. No observers/warmup/GPU mocks, no copied preparation loop.

Canonical replay profile b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47,
external183_cuda_synthetic_author_draft_v9.json (identical tracked183_cuda_stream_author_v1).
Go2 profile canonical79c3580baff1ef3a019c97a07177558808918b7e133caa430239e25b82853919,
raw47f755f6dc229858a9bf0b59673010a6aeac30a92a9afe9fbd5487300e7e8f71;
external183_cuda_go2_author_proposed_v1.json identical tracked183_cuda_go2_author_v1.
Real frozen Go2 path /data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2.
Existing asset preflight remains mandatory, no replacement fixture.

Predecessor collab q54-optical-lab-p12-4c4c2d-torch-upload-scope__gap-review__codex__v1.md
SHA d7f4eed814649296ea5776c39795cc179343dac5ecde6b54d957084ecd9362a6
approved asym only. C1 addressed by unchanged wrapper rejecting12 env variables,
recording snapshot and execve with exact env. C2 retained: RELEASED only clears
guardian owner table; bindings/module/native caches may live until process exit.
No claim of immediate all-native-free or destructor audit. Keep these conditions.
Go2 descriptor gap-review v1 SHA2cfc65675b11526f82de0c65bd1e2766bbc6803631a640d16e29747abc7d5219
approved descriptor only; its union/native publication/absence provenance is
available to check, not a substitute for actual combined runtime execution.
Governing runtime-adoption design v2, guarded-custody v2 and CUDA ambient v1.

New actual asym evidence: local c4_assets/c2d3-a3d750b-cuda-upload-author-v1 and
remote /data2/shared/ang.gao/c2d3-a3d750b-cuda-upload-author-v1.
Parent431190 (execve samePID), supervisor431371, worker431372; wrapper431188.
Worker0 observed/reaped; scoped ps all absent.7ARM/6COMPLETE,43allocation,
42H2D/42D2H,2 each event create/record/wait,3runtime import/init;136entered leaves.
One terminal target sync; guardian table RELEASED then hard exit; no ordinaryatexit.
UNKNOWNfalse, dirty scope[1]. No new raw/logical/context observation in this run.
NVML residual showed preexistingMPS2238410/60MiB plus transient unowned436331/python/
target1/582MiB. Subsequent scoped ps436331 absent before ownership could be read;
do not assert its user or call that an owned leak. No signal issued.8H200 mapping
still device1GPU-3a3112ff-837c-3b87-38d1-cb203d656589,driver595.71.05.
Raw SHA: custody_artifact6f30c50fc0d5aa069cd08a174153d176532f9ade089d4236f36419d466ca4668;
worker717b8ec97070bef3274c87f7afca0e732fa2210a602734c15dc08af756202315;
input_validation_report334f0bf29d401bd966e2153e7839babc740a8f799e1448f8341e1c24c2e1a760;
supervisor_stdoutb371e095d0d2e19029fb70d213e551cb0bb8f7929ec329b4f16a6278a4f029c8.
Existing custody/native-call/replacement/alias components replayed, all42array
names/content/topology/manifest matched a freshly regenerated CPU reference.
No aggregate runtime-custody or full ambient result claimed.

Fresh CPU manifests all four cases: c4_assets/c2d3-a3d750b-fresh-cpu-reference-v1,
also remote same basename. Script c2d3_expected_canonical_inputs_v1.py uses sole
canonical builders; CPU407517/407515 completed and absent.42fixed/27replay/42eachGo2.
Manifest replayfd63ec133149ac62ef49e37534783d8c3d4e9b6f21132fa90ac02b606187b023;
Go2-960ac3d950309917168d0bb5758e9916adcffcbb17bd68f6611e4fe0f3bdfed3a12;
Go2-10804785c42146d8707453654eefbc4c94e38e683ec4b5659536240e6b1048b2279e.
Old a3d750b Stream/C1 diagnostics and exact5811a8d native sources listed in prior
report remain available in c4_assets. Verify actual provenance as proportional.

This review requests only these three normal attempts, not failures or foreign
touch permission. Full ambient/dual-GPU, import/allocation/copy/event/adoption/
audit fault/hang/ACK/EOF/terminal matrix, final versioned aggregate schema and
whole C2d3 fresh review remain required; no later gate opens. Report sufficient
or missing native scope precisely; do not infer workload success from earlierasym.
Write complete report c4_assets/torch_upload_workloads_gap_review_v1.md with
exact snapshot, findings, original C1/C2 disposition, commands/results and bounds.
