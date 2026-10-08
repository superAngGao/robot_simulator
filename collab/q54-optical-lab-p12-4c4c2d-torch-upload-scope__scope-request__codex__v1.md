# C2d3 actual CUDA upload/readback bounded scope review v1

Frozen request: 2026-10-08. Code/gap review, not a new policy or C2d3 acceptance.
Local repository C:/Users/AG/新建文件夹/p12_4_c4_warp_timeline_design,
clean HEAD a30a761da355ca4cd884b202428e29d8f5418285; production code unchanged
from a3d750bbaf79d760a6727e56146d277e1cb3b770. Remote tileops-dev isolated
/data2/shared/ang.gao/robot_sim_c2d3_binding_wip stays clean at that a3d750b.
Local evidence-only base 96c0731a37ee07c436978e936b42cee6d8afeabb.

Request: independently inspect the existing shared captured production sequence
through allocation/H2D/ready-event/D2H/content validation/terminal disposition,
and determine whether its pinned CUDA dependency leaves substantiate exact-target
scope for the next actual run. Review code, native call chain and failure custody;
do not infer safety from prior approvals, Python signatures or successful counters.
No code changes are proposed. Report actual gaps rather than inventing a new gate.

Next execution uses the unchanged tests/support/native_runtime_process.py child
and tests/support/guarded_custody_process.py supervisor, fresh -I/-S/-B worker,
CUDA lane, asymmetric_synthetic_7x3_v1, target ordinal1, ACK1000ms/op120000ms,
environment external 183_cuda_synthetic_author_draft_v9.json (tracked identical
tests/data/runtime_environments/183_cuda_stream_author_v1.json), canonical digest
b4e47505a751377d48336beba5508cdbe2a2adcccdd10ae0e463e0767a537a47.
No monkeypatched GPU leaf, warmup, visibility remapping or ambient-device reset.
Normal driver terminates via its supervised sole target sync then process exit;
failure remains existing session/controller fail/quarantine. The author will
freeze a mechanical fixed invoker before execution, preserving raw failures.
Subsequent replay and real Go2 have separate descriptor closure obligations.

Required designs under collab prefix q54-optical-lab-p12-4c4c2d-:
runtime-adoption__design-request__codex__v2.md and design-review v2;
guarded-custody__design-request__codex__v2.md;
cuda-ambient__design-request__codex__v1.md and design-review v1.
Inspect native_bootstrap.py, native_setup.py, native_readback.py, native_contract.py,
guarded lifecycle/controller and the two driver files as needed. In particular:
explicit target torch.empty; nested outer device + inner stream for both copies;
CPU NumPy from_blob; event creation/record/wait; tensor storage inspection;
normal/partial-error ownership and terminal CUDA synchronize/release.

Pinned Torch source revision 5811a8d7da873dd699ff6687092c225caffcf1bb.
Outside-repo evidence root C:/Users/AG/新建文件夹/c4_assets:
torch_upload_sources_v1/manifest.json (7 files), SHA256
eb01115d801f3fd50f8d1b5377d8e33ed539b02877bc8c266119906c037cb6cd;
torch_upload_host_sources_v1/manifest.json (2 files), SHA256
d259493acba4aabac838f42dd9df61b9f0e27385567f83513bed1097c8fdc0d5;
torch-scope-upstream-5811a8d/, torch_native_scope_extra_upstream_v1/,
torch_deferred_sources_v1/, torch_native_scope_installed_read_v1.json,
c2d3-torch-init-native-audit-v1.json, torch_deferred_review_nvrtc_binary_v1.json.
Verify sources independently; official pinned raw GitHub available if needed.
Installed robot_sim Python3.12.13 at /home/ang.gao/miniforge3/envs/robot_sim/bin/python.

Recent evidence (claims to verify), collab same prefix:
torch-captured-stream__test-evidence__codex__v1.md (actual adopted binding inside
original armed runtime segment then intentional UNKNOWN before COMPLETE);
torch-copy-scope__test-evidence__codex__v1.md (ordinary supervised C1 production
copy vs stream-only negative: only latter creates context0; not full captured a2);
torch-deferred-scope__gap-review__codex__v2.md (bounded first Stream only).
Raw artifact directories local/remote c4_assets/shared respectively:
c2d3-a3d750b-captured-torch-stream-diagnostic-v3,
c2d3-a3d750b-torch-copy-scope-v1-production_nested,
c2d3-a3d750b-torch-copy-scope-v1-fault_stream_only.

No GPU execution by reviewer in this preflight; CPU/source/ELF checks allowed.
Do not mutate tracked files, alter dependencies, broad-kill or touch other users.
Owned temporary CPU processes must be checked/reaped. Local Python C:/Python314/python.exe.
Report unexecuted real validation honestly. Aggregate versioned runtime-custody
schema, both-lane canonical/replay/Go2/fault/hang/ACK/EOF/dual-GPU/terminal matrix
and whole fresh C2d3 review remain required. No later C2e/f,C4c3/c4,C4d/B3,
product preparation,a3b,result/baseline,host,R2/shipping gate opens.

Write complete unchanged review to outside-repo
C:/Users/AG/新建文件夹/c4_assets/torch_upload_scope_gap_review_v1.md,
including session identity, exact revisions, requirements, findings, inspected
native chain, commands/results, proof limitations and precise verdict scope.
