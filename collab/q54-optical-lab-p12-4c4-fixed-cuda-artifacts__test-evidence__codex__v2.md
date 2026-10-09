# Fixed artifacts: terminal acceptance

Initiative: Q54 / P12.4 / C4 scope convergence
Stage: terminal execution and independent acceptance evidence
Author: Codex
Version: v2
Date: 2026-10-09
Status: accepted
Related Files: fixed-cuda-artifacts test-evidence v1; acceptance-review v1
Owner Summary: All ten native cases completed and independent review approved
  the fixed-artifact replacement. Rendering correctness, performance and whole
  C4 completion remain unproved.

This terminal record supersedes the live-session status in test-evidence v1.
That historical build record and its original limitations remain unchanged.
Execution code: 66240e4fedc24ad8a5109690e3b1e78b450e2f62.
Reviewed frozen packet: 0df3e2609a7af03c789be819714f8d79f104ce63.

Original tool session94514 completed exit0 after accepting all ten cases:
cuda_fresh, cuda_reused_deployment, cuda_full_resolution, warp_go2,
warp_full_resolution, create_after_raise, exec_after_raise,
entry_adoption_raise, factory_after_raise, factory_hang.
The first five workers exited0, the four exception workers exited74,
and the hung worker exited-9 under its original supervisor deadline.
Parent2083102 and all20 recorded supervisor/worker PIDs were observed absent
by both author and reviewer. No matrix job remains live or needs restarting.

The hang reached cuda_lbvh_factory.after_adoption in original setup guard6.
Its supervisor records deadline_expired, kill_requested, no kill error and
complete EXITED. Its durable worker record is the preceding input-validation
prefix, with no terminal disposition or release authorization. This proves
supervised termination, not worker-reported quarantine or completed cleanup.

Remote raw root: /data2/shared/ang.gao/c4-fixed-artifact-runtime-v2.
results.json SHA256:
a82fe3dbb5ae47e0210fb35128746d700101fdb46b0314a7430bd957660b11d5.
The complete raw directory and composed environment are archived at
/data2/shared/ang.gao/c4-fixed-artifact-runtime-evidence-v2.tar.gz and copied to
C:/Users/AG/新建文件夹/c4_assets/c4-fixed-artifact-runtime-evidence-v2.tar.gz.
Local and remote archive SHA256 match:
4cb99bcb8523e74559f312cf54f680bbb5219e3017cb0efecf8cd7bdc6e34816.

Fresh independent reviewer /root/fixed_artifacts_acceptance_v1 approved the
bounded whole replacement with no substantive findings. Complete report copied
verbatim to q54-optical-lab-p12-4c4-fixed-cuda-artifacts__acceptance-review__codex__v1.md;
source and archived bytes have SHA256:
fb1da1aa38358c5a31ea988c97b39725176c50907389f07e03c274767e8cc972.
Review independently checked six manager cases, three actual builds, artifact
negatives, all ten native cases and public replay; fresh CPU tests yielded
596 passed and47 skipped. Skips do not establish native coverage.

Acceptance covers deployment, ordinary loading, ownership, factory/workspace
preparation, real Go2 input upload/readback and failure disposal. It does not
prove image rendering, sample execution, warmups, final readiness, performance,
or whole C2d3/C4. Remaining Warp module custody, timeline/arena, graph/seal/barrier,
C4c3/c4, C4d/B3 and whole-C4 structural refactor/review remain obligations.
No product PREPARE, benchmark/result/baseline or backend-selection gate opens.
This is an evidence-only record under collab/README.md; code and policy do not change.
