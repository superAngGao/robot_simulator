# Refresh custody after diagnostic RGB readback

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: bounded native test-driver repair
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: fixed-render-warmup code-review v2; warmup-bootstrap-repair code-review v1
Owner Summary: The first real Warp image was rendered and read back, but the driver compared its later readback lifecycle to the earlier warmup checkpoint. Refresh the existing validated checkpoint after the readback guard completes. Production rendering and source policy are unchanged.

Base: c83b35eeca9a9fec834425f066b41202c7d01059, evidence-only successor of approved0ede8d0.
Candidate: commit containing this note; exact HEAD in review packet.

Native attempt: exact0ede8d04a9ce99c9c2562cd1dfe1e692c7e5c526 at
tileops-dev:/data2/shared/ang.gao/robot_sim_render_warmup_wip, raw directory
/data2/shared/ang.gao/c4-render-warmup-author-v2. Warp Go2 960x640,
direct_light_full/shadows on/diagnostics off. Batch61466 ended exit1 because
worker returned74 at the later observe() check: normal custody checkpoint does
not match current lifecycle. This was after50 actual warmup launches, ten
terminal waits and the RGB copy, with25 normally completed setup guards.
Quarantine followed with no emergency sync, no kill/deadline expiry. Parent
3464184, supervisor3464352 and worker3464353 are absent; live PID/start ticks
were recorded in the original directory. CUDA and remaining cases did not run.

Failure archive c4-render-warmup-failure-evidence-v2.tar.gz is preserved remotely
under /data2/shared/ang.gao and locally under ../c4_assets, SHA256
0b7e668a1717f52683874ea831624e2aae0b38849d4b0b27cc3d7e78c115b8b1.
The unchanged _validate_render_warmups passes the original warmup/RGB artifacts;
this does not waive the failed whole-case terminal check.

Original frame10 RGB:7372800 bytes, SHA256
2bfa01fc5b39c61cce30c850a4d52183f3feb3a873a4ff7b1ff1e3c6931b7ba4.
Downloaded bytes match the recorded hash; all float32 values are finite, range
0..1.8693655729293823. Existing linear_rgb_to_preview_uint8 creates a diagnostic
PNG at ../c4_assets/warmup_image_warp_960_v2/warmup_rgb.png. Visual inspection
shows Go2/ground/shadows; no CPU-oracle correctness or speed claim is made.

The fix calls session.runtime_observation.checkpoint() after the driver's
additional readback guard exits successfully, before returning to observe().
It uses the original checkpoint parser and does not copy/rewrite old evidence
or relax the strict lifecycle comparison. Errors propagate to the existing
whole-process terminal path. No product source, catalog, policy, schema,
renderer, supervisor or deadline changes.

The existing positive support test now records actual lifecycle payloads at
checkpoint calls and requires the final checkpoint to equal the current
post-readback lifecycle. Before repair all four lane/diagnostic combinations
failed this assertion (4failed,2deselected in14.84s, handle58233 exit1).
After repair native_warmup_support and cuda_artifact_runtime tests passed36 in
34.24s (C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider; handle38858
exit0). These use explicit native doubles. Real positive rerun remains required.

The previous bootstrap reimport failure is now passed in real execution, but
this second failed attempt still is not positive native acceptance. Numerical
validation, failure matrices, fair performance, final readiness/graph/arena,
whole-C4 refactor/review and all previously closed product gates remain pending.
