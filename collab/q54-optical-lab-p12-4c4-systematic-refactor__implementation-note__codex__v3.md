Initiative: Q54 / P12.4 / C4
Stage: execution consolidation review repair and native regression evidence
Author: codex
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: systematic-refactor implementation-note v2, code-review v1; runtime-integration design-request v3
Owner Summary: Fix the independently reproduced resolver context-rebinding gap using existing template/authority validation. Empty LBVH stages no longer declare a nonexistent native call. Fresh native R0 checks cover the preceding consolidation revision only; final integration and C4 acceptance remain incomplete.

## Independent finding and repair

Fresh CLI review session `01a1237e-5d4d-7671-909d-86ca05f7eef1` reviewed clean
1934ac7673348c12ccb76a6e0b54ef883b17324b against c9d5377, with Request changes.
Built-in fresh spawning hit the existing thread limit; the owner-authorized
fresh CLI fallback supplied no implementation conversation or implementation role.
The complete report is archived as systematic-refactor code-review v1 in the
evidence-only commit 6f8759e. The CLI final-output option overwrote the report
path with the reviewer's short final response. Its full 19419-byte report was
recovered verbatim from the reviewer's successful PowerShell here-string write
in the raw JSONL; recovered byte length matches the recorded original stat.
Saved report SHA256:314621b6034a507beff37c90405594490bb112ea454c3889eb9358097897e052.
Original JSONL, short final response, exact repros and logs remain under local
c4_assets/execution_consolidation_review_v1. No finding was omitted or rewritten.

F1: active-sample identity checks did not freeze the contents of the retained
mutable execution context. A different stream or device during resolution
could pass the post-wait check, resolve the lease and drop its transaction hold.
The repaired common resolver validates the existing sealed transaction template,
minted live authority, exact completion/event and device/stream scope before
waiting and again after any DISARMED acknowledgement, before lease commitment.
It reuses existing validators; no new registry, state machine or copied seal.

C1: the LBVH launch hook now runs only when a native entry exists, retaining
the same diagnostic phase order and shared launch body. The remaining final
warmup audit literal now references SETUP_WARMUP_FRAME_COUNT.

Resolution tests now construct a real controlled template, transaction,
prepared capability, minted live authority and active lease and call actual
enqueue/finish. Only native event/stage operations and C3 inputs are CPU doubles.
They additionally reject device, handle, event-device and stream drift before
wait, during wait and during acknowledgement, including a self-consistent
replacement context stream/handle. Rejections retain both transaction and result
holds and credit zero resolutions/reclaims.

Unmodified reviewer repro files plus updated resolution/timeline tests passed
162 cases in10.38s. An intermediate implementation referred to a nonexistent
completion.device_ordinal field (17 failed,145 passed); it was corrected to the
actual completion.device field before that passing run. No failed run is counted
as successful evidence. Repair candidate is the commit containing this note.
Catalog v17:88 entries, SHA256
03b0e419b32e535758617db36b1ddfacb6b4faf2042d46f51c1417eece2f8951.

## Fresh 1934ac7 verification, not evidence of the later repair

Local complete selected consolidation suite:578 passed,4 skipped in162.58s.
The four skips require real Go2 assets/POSIX. Linux CPU-only source/source-policy/
recipe suite with actual Go2 assets and CUDA hidden:215 passed in39.92s, including
the actual POSIX committed-byte checks. Commands are the twelve unit files in
v2's scope, including render preparation and CUDA recipes; remote command and
process identity are retained in c4-execution-refactor-cpu-v1/launch.json.

Both lanes ran the unchanged supervised runtime_render_warmup_geometry driver,
real Go2 960x640, at exact clean 1934ac7 in
/data2/shared/ang.gao/robot_sim_execution_refactor_wip. Each performed ten actual
warmups and read frame10. All614400pixels and all ten channels pass the unchanged
original CPU oracle812ca29 comparison; RGB/intensity bytes match between lanes.
Maximum RGB absolute error3.7175754741269884e-7; intensity3.992438959432576e-7.
This does not exercise guarded final-arena resolution, R1, synthetic precision
counterexamples, compatibility GPU entrypoints,1920, timing or whole C4.

CUDA reused the exact previously built fixed kernel artifacts (kernel strings
and recipes were unchanged), through c4-precision-gap-cuda-environment-v1.json,
SHA000fe3bece3850327e3930630804fd5d5f5a9585172415dc7e0f34ec18ceb0e2.
The existing compilation artifact archive remains the deployment provenance;
no new compilation result is claimed. Warp used183_warp_go2_author_v4.json.

Raw directories:c4-execution-refactor-{cuda,warp}-960-geometry-v1.
Comparison SHA256s:
- CUDA:5b6269a92138c46bd0798463d2646ff4f9d3fdb26d4df85f1db0831d21432acb
- Warp:95e431727318e39c9483378aac64a2a03810b7ac59125a564f8480843fa1e147

Archive:c4-execution-refactor-native-evidence-v1.tar.gz,50722689bytes,100members,
SHAcec14bada1cc7d165d5d6cfc3cb3a401a6e91facb2fdd48d55a8591748aa2a02.
Retained locally and under /data2/shared/ang.gao; downloaded hash verified.
Closure JSON SHA97303284749bcd370bf59a4a20e926d08191a264b1c24dc029562d5c1e19512f.
All eight owned native/CPU job PIDs absent:
4028623,4028761,4028754,4041706,4041838,4041834,4028917,4028928.
The first local script-generation attempt failed on PowerShell pipe encoding,
so its remote launch found no script and started no GPU worker. Using the
working directory removed the path encoding issue; the successful runs above
each use unique output directories. A bundle command using unnamed SHA endpoints
also rejected an empty bundle before transfer; named HEAD packaging succeeded.

## Open scope

Fresh repair review is still required. M1/M4 final integration, both-lane final
readiness/origin/C3/arena/barrier, R1, lifecycle/failure/native precision matrices,
systematic whole review and all remaining C4 work remain open. No timing or
backend-selection claim exists. All later product/measurement gates stay closed.
