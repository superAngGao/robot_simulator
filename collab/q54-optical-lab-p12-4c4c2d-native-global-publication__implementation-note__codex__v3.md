Initiative: q54-optical-lab-p12-4c4c2d-native-global-publication
Stage: implementation-note
Author: codex
Version: v3
Date: 2026-10-08
Status: in_review
Related Files: native-global-publication implementation-gap-review v3; implementation-note v1/v2; approved design v1
Owner Summary: Both uninstrumented actual Go2 Warp sizes now complete native input upload/readback and terminal disposal on3646630. Independent gap-review v3 still requests one replay-order repair; this revision closes that specific same-call ordering gap. Full C2d3/C4 remain open.

## Exact review and repair

Fresh independent session `/root/native_global_implementation_gap_v3` reviewed
`3646630575509a075c8fc2f0d867a1dbaf025805` and returned Request changes for one P2.
The complete unchanged report is archived at `0c80a64`; SHA-256
`9f84c943d8b99a10e1091c4c463a8972999584e00b0aed8f4fe2e2a520d74c11`.
It closed all eight prior exact counterexamples, found no established defect in
the immutable configuration cache, and independently inspected both Go2 runs.

V3-F1 repair adds a per-call replay restriction: after any admitted event for a
call sequence, no later first observation can belong to that SAME sequence.
This follows the actual loader's full return checkpoint before its admission
loop. It does not prohibit a child admission before a still-running parent's
first observation, nor admissions for unrelated observers. Tests reject the
preserved reorder in complete, non-complete and genuine unrelated-import-failure
prefixes, and retain positive three-global return/request observations following
a completed child. The active catalog is repinned; no live loader, source policy,
cache, environment row, guard configuration or product gate changed in this repair.

Local mechanics validation:266 passed,3 skipped across native-global publication,
runtime environment and import-admission files. An initial collection attempt
used pytest's reserved parameter name `request`; renaming the test parameter
fixed collection before this successful full rerun. Skips are not acceptance.

## Actual uninstrumented Go2 evidence on3646630

The existing fixed supervised driver was used directly, without the diagnostic
faulthandler/runpy wrapper. Exact requests, original artifacts, commands/PIDs and
most hashes are independently recorded in gap-review v3 section5:

- Warp target1, Go2 960x640: `/data2/shared/ang.gao/c2d3-3646630-go2-960-guarded-author-v11`.
  Invoker `/data2/shared/ang.gao/run_c2d3_go2_guarded_author_v11.py`, SHA
  `5163f7f76be3d1c71c63dcac1eaba7ddf41a6ed534bf5d3345069da3dbc5bc19`.
- Warp target1, Go2 1920x1080: `/data2/shared/ang.gao/c2d3-3646630-go2-1080-guarded-author-v12`.
  Invoker `/data2/shared/ang.gao/run_c2d3_go2_guarded_author_v12.py`, SHA
  `ec68a4722f8492f368541034bc61cbf5ea8d363f964c0d45aa91c5c2040d8c42`.

Each invocation was `timeout 420` followed by the pinned robot_sim interpreter
`-I -S -B` and its invoker path. Both invokers/worker exited0, supervisor reaped
them without timeout. Original ACK1000ms/operation120000ms were unchanged.
Materialization segment durations were63.544s/63.334s, operational observations
only, not renderer benchmarks or proof of CUDA-vs-Warp speed. Both returned42
validated arrays, six normal completions, seven ARMs, one terminal target sync,
guardian RELEASED and release completed. All three configured globals have
actual pending/admitted events under the expected enclosing native execs.
No optical sample, final execution readiness or product output was issued.

Local read-only checker `../c4_assets/check_actual_go2_results_3646630.py`, SHA
`aa29123e8977c851027f29476a41410d7fbe06f8c727101344ebbfb824094e83`, replayed the
existing guarded-custody artifacts and checked fixed request/result/count fields.
It is not the mandatory aggregate runtime-custody schema or independent new
materialization. The reviewer separately replayed both native-global components
on pinned Linux. Both runs belong to3646630, not this later parser-only revision.

Scoped ps found none of the six owned parent/supervisor/worker PIDs after exit.
NVML showed pre-existing MPS2238410 plus unrelated tilesight-tl012 Python4184984
at740MiB; neither was touched. No globally empty GPU-process assertion is made.

## Remaining scope

Fresh independent review is still required for this delta. No bounded gap scan
is whole C2d3 acceptance. CUDA deployment closure and actual canonical/Go2 setup,
complete both-lane fault/hang/dual-GPU/terminal matrix, final versioned aggregate
runtime-custody schema and whole-stage review remain required. Continue these
authorized C4 prerequisites after closing the current repair loop. C2e/f,
C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline, host payload,
R2 and shipping remain closed; C4 is not complete.
