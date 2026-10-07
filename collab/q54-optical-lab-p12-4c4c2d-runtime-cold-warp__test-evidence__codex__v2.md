# C2d3 dynamic replay and Go2 import-policy gap

Initiative: q54-optical-lab-p12-4c4c2d-runtime-adoption
Stage: test-evidence
Author: codex
Version: v2
Date: 2026-10-07
Status: draft
Related Files: cold-warp test-evidence v1; runtime-adoption implementation-note v1;
optional-import gap-scan v1; tests/gpu/test_native_runtime_cold_warp.py
Owner Summary: Six real cold Warp tests passed, including dynamic replay and
supervised terminal failure/hang. Real Go2 CPU materialization still passes, but
independent review confirmed a policy/dependency gap in new guarded admission.
No whole C2d3 approval or gate activation; C4 remains incomplete.

Code candidate:a502ef1e01f16c45ab90faf58e3b3692122cafb2.
Primary183, hd03-gpu2-0031,8H200, robot_sim Python3.12.13, driver595.71.05.
Local/remote isolated candidate worktrees were clean throughout review.
No101 fallback, installs, original-worktree edits, unrelated kills or GPU resets.

## Actual cold Warp results

```text
ROBOT_SIM_TEST_C2D3_COLD_WARP=1 timeout --signal=TERM --kill-after=5s 240s \
 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -m pytest \
 tests/gpu/test_native_runtime_cold_warp.py -q -s
```

Result:6 passed in162.17s, no skips, wait/SSH exit0. Wrapper3341244,
Python3341246,parent3341214,start23:18:15+08. Continued observing the same
session89229; no restart on tool observation timeout. Outer240s bound did not fire.

| Workload | Fault | Supervisor/worker | Actual exit |
| --- | --- | --- | --- |
|asymmetric_synthetic_7x3_v1|success|3341316/3341317|0|
|synthetic_dynamic_rigid_960x640_v1|success|3341615/3341617|0|
|asymmetric_synthetic_7x3_v1|terminal sync raises|3341962/3341963|74|
|synthetic_dynamic_rigid_960x640_v1|terminal sync raises|3342486/3342487|74|
|asymmetric_synthetic_7x3_v1|terminal sync hangs|3342793/3342794|-9|
|synthetic_dynamic_rigid_960x640_v1|terminal sync hangs|3343523/3343524|-9|

The same frozen author environment, captured production upload/readback cores
and real guarded supervisor serve both cases. Parent expectations are derived
from the sole canonical scene/fixture/replay builders, not copied child output
or a second hand-written content oracle. Asymmetric has42 arrays; dynamic has27,
including both transform replay arrays and more than one frame. Successful
validation is input-only, not optical readiness/sample/result authority.

Terminal-raise retains/quarantines the graph after one attempt. Deliberate hangs
expire the actual30s operation watchdog with parent alive and actual child reap;
they are not successful release. The hang record remains the last worker prefix,
with the supervisor supplying terminal kill/reap evidence. No invented cleanup.

## Go2 asset and CPU regression

```text
ROBOT_SIM_TEST_GO2_MODEL_DIR=/data2/shared/ang.gao/c2d2_frozen_assets/mujoco_menagerie/unitree_go2 \
 timeout --signal=TERM --kill-after=5s 150s \
 /home/ang.gao/miniforge3/envs/robot_sim/bin/python -B -m pytest \
 tests/unit/optics/test_native_sources.py -k real_go2 -q -s
```

Result:2 passed,39 deselected in45.55s, wait/SSH0. Wrapper3348554,
Python3348556,parent3348552,start23:25:11+08,session44618.
Both lane-parameterized tests materialize both Go2 resolutions with actual asset
preflight. They do NOT use the new external runtime admission and are not its
acceptance evidence.

Asset HEAD remains da76818e269b82289eba39808e2fb91d679d6994. The historical
CRLF archive's stat/status M entries and existing core.autocrlf=true are recorded
in source-custody test-evidence v1. Fresh unstaged/staged diff name lists are empty,
and formal preflight/content validation succeeds. No reset, byte normalization,
Git configuration change or asset mutation was performed in this round.

## New gap and independent result

An author CPU-only -I -S -B observer confirmed normal trimesh5.0.0 import succeeds
while ray_pyembree and path.polygons bodies raise missing embreex/shapely errors
that parent modules catch. No Torch/Warp import. This is dependency observation,
not permission to ignore those failures in the guarded session.

Scratch probe:/data2/shared/ang.gao/c2d3-audit-trimesh-import-outcomes-v1.py,
matching local c4_assets/audit_trimesh_import_outcomes.py SHA-256
5794966dcab585470d9a57b66ff3ea63cb1cf7c6e0fea2d408b81b1f61c71a7c.
Retained local c4_assets/c2d3-trimesh-import-outcomes-author-v1.json SHA-256
54c1b6626fba8a807005c644abec75055e51a31adf0b4e1d7bf01d76009ced5d.
The observational hook is not production code, an admission fixture or a GPU proof.

Fresh independent session /root/c4_c2d3_optional_import_gap_review_v1 separately
reproduced both failures and ran the two relevant existing admission tests:
2 passed in0.20s. Its complete unchanged report is optional-import gap-scan v1,
SHA-256 4966fe2a37018f35dd7d83b1404c03027d2a2dbadff1f0dccb13f3d74b1e1109.
Verdict:Request changes for a bounded DESIGN decision, not whole-code review.
Current implementation correctly follows the accepted sticky-failure policy;
that policy does not yet accommodate required pinned Go2 dependency behavior.

Disposition: preserve all current failure rules and gates. Next draft a narrow
design amendment evaluating explicit, truthfully recorded pre-execution disabling
of exact optional adapters, as the reviewer recommends. Do not mark present files
absent, blanket-ignore ImportError, install packages, patch a dependency, replace
the canonical materializer or claim Go2 guarded success before design review and
required real-path tests. This evidence authorizes none of those changes.

## Hygiene and remaining scope

At23:23:40 and23:27:20+08, scoped OS checks found no ang.gao Python/test/worker/
supervisor/timeout residual; only user-session/current SSH readers. GPU inventory
contained only pre-existing MPS2238410,60MiB, untouched. Reviewer subsequently
checked its own exact probe/test exits. Local Python/SSH residual query empty.
Original robot_simulator remained clean f771e612beaeea126e26926738097512cd2d809c.
No claim of global-idle status or repair of the historical101 hang.

The evidence-only delta changes three collab documents, no code or gates.
Default staged diff-check reports ten trailing-space lines in the unchanged
reviewer report: intentional Markdown hard breaks from the returned report.
They are preserved to keep the archive verbatim. Diff-check restricted to the
two author documents passes; no full default diff-check pass is claimed for the
report archive. Candidate a502ef1 itself passed show-check before this archive.

Still required: resolve this scoped design gap; full CUDA/Go2 deployment admission;
both-lane real setup partial-failure/hang/parent-loss and dual-device matrix;
complete native closure/custody evidence schema; whole-candidate independent code
review. C2e/f, C4c3/c4, C4d/B3, product GPU preparation, a3b, result/baseline,
host payload and shipping remain closed. Existing shared workflow files unchanged.
