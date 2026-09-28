# Fixed shade checkpoint: review and regression evidence

Initiative: q54-optical-lab-p12-4c4c1b2-fixed-shade
Stage: test-evidence
Author: codex
Version: v1
Date: 2026-09-28
Status: accepted
Related Files: same-initiative independent review v1; implementation note v1
Owner Summary: Record scoped approval of 845dbb3; no gate or code changes.

Independent session `/root/c4_fixed_shade_review_v1` returned **Approve with
comments** for `845dbb3d923b73669154494ebe84270ab7164775`. Its complete report is
recorded unchanged beside this file. Its independent 163 repository tests and
56 exact counter/edge shader runs are separate from author regression evidence.

Author command on clean `/data3/ang.gao/robot_sim_c4_shade_review1`, authorized
101 fallback, conda `robot_sim`:

```text
python -m pytest -q -m 'not slow'
1906 passed, 39 skipped, 117 deselected, 1 warning in 371.67s
```

Log: `/data3/ang.gao/c4-fixed-shade-v1-full.log`. The warning is the existing
pytest class-scoped fixture instance-method deprecation. No performance sample,
baseline or result artifact was generated. The skipped/deselected tests are
not claimed as passed coverage.

Non-blocking comment C1 is accepted as a bounded C4c1c test-hardening follow-up:
retain equivalent hand-derived exact traversal-counter microfixtures in tracked
tests. Scratch source and digest are in the independent report. This is not a
new mathematical rule or activation condition.

Approval covers only this kernel/reset/ABI compatibility checkpoint. Camera,
factory, recorded bundle and C4c1c execution containment are unfinished. C4,
product GPU preparation, host payload/B3, a3b/result/baseline and shipping gates
remain closed. This evidence-only commit changes no executable/design/workflow
contract and needs no recursive review under collab/README.md.
