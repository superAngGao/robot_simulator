# Keep physical triangle range distinct from primary search pruning

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: implementation follow-up, pending fresh independent review
Author: codex
Version: v2
Date: 2026-10-10
Status: in_review
Related Files: intersection-precision implementation-note v1; code-review v1; actual native helpers and regression tests
Owner Summary: Fix reviewer F1 without rounding away physical/shadow range checks. Primary triangles are checked against the physical sensor maximum; the existing float32 comparator decides whether an accepted hit replaces the current result. No new tolerance, fallback or callback.

Rejected candidate c7d10114ccf9d29e688fa9f6a6387c6fc6509de8 remains in Git and
in the clean isolated remote checkout. The full independent Request changes
report is archived unchanged (SHA256
fb5ff4def3d21e5b8d709dcbdbaa68701021faaf3c40c5c12a763f870e54002a).
Its exact scratch repro remains c4_assets/intersection_review_boundary_v1.py.

F1 separated two meanings previously passed through max_t: physical range,
which the new double triangle calculation must enforce strictly, and rounded
primary search distance, which must not prematurely reject a distance allowed
by the existing float32 comparator. The revision passes the original physical
max_distance to primary triangle intersection. It removes redundant per-leaf
best-hit clipping; each physically valid returned hit reaches the original
comparator. Node/child search pruning still uses the existing tie window capped
by the physical range. Shadow callers retain their original physical cutoff.
Triangle validity still rejects true double t<0 or t>max_t before output
rounding; the proposed global rounded-range alternative was not implemented.

The exact original review repro now returns instance10 for both orders and
both leaf layouts in both source paths. Standalone helper clipping at the
rounded physical bound still rejects the outside hit, as required. No native
GPU execution or build occurred for the rejected candidate.

Regression coverage now includes the exact rounded-tie boundary, a candidate
one binary32 step outside the tie window, and a physical maximum that rejects
the farther face even though its returned rounded distance would equal that
maximum. Both orders, both leaf layouts and both actual source paths execute.
The shared primary/shadow helper additionally tests distances just below/at/
above the physical cutoff with equal rounded outputs and a negative distance
near zero. Existing captured Go2 predicates remain covered. Test harness
numeric/stack constants now come from the actual source rather than duplicated
test literals, including CUDA's actual stack32.

Author command `C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider
tests/unit/optics/test_native_intersection_precision.py` gives68 passed in1.98s,
exit0. The unchanged exact independent boundary script also exits0 with the
corrected full-kernel selections while retaining strict helper rejection.
git diff --check passes. Prior v1's51-test result remains its own revision's
evidence; native compilation, full Go2 geometry/RGB/CPU checks, speed impact,
whole C4 integration/refactor/review and all later gates remain pending.
