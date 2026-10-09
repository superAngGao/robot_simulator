# Consistent box arithmetic and conservative stored entries

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: implementation follow-up, pending fresh independent review
Author: codex
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: intersection-precision implementation-note v1, code-review v1/v2, review-followup v2; actual native helpers and regression tests
Owner Summary: Address F2 by using double box arithmetic, keeping physical validity separate from hit ordering, and rounding stored box entries downward. No new tolerance or runtime infrastructure. GPU compilation and full-image verification remain pending.

Parent is69c53d72a605e4ebd7d6530ab46e3ef2ca93f2c3; original base is
8cce978b73901eeac42a9e2fdc7e861a23a57890. The full independent v2 report is
archived unchanged with SHA256
0c4b3d206f12dfff33006cbcbd8c117297c406da698484e71d702bd4238a3570.
It rejects the parent for F2, while closing the exact original F1 example.

Both existing AABB helpers widen inputs before subtraction and divide directly
in double, avoiding the old separately rounded float reciprocal. Root/child
box validity uses physical max_distance, as triangle validity already does.
Popped-node pruning uses the same float32 distance difference as the existing
hit comparator, avoiding a rounded best_t + epsilon cap. Shared shadow callers
retain their physical range. Triangle validity and all existing thresholds,
comparators, arrays, ABI and output types are unchanged.

Double arithmetic alone was insufficient: an author search of5000 half-ULP
axis distances with varied triangle edges found357 entries rounded above the
triangle's returned distance. The stored nonnegative box entry now rounds
toward zero when nearest float32 rounding increased it. CUDA uses nextafterf;
Warp decrements the positive IEEE representation through its existing wp.cast
bit reinterpretation builtin. This preserves the computed double lower bound
through stack quantization without an unexplained padding epsilon. Installed
Warp1.12 builtin declarations/native template were inspected read-only; actual
Warp JIT lowering remains untested. This is not a proof of watertightness or
all possible double-arithmetic error bounds.

The exact F2 oblique-ray fixture is retained across both source bodies, both
storage orders and one/separate leaves, with outside-tie and physical-cutoff
variants. The original F1 cases remain. An additional half-ULP traversal case
uses origin x=-3.725290298461914e-9, far x=0.1076514795422554,
e1.y=3.5386428833007812 and e2.z=7.4439778327941895. The deterministic5000-case
stress test per lane checks actual box entry <= true axis distance and actual
returned triangle distance. It exercises both round-to-even parities; it does
not claim exhaustive arbitrary-direction coverage. All captured-edge and
strict primary/shadow predicate checks remain.

Fresh author command in this candidate, existing local Python3.14 and g++:
`C:/Python314/python.exe -B -m pytest -q -p no:cacheprovider tests/unit/optics/test_native_intersection_precision.py tests/unit/optics/test_warp_native_kernel_source.py tests/unit/optics/test_native_warmup_support.py`
completed115 passed in25.07s, tool session26154 terminal exit0. This includes
102 numerical checks and13 source/support checks; no host compiler skip.
git diff --check passes. The harness executes source bodies, not GPU kernels.
There was no native build or GPU launch for this revision.

The previous work interval had a read-only SSH grep stuck after Windows quote
loss. Only its verified local SSH PID40296 (parent44116, matching command)
was stopped; its tool session77296 ended exit1. A subsequent scoped remote
process query found no matching grep. No GPU process or other user's job was
stopped. All current author test handles are terminal.

Next: freeze and obtain a fresh independent code review, then use existing
fixed deployment tools to build the changed CUDA source and run supervised
Warp/CUDA Go2 geometry readback at1920 and960. Original artifacts remain
unchanged. Full arrays, original six RGB/two normal-only pixels, CPU checks
on actual rays, affected profiles/shadows and runtime lifecycle still require
verification. Double arithmetic may affect speed; no performance number or
acceptance is inferred from host tests. Whole C4 integration, systematic
refactor/whole review and later activation/result/backend gates remain open
work or closed gates as specified by the parent design.
