# Bind the reviewed numerical source into the existing release catalog

Initiative: Q54 / P12.4 / C4 numerical correctness
Stage: bounded release-data follow-up, pending independent review
Author: codex
Version: v4
Date: 2026-10-10
Status: in_review
Related Files: intersection-precision code-review v3; native_source_policy.py; data/native_sources/v16.json
Owner Summary: Repair omitted release pins after successful CUDA compilation. No numerical code, source inventory, AST rule, runtime mechanism or gate changes. Native rendering remains pending.

Base a42dbe0154e7c97dcd23a1de2514636551079d74 has bounded numerical code approval
from fresh reviewer /root/intersection_precision_code_review_v3. Full report
SHA2563402d0b1491eae97eb7b65a12f4edfb9069707f9aaeb1612bc2c8b57ead15219
is archived on the main branch at76a0d79, after merge f9a0cf6. Candidate source
and remote native checkout remained a42dbe0 for compilation.

The existing fixed deployment tool successfully built all three CUDA recipes.
Each service exited normally with status0; its exact cgroup was absent.
Deployment: /data2/shared/ang.gao/c4-intersection-cuda-artifacts-v1,
manifest SHA2568497e8410d17addb6e6e4739465bc6e636afc775ca181541c039023c2b1fab6d.
Derived environment c4-intersection-cuda-environment-v1.json has SHA256
c027ece426cb83961efb1887d9d6f563b0aa4b837e65d3045d4e9171ea753d67.
Build/raw observation archive c4-intersection-cuda-build-evidence-v1.tar.gz
has independently matching local/remote SHA256
c90fc21deb24556f1021c32be5f74dbc9550e765e5d5bf6c2aaf29195a948c34.
Build success is not native optical acceptance.

The first existing-runner invocation then failed in CPU custody preflight,
before supervisor creation, with source digest mismatch. Tool returned1;
parent3621401 is absent. c4-intersection-geometry-author-v1 contains only its
GPU mapping, and is retained. The observation record explicitly is an author
record of tool output, not an invented worker artifact. No native render or
Warp JIT was reached. This exposed omitted packaging updates in a42dbe0.

This delta updates exactly the existing renderer whole-file pin in
native_source_policy.py, the two changed numerical files' catalog digests,
the policy file's digest and the resulting policy digest. Catalog schema,
lane inventories, AST rules, restricted imports/attributes and snapshot
function pin are unchanged. The restricted _render_launch_snapshot function
is byte-identical with SHA256
725c7f315fdba4ad6df78e31a149865fd97e26da3c95df24678dbadba0459199.
Renderer whole-file SHA256 is now
e19dfc74327111fa70ce7d06aa9a4379a9ce50d8932664a0a2dedef53895886f.
Catalog SHA25643b90be45c9149c8395cd644b74ff421efce8111eaae00b43763e24297effe86.
All literal source entries pass the unchanged validator using LF release bytes.

Updating only the two catalog entries first reproduced13 Warp failures and13
CUDA passes in test_native_source_custody (76.05s, tool40649 exit1), identifying
the additional stale whole-file pin. After this complete delta, fresh command
`C:/Python314/python.exe -B -m pytest -x -q -p no:cacheprovider tests/unit/optics/test_native_source_policy.py tests/unit/optics/test_native_sources.py tests/unit/optics/test_native_source_custody.py`
passes232 with4 platform/asset skips in162.89s, tool38044 exit0. These include
the original source-custody failures. No regression tests or bypasses were
added. Numerical code/tests remain byte-identical to the approved base.

Existing verify_deployment checks exact SOURCE_FILES and recipe bytes rather
than requiring the later catalog-only Git revision equal the recorded build
revision. None of its source files changes here; the a42dbe0 build provenance
must remain truthful. Reuse that deployment only after the existing verifier
passes against this final release; no rebuild, manifest rewriting or new
artifact policy is needed. Freeze/review this delta, run cold preflight in
both lanes, then retry in new exclusive outputs, preserving failed v1.

All author build/test/launch handles are terminal. Full native Go2 geometry,
RGB/CPU comparison, speed, integrated C4 and systematic refactor/whole review
remain outstanding. No later gate is opened.
