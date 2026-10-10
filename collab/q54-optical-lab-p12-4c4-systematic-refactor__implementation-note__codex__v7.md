Initiative: Q54 / P12.4 / C4
Stage: R1 dynamic composition, work in progress
Author: Codex, implementation session
Version: v7
Date: 2026-10-10
Status: draft
Related Files: runtime-integration design-request/review v3 section6; systematic-refactor gap-review v1; implementation-note v6
Owner Summary: R1 work has begun by reusing existing scene/refit kernels and the renderer's guarded recording helper. This is an incomplete working change, not a new acceptance checkpoint or R1 approval.

The isolated c4_dynamic_graph branch starts at c3472f22f1d52bfb6c853bf95cef4040cf666316.
The M1 candidate in c4_setup_orchestration remains frozen for independent review.

WarpSceneUpdateCommandFactory.seal and WarpBvhRefitCommandFactory.seal now use
the existing _warp_command_recording core. Kernel operands, dispatch order and
evidence construction remain in their original factories. Each returned command
is retained before the next fallible operation. An internal optional context
connects the same construction body to the captured worker's original guard.
No second scene/refit executor or slot state machine is introduced.

Refit factory creation uses the existing native resource scope for its topology
ready-event dependency. A distinct slot BVH's ready-event dependency uses the
same scope before recording; registered captured calls check original ownership
and target. Legacy calls retain their signatures with an optional internal
context. Actual GPU verification of these newly scoped boundaries is pending.

CPU return-double tests exercise the actual scene/refit seal methods and actual
guard/custody code: second command failure, returned metadata/adoption failure,
first-command retention before the next leaf, original exception propagation,
scope restoration and quarantine without extra sync. Geometry/storage validators
are doubled in these tests, so they do not prove native geometry or ABI behavior.
Additional dependency tests cover wait-before-record, failed wait, foreign event
target and missing ownership. The twelve new checks pass in1.07s. The earlier
existing render-preparation/observer suites plus eight scene/refit probes passed
102 checks in36.18s. Native-command/device-scene contract checks also passed.
New test file Ruff and git diff --check pass.

Four pre-existing observer tests fail at the M1 candidate and its base because
their manually created SourceSession fixture lacks fixed_row. The independent
review reproduced that old failure. This working branch initializes fixed_row
to None in the fixture, matching its unadmitted-row setup; all four now pass.
Production does not gain a fallback for missing session fields.

All90 prospective LF source bodies pass the existing AST policy. Current draft
catalog SHA256:f41a8c69954859c7cfcaf73a9ee64da764f699a2b7e445ba31cf809ea1bb5db3.

Still required in the SAME coherent R1 implementation: guarded initial Warp
topology construction/upload using the existing median builder, independently
owned mutable slot world/bounds/render storage; CUDA existing world/LBVH rebuild
composition; exact replay/camera frame binding; shared five-phase setup/final
execution and C3/arena plans; ten observed warmups/final barrier; native and
numerical evidence and a fresh independent review. This partial change does not
open product PREPARE, measured submission, C2f or any performance/selection gate.
The full C4 goal and final systematic refactor review remain outstanding.
