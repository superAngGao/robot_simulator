# Exercise provisional recording before connecting execution

Initiative: Q54 / P12.4 / C4 render integration
Stage: native driver support
Author: codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: provisional-render-graph code-review v2; runtime-integration design-request v3
Owner Summary: Extend the existing supervised driver with two positive graph
  preparation cases. No production change, new supervisor or execution shortcut.

Base: 8315b008bf54f294af5bf0c3035d4bc9cc0e009b, whose bounded code review is
approved. Full review v2 is archived unchanged with this delta. Original worktree
remained frozen during that review; subsequent support was developed in the
separate c4_render_execution_next worktree from the same base.

The two cases reuse source intake, native upload/validation, CUDA factories or
Warp compilation/reuse, and the existing fixed-storage observer. They then call
the production prepare_fixed_render_graph method and perform original terminal
disposal. The original guard budgets and process manager are unchanged. Successful
paths have 13 normal guards for Warp / 9 for CUDA, followed by one original
terminal synchronization and exit. These cases do not launch optical kernels.

The observer examines actual returned native command/event custody at the existing
adoption boundary, after production's retention and before its next leaf. It does
not add observer-owned native resources. Each final frame's actual compiled entry
identity and production parameter ABI are checked. Raw observations preserve ten
frame IDs, command/executable or CUDA entry identities, unique event identities,
native target1/ambient0 scope, diagnostic phases and truthful warmup0. The replay
also checks the actual worker lifecycle's exact event create/record/wait counts
and final wait position in the graph guard. No allocations or warmup launches
are allowed in that guard. Ordinary loader restoration before Warp rendering is
necessary because the earlier compilation observer accepts Module objects while
the original render factory calls the public loader with a Python module name.

Four CPU tests exercise both lanes and diagnostic settings through the real
SourceSession method/guard/recording/event helpers. CUDA stages are actual
production constructors with resource doubles; Warp uses a factory double plus
the actual guarded command-recording helper. Raw evidence tampering and missing
wait descriptors reject. These tests do not prove native packed Warp recording.
Focused support/render/storage/artifact/scope/driver suites: 173 passed, 13 skipped
in 32.64s (session74725, observed PID3404 parent324). Initial selection included a
nonexistent test path and ran no tests; the corrected command is the result above.
The final event-trace additions were separately rerun in the four-case suite.

Native positive execution is pending. This is bring-up support for the actual
render path, not full graph native acceptance: native failure/retention/hang
behavior, actual optical execution, warmup evidence, final readiness/graph,
R1/arena, complete C4 validation/refactor and fresh whole review remain required.
Product/sample/performance/result/baseline gates remain closed. These fixed driver
selectors are comparison-stage scaffolding and follow the existing final backend
selection/cleanup rules; no permanent second rendering path is added.
