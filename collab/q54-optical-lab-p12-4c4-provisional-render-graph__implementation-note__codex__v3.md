# Repair Warp graph import closure and profile binding

Initiative: Q54 / P12.4 / C4 render integration
Stage: native bring-up repair
Author: codex
Version: v3
Date: 2026-10-10
Status: in_review
Related Files: provisional-render-graph code-review v2; render-graph-native-support code-review v1;
  runtime-integration design-request v3; native_setup.py; native_bootstrap.py
Owner Summary: The first real Warp graph preparation rejected a missing repository
  dependency. Admit the existing output-profile module and its scene dependency
  only in the Warp graph phases, and pass the existing normalized profile enum.

Base: f1ff683dd8ef01064b65e2b2f796486b17185622. Production before repair is
8315b00, with approved positive native support at50143c6. No timing, optical
execution, warmup authority or later product gate is added by this correction.

## Actual native failure

The eight-case positive graph campaign at exact clean50143c6 stopped at its first
case, Warp Go2 960x640, in fixed_render_graph. The worker raised SourceBootstrapError
with message "repository import outside CPU core or reload". The full raw trace
ends after render factory public loading and frame-view/context retention, before
any command-recording scope. The supervisor observed worker74, 12 normal completed
guards, 13 arms, no deadline/kill; the failed graph guard12 quarantined the complete
retained graph without emergency synchronization. Parent3328284, supervisor3328515
and worker3328520 are absent. The later seven campaign cases did not run.

Raw directory: tileops-dev:/data2/shared/ang.gao/c4-render-graph-author-v1.
Archive: c4-render-graph-failure-evidence-v1.tar.gz, remote /data2/shared/ang.gao
and local ../c4_assets; SHA256243685222f6487ace68a7786e0a6f6dffc1a9d7458bb68c88269f11c28a797af,
verified after transfer. The adjacent JSON records exact revision, original error,
worker/supervisor summaries and PID absence. The archive includes all raw case
files and the original batch/archive scripts. No native graph success is claimed.

## Diagnosis and repair

The original _WarpRenderBinding._validate imports OpticalOutputProfile from
optics.execution. That module was already frozen in the common source catalog,
but absent from the Warp phase admission map. Its direct scene dependency was
also absent. Static call-flow inspection locates this first deferred import at
the recorded failure boundary; the raw generic exception itself does not name
the requested module. The patch adds exactly optics.execution and optics.scene
for preparing_fixed_render_graph/fixed_render_graph_ready in Warp. Compilation,
storage and earlier phases keep their old closure, and CUDA modules remain closed
on Warp. Existing source verification/captured loaders remain unchanged.

Inspection also found the next real API mismatch masked by the integrated CPU
factory double: native graph passed a string profile, while the original Warp
binding requires the exact OpticalOutputProfile enum. It now resolves the profile
once through the existing normalize_output_profile and passes that value to every
frame. The graph's descriptive profile remains the frozen string. No second enum,
renderer variant or normalization implementation is introduced. This second
failure was found statically; the native run stopped before reaching it.

Twelve tests call the actual captured finder with retained source bytes for those
dependencies and a foreign CUDA module across early/graph phases; original session
phase logic and loader resolution run, with topology/controller metadata doubles.
The integrated Warp factory double now enforces the actual enum requirement in
all profile/shadow/diagnostic combinations. Source closure assertions cover both
lanes. Focused render/storage/support suites:131 passed in27.40s, session63391
exited0. Native source-policy/source suites passed206 with4 host/asset skips in
43.91s (session86853, exit0). Source regeneration retains all84
paths and the unchanged policy, updating exact bootstrap/class/file pins.
Catalog SHA2565fe8701d31b6953444e9c666620a5217ead719c744ad566a08d9d0b22c358504.

Fresh independent review and a new source-frozen native attempt are required.
CPU tests do not establish actual recording. Graph failure/hang acceptance,
real rendering, ten observed warmups, final readiness/graph, R1, whole-C4
validation/refactor/review and subsequent activation remain incomplete.
