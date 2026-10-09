# Connect provisional R0 rendering commands and events

Initiative: Q54 / P12.4 / C4 runtime integration
Stage: provisional render graph implementation
Author: Codex
Version: v1
Date: 2026-10-10
Status: in_review
Related Files: runtime-integration design-request codex v3 sections 5,5.1,6,8,9;
  remaining-setup gap-scan v1 G2; warp-module-preparation design-request v2;
  fixed-render-storage code-review v1; warp-module-preparation native-review v1
Owner Summary: Reuse the existing render builders to prepare ten exact provisional
  frames and initialized completion/diagnostic events, without running kernels.

Base15f64a96224a2ce27dabacf57c9aa6f2f1e288bc. The older native storage campaign
continues in its isolated3d6692b checkout and does not execute this delta.

SourceSession.prepare_fixed_render_graph is a private, single-use setup entry
following its own fixed storage. It validates mode/source/input/compiled owners,
uses the existing certifier to issue truthful R_setup with warmup0, and builds a
retained provisional graph under one original guarded setup segment. The literal
lane-specific source closure adds only two preparation phases. No product IPC,
PREPARE, launch, sample, lease, final readiness, measured result or timing gate
is opened. R1/final graphs and actual setup execution remain later composition.

Native resource preparation now shares the existing one-ticket scope mechanics.
The allocation helper delegates to that extracted scope; single-use input-context
registry, exact runtime/device/stream, active scope/original ticket checks,
per-operation sticky failure, pre-adopted partial-owner list and scope restoration
remain one implementation. This is not another lifecycle/authority registry.
Warp factory preparation retains its actual returned factory before public load;
recording retains operands and each returned command before device metadata or
adoption. Existing kernel definitions, command descriptor math and packed-command
validation remain unchanged. The public ordinary preparation route still uses
the same factory/recording implementation without a guarded context.

The provisional graph holds its exact compiled module set, R_setup, factory,
ten (parameter-record,bundle) pairs, one completion event and optional three
R0 diagnostic pairs. Each index0..9 uses the frozen canonical binary32 scalar
record. Warp frame-specific BVH metadata views share the original immutable
R0 allocations, with frame IDs matching the query. CUDA frame input views share
the same allocations and bind the same normalized scalars. Existing Warp packed
command and CUDA stage-evidence checks validate the actual parameter ABI.
Before each Warp recording batch and afterward, the actual compiled set is
revalidated. Every returned command must use the retained render ModuleExec and
Module. All provisional roots stay in the existing source-session guardian.

Each event is retained before metadata/adoption and recorded only after adoption.
Diagnostic events are absent when disabled. Terminal initialization is recorded
last, then an event-specific host wait runs under the same normal guard, covering
all earlier event initialization on the held stream. This is setup initialization,
not a completed optical transaction, final barrier, C4 lease or a warmup count.
Failures retain partial frames/commands/events and use the original failed guard;
no emergency retry or post-failure completion is introduced.

Author tests (Python3.14 -B, pytest -q -p no:cacheprovider): eight selected suites
native_render_preparation, native_output_allocation, prepared_fixed_render_inputs,
warp_render_command_contract, warp_render_buffer_contract, render_parameters,
cuda_artifact_runtime and warp_module_preparation:408 passed in33.19s, session18830
exited0. Two subsequently added foreign-event cases passed in0.52s. Scope source
pins/catalog regenerate all84 LF sources with unchanged policy and inventory;
only bootstrap/class and renderer whole-file pins change, not the packed-launch
snapshot function pin. Catalog SHA80459b9f68f9941c0cbe57ba379ccd116220e515e69890e5b9e3c5bc48f5766e.

Tests exercise actual production source-session entry, allocation/recording/event
helpers and lifecycle, actual CUDA stage constructors/ABI validation with array
metadata doubles, and original Warp factory prepare/seal methods with bounded
binding/constructor doubles. Full Warp integration uses a factory double here;
real Warp packed recording and native event behavior are not proven by CPU tests.
Coverage includes both diagnostics modes/profiles/shadows, repeated entry, lost
source/mode, partial frame/event failure, caught recording failure, factory-before-
load custody, every event create/adopt/record/wait failure and foreign return.
Existing allocation sticky-failure/original-ticket cases still pass.

Independent review, cold source/bootstrap regressions and actual captured native
recording/event validation remain required. No optical kernel has run through this
new graph. Ten actual warmups, setup execution authority, final readiness/origin,
new final factories/commands, full Warp timeline/arena, R1, C3/coverage/final
barrier, whole-C4 validation/refactor/review and later activation remain incomplete.
