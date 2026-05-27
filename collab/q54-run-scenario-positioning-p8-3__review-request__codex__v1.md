# Q54 `run_scenario(...)` Positioning / P8.3 Decision Note

Author: Codex
Date: 2026-05-26
Status: decision accepted; runner validation/naming follow-ups implemented; remaining follow-ups identified

## Summary

This note records the P8.3 decision for
`tools.optical_pipeline_lab.runner.run_scenario(...)`:

```text
run_scenario(...) is a developer/lab scenario executor.
It is not the user-facing simulation runtime.
```

Therefore `run_scenario(...)` should remain guarded for
`FrameSourceKind.PHYSICS_RUNTIME` after P8. This is not just a temporary scope
choice. It is a structural dependency-injection boundary:

```text
run_scenario(config, options)
  -> frozen value objects only

physics runtime execution
  -> live engine/registry/step function/runtime owner
```

`run_scenario(...)` should not silently construct a physics engine, choose
reset/action policy, and run a simulation loop.

## Current Role

Today `run_scenario(...)` belongs to the Optical Pipeline Lab. Its job is to
execute explicitly implemented optical/render lab presets:

- validate `OpticalLabScenarioConfig`;
- apply lab run options;
- write `scenario_config.json`;
- route static/synthetic optical benchmark paths;
- preserve timing/video/readback/delivery metadata;
- reject reserved or unsupported modes before side effects.

It is closer to:

```text
developer benchmark / validation harness
```

than to:

```text
user-facing robot simulation app
```

## What P7/P8 Already Proved

P7/P8 established the physics/render chain without changing
`run_scenario(...)`:

```text
PhysicsLabScenarioRuntime
  -> step_frame(i)
  -> run_physics_stepped_video_scenario(...)
  -> PhysicsFrameContextProvider
  -> OpticalLabRenderFrameContext
  -> render/video/delivery/timing
```

This proves that physics-owned time can drive optical rendering through explicit
lab helpers. It does not imply that the generic lab runner should own physics
engine lifecycle.

## Why `run_scenario(...)` Should Not Own Physics Runtime

The strongest reason is API shape. `run_scenario(config, options)` receives two
pure value objects. That works for presets whose complete execution description
is encoded by config/options:

```text
STATIC_ASSET_BUILDER      -> data source described by config, runner owns loop
SYNTHETIC_FRAME_SEQUENCE  -> data source described by config, runner owns loop
```

`PHYSICS_RUNTIME` is different. Its execution requires live dependencies:

```text
engine
registry
step_fn / runtime.step_frame(i)
runtime lifecycle owner
```

Those dependencies cannot be represented honestly by
`OpticalLabScenarioConfig`. Treating `PHYSICS_RUNTIME` as just another
`run_scenario(...)` preset would either hide construction behind a lab runner or
inject live state through a value-object API. Both would erase the ownership
boundary P8 just clarified.

If `run_scenario(...)` starts constructing `GpuEngine`, it would inherit several
responsibilities that do not belong to an optical lab scenario runner:

- physics scene construction;
- engine reset policy;
- action/control source;
- episode or rollout lifecycle;
- cleanup policy;
- future RL observation products;
- product selection across video, observations, debug readback, etc.

That would blur the boundary we just repaired:

```text
physics owns time
render consumes frames
video/delivery consume render products
workflow owns ordering
```

`run_scenario(...)` should not become the place where all of those ownership
questions are answered implicitly.

So the guard is intended to be permanent for this API, not a placeholder waiting
for the physics smoke path to become more mature.

## Current Design Debts Exposed By This Decision

Claude's review surfaced two related issues that should be tracked separately
from the P8.3 decision.

### `validate_implemented()` mixes two meanings

Today `OpticalLabScenarioConfig.validate_implemented()` appears to accept the
implemented physics runtime smoke config, but `run_scenario(...)` still raises
`NotImplementedError` for `FrameSourceKind.PHYSICS_RUNTIME`.

That creates a double-gating ambiguity:

```text
validate_implemented()
  -> "this config describes a valid implemented lab mode"

run_scenario(...)
  -> "this runner can execute this config"
```

Those are not the same predicate. We should clarify the semantics by separating:

```text
config structural/preset validity
runner executability
```

Preferred direction:

```text
Keep physics smoke valid as an implemented lab preset.
Add/rename a runner-level predicate for run_scenario executability.
```

In other words, do not make physics smoke "unimplemented" at the config layer
just because this particular runner should not execute it. Instead, introduce
something like:

```python
can_run_scenario(config) -> bool
validate_run_scenario_supported(config) -> None
```

or rename the existing validation stack so the split is explicit. The P8.3
guard remains correct, but the validation naming/contract should stop implying
that every implemented config is executable by `run_scenario(...)`.

### `FrameSourceKind` combines frame source and clock ownership

`FrameSourceKind` currently mixes two orthogonal concerns:

```text
STATIC_ASSET_BUILDER      -> static asset data, runner-owned loop
SYNTHETIC_FRAME_SEQUENCE  -> scripted frame data, runner-owned loop
PHYSICS_RUNTIME           -> physics data, externally owned physics clock
```

The first two variants differ mainly in frame data source. The third also
changes who owns time. That is why it does not fit the same runner API.

This does not need to be fixed inside P8.3, but it should be captured as
configuration design debt before we build a user-facing runtime or a
multi-product runtime layer.

## Note On The Physics Video Wrappers

`run_physics_stepped_video_scenario(...)` is currently a thin semantic wrapper,
not a distinct behavior layer. It adapts a "step physics to frame i" callback
into the "published frame for frame i" callback expected by
`run_physics_video_scenario(...)`.

The wrapper is worth keeping only while it documents a real architectural
distinction:

```text
run_physics_video_scenario(...)
  -> consume an externally supplied published-frame callback/replay source

run_physics_stepped_video_scenario(...)
  -> live-step physics as the frame source and clock owner
```

If the two paths later diverge, the split has a clear reason to exist. Examples:

- stepped execution needs warmup or pre-roll;
- stepped execution owns additional runtime lifecycle hooks;
- replay supports random access or cached published frames.

If the two paths remain behaviorally identical after more call sites exist, the
wrapper should be merged back into the shared callback API. It is documentation
intent today, not a permanent abstraction by default.

## Runner Boundary

Keep:

```text
run_scenario(...) stays guarded for FrameSourceKind.PHYSICS_RUNTIME.
```

Document the guard as a structural runtime-ownership boundary:

```text
value-object lab runner APIs do not own live physics runtime dependencies
```

Use explicit physics entries for lab/integration callers:

```python
run_physics_video_scenario(...)
run_physics_stepped_video_scenario(...)
create_physics_body_triangle_lab_runtime(...)
```

Do not add automatic physics engine construction to `run_scenario(...)`.

## User-Facing Runtime Should Be A Higher Layer

If the project needs a user-facing simulation command, it should be a separate
concept above the Optical Pipeline Lab, for example:

```text
SimulationRuntime
EnvironmentRuntime
run_simulation(...)
run_environment(...)
```

That higher layer can own:

- engine construction;
- reset/episode lifecycle;
- action or policy input;
- observation products;
- optional render/video/debug products;
- cleanup.

The optical lab can then plug in as one product/consumer path rather than
becoming the application runtime itself.

This higher layer should not be introduced before product boundaries are clear.
Video, future RL observations, and debug readback have different data flows and
ownership needs. A premature high-level runtime risks becoming a god object that
owns every product path without typed interfaces.

## Possible Future CLI

If we want a demo command soon, it should be explicit and lab-only, e.g.:

```bash
python -m tools.optical_pipeline_lab physics-smoke ...
```

or:

```bash
python -m tools.optical_pipeline_lab run-physics-smoke ...
```

But this should not be the generic `run` command automatically accepting
physics runtime presets. The name should communicate that it is a lab smoke/demo
around the synthetic body triangle runtime owner.

Even this explicit CLI should be deferred unless we actually need it, because it
still has to answer:

- how scripted heights/actions are configured;
- whether video is required or optional;
- how reset is represented;
- whether the command is a demo, benchmark, or future user workflow.

## P8.3 Decision

P8.3 is closed with this decision:

```text
run_scenario(...) remains a developer/lab scenario executor.
Physics runtime remains explicit.
User-facing simulation/runtime entry is deferred to a higher-level design.
```

Then move next architecture work toward multi-product runtime design:

```text
one physics-owned frame
  -> video product
  -> future observation product
  -> debug product
```

This is the point where future RL and sensor-loop ownership should be discussed,
rather than pushing more responsibility into `run_scenario(...)`.

## Implementation Follow-up Status

1. Validation split: implemented.

   `tools.optical_pipeline_lab.runner` now exposes runner-level
   `can_run_scenario(config)` and `validate_run_scenario_supported(config)`
   predicates. Physics smoke remains valid as an implemented lab preset, while
   `run_scenario(...)` explicitly rejects it because this value-object runner
   cannot own live physics runtime dependencies.

   The implementation also distinguishes invalid config from valid-but-not-runner
   supported config: `can_run_scenario(config)` returns `False` for the
   runner-ownership boundary and for runner-specific reserved scene/camera paths,
   while allowing config validity errors to propagate. The runner now avoids
   double-running `validate_implemented()` by sharing a private run-options
   validator after scenario validation has been performed.

2. Package-root export: deferred.

   `can_run_scenario(...)` and `validate_run_scenario_supported(...)` are
   available from `tools.optical_pipeline_lab.runner`, but are not added to
   `tools.optical_pipeline_lab.__init__` yet. This keeps them as explicit runner
   APIs until a CLI or workflow layer needs a stable package-root import.

3. Naming cleanup: implemented as contract documentation, not API rename.

   `validate_implemented()` now documents that it means lab-wide support by at
   least one path, not `run_scenario(...)` executability. The runner-level
   predicates carry the ordinary runner-executability meaning. This avoids a
   broad method rename while making the split explicit at the API boundary.

4. `FrameSourceKind` split: defer to P9 design before code changes.

   Current `frame_source` values are part of presets, validation gates, timing
   CSV defaults, GPU tests, and serialized `scenario_config.json` output. A
   direct enum split would therefore be a metadata/schema migration, not a
   local cleanup. The next design step should introduce a separate
   clock/loop-ownership concept, for example:

   ```text
   frame_source: static_asset | synthetic_sequence | physics_published_frame
   clock_owner: runner | external_physics_runtime
   ```

   Only after that contract is clear should the code migrate CSV/report fields
   and preset serialization.

5. If a CLI demo is needed, should it be a separate explicit lab subcommand
   rather than generic `run` accepting physics presets?

6. Is the next architecture topic multi-product runtime ownership
   (video/observation/debug) rather than CLI exposure?
