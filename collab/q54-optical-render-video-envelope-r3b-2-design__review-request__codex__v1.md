# Q54 Render Video Envelope R3b.2 Design Discussion

Date: 2026-05-13
Author: Codex
Status: review-request

## Context

R3b.1 landed the conservative bridge:

```text
RenderedVideoFrame.result: object                 # existing required field
RenderedVideoFrame.render: RenderResult | None    # new optional runtime link
```

The single production `_render_video_frame(...)` path fills both:

```text
result = render_result.compute
render = render_result
```

Tests still construct `RenderedVideoFrame(result=object(), ...)` cheaply. This
keeps the lab delivery tests small while production frames now carry the runtime
`RenderResult`.

R3b.2 should decide the next migration step. It should not start with code until
we agree which field moves first.

## Current Read Points

`RenderedVideoFrame.result` is read only by delivery submit paths:

```text
VideoDeliveryFacade._deliver_sync:
  _pack_for_delivery(rendered.result)

VideoDeliveryFacade._submit_async:
  _pack_for_delivery(rendered.result)
```

`RenderedVideoFrame.render_execute_ms` is read by:

```text
VideoDeliveryFacade._complete_job:
  overlap ratio input

VideoFrameTimingRowBuilder.build_row:
  frame_timing.csv render_execute_ms

VideoFrameTimingRowBuilder.progress_line:
  stdout render=...
```

`RenderedVideoFrame.render_profile_row` is read by:

```text
VideoFrameTimingRowBuilder.build_row:
  expands render_* profile columns

VideoFrameTimingRowBuilder.progress_line:
  profile text
```

`RenderedVideoFrame.render` is currently populated by production code but read
only by tests.

## Field Ownership

The target ownership is:

```text
RenderResult:
  compute result
  render timing summary
  render timing flat mapping

RenderedVideoFrame:
  lab video frame index
  camera
  camera/ray preparation timing
  geometry mode
  traversal diagnostics request bit
  compatibility views needed by delivery and CSV
```

`RenderResult` should not grow:

```text
camera
frame_index
geometry_mode
CSV/progress state
```

Those belong to the video envelope or frame context, not the compute output.

## Migration Options

### Option A — Move `result` First

Change `RenderedVideoFrame.result` from stored field to compatibility property:

```python
@property
def result(self):
    if self.render is not None:
        return self.render.compute
    return self._result
```

This would require a stored fallback, likely `_result`, because many tests still
construct fake frames without `RenderResult`.

Pros:

- delivery starts reading the compute result from the runtime object in
  production;
- simple semantic mapping.

Cons:

- dataclass field/property migration is awkward;
- requires renaming `result` field to `_result` or using `InitVar`;
- touches every test construction or adds compatibility plumbing;
- not much behavioral benefit yet, because production already stores both
  `result` and `render.compute` from the same source.

Risk: moderate churn for low immediate value.

### Option B — Move `render_execute_ms` First

Keep the stored field for now, but introduce an explicit method/property that
can read from `render` when present:

```python
def render_execute_ms_from_render_or_fallback(self) -> float:
    if self.render is not None and self.render.render_timing is not None:
        return self.render.render_timing.execute_ms
    if self.render is not None:
        return float(self.render.timing.get("render_execute_ms", float("nan")))
    return self.render_execute_ms
```

Then update internal call sites one at a time.

Pros:

- exercises the new `render` relationship without changing constructor shape;
- honors the explicit `None` vs `NaN` fallback rule from R3b.1;
- lower churn than moving `result`;
- render timing is exactly where the `RenderResult` link is most meaningful.

Cons:

- cannot use the same field name as a property while the dataclass field exists;
- would add a new helper name unless we rename the stored field;
- CSV still relies on a compatibility path.

Risk: low if implemented as a helper method first.

### Option C — Move `render_profile_row` First

Derive `render_profile_row` from `RenderResult.timing`.

Pros:

- removes one flattened duplicate from the envelope.

Cons:

- profile row shape is CSV-sensitive;
- `_render_profile_row_from_timing(...)` lives in `go2_backend.py`, not
  `delivery.py`;
- deriving this inside delivery would either duplicate key knowledge or create
  a new dependency direction;
- easy to cause CSV drift.

Risk: higher than the value for R3b.2.

### Option D — Stop After R3b.1 For Now

Keep the optional `render` link as documentation and future leverage, but do no
accessor migration yet.

Pros:

- no churn;
- production path already carries `RenderResult`;
- lets future CSV/timing cleanup decide the right shape.

Cons:

- `render` is not yet used by production delivery/row code;
- duplicate fields remain.

Risk: lowest.

## Recommended R3b.2 Shape

Do not property-ize `result` yet.

Do not derive `render_profile_row` yet.

If we do a small implementation step, make it a named helper for render timing:

```python
def render_execute_ms_value(self) -> float:
    if self.render is not None and self.render.render_timing is not None:
        return self.render.render_timing.execute_ms
    if self.render is not None:
        return float(self.render.timing.get("render_execute_ms", float("nan")))
    return self.render_execute_ms
```

Then update these call sites only:

```text
VideoDeliveryFacade._complete_job
VideoFrameTimingRowBuilder.build_row
VideoFrameTimingRowBuilder.progress_line
```

Keep the stored `render_execute_ms` field so tests and fallback construction
remain simple.

This gives production code a real read path through `RenderResult` while keeping
constructor compatibility and CSV behavior stable.

## Why Not Use A Property Named `render_execute_ms` Yet?

Because `render_execute_ms` is currently a dataclass field. Turning it into a
property safely requires a two-step migration:

```text
1. introduce helper method and update internal reads
2. later rename stored field to _render_execute_ms and add property
```

Doing both in one patch risks unnecessary test churn and hidden constructor
breakage.

## Test Strategy

If R3b.2 implements the helper:

- add a direct test where `render.render_timing.execute_ms` wins over stored
  fallback;
- add a direct test where `render_timing is None` falls back to
  `render.timing["render_execute_ms"]`;
- preserve a test where `render is None` uses stored `render_execute_ms`;
- keep row-builder tests unchanged except where they assert the helper behavior.

The existing full lab tests should remain:

```text
tests/unit/optics/test_render_api.py
tests/unit/optics/test_optical_pipeline_lab.py
```

## Open Questions

1. Should R3b.2 implement the `render_execute_ms_value()` helper now, or stop
   after R3b.1 and wait for a broader timing cleanup?

2. Is `render_execute_ms_value()` the right name, or should it be
   `render_execute_ms_for_csv()` / `render_execute_ms_observed()`?

3. Should the helper live on `RenderedVideoFrame`, or should
   `VideoFrameTimingRowBuilder` own the fallback logic?

4. Should `VideoDeliveryFacade._complete_job` use the same helper for overlap
   ratio, or should overlap continue using the stored field until a longer-run
   overlap benchmark exists?

## Codex Recommendation

```text
Q1: implement the helper now; it is small and validates the render link
Q2: use render_execute_ms_value()
Q3: put it on RenderedVideoFrame, because both delivery and row builder need it
Q4: use the helper in _complete_job too, keeping overlap semantics aligned
```

Do not migrate `result` or `render_profile_row` in R3b.2.
