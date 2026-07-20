# Optical Lab Examples

These examples use the P11 preset workflow API:

```python
run_optical_lab_preset(...)
```

The examples are intentionally written like user code. They should not import
or call legacy `go2_backend.py` paths.

## Physics Body Triangle Video + Debug

```bash
python examples/optical_lab/physics_body_triangle_video_debug.py --dry-run
```

Runs the reviewed physics-owned smoke preset with video and debug products.
Without `--dry-run`, this creates a live physics runtime and may require the
configured Warp/GPU environment.

## Physics Body Triangle Observation

```bash
python examples/optical_lab/physics_body_triangle_observation.py --dry-run
```

Shows how observation products stay explicit. The example builds an
`ObservationProductSpec` with schema, actuated indices, and contact body names
instead of using a product string like `"observation"`.

## Go2 Static Preset

No Go2 P11 example is included yet. Go2/Menagerie is a static asset preset, and
it should get a P11 example only after it can run through the same public
workflow without a legacy `go2_backend.render_many_views(...)` bypass.
