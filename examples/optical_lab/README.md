# Optical Lab Examples

These examples use the P11 preset workflow API:

```python
run_optical_lab_preset(...)
```

The examples are intentionally written like user code. They use public P11
workflow APIs instead of deleted legacy Go2 backend paths.

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

```bash
python examples/optical_lab/go2_video_ordered_static.py --dry-run
```

The reviewed Go2 static preset now runs through the same public workflow:

```python
run_optical_lab_preset("go2_video_ordered_static", products=("video", "debug"), ...)
```

Without `--dry-run`, this loads the Menagerie Go2 static asset scene and may
require the configured Warp/GPU environment plus mesh import dependencies.
