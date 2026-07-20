Initiative: q54-physics-owned-multi-product-runtime-p11
Stage: review-request
Author: Codex
Version: v1
Date: 2026-07-20
Status: ready_for_review
Related Commits: 6238c24, 35c8b27
Related Files: tools/optical_pipeline_lab/preset_workflows.py, tools/optical_pipeline_lab/__init__.py, tests/unit/optics/test_optical_pipeline_lab.py, MANIFEST.md, collab/q54-physics-owned-multi-product-runtime-p11-4__implementation-note__codex__v1.md
Owner Summary: Requesting review for the corrected P11.4 public preset workflow API. The prior P11.4 implementation was reverted because it duplicated P10 workflow semantics. The new implementation keeps P11 thin and delegates execution/materialization/cleanup to P10 `run_optical_lab_products(..., owns_runtime=True)`.

# Q54 P11.4 Public Preset Workflow Review Request

## Review Scope

Please review these two commits together:

```text
6238c24 Revert "feat: add P11.4 public preset workflow API"
35c8b27 feat: add thin preset workflow API
```

The revert is intentional. The first P11.4 implementation duplicated P10
workflow logic inside `preset_workflows.py`, which created two real problems:

```text
1. concrete FrameProduct pass-through could fail because P11 called .build(...)
   on every resolved product;
2. runtime cleanup did not cover errors before entering PhysicsOwnedProductWorkflow.
```

The replacement implementation is deliberately thinner.

## Intended Boundary

P11.4 should be a public preset facade:

```text
run_optical_lab_preset(...)
  -> resolve_lab_product_specs(...)
  -> create_runtime_for_lab_preset(...)
  -> run_optical_lab_products(..., owns_runtime=True)
```

P11.4 should not:

- build `ProductBuildContext` directly;
- call `ProductSpec.build(...)` directly;
- instantiate `PhysicsOwnedProductWorkflow` directly;
- write `scenario_config.json` directly;
- import or reference `go2_backend.py`.

Those responsibilities stay in P10 product workflow code.

## Changed Files

```text
tools/optical_pipeline_lab/preset_workflows.py
tools/optical_pipeline_lab/__init__.py
tests/unit/optics/test_optical_pipeline_lab.py
MANIFEST.md
collab/q54-physics-owned-multi-product-runtime-p11-4__implementation-note__codex__v1.md
```

## API Shape

New public entry:

```python
def run_optical_lab_preset(
    preset: str,
    *,
    frames: int,
    products: Iterable[ProductSelection],
    out: Path | None = None,
    output: ArtifactOutput | None = None,
    device: str | None = None,
    runtime_kwargs: Mapping[str, object] | None = None,
    **extra_runtime_kwargs: object,
) -> PhysicsProductRunResult:
    ...
```

Usage:

```python
result = run_optical_lab_preset(
    "physics_body_triangle_video_smoke",
    frames=120,
    products=("video", "debug"),
    out=Path("runs/p11/body_triangle"),
    device="cuda:0",
)
```

`runtime_kwargs={...}` and extra keyword arguments are both accepted. Extra
keyword arguments override duplicate keys from `runtime_kwargs`.

## Key Tests

Please focus on these tests:

```text
test_run_optical_lab_preset_delegates_to_p10_workflow
test_run_optical_lab_preset_accepts_frame_product_instances
test_run_optical_lab_preset_rejects_products_before_creating_runtime
test_run_optical_lab_preset_closes_runtime_on_p10_setup_error
test_run_optical_lab_preset_does_not_import_go2_backend
test_optical_pipeline_lab_exports_p9_product_contracts
```

Coverage intent:

- success path delegates to P10 and closes runtime;
- concrete `FrameProduct` pass-through is preserved;
- invalid product strings fail before runtime construction;
- P10 setup failures close the owned runtime;
- P11 public workflow does not load `go2_backend`;
- top-level lazy export is wired.

## Verification

Focused lint/format:

```bash
ruff check tools/optical_pipeline_lab/preset_workflows.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
ruff format --check tools/optical_pipeline_lab/preset_workflows.py \
  tools/optical_pipeline_lab/__init__.py \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
All checks passed
3 files already formatted
```

Focused unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py \
  -k "run_optical_lab_preset or optical_pipeline_lab_exports_p9_product_contracts"
```

Result:

```text
6 passed, 169 deselected
```

Full optical lab unit coverage:

```bash
conda run -n robot_sim env PYTHONPATH=. python -m pytest -q \
  tests/unit/optics/test_optical_pipeline_lab.py
```

Result:

```text
175 passed
```

## Review Questions

1. Is the cleanup ownership now correct across product-resolution errors and
   P10 setup/runtime errors?

2. Is accepting both `runtime_kwargs={...}` and extra keyword runtime options
   acceptable, or should P11 expose only one style before examples are written?

3. Should `run_optical_lab_preset(...)` require `out` in P11.4, or is the
   current `out | output` compatibility appropriate because it delegates to P10?

4. Are the no-`go2_backend` tests strong enough for P11.4, or should the review
   require a static import guard over `preset_workflows.py` source text too?

## Codex Recommendation

Approve if the boundary is acceptable.

The main thing to verify is that P11 remains a user-facing convenience layer,
not a second workflow implementation. The new code should be judged by whether
it stays thin and reuses P10 correctly.
