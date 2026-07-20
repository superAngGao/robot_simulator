"""Public preset workflow entry points for Optical Pipeline Lab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .preset_products import ProductSelection, resolve_lab_product_specs
from .preset_runtime import create_runtime_for_lab_preset
from .product_workflow import PhysicsProductRunResult, run_optical_lab_products
from .runner import ArtifactOutput


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
    """Run a reviewed Optical Pipeline Lab preset with selected products."""

    preset_name = str(preset)
    frame_count = int(frames)
    product_inputs = resolve_lab_product_specs(
        preset=preset_name,
        products=tuple(products),
    )
    runtime_options = dict(runtime_kwargs or {})
    runtime_options.update(extra_runtime_kwargs)
    runtime = create_runtime_for_lab_preset(
        preset_name,
        device=device,
        **runtime_options,
    )
    return run_optical_lab_products(
        preset=preset_name,
        runtime=runtime,
        products=product_inputs,
        output=output,
        out=out,
        frames=frame_count,
        owns_runtime=True,
    )
