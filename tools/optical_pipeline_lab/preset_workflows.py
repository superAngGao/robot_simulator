"""Public preset workflows for Optical Pipeline Lab.

P11 user-facing API for running reviewed lab presets.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .preset_products import ProductSelection, resolve_lab_product_specs
from .preset_runtime import create_runtime_for_lab_preset
from .presets import get_preset
from .product_specs import ProductBuildContext
from .product_workflow import PhysicsOwnedProductWorkflow, PhysicsProductRunResult
from .runner import ArtifactOutput


def run_optical_lab_preset(
    preset: str,
    *,
    frames: int,
    products: Iterable[ProductSelection],
    out: Path,
    device: str | None = None,
    **runtime_kwargs: object,
) -> PhysicsProductRunResult:
    """Run a reviewed Optical Pipeline Lab preset with specified products.

    This is the primary P11 public workflow for physics-owned presets.

    Args:
        preset: Preset identifier (e.g., "physics_body_triangle_video_smoke").
        frames: Number of simulation frames to run.
        products: Product selections — strings like "video"/"debug" or ProductInput instances.
        out: Output directory root for artifacts.
        device: Optional Warp device selection ("cuda:0", "cpu"). If None, uses runtime default.
        **runtime_kwargs: Additional runtime factory arguments.

    Returns:
        PhysicsProductRunResult with frame results, begin/end outputs, and artifacts.

    Raises:
        NotImplementedError: If preset has no registered runtime factory.
        ValueError: If product selection is invalid for the preset.

    Example:
        >>> result = run_optical_lab_preset(
        ...     preset="physics_body_triangle_video_smoke",
        ...     frames=120,
        ...     products=("video", "debug"),
        ...     out=Path("runs/examples/my_run"),
        ... )
    """

    preset_str = str(preset)
    frame_count = int(frames)
    out_path = Path(out)
    products_tuple = tuple(products)

    # Create runtime (workflow owns cleanup)
    runtime = create_runtime_for_lab_preset(
        preset_str,
        device=device,
        **runtime_kwargs,
    )

    # Resolve products
    product_specs = resolve_lab_product_specs(
        preset=preset_str,
        products=products_tuple,
    )

    # Configure output
    output = ArtifactOutput(out=out_path, frames=frame_count)

    # Get preset config for product build context
    config = get_preset(preset_str)

    # Build frame products
    context = ProductBuildContext(runtime=runtime, config=config, output=output)
    frame_products = tuple(spec.build(context) for spec in product_specs)

    # Write scenario config (matches P10 behavior)
    from .runner import write_scenario_config

    output.root.mkdir(parents=True, exist_ok=True)
    write_scenario_config(output.root / "scenario_config.json", config, output)

    # Run workflow with automatic cleanup
    with PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=frame_products,
        output=output,
        owns_runtime=True,
    ) as workflow:
        return workflow.run(frames=frame_count)
