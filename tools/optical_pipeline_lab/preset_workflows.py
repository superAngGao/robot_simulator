"""Public preset workflow entry points for Optical Pipeline Lab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .preset_products import ProductSelection, resolve_lab_product_specs
from .preset_runtime import create_runtime_for_lab_workflow
from .product_workflow import ProductRunResult, run_optical_lab_products
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
) -> ProductRunResult:
    """Run a reviewed Optical Pipeline Lab preset with selected products.

    Extra runtime keyword arguments override duplicate keys from
    ``runtime_kwargs``.
    """

    preset_name = str(preset)
    frame_count = int(frames)
    product_inputs = resolve_lab_product_specs(
        preset=preset_name,
        products=tuple(products),
    )
    artifact_output = _resolve_preset_output(output=output, out=out, frames=frame_count)
    runtime_options = dict(runtime_kwargs or {})
    runtime_options.update(extra_runtime_kwargs)
    runtime = create_runtime_for_lab_workflow(
        preset_name,
        output=artifact_output,
        device=device,
        runtime_kwargs=runtime_options,
    )
    return run_optical_lab_products(
        preset=preset_name,
        runtime=runtime,
        products=product_inputs,
        output=artifact_output,
        frames=None,
        owns_runtime=True,
    )


def _resolve_preset_output(
    *,
    output: ArtifactOutput | None,
    out: Path | None,
    frames: int,
) -> ArtifactOutput:
    if output is None:
        if out is None:
            raise TypeError("run_optical_lab_preset requires output or out")
        return ArtifactOutput(root=out, frames=frames)
    if out is not None and Path(out) != output.root:
        raise ValueError("run_optical_lab_preset received conflicting output root and out paths")
    frame_count = int(frames)
    if output._frames_explicit and output.frames != frame_count:
        raise ValueError("ArtifactOutput.frames conflicts with workflow run frames")
    if not output._frames_explicit:
        return output.replace_frames(frame_count)
    return output
