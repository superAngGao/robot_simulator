"""Physics-owned product workflow helpers for Optical Pipeline Lab runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .frame_products import FrameProduct, FrameProductResult, MultiProductFrameRunner
from .runner import ArtifactOutput, validate_run, write_scenario_config
from .scenarios import (
    ClockOwnerKind,
    OpticalLabScenarioConfig,
    is_physics_published_frame_source,
)


@dataclass(frozen=True)
class PhysicsProductRunResult:
    """Typed result for one physics-owned product workflow run."""

    frame_results: tuple[tuple[FrameProductResult | None, ...], ...]
    begin_outputs: Mapping[str, object | None]
    end_outputs: Mapping[str, object | None]
    artifacts: Mapping[str, object] = field(default_factory=dict)

    @cached_property
    def product_results(self) -> Mapping[str, tuple[FrameProductResult, ...]]:
        """Return non-empty per-frame product results grouped by product name."""

        grouped: dict[str, list[FrameProductResult]] = {}
        for frame in self.frame_results:
            for result in frame:
                if result is None:
                    continue
                grouped.setdefault(result.product_name, []).append(result)
        return {name: tuple(values) for name, values in grouped.items()}


@dataclass
class PhysicsOwnedProductWorkflow:
    """Run ordered products on a physics-owned simulation tick stream.

    P10.1 keeps this workflow intentionally one-shot. Reusable workflows need
    explicit artifact overwrite/subdirectory and product reset semantics.
    """

    runtime: object
    products: tuple[FrameProduct, ...]
    output: ArtifactOutput | None = None
    owns_runtime: bool = False
    _has_run: bool = field(default=False, init=False)

    def run(self, *, frames: int) -> PhysicsProductRunResult:
        """Run products for a fixed number of physics-owned frames."""

        frame_count = int(frames)
        if frame_count < 0:
            raise ValueError("frames must be >= 0")
        if self._has_run:
            raise RuntimeError("PhysicsOwnedProductWorkflow has already run")
        if self.output is not None and self.output._frames_explicit and self.output.frames != frame_count:
            raise ValueError("ArtifactOutput.frames conflicts with workflow run frames")

        runner = MultiProductFrameRunner(products=self.products)
        self._has_run = True
        begin_outputs = runner.begin_run()
        frame_results: list[tuple[FrameProductResult | None, ...]] = []
        for frame_index in range(frame_count):
            tick = self.runtime.step_tick(frame_index)
            frame_results.append(runner.step(tick))
        end_outputs = runner.end_run()
        return PhysicsProductRunResult(
            frame_results=tuple(frame_results),
            begin_outputs=begin_outputs,
            end_outputs=end_outputs,
            artifacts=self._artifacts(),
        )

    def close(self) -> None:
        """Close the runtime only when this workflow explicitly owns it."""

        if self.owns_runtime:
            self.runtime.close()

    def __enter__(self) -> "PhysicsOwnedProductWorkflow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _artifacts(self) -> Mapping[str, object]:
        if self.output is None:
            return {}
        return {"root": self.output.root}


def run_physics_products(
    *,
    runtime: object,
    products: Iterable[FrameProduct],
    frames: int,
    output: ArtifactOutput | None = None,
    owns_runtime: bool = False,
) -> PhysicsProductRunResult:
    """Run product instances on an existing physics-owned runtime."""

    workflow = PhysicsOwnedProductWorkflow(
        runtime=runtime,
        products=tuple(products),
        output=output,
        owns_runtime=owns_runtime,
    )
    try:
        return workflow.run(frames=frames)
    finally:
        workflow.close()


def run_physics_product_scenario(
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
    *,
    runtime: object,
    products: Iterable[object],
    frames: int | None = None,
    owns_runtime: bool = False,
) -> PhysicsProductRunResult:
    """Run products or product specs for one physics-backed lab scenario."""

    try:
        validate_physics_product_scenario(config, output)
        frame_count = _resolve_workflow_frame_count(output, frames)
        product_inputs = _validate_product_inputs(products)
        output.root.mkdir(parents=True, exist_ok=True)
        write_scenario_config(output.root / "scenario_config.json", config, output)
        concrete_products = _build_products_for_scenario(
            config=config,
            output=output,
            runtime=runtime,
            products=product_inputs,
        )
        return run_physics_products(
            runtime=runtime,
            products=concrete_products,
            frames=frame_count,
            output=output,
            owns_runtime=owns_runtime,
        )
    except Exception:
        if owns_runtime:
            runtime.close()
        raise


def run_physics_product_preset(
    preset: str,
    output: ArtifactOutput,
    *,
    runtime: object,
    products: Iterable[object],
    frames: int | None = None,
    owns_runtime: bool = False,
) -> PhysicsProductRunResult:
    """Run products or product specs for one named physics-backed lab preset."""

    from .presets import get_preset

    return run_physics_product_scenario(
        get_preset(preset),
        output,
        runtime=runtime,
        products=products,
        frames=frames,
        owns_runtime=owns_runtime,
    )


def run_optical_lab_workflow(
    *,
    runtime: object,
    products: Iterable[object],
    preset: str | None = None,
    config: OpticalLabScenarioConfig | None = None,
    output: ArtifactOutput | None = None,
    out: Path | None = None,
    frames: int | None = None,
    owns_runtime: bool = False,
) -> PhysicsProductRunResult:
    """Run a named or explicit Optical Pipeline Lab product workflow."""

    try:
        scenario_config = _resolve_workflow_config(preset=preset, config=config)
        artifact_output = _resolve_workflow_output(output=output, out=out, frames=frames)
    except Exception:
        if owns_runtime:
            runtime.close()
        raise
    return run_physics_product_scenario(
        scenario_config,
        artifact_output,
        runtime=runtime,
        products=products,
        frames=None,
        owns_runtime=owns_runtime,
    )


def run_optical_lab_products(
    *,
    runtime: object,
    products: Iterable[object],
    preset: str | None = None,
    config: OpticalLabScenarioConfig | None = None,
    output: ArtifactOutput | None = None,
    out: Path | None = None,
    frames: int | None = None,
    owns_runtime: bool = False,
) -> PhysicsProductRunResult:
    """Run Optical Pipeline Lab products with the workflow convenience API."""

    return run_optical_lab_workflow(
        runtime=runtime,
        products=products,
        preset=preset,
        config=config,
        output=output,
        out=out,
        frames=frames,
        owns_runtime=owns_runtime,
    )


def validate_physics_product_scenario(
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
) -> None:
    """Validate a generic physics-owned product workflow scenario."""

    if not is_physics_published_frame_source(config.frame_source):
        raise ValueError("run_physics_product_scenario requires frame_source='physics_published_frame'")
    if config.clock_owner is not ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME:
        raise ValueError("run_physics_product_scenario requires clock_owner='external_physics_runtime'")
    validate_run(config, output)


def _resolve_workflow_frame_count(output: ArtifactOutput, frames: int | None) -> int:
    frame_count = output.frames if frames is None else int(frames)
    if frame_count < 0:
        raise ValueError("frames must be >= 0")
    if frames is not None and output._frames_explicit and output.frames != frame_count:
        raise ValueError("ArtifactOutput.frames conflicts with workflow run frames")
    return frame_count


def _resolve_workflow_config(
    *,
    preset: str | None,
    config: OpticalLabScenarioConfig | None,
) -> OpticalLabScenarioConfig:
    if (preset is None) == (config is None):
        raise ValueError("provide exactly one of preset or config")
    if config is not None:
        return config

    from .presets import get_preset

    return get_preset(str(preset))


def _resolve_workflow_output(
    *,
    output: ArtifactOutput | None,
    out: Path | None,
    frames: int | None,
) -> ArtifactOutput:
    if output is None:
        if out is None:
            raise TypeError("run_optical_lab_workflow requires output or out")
        if frames is None:
            return ArtifactOutput(root=out)
        return ArtifactOutput(root=out, frames=frames)
    if out is not None and Path(out) != output.root:
        raise ValueError("run_optical_lab_workflow received conflicting output root and out paths")
    if frames is not None:
        frame_count = int(frames)
        if output._frames_explicit and output.frames != frame_count:
            raise ValueError("ArtifactOutput.frames conflicts with workflow run frames")
        if not output._frames_explicit:
            return output.replace_frames(frame_count)
    return output


def _build_products_for_scenario(
    *,
    config: OpticalLabScenarioConfig,
    output: ArtifactOutput,
    runtime: object,
    products: Iterable[object],
) -> tuple[FrameProduct, ...]:
    from .product_specs import ProductBuildContext, build_products

    return build_products(
        products,
        ProductBuildContext(
            runtime=runtime,
            config=config,
            output=output,
        ),
    )


def _validate_product_inputs(products: Iterable[object]) -> tuple[object, ...]:
    from .product_specs import validate_product_inputs

    return validate_product_inputs(products)
