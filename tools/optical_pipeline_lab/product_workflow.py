"""Physics-owned product workflow helpers for Optical Pipeline Lab runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

from .frame_products import FrameProduct, FrameProductResult, MultiProductFrameRunner
from .runner import ArtifactOutput


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
