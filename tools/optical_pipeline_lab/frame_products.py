"""Product contracts for tick-based Optical Pipeline Lab workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .frame_tick import SimulationFrameTick


@dataclass(frozen=True)
class FrameProductResult:
    """Typed per-frame result for one product consuming a simulation tick."""

    product_name: str
    frame_index: int
    frame_id: int
    sim_time: float
    env_idx: int
    payload: object | None = None
    timing: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_tick(
        cls,
        *,
        product_name: str,
        tick: SimulationFrameTick,
        payload: object | None = None,
        timing: Mapping[str, float] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "FrameProductResult":
        """Create a result whose identity fields mirror a simulation tick."""

        return cls(
            product_name=product_name,
            frame_index=tick.frame_index,
            frame_id=tick.frame_id,
            sim_time=tick.sim_time,
            env_idx=tick.env_idx,
            payload=payload,
            timing=dict(timing or {}),
            metadata=dict(metadata or {}),
        )


class FrameProduct(Protocol):
    """Consume simulation frame ticks and optionally produce per-frame results."""

    product_name: str

    def begin_run(self) -> object | None:
        """Prepare product-owned run state."""

    def consume(self, tick: SimulationFrameTick) -> FrameProductResult | None:
        """Consume one tick and optionally return a per-frame result."""

    def end_run(self) -> object | None:
        """Finalize product-owned run state."""


@dataclass(frozen=True)
class MultiProductFrameRunner:
    """Run ordered frame products for each simulation tick.

    This runner does not own frame-provider borrow/release. Products that need a
    frame context manage that lifecycle inside their own ``consume(tick)``.
    """

    products: tuple[FrameProduct, ...]

    def __post_init__(self) -> None:
        names = [product.product_name for product in self.products]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            joined = ", ".join(repr(name) for name in duplicate_names)
            raise ValueError(f"product_name values must be unique: {joined}")

    def begin_run(self) -> Mapping[str, object | None]:
        """Begin each product in order and return results keyed by product name."""

        return {product.product_name: product.begin_run() for product in self.products}

    def step(self, tick: SimulationFrameTick) -> tuple[FrameProductResult | None, ...]:
        """Consume one tick with every product, preserving product positions.

        Product execution is fail-fast: if one product raises, later products do
        not consume the tick and the original exception propagates.
        """

        return tuple(product.consume(tick) for product in self.products)

    def end_run(self) -> Mapping[str, object | None]:
        """End each product in order and return results keyed by product name."""

        return {product.product_name: product.end_run() for product in self.products}


@dataclass
class DebugFrameProduct:
    """Record tick identity and selected metadata without GPU or render work."""

    product_name: str = "debug"
    metadata_keys: tuple[str, ...] | None = None
    records: list[FrameProductResult] = field(default_factory=list, init=False)

    def begin_run(self) -> object | None:
        self.records.clear()
        return None

    def consume(self, tick: SimulationFrameTick) -> FrameProductResult:
        metadata = self._metadata_for_tick(tick)
        payload = {
            "frame_index": tick.frame_index,
            "frame_id": tick.frame_id,
            "sim_time": tick.sim_time,
            "env_idx": tick.env_idx,
            "metadata": metadata,
        }
        result = FrameProductResult.from_tick(
            product_name=self.product_name,
            tick=tick,
            payload=payload,
            metadata=metadata,
        )
        self.records.append(result)
        return result

    def end_run(self) -> object | None:
        return tuple(self.records)

    def _metadata_for_tick(self, tick: SimulationFrameTick) -> dict[str, object]:
        if self.metadata_keys is None:
            return dict(tick.metadata)
        return {key: tick.metadata[key] for key in self.metadata_keys if key in tick.metadata}
