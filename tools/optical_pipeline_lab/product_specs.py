"""Declarative product specs for Optical Pipeline Lab workflow helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from .frame_products import DebugFrameProduct, FrameProduct
from .runner import ArtifactOutput
from .scenarios import (
    ClockOwnerKind,
    OpticalLabScenarioConfig,
    is_physics_published_frame_source,
)

if TYPE_CHECKING:
    from rl_env.obs import ObsSchema


@dataclass(frozen=True)
class ProductBuildContext:
    """Context available while materializing declarative product specs."""

    runtime: object
    config: OpticalLabScenarioConfig
    output: ArtifactOutput


class ProductSpec(Protocol):
    """Build a concrete frame product for one workflow run."""

    product_name: str

    def build(self, context: ProductBuildContext) -> FrameProduct:
        """Return a concrete frame product bound to the workflow context."""


@dataclass(frozen=True)
class DebugProductSpec:
    """Declare a debug product without constructing product state early."""

    product_name: str = "debug"
    metadata_keys: tuple[str, ...] | None = None

    def build(self, context: ProductBuildContext) -> DebugFrameProduct:
        return DebugFrameProduct(
            product_name=self.product_name,
            metadata_keys=self.metadata_keys,
        )


@dataclass(frozen=True)
class ObservationProductSpec:
    """Declare a published-state observation product for physics frames."""

    schema: ObsSchema
    actuated_q_indices: object | None = None
    actuated_v_indices: object | None = None
    contact_body_names: Sequence[str] = ()
    root_body_idx: int = 0
    root_q_slice: slice = field(default_factory=lambda: slice(0, 7))
    product_name: str = "observation"
    engine: object | None = None

    @classmethod
    def from_scenario(
        cls,
        config: OpticalLabScenarioConfig,
        *,
        schema: ObsSchema,
        actuated_q_indices: object | None = None,
        actuated_v_indices: object | None = None,
        contact_body_names: Sequence[str] = (),
        root_body_idx: int = 0,
        root_q_slice: slice = slice(0, 7),
        product_name: str = "observation",
        engine: object | None = None,
    ) -> "ObservationProductSpec":
        """Create an observation spec for an explicit physics-backed scenario."""

        _validate_physics_product_config(config, caller="ObservationProductSpec.from_scenario")
        return cls(
            schema=schema,
            actuated_q_indices=actuated_q_indices,
            actuated_v_indices=actuated_v_indices,
            contact_body_names=tuple(contact_body_names),
            root_body_idx=root_body_idx,
            root_q_slice=root_q_slice,
            product_name=product_name,
            engine=engine,
        )

    def build(self, context: ProductBuildContext) -> FrameProduct:
        from .observation_products import PublishedStateObservationProduct

        engine = self.engine if self.engine is not None else getattr(context.runtime, "engine", None)
        if engine is None:
            raise ValueError("ObservationProductSpec requires engine or runtime.engine")
        return PublishedStateObservationProduct(
            engine=engine,
            schema=self.schema,
            root_body_idx=self.root_body_idx,
            root_q_slice=self.root_q_slice,
            actuated_q_indices=self.actuated_q_indices,
            actuated_v_indices=self.actuated_v_indices,
            contact_body_names=tuple(self.contact_body_names),
            product_name=self.product_name,
        )


ProductInput = FrameProduct | ProductSpec


def build_products(
    products: Iterable[ProductInput],
    context: ProductBuildContext,
) -> tuple[FrameProduct, ...]:
    """Materialize product instances from products or declarative specs."""

    return tuple(_build_product(product, context) for product in products)


def _build_product(product: ProductInput, context: ProductBuildContext) -> FrameProduct:
    if _looks_like_frame_product(product):
        return product
    build = getattr(product, "build", None)
    if callable(build):
        built = build(context)
        if _looks_like_frame_product(built):
            return built
        raise TypeError("ProductSpec.build() must return a FrameProduct")
    raise TypeError("products must be FrameProduct instances or ProductSpec values")


def _looks_like_frame_product(value: object) -> bool:
    return all(
        hasattr(value, attr)
        for attr in (
            "product_name",
            "begin_run",
            "consume",
            "end_run",
        )
    )


def _validate_physics_product_config(config: OpticalLabScenarioConfig, *, caller: str) -> None:
    if not is_physics_published_frame_source(config.frame_source):
        raise ValueError(f"{caller} requires frame_source='physics_published_frame'")
    if config.clock_owner is not ClockOwnerKind.EXTERNAL_PHYSICS_RUNTIME:
        raise ValueError(f"{caller} requires clock_owner='external_physics_runtime'")
