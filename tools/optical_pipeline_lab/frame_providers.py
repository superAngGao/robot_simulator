"""Product-facing frame-context provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .frame_contexts import PhysicsFrameContextProvider, StaticFrameContextProvider
from .frame_tick import SimulationFrameTick


class TickFrameContextProvider(Protocol):
    """Begin render frame contexts from shared simulation ticks."""

    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        """Return a context manager yielding one render frame context."""


@dataclass(frozen=True)
class PhysicsTickFrameContextProvider:
    """Adapt a physics frame-context provider to the product tick contract."""

    provider: PhysicsFrameContextProvider

    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        return self.provider.begin_frame(
            tick.frame_index,
            env_idx=tick.env_idx,
            published_frame=tick.published_frame,
        )


@dataclass(frozen=True)
class StaticTickFrameContextProvider:
    """Adapt a static frame-context provider to the product tick contract."""

    provider: StaticFrameContextProvider

    def begin_frame_for_tick(self, tick: SimulationFrameTick):
        return self.provider.begin_frame(
            tick.frame_index,
            env_idx=tick.env_idx,
        )


def physics_tick_frame_context_provider(
    provider: PhysicsFrameContextProvider,
) -> PhysicsTickFrameContextProvider:
    """Create a product-facing physics tick provider."""

    return PhysicsTickFrameContextProvider(provider=provider)


def static_tick_frame_context_provider(
    provider: StaticFrameContextProvider,
) -> StaticTickFrameContextProvider:
    """Create a product-facing static tick provider."""

    return StaticTickFrameContextProvider(provider=provider)
