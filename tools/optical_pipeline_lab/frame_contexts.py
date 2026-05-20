"""Frame-context provider helpers for Optical Pipeline Lab video paths."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from .physics_source import PhysicsLabRenderRuntime
from .render_session import OpticalLabRenderFrameContext, OpticalLabRenderPipeline


@dataclass(frozen=True)
class StaticFrameContextProvider:
    """Provide static lab render frame contexts from a pipeline."""

    pipeline: OpticalLabRenderPipeline

    @contextmanager
    def begin_frame(
        self,
        frame_index: int,
        *,
        env_idx: int = 0,
    ) -> Iterator[OpticalLabRenderFrameContext]:
        del frame_index
        yield self.pipeline.begin_frame(frame_inputs=None, env_idx=env_idx)


@dataclass(frozen=True)
class SyntheticFrameSequenceContextProvider:
    """Provide dynamic lab render frame contexts from a finite frame sequence.

    Out-of-range frame indexes use the underlying sequence's IndexError.
    """

    pipeline: OpticalLabRenderPipeline
    frame_inputs: Sequence[object]

    @contextmanager
    def begin_frame(
        self,
        frame_index: int,
        *,
        env_idx: int = 0,
    ) -> Iterator[OpticalLabRenderFrameContext]:
        frame_inputs = self.frame_inputs[frame_index]
        yield self.pipeline.begin_frame(frame_inputs=frame_inputs, env_idx=env_idx)


@dataclass(frozen=True)
class PhysicsFrameContextProvider:
    """Provide lab render frame contexts through a physics borrow lifecycle.

    `delivery_mode` is currently used only for construction-time compatibility
    validation; provider-backed warmup will make it operational later.
    """

    runtime: PhysicsLabRenderRuntime
    delivery_mode: str = "sync"

    def __post_init__(self) -> None:
        if self.delivery_mode == "torch_async":
            raise ValueError(
                "physics frame-context providers require provider-backed warmup "
                "before torch_async delivery can be used"
            )

    @contextmanager
    def begin_frame(
        self,
        frame_index: int,
        *,
        published_frame: object | None = None,
        env_idx: int = 0,
    ) -> Iterator[OpticalLabRenderFrameContext]:
        del frame_index
        with self.runtime.begin_frame(
            published_frame=published_frame,
            env_idx=env_idx,
        ) as lease:
            yield lease.frame_context


def static_frame_context_provider(pipeline: OpticalLabRenderPipeline) -> StaticFrameContextProvider:
    """Create a provider for static render frames."""

    return StaticFrameContextProvider(pipeline=pipeline)


def synthetic_frame_sequence_context_provider(
    pipeline: OpticalLabRenderPipeline,
    frame_inputs: Sequence[object],
) -> SyntheticFrameSequenceContextProvider:
    """Create a provider for a synthetic dynamic frame sequence."""

    return SyntheticFrameSequenceContextProvider(
        pipeline=pipeline,
        frame_inputs=frame_inputs,
    )


def physics_frame_context_provider(
    runtime: PhysicsLabRenderRuntime,
    *,
    delivery_mode: str = "sync",
) -> PhysicsFrameContextProvider:
    """Create a provider for physics-backed render frames."""

    return PhysicsFrameContextProvider(
        runtime=runtime,
        delivery_mode=delivery_mode,
    )
