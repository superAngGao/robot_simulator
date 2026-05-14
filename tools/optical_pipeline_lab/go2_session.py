"""Compatibility aliases for the pre-C1 Go2 render session names."""

from __future__ import annotations

from tools.optical_pipeline_lab.render_session import (
    OpticalLabRenderFrameContext,
    OpticalLabRenderPipeline,
    OpticalLabRenderSession,
    OpticalLabRenderWorkspace,
)

# transitional: remove in alias-deletion cleanup
Go2RenderWorkspace = OpticalLabRenderWorkspace
Go2RenderSession = OpticalLabRenderSession
Go2RenderFrameContext = OpticalLabRenderFrameContext
Go2RenderPipeline = OpticalLabRenderPipeline

__all__ = [
    "Go2RenderFrameContext",
    "Go2RenderPipeline",
    "Go2RenderSession",
    "Go2RenderWorkspace",
]
