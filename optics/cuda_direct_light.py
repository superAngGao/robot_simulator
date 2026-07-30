"""CUDA direct-light optical executor skeleton."""

from __future__ import annotations

from .execution import OpticalOutputProfile, normalize_output_profile

try:  # pragma: no cover - exercised in CUDA extension environments.
    import torch
    import warp as wp
    from torch.utils.cpp_extension import load_inline

    _HAS_CUDA_DIRECT_LIGHT_DEPS = True
except Exception:  # pragma: no cover - keeps CPU-only imports working.
    torch = None
    wp = None
    load_inline = None
    _HAS_CUDA_DIRECT_LIGHT_DEPS = False


class CudaDeviceBvhDirectLightOpticalExecutor:
    """CUDA direct-light executor over a device scene plus CUDA LBVH."""

    capabilities = frozenset(
        {
            "range_m",
            "hit_mask",
            "position_world",
            "normal_world",
            "numeric_instance_id",
            "rgb",
            "intensity",
        }
    )
    supported_profiles = frozenset(
        {
            OpticalOutputProfile.DIRECT_LIGHT_FULL,
            OpticalOutputProfile.RGB_PREVIEW,
            OpticalOutputProfile.RENDER_ONLY,
        }
    )

    def __init__(
        self,
        *,
        device=None,
        stream=None,
        shadows: bool = True,
        ambient_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shadow_bias: float = 1.0e-6,
    ) -> None:
        _require_cuda_direct_light_deps()
        wp.init()
        self.device = wp.get_device("cuda:0" if device is None else device)
        self.stream = stream
        self.shadows = bool(shadows)
        self.ambient_rgb = tuple(float(component) for component in ambient_rgb)
        self.background_rgb = tuple(float(component) for component in background_rgb)
        self.shadow_bias = float(shadow_bias)
        if len(self.ambient_rgb) != 3:
            raise ValueError("ambient_rgb must contain 3 components")
        if len(self.background_rgb) != 3:
            raise ValueError("background_rgb must contain 3 components")
        if self.shadow_bias < 0.0:
            raise ValueError("shadow_bias must be >= 0")

    def execute(
        self,
        snapshot,
        bvh,
        spec,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ):
        """Execute host-ray CUDA direct-light rendering once P12.3b lands."""

        self._validate_output_profile(output_profile)
        raise NotImplementedError("cuda_direct_light first-hit kernel is pending P12.3b")

    def execute_camera(
        self,
        snapshot,
        bvh,
        camera,
        *,
        output_profile: OpticalOutputProfile | str = OpticalOutputProfile.DIRECT_LIGHT_FULL,
        render_profile: list[tuple[str, float]] | None = None,
    ):
        """Execute camera-ray CUDA direct-light rendering once P12.3e lands."""

        self._validate_output_profile(output_profile)
        raise NotImplementedError("cuda_direct_light camera raygen is pending P12.3e")

    def _validate_output_profile(self, output_profile: OpticalOutputProfile | str) -> None:
        output_profile = normalize_output_profile(output_profile)
        if output_profile not in self.supported_profiles:
            raise ValueError(f"cuda_direct_light does not support output_profile={output_profile.value!r}")


def cuda_direct_light_available() -> bool:
    """Return whether CUDA direct-light dependencies are importable and CUDA is available."""

    return _HAS_CUDA_DIRECT_LIGHT_DEPS and bool(torch.cuda.is_available())


def _require_cuda_direct_light_deps() -> None:
    if not _HAS_CUDA_DIRECT_LIGHT_DEPS:
        raise ImportError("CUDA direct-light executor requires torch, warp, and torch CUDA extension tooling")
    if not torch.cuda.is_available():
        raise ImportError("CUDA direct-light executor requires torch CUDA availability")
