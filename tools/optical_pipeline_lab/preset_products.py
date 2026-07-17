"""Preset product selection for Optical Pipeline Lab workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from .preset_runtime import PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET
from .product_specs import DebugProductSpec, ProductInput, VideoProductSpec, validate_product_inputs

ProductSelection = str | ProductInput


def resolve_lab_product_specs(
    *,
    preset: str,
    products: Iterable[ProductSelection],
) -> tuple[ProductInput, ...]:
    """Resolve user-facing product selections into product specs or instances."""

    return tuple(_resolve_product_selection(preset=str(preset), product=product) for product in products)


def supported_lab_product_strings(*, preset: str | None = None) -> tuple[str, ...]:
    """Return product strings accepted by the preset product resolver."""

    if preset is None or str(preset) in _VIDEO_PRODUCT_FACTORIES:
        return ("debug", "video")
    return ("debug",)


def _resolve_product_selection(*, preset: str, product: ProductSelection) -> ProductInput:
    if isinstance(product, str):
        return _resolve_product_string(preset=preset, product=product)
    return validate_product_inputs((product,))[0]


def _resolve_product_string(*, preset: str, product: str) -> ProductInput:
    if product == "debug":
        return DebugProductSpec()
    if product == "video":
        return _video_product_spec_for_preset(preset)
    if product == "observation":
        raise ValueError(
            'Product string "observation" is not supported. '
            "Observation requires explicit robot metadata "
            "(schema, actuated indices, contact bodies). "
            "Use ObservationProductSpec.from_scenario(...) instead. "
            "See examples/optical_lab/physics_body_triangle_observation.py."
        )
    choices = ", ".join(supported_lab_product_strings(preset=preset))
    raise ValueError(f"Unsupported Optical Lab product string {product!r}; supported strings: {choices}")


def _video_product_spec_for_preset(preset: str) -> VideoProductSpec:
    try:
        factory = _VIDEO_PRODUCT_FACTORIES[preset]
    except KeyError as exc:
        choices = ", ".join(sorted(_VIDEO_PRODUCT_FACTORIES))
        raise NotImplementedError(
            f"Optical Lab video product is not registered for preset {preset!r}; "
            f"supported video presets: {choices}"
        ) from exc
    return factory()


def _physics_body_triangle_video_spec() -> VideoProductSpec:
    from . import go2_backend

    return VideoProductSpec(
        build_video_camera=go2_backend._build_video_camera,
        synchronize_event=getattr(go2_backend.wp, "synchronize_event", lambda event: None),
        pack_rgb8=go2_backend._pack_video_rgb8,
    )


_VIDEO_PRODUCT_FACTORIES: dict[str, Callable[[], VideoProductSpec]] = {
    PHYSICS_BODY_TRIANGLE_VIDEO_SMOKE_PRESET: _physics_body_triangle_video_spec,
}
