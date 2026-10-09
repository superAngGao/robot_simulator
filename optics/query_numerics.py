"""Fixed native query policy matching the unchanged CPU reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

BUILD_EPS = 1.0e-12
DIR_EPS = 1.0e-12
TIE_EPS = 1.0e-9


def outward_float32_bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enclose binary64 vertices with the nearest outward binary32 bounds."""
    import numpy as np

    lower = np.min(vertices, axis=-2)
    upper = np.max(vertices, axis=-2)
    with np.errstate(over="ignore"):
        lower32 = lower.astype(np.float32)
        upper32 = upper.astype(np.float32)
        lower32 = np.where(lower32.astype(np.float64) > lower,
                           np.nextafter(lower32, np.float32(-np.inf)), lower32)
        upper32 = np.where(upper32.astype(np.float64) < upper,
                           np.nextafter(upper32, np.float32(np.inf)), upper32)
    if not np.all(np.isfinite(lower32)) or not np.all(np.isfinite(upper32)):
        raise ValueError("world geometry bounds must be finite and representable as float32")
    return lower32, upper32
