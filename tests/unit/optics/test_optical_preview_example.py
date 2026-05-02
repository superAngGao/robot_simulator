from __future__ import annotations

import numpy as np
from PIL import Image

from examples.optical_direct_light_preview import render_preview, write_preview_images


def test_optical_direct_light_preview_example_writes_expected_images(tmp_path):
    image = render_preview(width=48, height=32)

    assert image.image_shape == (32, 48)
    assert image.channel("rgb").shape == (32, 48, 3)
    assert image.channel("depth_m").shape == (32, 48)
    assert image.channel("numeric_instance_id").shape == (32, 48)
    assert np.any(image.channel("hit_mask"))

    outputs = write_preview_images(image, tmp_path)

    assert set(outputs) == {"rgb", "depth", "segmentation", "panel"}
    for path in outputs.values():
        assert path.exists()

    with Image.open(outputs["rgb"]) as rgb:
        assert rgb.size == (48, 32)
        assert rgb.mode == "RGB"
    with Image.open(outputs["depth"]) as depth:
        assert depth.size == (48, 32)
        assert depth.mode == "L"
    with Image.open(outputs["segmentation"]) as segmentation:
        assert segmentation.size == (48, 32)
        assert segmentation.mode == "RGB"
