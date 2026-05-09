import numpy as np
import torch

from roi_ops import (
    corners_image_to_roi,
    corners_roi_to_image,
    crop_and_resize_roi,
    sanitize_bbox,
)


def test_sanitize_bbox_clamps_and_keeps_non_empty_crop():
    bbox = torch.tensor([-5.5, -2.0, -1.0, 0.0])

    assert sanitize_bbox(bbox, (10, 20, 3)) == (0, 0, 1, 1)


def test_crop_and_coordinate_round_trip():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    corners = torch.tensor(
        [
            [5.0, 4.0],
            [15.0, 4.0],
            [15.0, 14.0],
            [5.0, 14.0],
            [6.0, 5.0],
            [7.0, 6.0],
            [8.0, 7.0],
            [9.0, 8.0],
        ]
    )

    roi_img, transform = crop_and_resize_roi(image, [5, 4, 15, 14], (16, 16))
    corners_roi = corners_image_to_roi(corners, transform)
    corners_img = corners_roi_to_image(corners_roi, transform)

    assert roi_img.shape == (16, 16, 3)
    assert torch.allclose(corners_img, corners, atol=1e-5)
