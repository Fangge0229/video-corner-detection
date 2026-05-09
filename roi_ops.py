import torch
import cv2
import numpy as np
def sanitize_bbox(bbox, image_shape):
    H, W = image_shape[:2]
    bbox = torch.as_tensor(bbox, dtype=torch.float32).reshape(4)
    x1, y1, x2, y2 = bbox.tolist()

    x1 = int(np.floor(max(0.0, min(x1, W - 1))))
    y1 = int(np.floor(max(0.0, min(y1, H - 1))))
    x2 = int(np.ceil(max(0.0, min(x2, W))))
    y2 = int(np.ceil(max(0.0, min(y2, H))))

    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)

    return x1, y1, x2, y2

def crop_and_resize_roi(image, bbox, out_size):
    """
    image:
        H x W x 3
    bbox:
        [x1, y1, x2, y2]
    out_size:
        (Hr, Wr)

    return:
        roi_image:
            Hr x Wr x 3
        transform:
            dict
    """
    if image is None:
        raise ValueError("image is None")

    x1, y1, x2, y2 = sanitize_bbox(bbox, image.shape)
    crop = image[y1:y2, x1:x2]
    roi_image = cv2.resize(crop, (out_size[1], out_size[0]))
    
    scale_x = out_size[1] / max(x2 -x1, 1)
    scale_y = out_size[0] / max(y2 -y1, 1)
    
    transform = {
       "crop_x1": x1,
       "crop_y1": y1,
       "scale_x": scale_x,
       "scale_y": scale_y
    }
    return roi_image, transform
    
def corners_image_to_roi(corners_xy, transform):
    """
    corners_xy:
        Tensor[8, 2] 或 list[[x, y], ...]

    transform:
        crop_and_resize_roi 返回的 transform

    return:
        Tensor[8, 2]
    """
    x1 = transform["crop_x1"]
    y1 = transform["crop_y1"]
    sx = transform["scale_x"]
    sy = transform["scale_y"]
    
    corners_roi = []
    for x, y in corners_xy:
        x_roi = (x - x1) * sx
        y_roi = (y - y1) * sy
        corners_roi.append([x_roi, y_roi])
    return torch.tensor(corners_roi, dtype=torch.float32)

def corners_roi_to_image(corners_roi, transform):
    """
    corners_roi:
        Tensor[8, 2]

    transform:
        crop_and_resize_roi 返回的 transform

    return:
        Tensor[8, 2]
    """
    x1 = transform["crop_x1"]
    y1 = transform["crop_y1"]
    sx = transform["scale_x"]
    sy = transform["scale_y"]
    
    corners_xy = torch.empty_like(corners_roi)
    corners_xy[:, 0] = corners_roi[:, 0] / sx + x1
    corners_xy[:, 1] = corners_roi[:, 1] / sy + y1
    return corners_xy
