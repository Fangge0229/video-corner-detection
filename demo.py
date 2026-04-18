import cv2
import torch

def infer_one_instance(model, roi_seq, current_transform):
    model.eval()
    with torch.no_grad():
        pred = model(roi_seq)
    pred_corners_rois = pred["corners_pred"][-1]
    pred_conf = torch.sigmoid(pred["conf_logits_pred"][-1])

    pred_corners_img = corners_roi_to_img(pred_corners_rois, current_transform)
    return {
        "corners_roi":pred_corners_roi,
        "corners_img":pred_corners_img,
        "conf_logits":pred_conf
    }

def draw_corners(image, corners, conf=None ,thresh=0.5):
    for i, (x,y) in enumerate(corners):
        if conf is not None and conf[i] < thresh:
            continue
        
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), 2)
        cv2.putText(image, (int(x)-4, int(y)+4), str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image
