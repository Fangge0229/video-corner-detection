import torch
import torch.nn as nn
def corner_regression_loss(pred_corners, target_corners, target_vis):
    """
    pred_corners:   [B, 8, 2]
    target_corners: [B, 8, 2]
    target_vis:     [B, 8]
    """
    smooth_l1_loss = nn.SmoothL1Loss(reduction='none',beta = 2.0)
    diff = smooth_l1_loss(pred_corners,target_corners)
    mask = target_vis.unsqueeze(-1).float()
    diff = diff * mask

    denom = mask.sum() * 2.0 + 1e-6
    loss = torch.sum(diff) / denom
    return loss

def corner_confidence_loss(pred_conf_logits, target_vis):
    """
    pred_conf_logits: [B, 8]
    target_vis:       [B, 8]
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    return bce_loss(pred_conf_logits, target_vis.float())

def corner_loss(pred_corners, target_corners, pred_conf_logits, target_vis, lambda_conf=0.5):
    """
    pred_corners:       [B, 8, 2]
    target_corners:     [B, 8, 2]
    pred_conf_logits:   [B, 8]
    target_vis:         [B, 8]
    """
    reg_loss = corner_regression_loss(pred_corners, target_corners, target_vis)
    conf_loss = corner_confidence_loss(pred_conf_logits, target_vis)
    total = reg_loss + lambda_conf * conf_loss
    return {
        "loss": total,
        "loss_corner": reg_loss,
        "loss_conf": conf_loss
    }
