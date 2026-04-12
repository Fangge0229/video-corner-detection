import torch
import torch.nn as nn
def corner_regression_loss(pred,target):
    """
    pred_corners:   [B, 8, 2]
    target_corners: [B, 8, 2]
    target_vis:     [B, 8]
    """
    smooth_l1_loss = nn.SmoothL1Loss(reduction='none',beta = 2.0)
    diff = smooth_l1_loss(pred_corners,target_corners)
    mask = target_vis.unsqueeze(-1)
    diff = diff * mask

    denom = torch.sum(target_vis) * 2.0 + 1e-6
    loss = torch.sum(diff) / denom
    return loss

def corner_confidence_loss(pred,target):
    """
    pred_conf_logits: [B, 8]
    target_vis:       [B, 8]
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    loss = bce_loss(pred_conf_logits, target_vis.float())
    loss = loss.mean()
    return loss

def corner_loss(pred_corners, target_corners, pred_conf_logits, target_vis):
    """
    pred_corners:       [B, 8, 2]
    target_corners:     [B, 8, 2]
    pred_conf_logits:   [B, 8]
    target_vis:         [B, 8]
    """
    reg_loss = corner_regression_loss(pred_corners, target_corners, target_vis)
    conf_loss = corner_confidence_loss(pred_conf_logits, target_vis)
    return reg_loss + conf_loss