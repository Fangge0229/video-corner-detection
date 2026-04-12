import os
import torch
def train_one_step(model, batch, optimizer, device):
    model.train()
    
    roi_images = batch["roi_images"].to(device)
    target_corners = batch["target_corners"].to(device)
    target_vis = batch["target_vis"].to(device)
    
    batch_gpu = {
        "roi_images": roi_images,
        "target_corners": target_corners,
        "target_vis": target_vis
    }

    pred = model(batch_gpu["roi_images"])
    loss_dict = corner_loss(pred["corners_pred"], batch_gpu["target_corners"], pred["conf_logits_pred"], batch_gpu["target_vis"])
    
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()
    
    return {
        "loss": float(loss_dict["loss"].item()),
        "loss_corner": float(loss_dict["loss_corner"].item()),
        "loss_conf": float(loss_dict["loss_conf"].item()),
    }