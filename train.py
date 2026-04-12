import os
import torch
from loss import *
from video_corner_detection import *
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

def average_stats(meter):
    if len(meter) == 0:
        return {}
    keys = meter[0].keys()
    avg = {}
    for key in keys:
      avg[key] = sum(item[key] for item in meter) / len(meter)
    return avg

def train_one_epoch(model, dataloader, optimizer, device):
    meter = []
    for batch in dataloader:
        stats = train_one_step(model, batch, optimizer, device)
        meter.append(stats)
    meter = {k: [d[k] for d in meter] for k in meter[0]}
    return average_stats(meter)

def train(model, train_loader, val_loader, optimizer, device, num_epochs, checkpoint_path=None):
    for epoch in range(num_epochs):
        stats = train_one_epoch(model, train_loader, optimizer, device)
        print(epoch, stats)
    print("Training finished")

def update_sequence_memory(sequence_memory, sequence_id, roi_tensor, transform, max_len):
    if sequence_id not in sequence_memory:
        sequence_memory[sequence_id] = []
    sequence_memory[sequence_id].append({
        "roi_image": roi_tensor,
        "tranform": tranform
    })
    
    if len(sequence_memory[sequence_id]) > max_len:
        sequence_memory[sequence_id] = sequence_memory[sequence_id][-max_len:]
    return sequence_memory

def build_inference_sequence(sequence_memory, sequence_roi, sequence_id, current_roi, seq_len):
    history = sequence_memory.get(sequence_id, default=[])
    seq = [item["roi_image"] for item in history]
    seq.append(current_roi)

