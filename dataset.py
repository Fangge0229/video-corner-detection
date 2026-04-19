import os
import torch
import json
from pathlib import Path
import numpy as np

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

    while len(seq) < seq_len:
        seq.insert(0, seq[0])

    seq = seq[-seq_len:]
    return torch.stack(seq, dim=0)

def build_sequence_index(sequences, seq_len):
    index = []
    for seq_id, records in sequences.items():
        records = sort_by_frame_idx(records)
        
        if len(records) < seq_len:
            continue
        
        for end in range(seq_len-1, len(records)):
            window = records[end - seq_len + 1 : end + 1]
            index.append({
                "seq_id": seq_id,
                "start_frame": window[0]["frame_idx"],
                "end_frame": window[-1]["frame_idx"],
                "window": window
            })
    return index

def load_json(json_path: str | Path):
      json_path = Path(json_path)
      with json_path.open("r", encoding="utf-8") as f:
          return json.load(f)


def list_scene_dirs(dataset_root: str | Path):
      """
      返回 train_pbr 下所有 scene 目录，例如:
      dataset_root/train_pbr/000000
      dataset_root/train_pbr/000001
      """
      dataset_root = Path(dataset_root)
      train_pbr_dir = dataset_root / "train_pbr"

      if not train_pbr_dir.is_dir():
          raise FileNotFoundError(f"train_pbr directory not found: {train_pbr_dir}")

      scene_dirs = []
      for p in sorted(train_pbr_dir.iterdir()):
          if not p.is_dir():
              continue

          # 只保留像 000000、000001 这种场景目录
          if not p.name.isdigit():
              continue

          # 必须同时存在这三个标注文件
          required_files = [
              p / "scene_gt.json",
              p / "scene_camera.json",
              p / "scene_gt_info.json",
          ]
          if all(x.is_file() for x in required_files):
              scene_dirs.append(p)

      return scene_dirs

def corners_to_xyxy(corners_2d, image_size=None, clamp=False):
    """
    corners_2d: (8, 2)
    image_size: (height, weight)
    clamp: bool
    return: bbox (x1, y1, x2, y2)
    """
    valid_mask = torch.is_finite(corners_2d).all(dim=-1)
    corners_2d = torch.as_tensor(corners_2d, dtype=torch.float32).reshape(-1, 2)
    if valid_mask.sum() == 0:
        return None
    
    valid_points = corners_2d[valid_mask]

    x1 = min(valid_points[:,0])
    y1 = min(valid_points[:,1])
    x2 = max(valid_points[:,0])
    y2 = max(valid_points[:,1])
    bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
    
    if clamp and image_size is not None:
        h, w = image_size
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(w, bbox[2])
        bbox[3] = min(h, bbox[3])
    
    return bbox
    
    
def projects_3d_box_corners(obj_id, R, t, K, model_corner_cache):
    """
    obj_id: int
    R: (3, 3)
    t: (3,)
    K: (3, 3)
    model_corner_cache: Dict[int, corners_3d] corners_3d shape = [8, 3]
    return: corners_2d (8, 2)
    """
    
    corners_3d = torch.as_tensor(model_corner_cache[obj_id], dtype=torch.float32).reshape(-1, 3)
    R = torch.as_tensor(R, dtype=torch.float32).reshape(3, 3)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(3, 1)
    K = torch.as_tensor(K, dtype=torch.float32).reshape(3, 3)
    
    cam_points = corners_3d @ R.T + t.T
    corners_2d = torch.full((corners_3d.shape[0], 2), float("nan"), dtype=torch.float32)
    for i in range(8):
        xc,yc,zc = cam_points[i]
        if zc <= 0:
            continue
        pixel_h = K @ [xc, yc, zc]
        x = pixel_h[0] / pixel_h[2]
        y = pixel_h[1] / pixel_h[2]
        corners_2d[i] = [x, y]
    return conners_2d

def estimate_corner_visibility(corners_2d, infos, ann_idx, minvisib_fract=1e-6):
    """
    corners_2d:
        Tensor[8, 2]
    infos:
        当前 image_id 对应的 scene_gt_info 列表
    ann_idx:
        当前实例在 anns 中的索引

    return:
        corner_vis:
            Tensor[8], 每个元素是 0/1
    """
    corner_vis = torch.is_finite(corners_2d).all(dim=-1).float()
    if infos is None or ann_idx>= len(infos):
        return corner_vis
    
    info = infos[ann_idx]
    visib_fract = info.get("visib_fract", None)
    if visib_fract is not None and visib_fract <= minvisib_fract:
        return torch.zeros(8)
    bbox_visib = info.get("bbox_fract", None)
    if bbox_visib is not None:
        x, y, w, h = bbox_visib
        x2 = x + w
        y2 = y + h
        for i in range(8):
            if corner_vis[i] == 0:
                continue
            px, py = corners_2d[i]
            if not (x <= px <= x2 and y <= py <= y2):
                corner_vis[i] = 0
    return corner_vis
    

def build_sequences_from_bop_scenes(dataset_root):
    all_sequences = {}

    for scene_dir in list_dir(os.path.join(dataset_root, "train_pbr")):
        scene_gt = load_json(scene_dir / "scene_gt.json")
        scene_camera = load_json(scene_dir / "scene_camera.json")
        scene_gt_info = load_json(scene_dir / "scene_gt_info.json")
    
        for image_id in sorted(scene_gt.key()):
            anns = scene_gt[image_id]
            camera = scene_camera[image_id]
            infos = scene_gt_info.get(image_id, [])

            for ann_idx, ann in enumerate(anns):
                obj_id = ann["obj_id"]
                corners_2d = projects_3d_box_corners(
                    obj_id = obj_id,
                    R = ann["cam_R_m2c"],
                    t = ann["cam_t_m2c"],
                    K = camera["cam_K"],
                    model_corner_cache = model_corner_cache
                )
                bbox = corners_to_xyxy(corners_2d)
