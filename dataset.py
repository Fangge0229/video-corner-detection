import os
import torch
import json
from pathlib import Path
import numpy as np
from roi_ops import *

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
    
def load_ply_vertices_ascii(ply_path):
    try:
        text = Path(ply_path).read_txt(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read ply file: {ply_path}")
    
    lines = text.strip().split()
    if not line or lines[0].strp()!="ply":
        raise ValueError(f"Invalid ply file: {ply_path}")
    
    format_line = None
    vertex_count = None
    header_end = None

    for i, line in enumerate(lines):
        if line.startswith("format "):
            format_line = line.strip()
        elif line.startwith("format "):
            vertex_count = int(line.split()[-1])
        elif line.startwith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startwith("end_header"):
            header_end = i
            break
    
    if format_line != "format ascii 1.0":
        raise ValueError(f"Unsupported PLY format: {format_line}")
    
    if vertex_count is None or header_end is None:
        raise ValueError(f"Invalid ply file: {ply_path}")

    vertex_lines = lines[header_end + 1 : header_end + 1 + vertex_count]
    vertices = []
    for line in vertex_lines:
        parts = line.strip().split()
        if len(parts) < 3:
            raise ValueError(f"Invalid ply file: {ply_path}")
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        vertices.append([x, y, z])
    return np.array(vertices, dtype=np.float32)

def vertices_to_box_corners(vertices):
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)
    
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max],
    ])
    return corners

def load_ply_corners(ply_path):
    vertices = load_ply_vertices_ascii(ply_path)
    return vertices_to_box_corners(vertices)

def build_model_corner_caceh(models_dir):
    models_dir = Path(models_dir)
    cache = {}

    for ply_path in sorted(models_dir.glob("obj_*.ply")):
        obj_id = int(ply_path.stem.split("_")[-1])
        cache[obj_id] = load_ply_corners(ply_path)
    
    if len(cache) == 0:
        raise FileNotFoundError(f"No obj_*.ply files found in {models_dir}")
    
    return cache

def project_3d_box_corners(obj_id, R, t, K, model_corner_cache):
    """
    obj_id: int
    R: (3, 3)
    t: (3,)
    K: (3, 3)
    model_corner_cache: Dict[int, corners_3d] corners_3d shape = [8, 3]
    return: corners_2d (8, 2)
    """
    
    corners_3d = torch.as_tensor(model_corner_cache[obj_id], dtype=torch.float32).reshape(8, 3)
    R = torch.as_tensor(R, dtype=torch.float32).reshape(3, 3)
    t = torch.as_tensor(t, dtype=torch.float32).reshape(3, 1)
    K = torch.as_tensor(K, dtype=torch.float32).reshape(3, 3)
    
    cam_points = corners_3d @ R.T + t.T
    corners_2d = torch.full((8, 2), float("nan"), dtype=torch.float32)
    for i in range(8):
        xc,yc,zc = cam_points[i]
        if zc <= 0:
            continue
        pixel_h = K @ cam_points[i]
        x = pixel_h[0] / pixel_h[2]
        y = pixel_h[1] / pixel_h[2]
        corners_2d[i] = torch.tensor([x, y], dtype=torch.float32)
    return corners_2d

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
                corner_vis = estimate_corner_visibility(corners_2d, infos, ann_idx)

                record = {
                    "source_type": "bop_scene_clip",
                    "sequence_id": sequence_id,
                    "frame_id": int(image_id),
                    "image_path": str(scene_dir / "rgb" / f"{image_id:06d}.png"),
                    "bbox": bbox,
                    "corners": corners_2d,
                    "corner_vis": corner_vis,
                    "camera": camera,
                    "obj_id": obj_id,
                }
                all_sequences.setdefault(sequence_id, []).append(record)
    return all_sequences

def image_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = (tensor - mean) / std
    return tensor  
class VideoCornerDataset(Dataset):
    def __init__(self, anns, image_root, sesq_len=4, roi_size=(128, 128)):
        self.anns = anns
        self.image_root = image_root
        self.seq_len = seq_len
        self.roi_size = roi_size
        
        self.all_sequences = build_sequences_from_bop_scenes(anns, image_root)
        self.sequence_index = build_sequence_index(self.all_sequences, seq_len)

    def __len__(self):
        return len(self.sequence_index)

    def __getitem__(self, idx):
      item = self.sequencec_index[idx]
      window = item["window"]

      roi_images = []
      frame_ids = []
      target_corners = None
      target_vis = None
      target_bbox = None
      target_transform = None

      for i, record in enumerate(window):
        image = cv2.imread(record["image_path"])
        bbox = record["bbox"]
        roi_img, transform = crop_and_resize_roi(image, bbox, self.roi_size)
        roi_tensor = image_to_tensor(roi_image)
        roi_images.append(roi_tensor)
        frame_ids.append(record["frame_id"])
        if i == len(window) - 1:
            target_corners = corners_image_to_roi(record["corners"], transform)
            target_vis = torch.as_tensor(record["corner_vis"])
            target_bbox = torch.as_tensor(bbox)
            target_transform = transform
        
    return {
        "roi_images": torch.stack(roi_images, dim=0),
        "target_corners": target_corners,
        "target_vis": target_vis,
        "target_bbox": target_bbox,
        "target_transform": target_transform,
        "sequence_id": item["sequence_id"],
        "frame_ids": frame_ids
    }
            
def collate_fn(batch):
    return {
        "roi_images": torch.stack([x["roi_images"] for x in batch], dim=0),
        "target_corners": torch.stack([x["target_corners"] for x in batch], dim=0),
        "target_vis": torch.stack([x["target_vis"] for x in batch], dim=0),
        "target_bbox": torch.stack([x["target_bbox"] for x in batch], dim=0),
        "target_transform": [x["target_transform"] for x in batch],
        "sequence_id": [x["sequence_id"] for x in batch],
        "frame_ids": [x["frame_ids"] for x in batch]
    }