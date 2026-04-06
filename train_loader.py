import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _default_sequence_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def _has_video_corner_labels(dataset_root):
    clips_path = os.path.join(dataset_root, 'video_corner_labels', 'clips.json')
    return os.path.isfile(clips_path)


def _normalize_annotation_to_corners(annotation):
    if not isinstance(annotation, dict):
        raise ValueError("video_corner_labels annotation must be an object")

    if 'corners_per_class' in annotation:
        corners_per_class = annotation['corners_per_class']
        if not isinstance(corners_per_class, list) or len(corners_per_class) != 8:
            raise ValueError("video_corner_labels annotation must contain 8 corner classes")

        normalized = [[] for _ in range(8)]
        for class_id, corners in enumerate(corners_per_class):
            if corners is None:
                continue
            if not isinstance(corners, list):
                raise ValueError("Each class in corners_per_class must be a list")
            for corner in corners:
                if not isinstance(corner, (list, tuple)) or len(corner) < 2:
                    raise ValueError("Each corner must be an [x, y] pair")
                x, y = corner[:2]
                normalized[class_id].append([float(x), float(y)])
        return normalized

    if 'keypoints' in annotation:
        keypoints = annotation['keypoints']
        if not isinstance(keypoints, list):
            raise ValueError("annotation keypoints must be a list")

        normalized = [[] for _ in range(8)]
        for class_id in range(8):
            offset = class_id * 3
            if offset + 2 >= len(keypoints):
                break
            x, y, v = keypoints[offset:offset + 3]
            if v > 0:
                normalized[class_id].append([float(x), float(y)])
        return normalized

    if 'corners' in annotation:
        corners = annotation['corners']
        if not isinstance(corners, list) or len(corners) != 8:
            raise ValueError("video_corner_labels annotation corners must contain 8 classes")

        normalized = [[] for _ in range(8)]
        for class_id, class_corners in enumerate(corners):
            if class_corners is None:
                continue
            if not isinstance(class_corners, list):
                raise ValueError("Each class in corners must be a list")
            for corner in class_corners:
                if not isinstance(corner, (list, tuple)) or len(corner) < 2:
                    raise ValueError("Each corner must be an [x, y] pair")
                x, y = corner[:2]
                normalized[class_id].append([float(x), float(y)])
        return normalized

    raise ValueError("video_corner_labels annotation must contain corners_per_class, corners, or keypoints")


def _load_video_corner_label_clips(dataset_root):
    clips_path = os.path.join(dataset_root, 'video_corner_labels', 'clips.json')
    with open(clips_path, 'r', encoding='utf-8') as f:
        clips_data = json.load(f)

    if not isinstance(clips_data, dict) or 'clips' not in clips_data or not isinstance(clips_data['clips'], list):
        raise ValueError("video_corner_labels/clips.json must contain a 'clips' list")

    annotations_path = os.path.join(dataset_root, 'video_corner_labels', 'annotations.json')
    annotations_by_clip_frame = {}
    if os.path.isfile(annotations_path):
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations_data = json.load(f)
        if not isinstance(annotations_data, dict) or 'annotations' not in annotations_data or not isinstance(annotations_data['annotations'], list):
            raise ValueError("video_corner_labels/annotations.json must contain an 'annotations' list")
        annotations = annotations_data['annotations']
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError("Each annotation entry must be an object")
            clip_id = annotation.get('clip_id')
            frame_idx = annotation.get('frame_idx')
            if clip_id is None or frame_idx is None:
                raise ValueError("Each annotation entry must contain clip_id and frame_idx")
            annotations_by_clip_frame[(clip_id, frame_idx)] = _normalize_annotation_to_corners(annotation)

    clips = []
    for clip_idx, raw_clip in enumerate(clips_data['clips']):
        if not isinstance(raw_clip, dict):
            raise ValueError("Each clip entry in video_corner_labels/clips.json must be an object")

        if 'frames' not in raw_clip or not isinstance(raw_clip['frames'], list):
            raise ValueError(f"Clip entry {clip_idx} in video_corner_labels/clips.json must contain a 'frames' list")

        clip_id = raw_clip.get('clip_id', f'clip_{clip_idx:03d}')
        normalized_frames = []
        for frame_idx, frame in enumerate(raw_clip['frames']):
            if isinstance(frame, str):
                frame = {'image_path': frame}
            elif not isinstance(frame, dict):
                raise ValueError(f"Frame entry {frame_idx} in clip '{clip_id}' must be a string or object")

            image_path = frame.get('image_path') or frame.get('file_name') or frame.get('path')
            if not image_path:
                raise ValueError(f"Frame entry {frame_idx} in clip '{clip_id}' must contain an image path")

            resolved_frame_idx = frame.get('frame_idx', frame_idx)
            annotation = frame.get('annotation')
            image_id = frame.get('image_id')
            if annotation is not None:
                corners_list = _normalize_annotation_to_corners(annotation)
                if image_id is None:
                    image_id = annotation.get('image_id')
            else:
                corners_list = annotations_by_clip_frame.get((clip_id, resolved_frame_idx), [[] for _ in range(8)])
                if image_id is None:
                    image_id = resolved_frame_idx

            normalized_frames.append({
                'frame_idx': resolved_frame_idx,
                'image_id': image_id,
                'image_path': image_path,
                'corners_list': corners_list,
                'source_type': 'video_corner_labels',
            })

        normalized_frames.sort(key=lambda item: item['frame_idx'])
        clips.append({
            'clip_id': clip_id,
            'frames': normalized_frames,
            'source_type': 'video_corner_labels',
        })

    return clips


def corners_to_heatmap(corners_list, height, width, sigma=2.0, num_classes=8):
    heatmap = torch.zeros((num_classes, height, width), dtype=torch.float32)
    xx, yy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    xx = xx.float()
    yy = yy.float()

    for class_id in range(num_classes):
        corners = corners_list[class_id]
        for corner in corners:
            x, y = corner
            x = torch.tensor(x, dtype=torch.float32).clip(0, width - 1)
            y = torch.tensor(y, dtype=torch.float32).clip(0, height - 1)
            gaussian = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            heatmap[class_id] = torch.maximum(heatmap[class_id], gaussian)
    return heatmap


def load_ply_corners(model_path):
    try:
        with open(model_path, 'r', encoding='utf-8', errors='strict') as f:
            lines = f.readlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unable to read PLY file as UTF-8 text: {model_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read PLY file: {model_path}") from exc

    vertex_count = None
    ply_format = None
    header_end_index = None

    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        if line == 'ply':
            continue
        if line.startswith('format '):
            ply_format = line.split(maxsplit=2)[1]
            if ply_format != 'ascii':
                raise ValueError(f"Unsupported PLY format '{ply_format}' in {model_path}")
            continue
        if line.startswith('element vertex '):
            try:
                vertex_count = int(line.split()[-1])
            except ValueError as exc:
                raise ValueError(f"Invalid vertex count in PLY file: {model_path}") from exc
            continue
        if line == 'end_header':
            header_end_index = line_index + 1
            break

    if header_end_index is None:
        raise ValueError(f"PLY file is missing end_header: {model_path}")
    if vertex_count is None:
        raise ValueError(f"PLY file is missing vertex element count: {model_path}")

    vertices = []
    for raw_line in lines[header_end_index:]:
        if len(vertices) >= vertex_count:
            break

        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError as exc:
            raise ValueError(f"Invalid vertex coordinates in PLY file: {model_path}") from exc

    if len(vertices) != vertex_count:
        raise ValueError(f"PLY file ended before reading {vertex_count} vertices: {model_path}")

    vertices = np.asarray(vertices, dtype=np.float32)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    return np.asarray([
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
    ], dtype=np.float32)


def project_corners_to_2d(corners_3d, R, t, K, width, height):
    corners_3d = np.asarray(corners_3d, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    K = np.asarray(K, dtype=np.float32)

    cam_points = (R @ corners_3d.T).T + t.reshape(1, 3)
    projected = np.full((corners_3d.shape[0], 2), np.nan, dtype=np.float32)
    visibility = []

    for idx, cam_point in enumerate(cam_points):
        z = float(cam_point[2])
        if z <= 0:
            visibility.append(0)
            continue

        pixel_h = K @ cam_point
        x = float(pixel_h[0] / pixel_h[2])
        y = float(pixel_h[1] / pixel_h[2])
        projected[idx] = [x, y]

        is_visible = 0 <= x < width and 0 <= y < height
        visibility.append(1 if is_visible else 0)

    return projected, visibility


def _frame_is_visible(scene_gt_info_entry, min_visib_fract=1e-6):
    if scene_gt_info_entry is None:
        return False

    if isinstance(scene_gt_info_entry, (list, tuple)):
        return any(_frame_is_visible(entry, min_visib_fract=min_visib_fract) for entry in scene_gt_info_entry)

    if not isinstance(scene_gt_info_entry, dict):
        return False

    visib_fract = scene_gt_info_entry.get('visib_fract')
    if visib_fract is None:
        return False

    try:
        return float(visib_fract) > float(min_visib_fract)
    except (TypeError, ValueError):
        return False


class VideoCornerDataset(Dataset):
    def __init__(self, dataset_root, seq_len=4, stride=1, transform=None, phase='train'):
        if seq_len <= 0 or stride <= 0:
            raise ValueError("seq_len and stride must both be positive")

        self.dataset_root = dataset_root
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform or _default_sequence_transform()
        self.phase = phase
        self.max_corners = 32

        self.clips = self._load_clips()
        self.sequence_starts = self._build_windows()
        print(f"找到 {len(self.clips)} 个 clips，可构造 {len(self.sequence_starts)} 个训练样本 ({phase}阶段)")

    def _load_clips(self):
        if _has_video_corner_labels(self.dataset_root):
            clips = _load_video_corner_label_clips(self.dataset_root)
            return [clip for clip in clips if len(clip['frames']) >= self.seq_len]

        raise FileNotFoundError(
            "video_corner_labels dataset not found; BOP fallback is not implemented in this task"
        )

    def _build_windows(self):
        windows = []
        for clip_index, clip in enumerate(self.clips):
            frame_count = len(clip['frames'])
            if frame_count < self.seq_len:
                continue

            for start_idx in range(0, frame_count - self.seq_len + 1, self.stride):
                windows.append({
                    'clip_index': clip_index,
                    'start_idx': start_idx,
                    'clip_id': clip['clip_id'],
                })

        return windows

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, idx):
        window = self.sequence_starts[idx]
        clip = self.clips[window['clip_index']]
        frames = clip['frames'][window['start_idx']:window['start_idx'] + self.seq_len]

        seq_images = []
        seq_heatmaps = []
        seq_corners_list = []
        seq_source_types = []
        seq_image_paths = []
        seq_frame_indices = []
        seq_image_ids = []

        for frame in frames:
            image_path = frame['image_path']
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.dataset_root, 'video_corner_labels', image_path)

            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size

            corners_list = frame.get('corners_list', [[] for _ in range(8)])

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            if not torch.is_tensor(image):
                image = transforms.ToTensor()(image)

            target_h, target_w = image.shape[1], image.shape[2]
            scale_x = float(target_w) / float(orig_w)
            scale_y = float(target_h) / float(orig_h)

            scaled_corners_list = [[] for _ in range(8)]
            for class_id in range(8):
                for x, y in corners_list[class_id]:
                    scaled_corners_list[class_id].append([x * scale_x, y * scale_y])

            heatmap = corners_to_heatmap(scaled_corners_list, target_h, target_w, num_classes=8)

            seq_images.append(image)
            seq_heatmaps.append(heatmap)
            seq_corners_list.append(scaled_corners_list)
            seq_source_types.append(frame.get('source_type', clip['source_type']))
            seq_image_paths.append(image_path)
            seq_frame_indices.append(frame['frame_idx'])
            seq_image_ids.append(frame.get('image_id', frame['frame_idx']))

        return {
            'clip_id': clip['clip_id'],
            'images': torch.stack(seq_images, dim=0),
            'heatmaps': torch.stack(seq_heatmaps, dim=0),
            'corners_list': seq_corners_list,
            'source_types': seq_source_types,
            'image_paths': seq_image_paths,
            'frame_indices': seq_frame_indices,
            'image_ids': seq_image_ids,
        }


class BOPCornerDataset(Dataset):
    def __init__(self, scene_dir, seq_len=4, stride=1, transform=None, phase='train'):
        self.scene_dir = scene_dir
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.phase = phase
        self.max_corners = 32

        coco_path = os.path.join(scene_dir, 'scene_gt_coco.json')
        with open(coco_path, 'r') as f:
            self.coco_data = json.load(f)

        self.images = sorted(self.coco_data['images'], key=lambda x: x['id'])
        self.annotations = self.coco_data['annotations']

        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

        self.sequence_starts = list(range(0, len(self.images) - self.seq_len + 1, self.stride))
        print(f"找到 {len(self.images)} 帧，可构造 {len(self.sequence_starts)} 个训练样本 ({phase}阶段)")

    def __len__(self):
        return len(self.sequence_starts)

    def __getitem__(self, idx):
        start_idx = self.sequence_starts[idx]
        seq_images = []
        seq_heatmaps = []
        seq_image_ids = []
        seq_image_paths = []

        for offset in range(self.seq_len):
            img_info = self.images[start_idx + offset]
            img_id = img_info['id']
            img_filename = img_info['file_name']

            if img_filename.startswith('rgb/'):
                img_filename = img_filename[4:]

            img_path = os.path.join(self.scene_dir, 'rgb', img_filename)
            image = Image.open(img_path)

            if image.mode in ['I', 'L']:
                img_array = np.array(image)
                if img_array.dtype == np.uint16:
                    img_array = (img_array / 65535.0 * 255).astype(np.uint8)
                elif img_array.dtype != np.uint8:
                    img_array = (img_array * 255).astype(np.uint8)
                image = Image.fromarray(img_array, mode='L').convert('RGB')
            else:
                image = image.convert('RGB')

            orig_w, orig_h = image.size

            anns = self.img_id_to_anns.get(img_id, [])
            corners_per_class = [[] for _ in range(8)]
            for ann in anns:
                if ann.get('ignore', False):
                    continue
                if 'keypoints' not in ann:
                    continue

                keypoints = ann['keypoints']
                for i in range(8):
                    k = i * 3
                    if k + 2 < len(keypoints):
                        x, y, v = keypoints[k:k + 3]
                        if v > 0:
                            corners_per_class[i].append([float(x), float(y)])

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            target_h, target_w = image.shape[1], image.shape[2]
            scale_x = float(target_w) / float(orig_w)
            scale_y = float(target_h) / float(orig_h)

            scaled_corners_per_class = [[] for _ in range(8)]
            for i in range(8):
                for x, y in corners_per_class[i]:
                    scaled_corners_per_class[i].append([x * scale_x, y * scale_y])

            heatmap = corners_to_heatmap(scaled_corners_per_class, target_h, target_w, num_classes=8)

            seq_images.append(image)
            seq_heatmaps.append(heatmap)
            seq_image_ids.append(img_id)
            seq_image_paths.append(img_path)

        return {
            'images': torch.stack(seq_images, dim=0),
            'heatmaps': torch.stack(seq_heatmaps, dim=0),
            'image_ids': seq_image_ids,
            'image_paths': seq_image_paths
        }


def collate_fn(batch):
    images = []
    heatmaps = []
    image_ids = []
    image_paths = []
    corners_list = []
    source_types = []
    clip_ids = []
    frame_indices = []

    for item in batch:
        images.append(item['images'])
        heatmaps.append(item['heatmaps'])
        image_ids.append(item.get('image_ids'))
        image_paths.append(item['image_paths'])
        corners_list.append(item.get('corners_list'))
        source_types.append(item.get('source_types'))
        clip_ids.append(item.get('clip_id'))
        frame_indices.append(item.get('frame_indices'))

    return {
        'images': torch.stack(images, dim=0),
        'heatmaps': torch.stack(heatmaps, dim=0),
        'image_ids': image_ids,
        'image_paths': image_paths,
        'corners_list': corners_list,
        'source_types': source_types,
        'clip_ids': clip_ids,
        'frame_indices': frame_indices,
    }


def create_bop_data_loader(scene_dir, batch_size=2, num_workers=0, phase='train', seq_len=4, stride=1):
    transform = _default_sequence_transform()

    dataset = BOPCornerDataset(
        scene_dir=scene_dir,
        seq_len=seq_len,
        stride=stride,
        transform=transform,
        phase=phase
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return data_loader
