import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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

    for item in batch:
        images.append(item['images'])
        heatmaps.append(item['heatmaps'])
        image_ids.append(item['image_ids'])
        image_paths.append(item['image_paths'])

    return {
        'images': torch.stack(images, dim=0),
        'heatmaps': torch.stack(heatmaps, dim=0),
        'image_ids': image_ids,
        'image_paths': image_paths
    }


def create_bop_data_loader(scene_dir, batch_size=2, num_workers=0, phase='train', seq_len=4, stride=1):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

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
