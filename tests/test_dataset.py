import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import VideoCornerDataset, collate_fn, load_ply_vertices_ascii


def _write_ascii_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    body = [f"{x} {y} {z}" for x, y, z in vertices]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def _write_bop_dataset(root: Path, frame_count: int = 4) -> None:
    _write_ascii_ply(root / "models" / "obj_000001.ply")

    scene_dir = root / "train_pbr" / "000001"
    rgb_dir = scene_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    scene_gt = {}
    scene_camera = {}
    scene_gt_info = {}
    for image_id in range(frame_count):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(rgb_dir / f"{image_id:06d}.jpg"), image)
        scene_gt[str(image_id)] = [
            {
                "obj_id": 1,
                "cam_R_m2c": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "cam_t_m2c": [0.0, 0.0, 100.0],
            }
        ]
        scene_camera[str(image_id)] = {
            "cam_K": [
                100.0,
                0.0,
                32.0,
                0.0,
                100.0,
                32.0,
                0.0,
                0.0,
                1.0,
            ]
        }
        scene_gt_info[str(image_id)] = [{"visib_fract": 1.0}]

    (scene_dir / "scene_gt.json").write_text(json.dumps(scene_gt), encoding="utf-8")
    (scene_dir / "scene_camera.json").write_text(json.dumps(scene_camera), encoding="utf-8")
    (scene_dir / "scene_gt_info.json").write_text(json.dumps(scene_gt_info), encoding="utf-8")


def test_load_ply_vertices_ascii_reads_vertices(tmp_path):
    ply_path = tmp_path / "models" / "obj_000001.ply"
    _write_ascii_ply(ply_path)

    vertices = load_ply_vertices_ascii(ply_path)

    assert vertices.shape == (8, 3)


def test_video_corner_dataset_returns_full_roi_sequence(tmp_path):
    _write_bop_dataset(tmp_path)
    dataset = VideoCornerDataset(str(tmp_path), seq_len=4, roi_size=(32, 32))

    sample = dataset[0]
    batch = collate_fn([sample])

    assert sample["roi_images"].shape == (4, 3, 32, 32)
    assert sample["target_corners"].shape == (8, 2)
    assert sample["target_vis"].shape == (8,)
    assert sample["frame_ids"] == [0, 1, 2, 3]
    assert batch["roi_images"].shape == (1, 4, 3, 32, 32)
    assert torch.isfinite(batch["target_corners"]).all()
