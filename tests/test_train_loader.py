import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_loader
from train_loader import VideoCornerDataset, collate_fn


def _write_png(path: Path, size=(2, 2)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_ascii_ply(path: Path, vertices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _build_video_corner_labels_dataset(root: Path) -> Path:
    labels_root = root / "video_corner_labels"
    clips = []
    annotations = []

    for clip_idx, clip_len in enumerate((4, 5)):
        clip_id = f"clip_{clip_idx:03d}"
        frame_entries = []
        for frame_idx in range(clip_len):
            image_relpath = f"{clip_id}/rgb/{frame_idx:06d}.png"
            image_path = labels_root / image_relpath
            _write_png(image_path)
            frame_entries.append(
                {
                    "frame_idx": frame_idx,
                    "image_path": image_relpath,
                }
            )
            annotations.append(
                {
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "corners_per_class": [[] for _ in range(8)],
                }
            )

        clips.append(
            {
                "clip_id": clip_id,
                "frames": frame_entries,
            }
        )

    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / "clips.json").write_text(
        json.dumps({"clips": clips}, indent=2),
        encoding="utf-8",
    )
    (labels_root / "annotations.json").write_text(
        json.dumps({"annotations": annotations}, indent=2),
        encoding="utf-8",
    )
    return root


def _write_annotations_json(root: Path, payload) -> None:
    labels_root = root / "video_corner_labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / "annotations.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_clips_json(root: Path, clips_payload) -> Path:
    labels_root = root / "video_corner_labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / "clips.json").write_text(
        json.dumps(clips_payload, indent=2),
        encoding="utf-8",
    )
    return root


def _write_bop_scene(root: Path, frame_count: int) -> None:
    scene_root = root / "scene_000001"
    rgb_root = scene_root / "rgb"
    rgb_root.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    for image_id in range(frame_count):
        file_name = f"{image_id:06d}.png"
        _write_png(rgb_root / file_name, size=(64, 64))
        images.append({"id": image_id, "file_name": file_name})
        annotations.append(
            {
                "image_id": image_id,
                "keypoints": [0.0, 0.0, 2] * 8,
            }
        )

    (scene_root / "scene_gt_coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations}, indent=2),
        encoding="utf-8",
    )


def _write_bop_scene_at(root: Path, scene_name: str, frame_count: int, low_visibility_frames=None) -> None:
    low_visibility_frames = set(low_visibility_frames or set())
    scene_root = root / "train_pbr" / scene_name
    rgb_root = scene_root / "rgb"
    rgb_root.mkdir(parents=True, exist_ok=True)

    scene_gt = {}
    scene_camera = {}
    scene_gt_info = {}
    for image_id in range(frame_count):
        file_name = f"{image_id:06d}.png"
        _write_png(rgb_root / file_name, size=(64, 64))
        scene_gt[str(image_id)] = [
            {
                "obj_id": 1,
                "cam_R_m2c": [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0],
                "cam_t_m2c": [0.0, 0.0, 100.0],
            }
        ]
        scene_camera[str(image_id)] = {
            "cam_K": [100.0, 0.0, 32.0,
                      0.0, 100.0, 32.0,
                      0.0, 0.0, 1.0],
        }
        scene_gt_info[str(image_id)] = [
            {
                "visib_fract": 0.0 if image_id in low_visibility_frames else 1.0,
            }
        ]

    (scene_root / "scene_gt.json").write_text(
        json.dumps(scene_gt, indent=2),
        encoding="utf-8",
    )
    (scene_root / "scene_camera.json").write_text(
        json.dumps(scene_camera, indent=2),
        encoding="utf-8",
    )
    (scene_root / "scene_gt_info.json").write_text(
        json.dumps(scene_gt_info, indent=2),
        encoding="utf-8",
    )


def _write_bop_scene_with_mixed_visibility(root: Path) -> None:
    scene_root = root / "train_pbr" / "000001"
    rgb_root = scene_root / "rgb"
    rgb_root.mkdir(parents=True, exist_ok=True)

    _write_png(rgb_root / "000000.png", size=(64, 64))

    scene_gt = {
        "0": [
            {
                "obj_id": 1,
                "cam_R_m2c": [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0],
                "cam_t_m2c": [0.0, 0.0, 100.0],
            },
            {
                "obj_id": 2,
                "cam_R_m2c": [1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, 1.0],
                "cam_t_m2c": [30.0, 0.0, 100.0],
            },
        ]
    }
    scene_camera = {
        "0": {
            "cam_K": [100.0, 0.0, 32.0,
                      0.0, 100.0, 32.0,
                      0.0, 0.0, 1.0],
        }
    }
    scene_gt_info = {
        "0": [
            {"visib_fract": 1.0},
            {"visib_fract": 0.0},
        ]
    }

    (scene_root / "scene_gt.json").write_text(json.dumps(scene_gt, indent=2), encoding="utf-8")
    (scene_root / "scene_camera.json").write_text(json.dumps(scene_camera, indent=2), encoding="utf-8")
    (scene_root / "scene_gt_info.json").write_text(json.dumps(scene_gt_info, indent=2), encoding="utf-8")


def _build_bop_dataset(root: Path, frame_count: int = 4, low_visibility_frames=None) -> Path:
    dataset_root = root

    models_root = dataset_root / "models"
    _write_ascii_ply(
        models_root / "obj_000001.ply",
        vertices=[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    _write_bop_scene_at(dataset_root, "000001", frame_count, low_visibility_frames=low_visibility_frames)
    return dataset_root


def test_dataset_root_scans_multiple_clips_and_builds_windows(tmp_path):
    dataset_root = _build_video_corner_labels_dataset(tmp_path)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    assert len(dataset.clips) == 2
    assert len(dataset) == 3
    assert dataset.sequence_starts == [
        {"clip_index": 0, "start_idx": 0, "clip_id": "clip_000"},
        {"clip_index": 1, "start_idx": 0, "clip_id": "clip_001"},
        {"clip_index": 1, "start_idx": 1, "clip_id": "clip_001"},
    ]


@pytest.mark.parametrize("seq_len,stride", [(0, 1), (4, 0), (-1, 1), (4, -2)])
def test_dataset_rejects_non_positive_window_parameters(tmp_path, seq_len, stride):
    dataset_root = _build_video_corner_labels_dataset(tmp_path)

    with pytest.raises(ValueError, match="seq_len and stride must both be positive"):
        VideoCornerDataset(str(dataset_root), seq_len=seq_len, stride=stride, phase="train")


def test_dataset_rejects_invalid_clips_json_structure(tmp_path):
    with pytest.raises(ValueError, match="video_corner_labels/clips.json must contain a 'clips' list"):
        VideoCornerDataset(str(_write_clips_json(tmp_path, {"not_clips": []})), seq_len=4, stride=1, phase="train")


@pytest.mark.parametrize(
    "clips_payload, expected_message",
    [
        (
            {"clips": [{"clip_id": "clip_000"}]},
            "Clip entry 0 in video_corner_labels/clips.json must contain a 'frames' list",
        ),
        (
            {"clips": [{"clip_id": "clip_000", "frames": "not-a-list"}]},
            "Clip entry 0 in video_corner_labels/clips.json must contain a 'frames' list",
        ),
        (
            {"clips": [{"clip_id": "clip_000", "frames": [123]}]},
            "Frame entry 0 in clip 'clip_000' must be a string or object",
        ),
        (
            {"clips": [{"clip_id": "clip_000", "frames": [{}]}]},
            "Frame entry 0 in clip 'clip_000' must contain an image path",
        ),
    ],
)
def test_dataset_rejects_invalid_clip_and_frame_entries(tmp_path, clips_payload, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        VideoCornerDataset(str(_write_clips_json(tmp_path, clips_payload)), seq_len=4, stride=1, phase="train")


def test_prefers_video_corner_labels_when_both_sources_exist(tmp_path):
    dataset_root = _build_video_corner_labels_dataset(tmp_path)
    _write_bop_scene(dataset_root, frame_count=4)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    assert len(dataset) == 3

    sample = dataset[0]

    assert sample["images"].shape == (4, 3, 256, 256)
    assert sample["heatmaps"].shape == (4, 8, 256, 256)
    assert len(sample["corners_list"]) == 4
    assert sample["image_ids"] == [0, 1, 2, 3]
    assert sample["source_types"] == ["video_corner_labels"] * 4


def test_short_video_corner_clip_is_skipped(tmp_path):
    dataset_root = tmp_path
    labels_root = dataset_root / "video_corner_labels"
    clips = [
        {
            "clip_id": "clip_000",
            "frames": [
                {"frame_idx": 0, "image_path": "clip_000/rgb/000000.png"},
                {"frame_idx": 1, "image_path": "clip_000/rgb/000001.png"},
                {"frame_idx": 2, "image_path": "clip_000/rgb/000002.png"},
            ],
        }
    ]
    for frame_idx in range(3):
        _write_png(labels_root / f"clip_000/rgb/{frame_idx:06d}.png")

    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / "clips.json").write_text(
        json.dumps({"clips": clips}, indent=2),
        encoding="utf-8",
    )
    (labels_root / "annotations.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "clip_id": "clip_000",
                        "frame_idx": frame_idx,
                        "corners_per_class": [[] for _ in range(8)],
                    }
                    for frame_idx in range(3)
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_bop_scene(dataset_root, frame_count=4)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    assert len(dataset.clips) == 0
    assert len(dataset) == 0


def test_video_corner_labels_annotations_json_must_have_expected_top_level_shape(tmp_path):
    dataset_root = _build_video_corner_labels_dataset(tmp_path)
    _write_annotations_json(dataset_root, {"wrong_key": []})

    with pytest.raises(ValueError, match="video_corner_labels/annotations.json must contain an 'annotations' list"):
        VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")


def test_prefers_video_corner_labels_collate_fn_batches_image_ids_once(tmp_path):
    dataset_root = _build_video_corner_labels_dataset(tmp_path)
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["image_ids"] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]


def test_bop_root_discovers_multiple_scenes(tmp_path):
    dataset_root = _build_bop_dataset(tmp_path, frame_count=4)
    _write_bop_scene_at(dataset_root, "000002", frame_count=5)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    assert dataset.source_type == "bop_fallback"
    assert len(dataset.clips) == 2
    assert [clip["clip_id"] for clip in dataset.clips] == ["000001", "000002"]
    assert len(dataset) == 3
    assert dataset.sequence_starts == [
        {"clip_index": 0, "start_idx": 0, "clip_id": "000001"},
        {"clip_index": 1, "start_idx": 0, "clip_id": "000002"},
        {"clip_index": 1, "start_idx": 1, "clip_id": "000002"},
    ]


def test_bop_relative_dataset_root_loads_without_double_prepend(tmp_path, monkeypatch):
    relative_root = Path("relative_bop_dataset")
    dataset_root = tmp_path / relative_root
    _build_bop_dataset(dataset_root, frame_count=4)
    monkeypatch.chdir(tmp_path)

    dataset = VideoCornerDataset(str(relative_root), seq_len=4, stride=1, phase="train")

    sample = dataset[0]
    assert sample["image_paths"][0] == os.path.join(str(relative_root), "train_pbr", "000001", "rgb", "000000.png")
    assert sample["heatmaps"].shape == (4, 8, 256, 256)


def test_falls_back_to_bop_when_video_corner_labels_missing(tmp_path):
    dataset_root = _build_bop_dataset(tmp_path, frame_count=4)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    assert dataset.source_type == "bop_fallback"
    sample = dataset[0]
    assert sample["heatmaps"].shape == (4, 8, 256, 256)
    assert sample["source_types"] == ["bop_fallback"] * 4


def test_bop_visibility_allows_empty_heatmap_for_low_visibility_frame(tmp_path):
    dataset_root = _build_bop_dataset(tmp_path, frame_count=4, low_visibility_frames={1})

    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase="train")

    sample = dataset[0]
    assert torch.count_nonzero(sample["heatmaps"][1]) == 0


def test_bop_visibility_is_applied_per_annotation(tmp_path):
    dataset_root = tmp_path
    models_root = dataset_root / "models"
    _write_ascii_ply(
        models_root / "obj_000001.ply",
        vertices=[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    _write_ascii_ply(
        models_root / "obj_000002.ply",
        vertices=[
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    _write_bop_scene_with_mixed_visibility(dataset_root)

    dataset = VideoCornerDataset(str(dataset_root), seq_len=1, stride=1, phase="train")

    sample = dataset[0]

    visible_corners_2d, _ = train_loader.project_corners_to_2d(
        corners_3d=train_loader.load_ply_corners(models_root / "obj_000001.ply"),
        R=np.eye(3, dtype=np.float32),
        t=np.array([0.0, 0.0, 100.0], dtype=np.float32),
        K=np.array([100.0, 0.0, 32.0,
                    0.0, 100.0, 32.0,
                    0.0, 0.0, 1.0], dtype=np.float32),
        width=64,
        height=64,
    )
    hidden_corners_2d, _ = train_loader.project_corners_to_2d(
        corners_3d=train_loader.load_ply_corners(models_root / "obj_000002.ply"),
        R=np.eye(3, dtype=np.float32),
        t=np.array([30.0, 0.0, 100.0], dtype=np.float32),
        K=np.array([100.0, 0.0, 32.0,
                    0.0, 100.0, 32.0,
                    0.0, 0.0, 1.0], dtype=np.float32),
        width=64,
        height=64,
    )
    hidden_x, hidden_y = hidden_corners_2d[0]
    hidden_x = int(round(float(hidden_x)))
    hidden_y = int(round(float(hidden_y)))
    visible_x, visible_y = visible_corners_2d[0]
    visible_x = int(round(float(visible_x) * 4.0))
    visible_y = int(round(float(visible_y) * 4.0))
    hidden_x = int(round(float(hidden_x) * 4.0))
    hidden_y = int(round(float(hidden_y) * 4.0))

    assert sample["heatmaps"][0, 0, hidden_y, hidden_x] < 1e-6
    assert sample["heatmaps"][0, 0, visible_y, visible_x] > 0.01


def test_projected_points_outside_image_are_dropped(tmp_path):
    corners_2d, visibility = train_loader.project_corners_to_2d(
        corners_3d=np.array(
            [
                [0, 0, 1],
                [10, 0, 1],
                [0, 10, 1],
                [10, 10, 1],
                [0, 0, 2],
                [10, 0, 2],
                [0, 10, 2],
                [10, 10, 2],
            ],
            dtype=np.float32,
        ),
        R=np.eye(3, dtype=np.float32),
        t=np.array([10000, 10000, 0], dtype=np.float32),
        K=np.eye(3, dtype=np.float32),
        width=64,
        height=64,
    )

    assert visibility == [0] * 8
    assert isinstance(corners_2d, np.ndarray)
    assert corners_2d.shape == (8, 2)


def test_load_ply_corners_rejects_malformed_or_unsupported_ply(tmp_path):
    malformed_ply = tmp_path / "malformed.ply"
    malformed_ply.write_text(
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported PLY format"):
        train_loader.load_ply_corners(malformed_ply)

    non_utf8_ply = tmp_path / "non_utf8.ply"
    non_utf8_ply.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        b"\xff\xfe\xfd\n"
    )

    with pytest.raises(ValueError, match="Unable to read PLY file"):
        train_loader.load_ply_corners(non_utf8_ply)


def test_project_corners_to_2d_marks_behind_camera_points_invisible():
    corners_2d, visibility = train_loader.project_corners_to_2d(
        corners_3d=np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
        K=np.eye(3, dtype=np.float32),
        width=64,
        height=64,
    )

    assert visibility == [0, 1]
    assert np.isnan(corners_2d[0]).all()
    assert np.isfinite(corners_2d[1]).all()


def test_frame_is_visible_uses_dict_list_and_threshold_behavior():
    assert train_loader._frame_is_visible({"visib_fract": 0.5}, min_visib_fract=0.1) is True
    assert train_loader._frame_is_visible({"visib_fract": 0.1}, min_visib_fract=0.1) is False
    assert train_loader._frame_is_visible({"visib_fract": 0.01}, min_visib_fract=0.1) is False
    assert train_loader._frame_is_visible(
        [{"visib_fract": 0.0}, {"visib_fract": 0.2}],
        min_visib_fract=0.1,
    ) is True
    assert train_loader._frame_is_visible(
        [{"visib_fract": 0.0}, {"visib_fract": 0.01}],
        min_visib_fract=0.1,
    ) is False
