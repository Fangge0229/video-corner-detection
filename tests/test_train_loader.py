import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_loader import VideoCornerDataset


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)


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


def _write_clips_json(root: Path, clips_payload) -> Path:
    labels_root = root / "video_corner_labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    (labels_root / "clips.json").write_text(
        json.dumps(clips_payload, indent=2),
        encoding="utf-8",
    )
    return root


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
