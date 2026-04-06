# Video Corner Train Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training loader for `video-corner-detection` that reads a dataset root, prefers `video_corner_labels`, falls back to `train_pbr` BOP annotations, and emits sliding-window video samples with 8-channel heatmaps.

**Architecture:** Replace the current single-scene COCO-only loader with a dataset-root loader that normalizes two label backends into one clip/frame representation. Keep the public API close to the existing loader style, then generate sliding windows and batched tensors from that unified representation.

**Tech Stack:** Python, PyTorch, torchvision, PIL, NumPy, pytest

---

## File Structure

- Modify: `train_loader.py`
  - Main implementation file
  - Contains dataset-root scanning, backend selection, sliding-window indexing, heatmap generation, and public loader API
- Create: `tests/test_train_loader.py`
  - Focused unit/integration tests using tiny synthetic datasets under `tmp_path`
- Optional create only if needed during implementation: `tests/__init__.py`
  - Only if pytest import discovery needs it; otherwise skip

Keep everything in `train_loader.py` unless the file becomes too large during implementation. Do not introduce extra modules unless the test-driven work clearly requires them.

## Implementation Notes

- Follow `@superpowers:test-driven-development` strictly.
- Preserve the familiar return fields from the existing loader where practical: `images`, `heatmaps`, `corners_list`, `image_ids`, `image_paths`.
- Add the new sequence-aware metadata fields: `clip_ids`, `frame_indices`, `source_types`.
- Use `video_corner_labels` when both sources are available.
- For BOP fallback, do online 3D-corner projection from `models/*.ply` + `scene_gt.json` + `scene_camera.json` + `scene_gt_info.json`.
- Implement the accepted visibility rules:
  - projected point must be in-image and have positive depth
  - low-visibility frames may legitimately produce empty heatmaps
  - never replace missing keypoints with bbox corners

## Task 1: Replace the old single-scene assumptions with dataset-root indexing

**Files:**
- Modify: `train_loader.py`
- Test: `tests/test_train_loader.py`

- [ ] **Step 1: Write the failing test for dataset-root scanning and sliding windows**

```python
def test_dataset_root_scans_multiple_clips_and_builds_windows(tmp_path):
    dataset_root = build_video_corner_labels_dataset(tmp_path, clip_lengths=[4, 5])
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')

    assert len(dataset.clips) == 2
    assert len(dataset) == 3  # 1 window from len=4 clip, 2 from len=5 clip
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_dataset_root_scans_multiple_clips_and_builds_windows -v
```

Expected: FAIL because `VideoCornerDataset` still expects `scene_dir` / `scene_gt_coco.json`.

- [ ] **Step 3: Add minimal dataset-root indexing implementation**

Implement in `train_loader.py`:

```python
class VideoCornerDataset(Dataset):
    def __init__(self, dataset_root, seq_len=4, stride=1, transform=None, phase='train', ...):
        self.dataset_root = dataset_root
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.phase = phase
        self.clips = self._load_clips()
        self.windows = self._build_windows(self.clips)
```

Also add `_build_windows()` that skips clips shorter than `seq_len`.

- [ ] **Step 4: Run the test to verify it passes**

Run the same pytest command.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "feat: index video clips from dataset root"
```

## Task 2: Prefer `video_corner_labels` and normalize it into per-frame corners

**Files:**
- Modify: `train_loader.py`
- Test: `tests/test_train_loader.py`

- [ ] **Step 1: Write the failing tests for preferred backend and sample structure**

```python
def test_prefers_video_corner_labels_when_both_sources_exist(tmp_path):
    dataset_root = build_dual_source_dataset(tmp_path)
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')

    assert dataset.source_type == 'video_corner_labels'
    sample = dataset[0]
    assert sample['images'].shape == (4, 3, 256, 256)
    assert sample['heatmaps'].shape == (4, 8, 256, 256)
    assert len(sample['corners_list']) == 4
    assert sample['source_types'] == ['video_corner_labels'] * 4
```

```python
def test_short_video_corner_clip_is_skipped(tmp_path):
    dataset_root = build_video_corner_labels_dataset(tmp_path, clip_lengths=[3])
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')
    assert len(dataset) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_prefers_video_corner_labels_when_both_sources_exist tests/test_train_loader.py::test_short_video_corner_clip_is_skipped -v
```

Expected: FAIL because `video_corner_labels` backend does not exist yet.

- [ ] **Step 3: Implement minimal `video_corner_labels` backend parsing**

Add helpers in `train_loader.py`:

```python
def _has_video_corner_labels(dataset_root): ...
def _load_video_corner_label_clips(dataset_root): ...
def _normalize_annotation_to_corners(annotation): ...
```

Requirements:
- read `video_corner_labels/clips.json`
- read `video_corner_labels/annotations.json`
- build clip records with ordered frames
- produce `corners_per_class` as length-8 nested lists
- attach `image_path`, `image_id`, `frame_idx`, `clip_id`, `source_type`

- [ ] **Step 4: Run the tests to verify they pass**

Run the same pytest command.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "feat: load preferred video corner label clips"
```

## Task 3: Add failing BOP geometry tests before writing fallback code

**Files:**
- Modify: `train_loader.py`
- Test: `tests/test_train_loader.py`

- [ ] **Step 1: Write the failing tests for BOP fallback projection and visibility**

```python
def test_falls_back_to_bop_when_video_corner_labels_missing(tmp_path):
    dataset_root = build_bop_dataset(tmp_path, num_scenes=1, frames_per_scene=4)
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')

    assert dataset.source_type == 'bop_fallback'
    sample = dataset[0]
    assert sample['heatmaps'].shape == (4, 8, 256, 256)
    assert sample['source_types'] == ['bop_fallback'] * 4
```

```python
def test_bop_visibility_allows_empty_heatmap_for_low_visibility_frame(tmp_path):
    dataset_root = build_bop_dataset(tmp_path, num_scenes=1, frames_per_scene=4, low_visibility_frames={1})
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')

    sample = dataset[0]
    assert torch.count_nonzero(sample['heatmaps'][1]) == 0
```

```python
def test_projected_points_outside_image_are_dropped(tmp_path):
    corners_2d, visibility = project_corners_to_2d(
        corners_3d=np.array([[0,0,1],[10,0,1],[0,10,1],[10,10,1],[0,0,2],[10,0,2],[0,10,2],[10,10,2]], dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        t=np.array([10000, 10000, 0], dtype=np.float32),
        K=np.eye(3, dtype=np.float32),
        width=64,
        height=64,
    )
    assert visibility == [0] * 8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_falls_back_to_bop_when_video_corner_labels_missing tests/test_train_loader.py::test_bop_visibility_allows_empty_heatmap_for_low_visibility_frame tests/test_train_loader.py::test_projected_points_outside_image_are_dropped -v
```

Expected: FAIL because BOP parsing / projection helpers do not exist yet.

- [ ] **Step 3: Implement minimal geometry helpers in `train_loader.py`**

Add:

```python
def load_ply_corners(model_path): ...
def project_corners_to_2d(corners_3d, R, t, K, width, height): ...
def _frame_is_visible(scene_gt_info_entry, min_visib_fract=1e-6): ...
```

Requirements:
- compute 8 3D AABB corners from PLY vertices
- project using `R @ X + t`
- mark invisible when depth <= 0 or out of bounds
- allow zero-supervision frame if visibility test fails

- [ ] **Step 4: Run the tests to verify projection helper behavior passes**

Run the same pytest command.
Expected: at least the geometry-focused test passes; if the dataset fallback tests still fail, continue to Task 4.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "feat: add BOP projection helpers for video loader"
```

## Task 4: Implement the BOP fallback backend end-to-end

**Files:**
- Modify: `train_loader.py`
- Test: `tests/test_train_loader.py`

- [ ] **Step 1: Extend the failing tests to cover BOP clip parsing across scenes**

```python
def test_bop_root_discovers_multiple_scenes(tmp_path):
    dataset_root = build_bop_dataset(tmp_path, num_scenes=2, frames_per_scene=4)
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')
    assert len(dataset.clips) == 2
    assert len(dataset) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_bop_root_discovers_multiple_scenes -v
```

Expected: FAIL because scene discovery / JSON parsing is incomplete.

- [ ] **Step 3: Implement minimal BOP fallback clip loading**

Add helpers in `train_loader.py`:

```python
def _has_bop_root(dataset_root): ...
def _load_bop_clips(dataset_root): ...
def _load_scene_records(scene_dir, model_corner_cache, min_visib_fract=1e-6): ...
```

Requirements:
- scan `train_pbr/*`
- sort scene directories numerically when possible
- sort frame file names numerically
- read `scene_gt.json`, `scene_camera.json`, `scene_gt_info.json`
- cache model corners by `obj_id`
- for each frame, either emit 8 visible corners or empty lists
- include `clip_id`, `frame_idx`, `image_id`, `image_path`, `source_type='bop_fallback'`

- [ ] **Step 4: Run the BOP tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_falls_back_to_bop_when_video_corner_labels_missing tests/test_train_loader.py::test_bop_visibility_allows_empty_heatmap_for_low_visibility_frame tests/test_train_loader.py::test_bop_root_discovers_multiple_scenes -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "feat: add BOP fallback backend for video loader"
```

## Task 5: Restore the public loader API and collate behavior

**Files:**
- Modify: `train_loader.py`
- Test: `tests/test_train_loader.py`

- [ ] **Step 1: Write the failing tests for batch collation and API shape**

```python
def test_create_video_data_loader_batches_sequence_samples(tmp_path):
    dataset_root = build_video_corner_labels_dataset(tmp_path, clip_lengths=[4, 5])
    loader = create_video_data_loader(str(dataset_root), batch_size=2, num_workers=0, seq_len=4, stride=1)
    batch = next(iter(loader))

    assert batch['images'].shape == (2, 4, 3, 256, 256)
    assert batch['heatmaps'].shape == (2, 4, 8, 256, 256)
    assert len(batch['clip_ids']) == 2
    assert len(batch['frame_indices']) == 2
```

```python
def test_heatmap_generation_respects_scaled_keypoints(tmp_path):
    dataset_root = build_video_corner_labels_dataset(tmp_path, clip_lengths=[4], place_corner_at_center=True)
    dataset = VideoCornerDataset(str(dataset_root), seq_len=4, stride=1, phase='train')
    sample = dataset[0]

    center_value = sample['heatmaps'][0, 0, 128, 128]
    assert center_value > 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_create_video_data_loader_batches_sequence_samples tests/test_train_loader.py::test_heatmap_generation_respects_scaled_keypoints -v
```

Expected: FAIL because public API / collation still mismatch the new contract.

- [ ] **Step 3: Implement minimal public API completion**

In `train_loader.py`:
- rename or replace `create_bop_data_loader(...)` with `create_video_data_loader(...)`
- optionally keep `create_bop_data_loader = create_video_data_loader` as a compatibility alias if helpful
- make `collate_fn(...)` batch these fields:

```python
{
    'images': torch.stack(..., dim=0),
    'heatmaps': torch.stack(..., dim=0),
    'corners_list': [...],
    'image_ids': [...],
    'image_paths': [...],
    'clip_ids': [...],
    'frame_indices': [...],
    'source_types': [...],
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_train_loader.py::test_create_video_data_loader_batches_sequence_samples tests/test_train_loader.py::test_heatmap_generation_respects_scaled_keypoints -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "feat: finalize video train loader batching api"
```

## Task 6: Full regression pass before claiming completion

**Files:**
- Modify if needed: `train_loader.py`, `tests/test_train_loader.py`
- Verify only if clean: no code changes required

- [ ] **Step 1: Run the full loader test file**

Run:

```bash
python3 -m pytest tests/test_train_loader.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run a quick manual smoke check**

Run:

```bash
python3 - <<'PY'
from train_loader import VideoCornerDataset
# Replace with a tiny local fixture or temp dataset path during implementation
print('Import smoke check passed')
PY
```

Expected: no import errors.

- [ ] **Step 3: Refactor only if tests stay green**

Possible safe cleanups:
- deduplicate frame parsing helpers
- cache mesh corners cleanly
- improve error messages / statistics

Do not change behavior without adding a new failing test first.

- [ ] **Step 4: Re-run the full test file after refactor**

Run:

```bash
python3 -m pytest tests/test_train_loader.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add train_loader.py tests/test_train_loader.py
git commit -m "test: verify video train loader end-to-end"
```

## Final verification checklist

- [ ] Loader accepts dataset root, not just scene dir
- [ ] Default source priority is `video_corner_labels`
- [ ] BOP fallback works without `video_corner_labels`
- [ ] Sliding-window samples honor `seq_len=4`, `stride=1`
- [ ] Empty/low-visibility frames produce zero heatmaps without bbox fakery
- [ ] Batch output contains both tensors and sequence metadata
- [ ] `python3 -m pytest tests/test_train_loader.py -v` passes

