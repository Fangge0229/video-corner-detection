# video-corner-detection

这是一个从 `corner-detection` 演化而来的时序角点检测项目，目标是把**单帧角点检测**扩展到**视频/clip 角点检测**。

当前项目已经补齐了新的 `train_loader.py`，可直接读取由 HCCEPose clip 数据流程导出的训练数据，并同时兼容两类数据来源：

1. **优先**读取 `video_corner_labels/`
2. 若不存在，则**回退**读取 `train_pbr/` BOP 风格数据

---

## 1. 支持的数据目录格式

### 1.1 首选：`video_corner_labels`

数据集根目录示例：

```text
dataset_root/
├── video_corner_labels/
│   ├── clips.json
│   └── annotations.json
└── ...
```

这是当前推荐的数据格式。

### 1.2 回退：BOP `train_pbr`

如果没有 `video_corner_labels/`，则会自动回退到：

```text
dataset_root/
├── models/
│   ├── obj_000001.ply
│   └── ...
└── train_pbr/
    ├── 000001/
    │   ├── rgb/
    │   ├── scene_gt.json
    │   ├── scene_camera.json
    │   └── scene_gt_info.json
    └── 000002/
        └── ...
```

这个结构对应 `s2_p1_gen_pbr_data_clip.py` 的输出。

---

## 2. train_loader 的主要能力

`train_loader.py` 当前支持：

- 从**数据集根目录**构造训练集，而不是只读单个 scene
- 优先使用 `video_corner_labels`
- 自动回退到 BOP `train_pbr`
- 将一个 clip 切成**滑动窗口时序样本**
- 自动把 8 个角点转换成 **8 通道 heatmap**
- 输出适合视频模型训练的 batch：
  - `images: (B, T, 3, 256, 256)`
  - `heatmaps: (B, T, 8, 256, 256)`

默认参数：

- `seq_len=4`
- `stride=1`

---

## 3. 推荐使用方式

### 3.1 新接口：读取整个数据集根目录

```python
from train_loader import create_video_data_loader

loader = create_video_data_loader(
    dataset_root="/path/to/dataset_root",
    batch_size=2,
    num_workers=0,
    phase="train",
    seq_len=4,
    stride=1,
)

batch = next(iter(loader))
print(batch["images"].shape)    # (B, T, 3, 256, 256)
print(batch["heatmaps"].shape)  # (B, T, 8, 256, 256)
```

### 3.2 直接使用 Dataset

```python
from train_loader import VideoCornerDataset

dataset = VideoCornerDataset(
    dataset_root="/path/to/dataset_root",
    seq_len=4,
    stride=1,
    phase="train",
)

sample = dataset[0]
print(sample["images"].shape)    # (T, 3, 256, 256)
print(sample["heatmaps"].shape)  # (T, 8, 256, 256)
```

---

## 4. 输出字段说明

### 4.1 单个 sample（`dataset[i]`）

`VideoCornerDataset` 返回一个字典，主要字段如下：

```python
{
    "clip_id": str,
    "images": Tensor[T, 3, 256, 256],
    "heatmaps": Tensor[T, 8, 256, 256],
    "corners_list": list,
    "source_types": list,
    "image_paths": list,
    "frame_indices": list,
    "image_ids": list,
}
```

含义：

- `clip_id`：该时序样本所属 clip/scene 标识
- `images`：长度为 `T` 的图像序列
- `heatmaps`：长度为 `T` 的 8 通道角点监督热图
- `corners_list`：每帧的角点坐标（已按 resize 比例缩放）
- `source_types`：每帧来源，常见值：
  - `video_corner_labels`
  - `bop_fallback`
  - `bop_legacy_coco`
- `image_paths`：每帧图像路径
- `frame_indices`：每帧在 clip 中的帧号
- `image_ids`：每帧对应的 image id

### 4.2 一个 batch（`next(iter(loader))`）

`collate_fn` 返回：

```python
{
    "images": Tensor[B, T, 3, 256, 256],
    "heatmaps": Tensor[B, T, 8, 256, 256],
    "corners_list": list,
    "image_ids": list,
    "image_paths": list,
    "clip_ids": list,
    "frame_indices": list,
    "source_types": list,
}
```

---

## 5. 向后兼容接口

为了兼容旧代码，保留了：

```python
from train_loader import create_bop_data_loader
```

它支持两种旧入口：

1. **旧 COCO 场景目录**（带 `scene_gt_coco.json`）
2. **单个 BOP scene 目录**（带 `scene_gt.json` / `scene_camera.json`）

示例：

```python
loader = create_bop_data_loader(
    scene_dir="/path/to/train_pbr/000001",
    batch_size=1,
    num_workers=0,
    phase="val",
    seq_len=4,
    stride=1,
)
```

---

## 6. 参数约束

以下参数必须为正数：

- `seq_len > 0`
- `stride > 0`

否则会抛出：

```python
ValueError("seq_len and stride must both be positive")
```

---

## 7. 当前测试状态

当前 `train_loader.py` 已覆盖并通过以下测试：

- 数据集根目录扫描
- `video_corner_labels` 优先级
- `train_pbr` BOP fallback
- 多 scene 发现
- 相对路径数据集根目录
- 每对象可见性处理
- heatmap 缩放位置正确性
- `collate_fn` 批处理元数据
- 旧 `create_bop_data_loader(...)` 兼容路径
- 非法 `seq_len/stride` 参数校验

本地最新验证结果：

```text
35 passed
```

---

## 8. 适用数据链路

这套 loader 主要面向下面这条数据链路：

```text
s2_p1_gen_pbr_data_clip.py
-> train_pbr/
-> (可选) s2_p2_export_video_corner_labels.py
-> video_corner_labels/
-> video-corner-detection/train_loader.py
```

也就是说：

- 如果你已经导出了 `video_corner_labels`，会优先读取它
- 如果还没导出，只要有 `train_pbr + models`，也可以直接训练

