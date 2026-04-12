# video-corner-detection

`video-corner-detection` 是一个面向视频序列的角点检测实验仓库，目标是把单帧角点检测扩展为适用于 clip / video 的时序建模流程。当前仓库的核心价值主要集中在数据读取与训练样本组织：它已经可以从视频角点标注或 BOP 风格数据中构造时序样本，并输出适合视频模型训练的张量批次。

这个仓库同时保留了几条不同阶段的建模尝试，包括单帧角点热图模型、基于 ConvLSTM 的视频角点模型，以及一个尚在草稿阶段的 Transformer 风格 `model.py`。如果你是第一次进入这个项目，建议优先关注 `train_loader.py` 和 `video_corner_detection.py`，再根据需要查看其他文件。

## Project Snapshot

- 任务目标：从视频帧序列中预测目标角点
- 当前最稳定模块：`train_loader.py`
- 当前已验证内容：训练数据加载、时序窗口切分、heatmap 监督生成、兼容 `video_corner_labels` 和 `train_pbr`
- 当前测试状态：`pytest -q` 本地验证 `35 passed`
- 当前代码形态：数据链路较完整，训练入口与新模型文件仍处于持续整理中

## What The Repository Contains

这个项目目前同时包含“可直接使用的代码”和“仍在迭代中的实验代码”。

### 已相对稳定的部分

- `train_loader.py`
  提供视频角点数据集与 `DataLoader` 构建逻辑，是当前最成熟、测试最完整的模块。

- `tests/test_train_loader.py`
  覆盖 `train_loader.py` 的主要行为，包括数据扫描、优先级选择、fallback、批处理输出和参数校验。

- `video_corner_detection.py`
  包含单帧角点热图模型和基于 ConvLSTM 的视频角点检测模型，也包含当前项目里实际定义好的损失函数实现。

### 仍在整理或实验中的部分

- `train.py`
  当前为空文件，尚未形成正式训练入口。

- `model.py`
  新建的时序模型草稿文件，意图是构建基于 ROI encoder + temporal transformer 的视频角点模型，但目前还不是可直接使用的稳定实现。

- `loss.py`
  当前为空文件，尚未承载正式损失逻辑。

- `video.py`
  主要是基于外部 HCCEPose 代码路径的视频推理脚本，不是一个独立、可直接复用的通用入口。

## Repository Structure

当前仓库的主要文件结构如下：

```text
video-corner-detection/
├── .gitignore
├── README.md
├── loss.py
├── model.py
├── tests/
│   └── test_train_loader.py
├── train.py
├── train_loader.py
├── video.py
└── video_corner_detection.py
```

## Core Workflow

从现有代码来看，推荐把项目理解成下面这条主链路：

```text
dataset_root
├── video_corner_labels/        # 推荐
│   ├── clips.json
│   └── annotations.json
└── train_pbr/                  # fallback

        |
        v
train_loader.py
        |
        v
VideoCornerDataset / DataLoader
        |
        v
video_corner_detection.py
```

也就是说，当前仓库最适合承担的角色是：

1. 读取视频或 BOP 风格数据
2. 切分为固定长度的时序窗口
3. 将角点监督转换为 heatmap
4. 将数据喂给视频角点检测模型

## Data Formats

`train_loader.py` 当前支持两类数据来源，并按优先级自动选择。

### 1. 首选：`video_corner_labels`

如果数据集根目录下存在 `video_corner_labels/clips.json`，则优先走这一套格式。

目录示例：

```text
dataset_root/
├── video_corner_labels/
│   ├── clips.json
│   └── annotations.json
└── ...
```

这个格式更贴近“视频 clip + 帧级标注”的使用方式，也是当前更推荐的输入。

### 2. 回退：BOP 风格 `train_pbr`

如果没有 `video_corner_labels/`，则自动回退到 `train_pbr/`。

目录示例：

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

这条路径适合从现有 BOP / PBR 数据流程直接接入，不必先导出 `video_corner_labels`。

## Data Loader Details

`train_loader.py` 当前完成了这个项目里最关键的工程化部分。

它支持：

- 从整个数据集根目录扫描可用数据
- 优先读取 `video_corner_labels`
- 在缺少该目录时自动回退到 `train_pbr`
- 将 clip 或 scene 按滑动窗口切成时序样本
- 将 8 个角点转换为 8 通道监督 heatmap
- 输出适配视频模型训练的批次张量

默认配置：

- `seq_len=4`
- `stride=1`
- 图像 resize 到 `256 x 256`

### Dataset 输出

`VideoCornerDataset` 返回单个样本时，核心字段包括：

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

### Batch 输出

`collate_fn` 聚合后的批次，核心张量结构为：

```python
{
    "images": Tensor[B, T, 3, 256, 256],
    "heatmaps": Tensor[B, T, 8, 256, 256],
    ...
}
```

这意味着下游模型只要接受 `(B, T, C, H, W)` 输入，就能较直接地对接当前 loader。

## Quick Start

如果你只想快速验证数据加载是否正常，建议从这里开始。

### 1. 创建 DataLoader

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

### 2. 直接使用 Dataset

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

## Model Files

这一部分更偏向开发维护说明，帮助你快速判断哪些代码适合继续扩展。

### `video_corner_detection.py`

这个文件目前包含两类模型实现：

- `CornerDetectionModel`
  单帧角点热图检测模型，结构上是 ResNet18 backbone + task head + upsampling。

- `VideoCornerDetectionModel`
  视频角点检测模型，流程为逐帧 CNN 特征提取，再用 `ConvLSTM` 做时序建模，最后逐帧输出 8 通道角点热图。

该文件还定义了当前真正可用的损失函数：

- `detection_loss`
- `temporal_loss`
- `total_loss`
- `criterion`

如果你要继续推进训练主线，优先从这里扩展通常最稳妥。

### `model.py`

`model.py` 是一个新的实验方向，意图大致是：

- 使用 `ROIEncoder` 做帧级特征编码
- 使用 `TemporalTransformer` 建模时序
- 预测角点与置信度

不过它目前仍处于草稿状态，至少包括以下特征：

- 尚未形成与现有训练链路的稳定对接
- 存在明显未完成或未验证实现
- 不适合作为当前项目默认模型入口

如果你打算继续发展 Transformer 路线，建议把它视为“待完善实验分支”而不是现成主干。

### `loss.py`

`loss.py` 目前为空文件，说明损失模块还没有从 `video_corner_detection.py` 中正式拆分出来。

### `train.py`

`train.py` 目前为空文件，表明仓库还缺一个正式的统一训练入口脚本。

## Backward Compatibility

为了兼容旧代码，`train_loader.py` 仍保留：

```python
from train_loader import create_bop_data_loader
```

它兼容的旧入口包括：

- 带 `scene_gt_coco.json` 的旧 COCO 风格场景目录
- 带 `scene_gt.json / scene_camera.json` 的单个 BOP scene 目录

这让旧数据流程不必一次性全部迁移。

## Validation And Tests

当前项目最可靠的验证方式是直接运行测试：

```bash
pytest -q
```

本地最新验证结果：

```text
35 passed
```

这些测试主要覆盖：

- 数据集根目录扫描
- `video_corner_labels` 优先级
- `train_pbr` fallback
- 多 scene 发现
- 相对路径数据集根目录
- 可见性过滤
- heatmap 缩放位置
- `collate_fn` 元数据聚合
- 旧接口兼容路径
- `seq_len` / `stride` 参数校验

## Current Limitations

为了避免 README 过度承诺，下面是当前仓库比较实际的状态总结。

- 训练入口尚未统一，`train.py` 还未落地
- `model.py` 仍是实验草稿，不应默认视为可训练主模型
- `loss.py` 还未承担正式逻辑
- `video.py` 依赖外部 HCCEPose 环境，不属于完全自包含脚本
- 当前项目最成熟的是“数据组织与监督生成”，不是完整的一键训练闭环

## Recommended Development Direction

如果接下来继续推进这个仓库，通常比较自然的顺序是：

1. 先把 `train.py` 建成正式训练入口
2. 明确默认模型使用 `video_corner_detection.py` 还是 `model.py`
3. 将损失函数、训练配置、验证逻辑逐步拆分成独立模块
4. 根据数据链路补充推理与可视化脚本

## Practical Advice For New Contributors

如果你是新接手这个仓库的人，最推荐的阅读顺序是：

1. 先看 `README.md`
2. 再看 `train_loader.py`
3. 然后看 `tests/test_train_loader.py`
4. 最后再看 `video_corner_detection.py` 和 `model.py`

这样能先理解“数据如何进入系统”，再理解“模型准备如何消费这些数据”。

## Summary

这个仓库现在最强的部分，是把视频角点任务的数据读取、样本切分和监督生成整理成了一个可测试、可复用的 loader；它最需要继续完善的部分，是统一训练入口以及明确后续主模型路线。

如果你的目标是尽快开始实验，建议先基于 `train_loader.py + video_corner_detection.py` 形成可运行训练闭环，再决定是否把 `model.py` 演化成下一代默认模型。
