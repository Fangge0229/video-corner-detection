# Video Corner Train Loader Design

**Date:** 2026-04-06

## Goal

为 `video-corner-detection` 设计一个新的训练数据加载器，使其能够从数据集根目录读取由 `s2_p1_gen_pbr_data_clip.py` 生成的 clip 数据，优先消费 `video_corner_labels`，并在缺失时回退到 `train_pbr` 的原始 BOP 标注来在线计算 8 个角点热图监督信号。

## Background

参考实现位于 `Desktop/corner-detection/train_loader_bop.py`，其特点是：

- 使用 `Dataset + DataLoader + collate_fn`
- 读取图像并 resize 到 `256x256`
- 将 8 个角点转换为 8 通道 heatmap
- 返回图像、heatmap、角点列表、图像路径等字段

当前 `video-corner-detection/train_loader.py` 已支持“单个 scene_dir + COCO keypoints + 时序窗口”的基础模式，但目标数据源已变为：

1. 数据集根目录下的 `video_corner_labels/annotations.json` 与 `clips.json`
2. 数据集根目录下的 `train_pbr/<scene_id>/...` BOP 场景目录
3. 数据集根目录下的 `models/*.ply` 模型文件（仅在 BOP 回退模式使用）

## User Decisions Captured

- 输入粒度：**整个数据集根目录**，不是单个 scene 目录
- 标签源：**同时支持两种来源**
- 优先级：**默认优先 `video_corner_labels`**
- 样本单位：**滑动窗口子序列**
- 默认窗口参数：**`seq_len=4`, `stride=1`**
- 可见性策略：
  1. 关键点投影后在图像内且深度为正，记为可见
  2. 若 `scene_gt_info` 表明目标几乎不可见，允许该帧 heatmap 为空
  3. 不强行使用 bbox 替代角点

## Output Contract

新的 loader 对训练代码暴露统一接口：

```python
dataset = VideoCornerDataset(
    dataset_root="/path/to/dataset",
    seq_len=4,
    stride=1,
    transform=...,
    phase="train",
)

loader = create_video_data_loader(
    dataset_root="/path/to/dataset",
    batch_size=2,
    num_workers=0,
    phase="train",
    seq_len=4,
    stride=1,
)
```

每个 batch 预计返回：

```python
{
    "images": Tensor[B, T, 3, H, W],
    "heatmaps": Tensor[B, T, 8, H, W],
    "corners_list": list,
    "image_ids": list,
    "image_paths": list,
    "clip_ids": list,
    "frame_indices": list,
    "source_types": list,
}
```

其中：

- `B` = batch size
- `T` = `seq_len`
- `H, W` = 变换后的图像大小，默认 `256x256`

## Architecture

采用统一 loader + 双后端解析器架构。

### 1. `VideoCornerDataset`

职责：

- 接收 `dataset_root`
- 检测可用标签源
- 构建统一的 clip 元数据
- 基于 clip 生成滑动窗口样本索引
- 在 `__getitem__` 中返回统一格式的图像序列与 heatmap 序列

### 2. `VideoCornerLabelsBackend`

职责：

- 读取 `video_corner_labels/annotations.json`
- 读取 `video_corner_labels/clips.json`
- 将标注转换为统一的 clip/frame 结构
- 提供逐帧 8 角点坐标

预期行为：

- 按 clip 组织帧
- 按 frame 顺序输出
- 保留角点与可见性信息
- 只要该后端可用，就默认被优先选中

### 3. `BOPFallbackBackend`

职责：

- 遍历 `train_pbr/<scene_id>`
- 读取每个 scene 的：
  - `rgb/`
  - `scene_gt.json`
  - `scene_camera.json`
  - `scene_gt_info.json`
- 读取 `models/*.ply`
- 计算模型 3D 轴对齐包围盒 8 个角点
- 使用 `cam_R_m2c`, `cam_t_m2c`, `cam_K` 在线投影到 2D
- 产出与 `video_corner_labels` 同构的逐帧角点表示

### 4. 中间统一表示

无论来源如何，内部统一为：

```python
clip_record = {
    "clip_id": str,
    "source_type": "video_corner_labels" | "bop_fallback",
    "frames": [
        {
            "frame_idx": int,
            "image_id": int | str,
            "image_path": str,
            "corners_per_class": list[list[list[float]]],
            "visibility": list[int] | None,
            "bbox": list[float] | None,
        },
        ...
    ]
}
```

说明：

- `corners_per_class` 保持与参考项目一致的形状：长度为 8 的列表，每类下是该角点类别的若干坐标点
- 对单实例目标，通常每个类别只有 0 或 1 个点
- 当帧不可监督时，对应类别列表为空

## Data Flow

### A. `video_corner_labels` 路径

1. 扫描 `dataset_root/video_corner_labels`
2. 读取 `clips.json` 建立 clip 与帧集合映射
3. 读取 `annotations.json` 建立逐帧角点字典
4. 合并为统一 `clip_record`
5. 对每个 clip 生成滑动窗口索引
6. 读取图像并生成 heatmap

### B. `train_pbr` 回退路径

1. 扫描 `dataset_root/train_pbr/*`
2. 读取每个 scene 的 RGB 与 JSON
3. 读取 `dataset_root/models/*.ply`
4. 为每个 `obj_id` 缓存 3D 8 角点
5. 按帧读取 pose 与 camera 参数
6. 根据可见性规则生成逐帧角点
7. 组织成 `clip_record`
8. 生成滑动窗口索引并输出 heatmap

## Sliding Window Policy

默认参数：

- `seq_len=4`
- `stride=1`

对一个长度为 `N` 的 clip：

- 若 `N < seq_len`，该 clip 默认跳过
- 若 `N >= seq_len`，窗口起点为：
  - `0, stride, 2*stride, ...`
  - 直到 `start + seq_len <= N`

例如 `N=24` 时，默认得到 `21` 个训练样本。

## Image Preprocessing

延续参考项目风格：

- 读取 RGB 图像
- 强制转为 `RGB`
- `Resize((256, 256))`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

角点也按同样比例缩放到目标尺寸后再生成 heatmap。

## Heatmap Generation

继续沿用 `corners_to_heatmap(...)` 思路：

- 输出 shape：`(8, H, W)`
- 默认 `sigma=2.0`
- 每个类别单独一张 heatmap
- 多个点时取逐像素最大值

时序样本的 heatmap shape 为：

```python
(T, 8, H, W)
```

## Visibility Rules

已确认采用以下规则：

1. **投影点在图像内且深度为正**：记为可见
2. **若 `scene_gt_info` 显示目标几乎不可见**：允许该帧 heatmap 为空
3. **绝不使用 bbox 伪造角点**

建议具体落地时补充一个可配置阈值，例如：

- `min_visib_fract = 1e-6` 或 `0.0`

实现时，如果一帧没有任何有效角点，应允许返回全零 heatmap，而不是报错。

## Error Handling

### 硬错误（直接报错）

- 数据集根目录不存在
- 同时缺失 `video_corner_labels` 与 `train_pbr`
- BOP 回退模式下缺失 `models/`
- scene 缺失关键 JSON：`scene_gt.json` / `scene_camera.json`

### 软错误（跳过并记录）

- 某个 clip 帧数不足 `seq_len`
- 某张图像缺失
- 某个 `obj_id` 没有匹配模型
- 单帧无可用角点
- `video_corner_labels` 中个别 annotation 不完整

建议在构建索引阶段打印汇总统计，而不要在训练时反复刷屏。

## Testing Strategy

采用 TDD，至少覆盖以下行为：

1. **优先级测试**
   - 同时存在两类标签源时，默认选择 `video_corner_labels`
2. **回退测试**
   - 缺失 `video_corner_labels` 时，自动回退到 BOP
3. **根目录扫描测试**
   - 能自动发现多个 `train_pbr/<scene>`
4. **滑动窗口测试**
   - `seq_len=4, stride=1` 时窗口数正确
5. **长度不足测试**
   - 帧数不足的 clip 被跳过
6. **heatmap 形状测试**
   - 返回 `(T, 8, 256, 256)`
7. **batch 形状测试**
   - `collate_fn` 输出 `(B, T, 3, H, W)` 与 `(B, T, 8, H, W)`
8. **BOP 投影测试**
   - 已知简单 pose + K 下投影结果可预测
9. **可见性测试**
   - 出界点、负深度点、低可见帧返回空监督
10. **字段兼容性测试**
    - 保留 `image_paths`, `image_ids`, `corners_list` 等关键字段

## Files Expected to Change

核心应只涉及少量文件：

- 修改：`train_loader.py`
- 新增：`tests/test_train_loader.py` 或同类测试文件
- 如有必要新增小型辅助模块（例如 `bop_geometry.py`），但优先保持最小改动

## Why This Design

该设计兼顾了：

- 与参考项目接口风格的一致性
- 对新数据流程 `video_corner_labels` 的优先支持
- 对 `s2_p1_gen_pbr_data_clip.py` 原始输出的兼容性
- 对时序模型训练所需的 clip/window 组织能力
- 对未来不同数据集与不同 clip 长度的扩展性

## Non-Goals

本次不包含：

- 训练脚本重构
- 模型结构调整
- 离线标签转换器重写
- 可视化工具新增
- 数据增强策略扩展

