# video-corner-detection

`video-corner-detection` 是一个面向视频 ROI 序列的 3D box 角点检测实验项目。当前主线目标是：从 BOP 风格数据集中构造连续 ROI 序列，模型只使用最后一个时间步输出 8 个角点坐标和 8 个角点可见性 logits。

这个仓库仍是整理中的研究代码，不是已经稳定封装好的训练框架。

## Current Mainline Contract

当前代码应统一使用下面的接口命名。

模型输出：

```python
{
    "corners_pred": corner_pred,
    "conf_logits_pred": confidence_pred,
}
```

训练 batch：

```python
{
    "roi_images": roi_images,
    "target_corners": target_corners,
    "target_vis": target_vis,
}
```

loss 返回：

```python
{
    "loss": total,
    "loss_corner": reg_loss,
    "loss_conf": conf_loss,
}
```

ROI 坐标转换函数：

```python
corners_image_to_roi(...)
corners_roi_to_image(...)
```

不要再使用旧命名：

```text
pred_corners
pred_conf_logits
moss_conf
corners_roi_to_img
pred_corners_rois
tranform
sequence_memory.get(..., default=...)
```

## Repository Structure

```text
video-corner-detection/
├── README.md
├── dataset.py
├── demo.py
├── loss.py
├── model.py
├── roi_ops.py
├── tests/
│   └── test_train_loader.py
├── train.py
└── video.py
```

注意：当前项目目录中没有 `train_loader.py`。`tests/test_train_loader.py` 仍然是旧测试，应移动到 `legacy_tests/` 或加 legacy skip。

## Module Overview

### `roi_ops.py`

负责 ROI 裁剪和角点坐标系转换：

- `sanitize_bbox(...)`
- `crop_and_resize_roi(...)`
- `corners_image_to_roi(...)`
- `corners_roi_to_image(...)`

仍需修正 `sanitize_bbox(...)` 中的 `dtupe` 拼写错误，并保证 bbox 能从 tensor/list/ndarray 安全转换和 clamp 到图像范围。

### `model.py`

模型结构包括：

- `ROIEncoder`
- `TemporalTransformer`
- `CornerHead`
- `ConfidenceHead`
- `VideoCornerModel`

主线设计是先用 ResNet 卷积特征编码每帧 ROI，再用 Transformer 做时序建模，最后只用最后一个时间步预测：

- `corners_pred`: shape `[B, 8, 2]`
- `conf_logits_pred`: shape `[B, 8]`

仍需修正 `nn.Linear(..., out_channels=...)`、`nn.Linear(in_channels=..., out_channels=...)` 和 `torchvision.models.resnet18(...)` 的调用方式。

### `loss.py`

损失函数包括：

- `corner_regression_loss(...)`
- `corner_confidence_loss(...)`
- `corner_loss(...)`

其中角点回归只在 `target_vis` 可见位置计算，置信度分支使用 `BCEWithLogitsLoss`。当前仍需修正 `corner_loss(...)` 里的未定义变量 `reg`，应使用 `reg_loss`。

### `train.py`

训练骨架包括：

- `train_one_step(...)`
- `average_stats(...)`
- `train_one_epoch(...)`
- `train(...)`

`train_one_epoch(...)` 应保持 `meter` 为 list-of-dicts，然后直接调用 `average_stats(meter)`。不要把它转换成 dict-of-lists，否则会和 `average_stats(...)` 的输入格式不匹配。

### `demo.py`

包含推理和可视化辅助函数：

- `infer_one_instance(...)`
- `draw_corners(...)`
- `update_sequence_memory(...)`
- `build_inference_sequence(...)`

仍需统一使用 `corners_roi_to_image(...)`，并修正 `tranform` 拼写错误。`build_inference_sequence(...)` 中应使用 `sequence_memory.get(sequence_id, [])`。

### `dataset.py`

当前数据主线面向 BOP 风格目录，负责：

- 扫描 `train_pbr`
- 读取 `scene_gt.json`、`scene_camera.json`、`scene_gt_info.json`
- 从模型 PLY 构造 3D box 角点缓存
- 投影 3D box 角点到图像平面
- 裁剪 ROI 序列并返回训练 batch

仍需重点检查：

- `torch.is_finite` 应统一为 `torch.isfinite`
- `load_ply_vertices_ascii(...)` 里 `if not line` 应改为检查 `lines`
- `VideoCornerDataset.__init__` 中 `dataset_root` 应先转换为 `Path`
- `__getitem__(...)` 的 `return` 目前缩进在循环内部，会导致只返回第 1 帧 ROI，而不是完整序列
- 当 `corners_to_xyxy(...)` 返回 `None` 时，后续 ROI 裁剪需要有明确处理策略

### `video.py`

保留的视频脚本，不是当前主线训练入口。

## Remaining Fix Checklist

当前再次检查后，还需要修正：

1. `tests/test_train_loader.py` 仍引用不存在的 `train_loader.py`。
2. `roi_ops.py` 中 `torch.as_tensor(..., dtupe=...)` 会运行时报错。
3. `model.py` 中 `torchvision` 未按 import 名称使用，且 `nn.Linear` 参数名错误。
4. `loss.py` 中 `total = reg + ...` 使用了未定义变量。
5. `demo.py` 中仍调用不存在的 `corners_roi_to_img(...)`，并引用未定义的 `tranform`。
6. `dataset.py` 中仍有运行期问题，尤其是 `line` 未定义、`dataset_root / "models"` 可能因字符串路径报错、`__getitem__` 提前 return。
7. 需要补充新的主线测试：`test_roi_ops.py`、`test_model.py`、`test_loss.py`、`test_train.py`、`test_dataset.py`。

## Testing

当前可以先做基础语法检查：

```bash
python3 -m py_compile roi_ops.py model.py loss.py train.py demo.py dataset.py
```

在旧测试迁移或 skip 之前，直接运行：

```bash
pytest -q
```

大概率会因为 `tests/test_train_loader.py` 引用不存在的 `train_loader.py` 而失败。

## Recommended Reading Order

1. `README.md`
2. `roi_ops.py`
3. `dataset.py`
4. `model.py`
5. `loss.py`
6. `train.py`
7. `demo.py`

## Summary

当前项目已经形成了 ROI 序列角点检测的模块边界：ROI 操作、数据集、模型、损失、训练和 demo 分开维护。但代码还没有进入稳定可训练状态，下一步应优先修正上面的运行期错误，再补齐主线单元测试。
