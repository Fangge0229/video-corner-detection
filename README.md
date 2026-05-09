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
├── legacy_tests/
│   └── test_train_loader.py
├── loss.py
├── model.py
├── roi_ops.py
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_model.py
│   ├── test_roi_ops.py
│   └── test_train.py
├── train.py
└── video.py
```

注意：当前项目目录中没有 `train_loader.py`。旧的 `test_train_loader.py` 已移到 `legacy_tests/`，并在模块级 skip，避免阻塞当前 ROI 主线测试。

## Module Overview

### `roi_ops.py`

负责 ROI 裁剪和角点坐标系转换：

- `sanitize_bbox(...)`
- `crop_and_resize_roi(...)`
- `corners_image_to_roi(...)`
- `corners_roi_to_image(...)`

`sanitize_bbox(...)` 已按当前教程修正为安全转换和 clamp 到图像范围，避免空 crop 进入 `cv2.resize(...)`。

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

`ROIEncoder` 使用 ResNet 卷积特征，`nn.Linear(...)` 和 `torchvision` 调用已经按当前教程修正。

### `loss.py`

损失函数包括：

- `corner_regression_loss(...)`
- `corner_confidence_loss(...)`
- `corner_loss(...)`

其中角点回归只在 `target_vis` 可见位置计算，置信度分支使用 `BCEWithLogitsLoss`。`corner_loss(...)` 返回统一的 `loss`、`loss_corner`、`loss_conf`。

### `train.py`

训练骨架包括：

- `train_one_step(...)`
- `average_stats(...)`
- `train_one_epoch(...)`
- `train(...)`

`train_one_epoch(...)` 保持 `meter` 为 list-of-dicts，然后直接调用 `average_stats(meter)`。

### `demo.py`

包含推理和可视化辅助函数：

- `infer_one_instance(...)`
- `draw_corners(...)`
- `update_sequence_memory(...)`
- `build_inference_sequence(...)`

推理侧已统一使用 `corners_roi_to_image(...)`。`build_inference_sequence(...)` 中使用 `sequence_memory.get(sequence_id, [])`。

### `dataset.py`

当前数据主线面向 BOP 风格目录，负责：

- 扫描 `train_pbr`
- 读取 `scene_gt.json`、`scene_camera.json`、`scene_gt_info.json`
- 从模型 PLY 构造 3D box 角点缓存
- 投影 3D box 角点到图像平面
- 裁剪 ROI 序列并返回训练 batch

已经按当前教程修正：

- `torch.is_finite` 已统一为 `torch.isfinite`
- `load_ply_vertices_ascii(...)` 中改为检查 `lines`
- `VideoCornerDataset.__init__` 中 `dataset_root` 已转换为 `Path`
- `__getitem__(...)` 的 `return` 已移出循环，返回完整 ROI 序列
- `corners_to_xyxy(...)` 返回 `None` 时跳过该 record

### `video.py`

保留的视频脚本，不是当前主线训练入口。

## Validation Status

当前按 2026-05-09 教程已经修正主要运行期错误，并补了最小主线测试：

1. `test_roi_ops.py`
2. `test_model.py`
3. `test_loss.py`
4. `test_train.py`
5. `test_dataset.py`

剩余工作：

1. 整理完整训练入口，包括配置、日志、验证和 checkpoint。
2. 用真实 BOP 数据跑一次 dataset 到 train step 的端到端验证。

## Testing

当前可以先做基础语法检查：

```bash
python3 -m py_compile roi_ops.py model.py loss.py train.py demo.py dataset.py
```

当前旧测试已迁移到 `legacy_tests/` 并跳过。主线测试运行方式：

```bash
pytest -q
```

## Recommended Reading Order

1. `README.md`
2. `roi_ops.py`
3. `dataset.py`
4. `model.py`
5. `loss.py`
6. `train.py`
7. `demo.py`

## Summary

当前项目已经形成了 ROI 序列角点检测的模块边界：ROI 操作、数据集、模型、损失、训练和 demo 分开维护。当前最小主线测试已经通过，下一步是整理完整训练入口，并用真实 BOP 数据做端到端验证。
