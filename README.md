# video-corner-detection

`video-corner-detection` 是一个面向视频序列角点检测的实验仓库。当前代码重点已经从早期的单文件实现，重构为按职责拆分的训练草稿、数据草稿、损失函数和推理辅助模块。

这个仓库目前更适合被理解为“正在整理中的研究代码”，而不是已经稳定封装好的训练框架。

## Current Status

- 当前主线是把视频 ROI 序列送入时序模型，预测每帧 8 个角点及其置信度
- 旧的 `video_corner_detection.py` 已移除，代码正在拆分到更小的模块中
- `model.py`、`dataset.py`、`demo.py`、`roi_ops.py` 目前都还处于草稿阶段
- `train.py` 仍然只是一个轻量训练循环骨架，不是完整训练入口

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
├── train_loader.py
└── video.py
```

## Module Overview

### `train_loader.py`

仓库里目前最成熟的模块，负责：

- 从数据集根目录扫描样本
- 支持 `video_corner_labels` 和 `train_pbr`
- 生成固定长度的视频片段
- 构建训练需要的图像与 heatmap 张量

如果你要理解当前项目里最可靠的代码，优先看这个文件和对应测试。

### `train.py`

当前包含基础训练骨架：

- `train_one_step(...)`
- `train_one_epoch(...)`
- `train(...)`

它现在主要负责调用模型、计算损失并执行一次优化步骤，还没有包含完整的配置管理、日志、验证和 checkpoint 流程。

### `model.py`

当前实验模型方向是：

- `ROIEncoder`
- `TemporalTransformer`
- `CornerHead`
- `ConfidenseHead`
- `VideoCornerModel`

设计目标是先编码 ROI，再做时序建模，最后输出角点和置信度。这个文件目前仍是实验实现，接口和细节还没有完全稳定。

### `loss.py`

当前拆分出的损失模块包括：

- `corner_regression_loss`
- `corner_confidence_loss`
- `corner_loss`

这部分表达了新训练链路的目标形式，但和模型、数据之间的接口仍在继续对齐。

### `dataset.py`

这个文件开始承载时序数据组织相关逻辑，例如：

- 维护序列缓存
- 构造推理时输入序列
- 从 BOP 风格场景构建时序索引
- 根据 2D 角点生成包围框

目前仍是开发中的草稿，尚未形成稳定可直接复用的数据模块。

### `demo.py`

包含简单的推理和可视化辅助函数：

- `infer_one_instance(...)`
- `draw_corners(...)`

适合后续扩展成独立 demo，但现在还不是完整脚本入口。

### `roi_ops.py`

当前是预留文件，后续可用于放置 ROI 裁剪、坐标映射和图像空间变换相关操作。

### `video.py`

保留了一个与外部流程耦合较强的视频脚本，不是当前仓库的主开发入口。

## Recommended Reading Order

如果你是第一次接手这个仓库，推荐顺序：

1. `README.md`
2. `train_loader.py`
3. `tests/test_train_loader.py`
4. `train.py`
5. `model.py`
6. `dataset.py`
7. `demo.py`

## Testing

当前最明确的本地验证方式仍然是：

```bash
pytest -q
```

现有测试主要覆盖 `train_loader.py` 这条数据加载链路；新拆分出来的模型、训练和推理模块还缺少系统性测试。

## Current Limitations

- 训练入口尚未整理为可直接启动的完整脚本
- 新拆分模块仍然有明显实验性质
- 文档描述的是当前代码组织方向，不代表所有模块都已稳定可运行
- 仓库当前最可靠的能力仍然是数据加载与样本组织，不是完整训练闭环

## Summary

这次改动的核心方向，是把原来集中在单个文件里的视频角点检测实验代码拆分成更清晰的模块边界：数据、模型、损失、训练和 demo 分开维护。当前仓库已经更接近一个可继续演化的研究原型，但离稳定训练框架还有一段整理工作。
