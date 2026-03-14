# Video Corner Detection with ConvLSTM

基于 ConvLSTM 的视频角点检测模型，结合 CNN 特征提取和 LSTM 时序建模。

## 项目结构

```
video-corner-detection/
├── video_corner_detection.py  # 主代码文件
├── README.md                  # 项目说明
└── .gitignore                 # Git 忽略文件
```

## 模型架构

### 1. 单帧角点检测模型 (CornerDetectionModel)

```
输入图像 (3, H, W)
    ↓
Backbone (ResNet18) → 特征图 (256, H/16, W/16)
    ↓
Task Head → 角点热图 (8, H/16+4, W/16+4)
    ↓
Upsampling → 最终输出 (8, H, W)
```

### 2. ConvLSTM Cell

将标准 LSTM 的全连接层替换为卷积层，保留空间信息：

```
输入 x_t + 隐藏状态 h_{t-1}
    ↓
拼接 + 卷积 (Conv2d)
    ↓
分割为4个门: i (输入门), f (遗忘门), o (输出门), g (候选记忆)
    ↓
细胞状态更新: C_t = f * C_{t-1} + i * g
隐藏状态更新: h_t = o * tanh(C_t)
```

### 3. 多层 ConvLSTM

支持多层堆叠，逐层提取更抽象的时空特征：

```
Layer 1: 输入 → ConvLSTM → 输出1
Layer 2: 输出1 → ConvLSTM → 输出2
Layer 3: 输出2 → ConvLSTM → 输出3 (最终输出)
```

### 4. 视频角点检测模型 (VideoCornerDetectionModel)

```
视频序列 (T, 3, H, W)
    ↓
CNN Backbone (每帧) → 特征序列 (T, 256, H', W')
    ↓
特征投影 → (T, hidden_dim, H', W')
    ↓
ConvLSTM (多层) → 时序特征 (T, hidden_dim, H', W')
    ↓
特征反投影 → (T, 256, H', W')
    ↓
Task Head (每帧) → 角点热图序列 (T, 8, H, W)
```

## 核心组件

### ConvLSTMCell

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        # input_dim: 输入通道数
        # hidden_dim: 隐藏状态通道数
        # kernel_size: 卷积核大小
```

**关键特性：**
- 使用卷积代替全连接，保留空间结构
- 4个门控机制（输入门、遗忘门、输出门、候选记忆）
- Xavier 初始化确保训练稳定

### ConvLSTM (多层)

```python
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        # num_layers: LSTM 层数
```

**关键特性：**
- 支持多层堆叠
- 每层独立维护隐藏状态和细胞状态
- 第 i 层输出作为第 i+1 层输入

## 损失函数

### 1. 检测损失 (Detection Loss)

```python
def detection_loss(output, target):
    return F.mse_loss(output, target)
```

### 2. 时序平滑损失 (Temporal Loss)

```python
def temporal_loss(output):
    diff = output[:, 1:] - output[:, :-1]
    return (diff ** 2).mean()
```

鼓励相邻帧预测的一致性。

### 3. 总损失

```python
def total_loss(output, target, lambda_temporal=0.2):
    return detection_loss(output, target) + lambda_temporal * temporal_loss(output)
```

## 使用示例

### 单帧检测

```python
from video_corner_detection import CornerDetectionModel

model = CornerDetectionModel(H=224, W=224)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)  # (1, 8, 224, 224)
```

### 视频检测

```python
from video_corner_detection import VideoCornerDetectionModel

model = VideoCornerDetectionModel(
    H=224, W=224,
    hidden_dim=64,
    num_layers=2
)
video = torch.randn(1, 5, 3, 224, 224)  # (batch, seq, C, H, W)
output = model(video)  # (1, 5, 8, 224, 224)
```

### ConvLSTM 独立使用

```python
from video_corner_detection import ConvLSTM

convlstm = ConvLSTM(
    input_dim=64,
    hidden_dim=64,
    kernel_size=3,
    num_layers=2
)
video_features = torch.randn(2, 10, 64, 32, 32)  # (batch, seq, C, H, W)
output, states = convlstm(video_features)
```

## 运行测试

```bash
python video_corner_detection.py
```

输出示例：

```
============================================================
测试单帧角点检测模型
============================================================
输入形状: torch.Size([1, 3, 224, 224])
输出形状: torch.Size([1, 8, 224, 224])

============================================================
测试 ConvLSTM Cell
============================================================
输入 x 形状: torch.Size([2, 64, 32, 32])
输出 h_next 形状: torch.Size([2, 64, 32, 32])
输出 c_next 形状: torch.Size([2, 64, 32, 32])

============================================================
测试多层 ConvLSTM
============================================================
输入形状: torch.Size([2, 10, 64, 32, 32])
输出形状: torch.Size([2, 10, 64, 32, 32])
层数: 2

============================================================
测试视频角点检测模型
============================================================
输入形状: torch.Size([1, 5, 3, 224, 224])
输出形状: torch.Size([1, 5, 8, 224, 224])
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11

安装：

```bash
pip install torch torchvision
```

## 技术要点

### 1. 为什么使用 ConvLSTM？

标准 LSTM 将输入展平为向量，丢失了空间信息。ConvLSTM 使用卷积操作，可以：
- 保留特征图的空间结构
- 建模时空依赖关系
- 适用于视频、气象数据等时空序列

### 2. 多层 vs 单层

| 特性 | 单层 | 多层 |
|------|------|------|
| 表达能力 | 较弱 | 更强 |
| 特征层次 | 单一 | 分层抽象 |
| 参数量 | 较少 | 较多 |
| 适用场景 | 简单模式 | 复杂时空模式 |

### 3. 初始化策略

- **权重**: Xavier 均匀初始化，保持信号方差稳定
- **偏置**: 初始化为0，使门控最初处于"中立"状态（sigmoid(0) = 0.5）

## 参考

- Shi et al. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" (2015)
- Hochreiter & Schmidhuber "Long Short-Term Memory" (1997)

## License

MIT License
