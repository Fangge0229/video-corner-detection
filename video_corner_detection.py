import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class backbone(nn.Module):
    """特征提取主干网络，基于 ResNet18"""
    def __init__(self, H, W):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class taskhead(nn.Module):
    """任务头，用于生成角点预测热图"""
    def __init__(self, H, W):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        # 产生8个通道的角点预测热图
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=8, kernel_size=3, stride=1, padding=0)

        nn.init.normal_(self.conv1.weight, std=0.01, mean=0.0)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, std=0.01, mean=0.0)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class upsampling(nn.Module):
    """上采样模块，将特征图恢复到原始尺寸"""
    def __init__(self, H, W):
        super().__init__()
        self.upsample = nn.Upsample(size=(H, W), mode='bicubic', align_corners=True)
    
    def forward(self, x):
        return torch.sigmoid(self.upsample(x))


class CornerDetectionModel(nn.Module):
    """单帧角点检测模型"""
    def __init__(self, H, W):
        super().__init__()
        self.backbone = backbone(H, W)
        self.taskhead = taskhead(H, W)
        self.upsampling = upsampling(H, W)
    
    def forward(self, x):
        features = self.backbone(x)      
        detection = self.taskhead(features)  
        output = self.upsampling(detection)   
        return output


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元 - 将 LSTM 的全连接替换为卷积
    适用于处理具有空间结构的时间序列数据（如视频帧）
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 将输入和隐藏状态拼接后进行卷积
        # 输出通道为 4 * hidden_dim，对应4个门（输入门、遗忘门、输出门、候选记忆）
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # Xavier 初始化权重，偏置初始化为0
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, h, c):
        """
        前向传播
        
        参数:
            x: 当前输入 (batch_size, input_dim, height, width)
            h: 上一时刻隐藏状态 (batch_size, hidden_dim, height, width)
            c: 上一时刻细胞状态 (batch_size, hidden_dim, height, width)
        
        返回:
            h_next: 下一时刻隐藏状态
            c_next: 下一时刻细胞状态
        """
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        
        # 卷积计算所有门
        conv_output = self.conv(combined)
        
        # 分割为4个门
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选记忆
        
        # 更新细胞状态和隐藏状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        """
        初始化隐藏状态
        
        参数:
            batch_size: 批次大小
            image_size: (height, width) 图像尺寸
            device: 计算设备
        
        返回:
            h: 初始隐藏状态 (batch_size, hidden_dim, height, width)
            c: 初始细胞状态 (batch_size, hidden_dim, height, width)
        """
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM 网络
    支持多层堆叠，适用于视频等时空序列数据
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None):
        """
        前向传播
        
        参数:
            x: 输入序列 (batch_size, seq_len, channels, height, width)
            hidden_state: 初始隐藏状态，None则自动初始化
        
        返回:
            layer_output: 最后一层的输出 (batch_size, seq_len, hidden_dim, height, width)
            last_state_list: 每层的最终状态列表
        """
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = []
            for i in range(self.num_layers):
                h, c = self.cell_list[i].init_hidden(batch_size, (height, width), device)
                hidden_state.append((h, c))
        
        # 逐层处理
        cur_layer_input = x
        last_state_list = []
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # 遍历时间步
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t], h, c)
                output_inner.append(h)
            
            # 堆叠该层所有时间步的输出
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output  # 作为下一层的输入
            last_state_list.append((h, c))
        
        return layer_output, last_state_list


class VideoCornerDetectionModel(nn.Module):
    """
    视频角点检测模型
    结合 CNN 特征提取和 ConvLSTM 时序建模
    """
    def __init__(self, H, W, hidden_dim=64, num_layers=2):
        super().__init__()
        self.H = H
        self.W = W
        
        # CNN 特征提取
        self.backbone = backbone(H, W)
        
        # 特征投影层（将 backbone 输出映射到 ConvLSTM 输入维度）
        self.feature_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        
        # ConvLSTM 时序建模
        self.convlstm = ConvLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=3,
            num_layers=num_layers
        )
        
        # 特征反投影层（将 ConvLSTM 输出映射回 256 通道）
        self.feature_reproj = nn.Conv2d(hidden_dim, 256, kernel_size=1)
        
        # 任务头
        self.taskhead = taskhead(H, W)
        self.upsampling = upsampling(H, W)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 视频序列 (batch_size, seq_len, 3, H, W)
        
        返回:
            output: 角点预测热图序列 (batch_size, seq_len, 8, H, W)
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # 提取每帧的 CNN 特征
        features = []
        for t in range(seq_len):
            feat = self.backbone(x[:, t])  # (batch, 256, H', W')
            feat = self.feature_proj(feat)  # (batch, hidden_dim, H', W')
            features.append(feat)
        
        # 堆叠为序列 (batch, seq_len, hidden_dim, H', W')
        features = torch.stack(features, dim=1)
        
        # ConvLSTM 时序建模
        lstm_out, _ = self.convlstm(features)  # (batch, seq_len, hidden_dim, H', W')
        
        # 生成角点预测
        outputs = []
        for t in range(seq_len):
            feat = self.feature_reproj(lstm_out[:, t])  # (batch, 256, H', W')
            detection = self.taskhead(feat)
            output = self.upsampling(detection)
            outputs.append(output)
        
        # 堆叠输出
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, 8, H, W)
        return output


def detection_loss(output, target):
    """检测损失（MSE）"""
    return F.mse_loss(output, target)


def temporal_loss(output):
    """时序平滑损失，鼓励相邻帧预测的一致性"""
    if output.size(1) <= 1:
        return torch.tensor(0.0, device=output.device)
    diff = output[:, 1:] - output[:, :-1]
    return (diff ** 2).mean()


def total_loss(output, target, lambda_temporal=0.2):
    """总损失 = 检测损失 + 时序平滑损失"""
    return detection_loss(output, target) + lambda_temporal * temporal_loss(output)


if __name__ == '__main__':
    # 测试单帧模型
    print("=" * 60)
    print("测试单帧角点检测模型")
    print("=" * 60)
    model = CornerDetectionModel(224, 224)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print()
    
    # 测试 ConvLSTM Cell
    print("=" * 60)
    print("测试 ConvLSTM Cell")
    print("=" * 60)
    cell = ConvLSTMCell(input_dim=64, hidden_dim=64)
    x = torch.randn(2, 64, 32, 32)
    h = torch.randn(2, 64, 32, 32)
    c = torch.randn(2, 64, 32, 32)
    h_next, c_next = cell(x, h, c)
    print(f"输入 x 形状: {x.shape}")
    print(f"输出 h_next 形状: {h_next.shape}")
    print(f"输出 c_next 形状: {c_next.shape}")
    print()
    
    # 测试多层 ConvLSTM
    print("=" * 60)
    print("测试多层 ConvLSTM")
    print("=" * 60)
    convlstm = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=3, num_layers=2)
    video = torch.randn(2, 10, 64, 32, 32)  # (batch, seq, channels, H, W)
    output, states = convlstm(video)
    print(f"输入形状: {video.shape}")
    print(f"输出形状: {output.shape}")
    print(f"层数: {len(states)}")
    print()
    
    # 测试视频角点检测模型
    print("=" * 60)
    print("测试视频角点检测模型")
    print("=" * 60)
    video_model = VideoCornerDetectionModel(224, 224, hidden_dim=64, num_layers=2)
    video_input = torch.randn(1, 5, 3, 224, 224)  # (batch, seq, C, H, W)
    video_output = video_model(video_input)
    print(f"输入形状: {video_input.shape}")
    print(f"输出形状: {video_output.shape}")
