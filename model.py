import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CornerHead(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.linear1 = nn.Linear(in_channels=256,out_channels=128)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_channels=128,out_channels=16)

        nn.init.normal_(self.linear1.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.linear1.bias,0)
        nn.init.normal_(self.linear2.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.linear2.bias,0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x.view(-1, 8, 2)

class ConfidenceHead(nn.Module):
    def __init__(self, in_dim=256):
        self.linear1 = nn.Linear(in_channels=256,out_channels=128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_channels=128,out_channels=1)

        nn.init.normal_(self.linear1.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.linear1.bias,0)
        nn.init.normal_(self.linear2.weight,std=0.01,mean=0.0)
        nn.init.constant_(self.linear2.bias,0)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, max_len=16):
        super().__init__()
        self.time_embed = nn.Parameter(torch.randn(max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        B,T,D = x.shape
        x = x + self.time_embed[:T, :]
        x = self.transformer_encoder(x)
        return x

class ROIEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=weight)
        self.backbone = models.resnet18(pretrained=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(in_channels=512,out_channels=out_dim)
    
    def forward(self,x):
        feat = self.backbone(x)
        feat = self.pool(feat)
        feat = feat.view(feat.shape[0],-1)#torch.flatten(feat, start_dim=1)有什么区别
        feat = self.linear(feat)
        return feat

class VideoCornerModel(nn.Module):
    def __init__(self, feat_dim=256,nhead=8,num_layers=4,max_len=8):
        super().__init__()
        self.roi_encoder = ROIEncoder(out_dim=feat_dim)
        self.temporal_transformer = TemporalTransformer(d_model=feat_dim,nhead=nhead,num_layers=num_layers,max_len=max_len)
        self.corner_head = CornerHead(in_dim=feat_dim)
        self.confidence_head = ConfidenseHead(in_dim=feat_dim)
    
    def forward(self,x):
        B,T,C,H,W = x.shape
        x = x.view(B*T,C,H,W)
        feat = self.roi_encoder(x)
        feat = feat.view(B,T,-1)
        feat = self.temporal_transformer(feat)
        corner_pred = self.corner_head(feat)
        confidence_pred = self.confidence_head(feat)
        return{
            "corners_pred": corner_pred,
            "conf_logits_pred": confidence_pred
        }