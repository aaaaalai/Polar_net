# modules/attention/cbam.py
import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    通道注意力：AvgPool + MaxPool → 共享 MLP（用 1x1 conv 实现）→ Sigmoid
    参考 CBAM 论文，MLP 共享权重，reduction 默认 16。
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        # 用 1x1 conv 等价替代 FC，便于处理任意空间大小
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    """
    空间注意力：在通道维做 Avg/Max 聚合 → 拼接 → 7x7 卷积 → Sigmoid
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, maxv], dim=1)
        attn = self.conv(x_cat)
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    """
    标准 CBAM：Channel → Spatial
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class CBAMStack(nn.Module):
    """
    针对 FPN/Neck 输出为 list/tuple 的情况，逐尺度套 CBAM。
    `channels_per_level` 为每个尺度的通道数列表。
    """
    def __init__(self, channels_per_level, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.blocks = nn.ModuleList([CBAM(c, reduction, spatial_kernel) for c in channels_per_level])

    def forward(self, feats):
        if isinstance(feats, (list, tuple)):
            return [blk(f) for blk, f in zip(self.blocks, feats)]
        return self.blocks[0](feats)
