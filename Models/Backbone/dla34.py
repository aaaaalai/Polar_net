# ##################################################原来####################################################
# import os
# import math
# import logging
# import numpy as np
# from os.path import join
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
#
# BN_MOMENTUM = 0.1
# logger = logging.getLogger(__name__)
#
#
# def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
#     return join('http://dl.yf.io/dla/models', data,
#                 '{}-{}.pth'.format(name, hash))
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=1,
#                      bias=False)
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes,
#                                planes,
#                                kernel_size=3,
#                                stride=stride,
#                                padding=dilation,
#                                bias=False,
#                                dilation=dilation)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(planes,
#                                planes,
#                                kernel_size=3,
#                                stride=1,
#                                padding=dilation,
#                                bias=False,
#                                dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 2
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(Bottleneck, self).__init__()
#         expansion = Bottleneck.expansion
#         bottle_planes = planes // expansion
#         self.conv1 = nn.Conv2d(inplanes,
#                                bottle_planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes,
#                                bottle_planes,
#                                kernel_size=3,
#                                stride=stride,
#                                padding=dilation,
#                                bias=False,
#                                dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes,
#                                planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class BottleneckX(nn.Module):
#     expansion = 2
#     cardinality = 32
#
#     def __init__(self, inplanes, planes, stride=1, dilation=1):
#         super(BottleneckX, self).__init__()
#         cardinality = BottleneckX.cardinality
#         # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
#         # bottle_planes = dim * cardinality
#         bottle_planes = planes * cardinality // 32
#         self.conv1 = nn.Conv2d(inplanes,
#                                bottle_planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv2 = nn.Conv2d(bottle_planes,
#                                bottle_planes,
#                                kernel_size=3,
#                                stride=stride,
#                                padding=dilation,
#                                bias=False,
#                                dilation=dilation,
#                                groups=cardinality)
#         self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
#         self.conv3 = nn.Conv2d(bottle_planes,
#                                planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#
#     def forward(self, x, residual=None):
#         if residual is None:
#             residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Root(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, residual):
#         super(Root, self).__init__()
#         self.conv = nn.Conv2d(in_channels,
#                               out_channels,
#                               1,
#                               stride=1,
#                               bias=False,
#                               padding=(kernel_size - 1) // 2)
#         self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=True)
#         self.residual = residual
#
#     def forward(self, *x):
#         children = x
#         x = self.conv(torch.cat(x, 1))
#         x = self.bn(x)
#         if self.residual:
#             x += children[0]
#         x = self.relu(x)
#
#         return x
#
#
# class Tree(nn.Module):
#     def __init__(self,
#                  levels,
#                  block,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  level_root=False,
#                  root_dim=0,
#                  root_kernel_size=1,
#                  dilation=1,
#                  root_residual=False):
#         super(Tree, self).__init__()
#         if root_dim == 0:
#             root_dim = 2 * out_channels
#         if level_root:
#             root_dim += in_channels
#         if levels == 1:
#             self.tree1 = block(in_channels,
#                                out_channels,
#                                stride,
#                                dilation=dilation)
#             self.tree2 = block(out_channels,
#                                out_channels,
#                                1,
#                                dilation=dilation)
#         else:
#             self.tree1 = Tree(levels - 1,
#                               block,
#                               in_channels,
#                               out_channels,
#                               stride,
#                               root_dim=0,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation,
#                               root_residual=root_residual)
#             self.tree2 = Tree(levels - 1,
#                               block,
#                               out_channels,
#                               out_channels,
#                               root_dim=root_dim + out_channels,
#                               root_kernel_size=root_kernel_size,
#                               dilation=dilation,
#                               root_residual=root_residual)
#         if levels == 1:
#             self.root = Root(root_dim, out_channels, root_kernel_size,
#                              root_residual)
#         self.level_root = level_root
#         self.root_dim = root_dim
#         self.downsample = None
#         self.project = None
#         self.levels = levels
#         if stride > 1:
#             self.downsample = nn.MaxPool2d(stride, stride=stride)
#         if in_channels != out_channels:
#             self.project = nn.Sequential(
#                 nn.Conv2d(in_channels,
#                           out_channels,
#                           kernel_size=1,
#                           stride=1,
#                           bias=False),
#                 nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
#
#     def forward(self, x, residual=None, children=None):
#         children = [] if children is None else children
#         bottom = self.downsample(x) if self.downsample else x
#         residual = self.project(bottom) if self.project else bottom
#         if self.level_root:
#             children.append(bottom)
#         x1 = self.tree1(x, residual)
#         if self.levels == 1:
#             x2 = self.tree2(x1)
#             x = self.root(x2, x1, *children)
#         else:
#             children.append(x1)
#             x = self.tree2(x1, children=children)
#         return x
#
#
# class DLA(nn.Module):
#     def __init__(self,
#                  levels,
#                  channels,
#                  num_classes=1000,
#                  block=BasicBlock,
#                  residual_root=False,
#                  linear_root=False):
#         super(DLA, self).__init__()
#         self.channels = channels
#         self.num_classes = num_classes
#         self.base_layer = nn.Sequential(
#             nn.Conv2d(3,
#                       channels[0],
#                       kernel_size=7,
#                       stride=1,
#                       padding=3,
#                       bias=False),
#             nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
#             nn.ReLU(inplace=True))
#         self.level0 = self._make_conv_level(channels[0], channels[0],
#                                             levels[0])
#         self.level1 = self._make_conv_level(channels[0],
#                                             channels[1],
#                                             levels[1],
#                                             stride=2)
#         self.level2 = Tree(levels[2],
#                            block,
#                            channels[1],
#                            channels[2],
#                            2,
#                            level_root=False,
#                            root_residual=residual_root)
#         self.level3 = Tree(levels[3],
#                            block,
#                            channels[2],
#                            channels[3],
#                            2,
#                            level_root=True,
#                            root_residual=residual_root)
#         self.level4 = Tree(levels[4],
#                            block,
#                            channels[3],
#                            channels[4],
#                            2,
#                            level_root=True,
#                            root_residual=residual_root)
#         self.level5 = Tree(levels[5],
#                            block,
#                            channels[4],
#                            channels[5],
#                            2,
#                            level_root=True,
#                            root_residual=residual_root)
#
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         #         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         m.weight.data.fill_(1)
#         #         m.bias.data.zero_()
#
#     def _make_level(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes:
#             downsample = nn.Sequential(
#                 nn.MaxPool2d(stride, stride=stride),
#                 nn.Conv2d(inplanes,
#                           planes,
#                           kernel_size=1,
#                           stride=1,
#                           bias=False),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample=downsample))
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
#         modules = []
#         for i in range(convs):
#             modules.extend([
#                 nn.Conv2d(inplanes,
#                           planes,
#                           kernel_size=3,
#                           stride=stride if i == 0 else 1,
#                           padding=dilation,
#                           bias=False,
#                           dilation=dilation),
#                 nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
#                 nn.ReLU(inplace=True)
#             ])
#             inplanes = planes
#         return nn.Sequential(*modules)
#
#     def forward(self, x):
#         y = []
#         x = self.base_layer(x)
#         for i in range(6):
#             x = getattr(self, 'level{}'.format(i))(x)
#             y.append(x)
#         return y[2:]
#
#     def load_pretrained_model(self,
#                               data='imagenet',
#                               name='dla34',
#                               hash='ba72cf86'):
#         # fc = self.fc
#         if name.endswith('.pth'):
#             model_weights = torch.load(data + name)
#         else:
#             model_url = get_model_url(data, name, hash)
#             model_weights = model_zoo.load_url(model_url)
#         self.load_state_dict(model_weights, strict=False)
#         # self.fc = fc
#
#
# def dla34(pretrained=True, levels=None, in_channels=None, **kwargs):  # DLA-34
#     model = DLA(levels=levels,
#                 channels=in_channels,
#                 block=BasicBlock,
#                 **kwargs)
#     if pretrained:
#         model.load_pretrained_model(data='imagenet',
#                                     name='dla34',
#                                     hash='ba72cf86')
#     return model
#
#
# class DLAWrapper(nn.Module):
#     def __init__(self,
#                  dla='dla34',
#                  pretrained=True,
#                  levels=[1, 1, 1, 2, 2, 1],
#                  in_channels=[16, 32, 64, 128, 256, 512],
#                  cfg=None):
#         super(DLAWrapper, self).__init__()
#         self.cfg = cfg
#         self.in_channels = in_channels
#
#         self.model = eval(dla)(pretrained=pretrained,
#                                levels=levels,
#                                in_channels=in_channels)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
#
# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#
#     def forward(self, x):
#         return x
#
#
# def fill_fc_weights(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#
# def fill_up_weights(up):
#     w = up.weight.data
#     f = math.ceil(w.size(2) / 2)
#     c = (2 * f - 1 - f % 2) / (2. * f)
#     for i in range(w.size(2)):
#         for j in range(w.size(3)):
#             w[0, 0, i, j] = \
#                 (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
#     for c in range(1, w.size(0)):
#         w[c, 0, :, :] = w[0, 0, :, :]

####################################################主干网络2，3，4层的Tree根节点和IDA融合后加CBAM#############################################
# -*- coding: utf-8 -*-
import os
import math
import logging
from os.path import join
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# ====================== 基础配置 =========================
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data,
                '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# =========================================================
# ====================== CBAM 模块定义 =====================
# =========================================================

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_planes, in_planes // ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_planes // ratio, in_planes, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         avg_out = self.mlp(x.mean(dim=(2, 3), keepdim=False))
#         max_out = self.mlp(x.amax(dim=(2, 3), keepdim=False))
#         out = avg_out + max_out
#         scale = self.sigmoid(out).view(b, c, 1, 1)
#         return x * scale
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv(out)
#         scale = self.sigmoid(out)
#         return x * scale
#
#
# class CBAM(nn.Module):
#     """CBAM: Convolutional Block Attention Module"""
#     def __init__(self, in_planes, ratio=16, spatial_kernel=7):
#         super(CBAM, self).__init__()
#         self.channel_att = ChannelAttention(in_planes, ratio)
#         self.spatial_att = SpatialAttention(spatial_kernel)
#
#     def forward(self, x):
#         out = self.channel_att(x)
#         out = self.spatial_att(out)
#         return out
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    保持原版 CBAM 通道注意力不变：
    GAP + GMP -> shared MLP -> sigmoid -> channel-wise scale
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, A, B] (A/B could be Theta/R)
        b, c, _, _ = x.size()
        avg_out = self.mlp(x.mean(dim=(2, 3), keepdim=False))   # [B, C]
        max_out = self.mlp(x.amax(dim=(2, 3), keepdim=False))   # [B, C]
        out = avg_out + max_out
        scale = self.sigmoid(out).view(b, c, 1, 1)              # [B, C, 1, 1]
        return x * scale


class AxialPolarSpatialAttention(nn.Module):
    """
    Polar Spatial Attention（轴向解耦 + θ循环填充）
    - 将 2D 空间注意力分解为两个 1D 注意力：M(θ,r)=Mθ(θ)*Mr(r)
    - θ 维做 circular padding，避免角度边界不连续（0°~360°）
    - 支持 θ 在 dim=2 或 dim=3（通过 theta_dim 指定）
    """
    def __init__(
        self,
        kernel_theta: int = 7,
        kernel_r: int = 7,
        theta_dim: int = 2,   # 2 表示 [B,C,Theta,R]；3 表示 [B,C,R,Theta]
    ):
        super().__init__()
        assert kernel_theta in (3, 5, 7, 9)
        assert kernel_r in (3, 5, 7, 9)
        assert theta_dim in (2, 3)

        self.theta_dim = theta_dim
        self.pad_t = (kernel_theta - 1) // 2

        # θ 分支：输入 [B,2,Theta] -> Conv1d -> [B,1,Theta]
        self.conv_theta = nn.Conv1d(2, 1, kernel_size=kernel_theta, padding=0, bias=False)

        # r 分支：输入 [B,2,R] -> Conv1d -> [B,1,R]
        pad_r = (kernel_r - 1) // 2
        self.conv_r = nn.Conv1d(2, 1, kernel_size=kernel_r, padding=pad_r, bias=False)

        self.sigmoid = nn.Sigmoid()

    def _to_theta_r(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入统一成 [B, C, Theta, R]
        若输入是 [B,C,R,Theta] 且 theta_dim=3，则 permute 成 [B,C,Theta,R]
        """
        if self.theta_dim == 2:
            return x  # already [B,C,Theta,R]
        # theta_dim == 3: x is [B,C,R,Theta] -> [B,C,Theta,R]
        return x.permute(0, 1, 3, 2).contiguous()

    def _restore(self, x_tr: torch.Tensor) -> torch.Tensor:
        """
        将 [B,C,Theta,R] 还原回输入布局
        """
        if self.theta_dim == 2:
            return x_tr
        # back to [B,C,R,Theta]
        return x_tr.permute(0, 1, 3, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # unify to [B,C,Theta,R]
        x_tr = self._to_theta_r(x)
        b, c, t, r = x_tr.size()

        # ---------------- θ-attention：先对 r 聚合，再在 θ 上做 1D conv ----------------
        # 先对 r 聚合得到 [B,C,Theta]，再对 C 聚合得到 [B,1,Theta]
        avg_t = x_tr.mean(dim=3)                # [B, C, Theta]
        max_t = x_tr.amax(dim=3)                # [B, C, Theta]
        avg_t = avg_t.mean(dim=1, keepdim=True) # [B, 1, Theta]
        max_t = max_t.mean(dim=1, keepdim=True) # [B, 1, Theta]
        feat_t = torch.cat([avg_t, max_t], dim=1)  # [B, 2, Theta]

        # θ 维 circular padding（关键创新点）
        if self.pad_t > 0:
            feat_t = F.pad(feat_t, (self.pad_t, self.pad_t), mode="circular")

        attn_t = self.conv_theta(feat_t)              # [B, 1, Theta]
        attn_t = self.sigmoid(attn_t).view(b, 1, t, 1)  # [B,1,Theta,1]

        # ---------------- r-attention：先对 θ 聚合，再在 r 上做 1D conv ----------------
        avg_r = x_tr.mean(dim=2)                # [B, C, R]
        max_r = x_tr.amax(dim=2)                # [B, C, R]
        avg_r = avg_r.mean(dim=1, keepdim=True) # [B, 1, R]
        max_r = max_r.mean(dim=1, keepdim=True) # [B, 1, R]
        feat_r = torch.cat([avg_r, max_r], dim=1)  # [B, 2, R]

        attn_r = self.conv_r(feat_r)              # [B, 1, R]
        attn_r = self.sigmoid(attn_r).view(b, 1, 1, r)  # [B,1,1,R]

        # ---------------- 合成 2D 注意力并作用到特征 ----------------
        attn = attn_t * attn_r                    # [B,1,Theta,R]
        out = x_tr * attn                         # [B,C,Theta,R]

        # restore to original layout
        return self._restore(out)


class CBAM(nn.Module):
    """
    Polar-CBAM（推荐命名）
    - ChannelAttention：原版 CBAM 通道注意力
    - SpatialAttention：极坐标适配的轴向解耦 + θ循环填充
    """
    def __init__(
        self,
        in_planes: int,
        ratio: int = 16,
        kernel_theta: int = 7,
        kernel_r: int = 7,
        theta_dim: int = 2,
    ):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = AxialPolarSpatialAttention(
            kernel_theta=kernel_theta,
            kernel_r=kernel_r,
            theta_dim=theta_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
# =========================================================
# ==================== DLA 基础结构定义 ====================
# =========================================================

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1,
                              bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual
        # === 在 Root 输出后插入 CBAM ===
        self.cbam = CBAM(out_channels)

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        # 插入 CBAM 注意力增强
        x = self.cbam(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1, dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride,
                              root_dim=0, root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)

        # === 在 level2、3、4 的 Tree 结构（Root 已自动含 CBAM） ===
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # === 在 IDA 融合后再加一个 CBAM，用于最终融合特征增强 ===
        self.cbam_ida = CBAM(channels[5])

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        # 因为我们改了部分层类型，用 strict=False 兼容
        self.load_state_dict(model_weights, strict=False)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f'level{i}')(x)
            y.append(x)
        # === IDA 融合后的 CBAM 注意力增强 ===
        y[-1] = self.cbam_ida(y[-1])
        return y[2:]  # 返回level2~5


# =========================================================
# ==================== 封装 Wrapper ========================
# =========================================================
def dla34(pretrained=True, levels=None, in_channels=None, **kwargs):
    model = DLA(levels=levels, channels=in_channels, block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet',
                                    name='dla34',
                                    hash='ba72cf86')
    return model


class DLAWrapper(nn.Module):
    def __init__(self,
                 dla='dla34',
                 pretrained=True,
                 levels=[1, 1, 1, 2, 2, 1],
                 in_channels=[16, 32, 64, 128, 256, 512],
                 cfg=None):
        super(DLAWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.model = eval(dla)(pretrained=pretrained,
                               levels=levels,
                               in_channels=in_channels)

    def forward(self, x):
        return self.model(x)


# =========================================================
# =================== 自测入口 =============================
# =========================================================
if __name__ == "__main__":
    model = DLAWrapper(pretrained=False)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 288, 800)
        outs = model(x)
        for i, o in enumerate(outs):
            print(i, o.shape)
