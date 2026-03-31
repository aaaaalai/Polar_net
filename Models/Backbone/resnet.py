#
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.hub import load_state_dict_from_url
#
#
# model_urls = {
#     'resnet18':
#     'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34':
#     'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50':
#     'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101':
#     'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152':
#     'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d':
#     'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d':
#     'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2':
#     'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2':
#     'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }
#
#
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes,
#                      out_planes,
#                      kernel_size=3,
#                      stride=stride,
#                      padding=dilation,
#                      groups=groups,
#                      bias=False,
#                      dilation=dilation)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes,
#                      out_planes,
#                      kernel_size=1,
#                      stride=stride,
#                      bias=False)
#
# import os, sys
# # 确保能找到 my_trick/FDConv.py
# proj_root = r"D:\Desktop\PolarRCNN-master"
# if proj_root not in sys.path:
#     sys.path.insert(0, proj_root)
#
# from my_trick.FDConv import FDConv  # 你的 FDConv 类
#
# import torch
# import torch.nn as nn
#
# def conv3x3(in_planes, out_planes, stride=1, dilation=1):
#     # 你项目里已有该函数就保留原来的；这里给一个兜底实现
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3,
#                      stride=stride, padding=dilation, dilation=dilation, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None,
#                  # 新增两个参数：是否在下采样块用 FDConv，以及 FDConv 的超参
#                  use_fdconv_downsample=False,
#                  fdconv_kwargs=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#
#         if fdconv_kwargs is None:
#             fdconv_kwargs = {}
#
#         # --- 关键修改：当该块负责下采样（stride != 1）且开关为 True，用 FDConv 替换 conv1 ---
#         if use_fdconv_downsample and stride != 1:
#             self.conv1 = FDConv(
#                 in_channels=inplanes,
#                 out_channels=planes,
#                 kernel_size=3,
#                 stride=stride,      # =2 的块会在这里下采样
#                 padding=1,
#                 bias=False,
#                 # 典型的 FDConv 默认超参（可按需在 fdconv_kwargs 里覆盖）
#                 kernel_num=8,
#                 use_fbm_for_stride=True,
#                 fbm_cfg=dict(k_list=[2, 4, 8], lowfreq_att=False,
#                              fs_feat='feat', act='sigmoid',
#                              spatial='conv', spatial_group=1,
#                              spatial_kernel=3, init='zero'),
#                 ksm_global_act='sigmoid',
#                 ksm_local_act='sigmoid',
#                 convert_param=False,     # 先 False 以便兼容预训练；稳定后可改 True
#                 **fdconv_kwargs
#             )
#         else:
#             # 非下采样块仍用普通 3x3
#             self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
#
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 第二个 3x3 保持不变
#         self.conv2 = conv3x3(planes, planes, dilation=dilation)
#         self.bn2 = norm_layer(planes)
#
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError(
#                 'BasicBlock only supports groups=1 and base_width=64')
#         # if dilation > 1:
#         #     raise NotImplementedError(
#         #         "Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes, dilation=dilation)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
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
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNetWrapper(nn.Module):
#     def __init__(self,
#                  resnet='resnet18',
#                  pretrained=True,
#                  replace_stride_with_dilation=[False, False, False],
#                  out_conv=False,
#                  fea_stride=8,
#                  out_channel=128,
#                  in_channels=[64, 128, 256, 512],
#                  cfg=None):
#         super(ResNetWrapper, self).__init__()
#         self.cfg = cfg
#         self.in_channels = in_channels
#
#         self.model = eval(resnet)(
#             pretrained=pretrained,
#             replace_stride_with_dilation=replace_stride_with_dilation,
#             in_channels=self.in_channels)
#         self.out = None
#         if out_conv:
#             out_channel = 512
#             for chan in reversed(self.in_channels):
#                 if chan < 0: continue
#                 out_channel = chan
#                 break
#             self.out = conv1x1(out_channel * self.model.expansion,
#                                cfg.featuremap_out_channel)
#
#     def forward(self, x):
#         x = self.model(x)
#         if self.out:
#             x[-1] = self.out(x[-1])
#         return x
#
#
# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  zero_init_residual=False,
#                  groups=1,
#                  width_per_group=64,
#                  replace_stride_with_dilation=None,
#                  norm_layer=None,
#                  in_channels=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(
#                                  replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3,
#                                self.inplanes,
#                                kernel_size=7,
#                                stride=2,
#                                padding=3,
#                                bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.in_channels = in_channels
#         self.layer1 = self._make_layer(block, in_channels[0], layers[0])
#         self.layer2 = self._make_layer(block,
#                                        in_channels[1],
#                                        layers[1],
#                                        stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block,
#                                        in_channels[2],
#                                        layers[2],
#                                        stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         if in_channels[3] > 0:
#             self.layer4 = self._make_layer(
#                 block,
#                 in_channels[3],
#                 layers[3],
#                 stride=2,
#                 dilate=replace_stride_with_dilation[2])
#         self.expansion = block.expansion
#
#         # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(
#             block(self.inplanes, planes, stride, downsample, self.groups,
#                   self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(self.inplanes,
#                       planes,
#                       groups=self.groups,
#                       base_width=self.base_width,
#                       dilation=self.dilation,
#                       norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         out_layers = []
#         for name in ['layer1', 'layer2', 'layer3', 'layer4']:
#             if not hasattr(self, name):
#                 continue
#             layer = getattr(self, name)
#             x = layer(x)
#             out_layers.append(x)
#
#         return out_layers
#
#
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         print('pretrained model: ', model_urls[arch])
#         state_dict = load_state_dict_from_url(model_urls[arch])
#         model.load_state_dict(state_dict, strict=False)
#     return model
#
#
# def resnet18(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-18 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)
#
#
# def resnet34(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-34 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet50(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-50 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
#                    **kwargs)
#
#
# def resnet101(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-101 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
#                    progress, **kwargs)
#
#
# def resnet152(pretrained=False, progress=True, **kwargs):
#     r"""ResNet-152 model from
#     `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
#                    progress, **kwargs)
#
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-50 32x4d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
#                    progress, **kwargs)
#
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     r"""ResNeXt-101 32x8d model from
#     `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
#                    progress, **kwargs)
#
#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
#                    progress, **kwargs)
#
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     r"""Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
#                    progress, **kwargs)
# import os
# import sys
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.hub import load_state_dict_from_url

# # ----------------------- 以下代码为fdconv 以及cbam-zong-----------------------
# model_urls = {
#     'resnet18':
#         'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34':
#         'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50':
#         'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101':
#         'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152':
#         'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
#     'resnext50_32x4d':
#         'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d':
#         'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2':
#         'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2':
#         'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }
#
# # ----------------------- 基础卷积封装 -----------------------
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=3,
#         stride=stride, padding=dilation,
#         groups=groups, bias=False, dilation=dilation
#     )
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(
#         in_planes, out_planes, kernel_size=1, stride=stride, bias=False
#     )
#
# # ----------------------- 引入你的 FDConv -----------------------
# # 确保能找到 D:\Desktop\PolarRCNN-master\my_trick\FDConv.py
# proj_root = r"D:\Desktop\PolarRCNN-master"
# if proj_root not in sys.path:
#     sys.path.insert(0, proj_root)
# from my_trick.FDConv import FDConv
#
#
# # ----------------------- ResNet Blocks -----------------------
# class BasicBlock(nn.Module):
#     """在下采样块（stride!=1）的 conv1 使用 FDConv，其它保持 3x3。"""
#     expansion = 1
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None,
#                  # 仅入口块使用
#                  use_fdconv_downsample=False,
#                  fdconv_kwargs=None):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#
#         if fdconv_kwargs is None:
#             fdconv_kwargs = {}
#
#         # 关键：入口块且 stride!=1 时，用 FDConv 做下采样
#         if use_fdconv_downsample and stride != 1:
#             self.conv1 = FDConv(
#                 in_channels=inplanes,
#                 out_channels=planes,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#                 bias=False,
#                 # 典型默认超参（你可在 _make_layer 里通过 fdconv_kwargs 覆盖）
#                 # kernel_num=8,
#                 # use_fbm_for_stride=True,
#                 # fbm_cfg=dict(
#                 #     k_list=[2, 4, 8], lowfreq_att=False,
#                 #     fs_feat='feat', act='sigmoid',
#                 #     spatial='conv', spatial_group=1,
#                 #     spatial_kernel=3, init='zero'
#                 # ),
#                 # ksm_global_act='sigmoid',
#                 # ksm_local_act='sigmoid',
#                 # convert_param=False,  # 先 False 以便用 ImageNet 预训练，后续可改 True
#                 **fdconv_kwargs
#             )
#         else:
#             self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
#
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
#         self.bn2 = norm_layer(planes)
#
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
#         out = self.conv2(out); out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self,
#                  inplanes,
#                  planes,
#                  stride=1,
#                  downsample=None,
#                  groups=1,
#                  base_width=64,
#                  dilation=1,
#                  norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
#         out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
#         out = self.conv3(out); out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# # ----------------------- ResNet Backbone -----------------------
# class ResNet(nn.Module):
#     def __init__(self,
#                  block,
#                  layers,
#                  zero_init_residual=False,
#                  groups=1,
#                  width_per_group=64,
#                  replace_stride_with_dilation=None,
#                  norm_layer=None,
#                  in_channels=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
#
#         self.groups = groups
#         self.base_width = width_per_group
#
#         # stem
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # 方便外部读取各 stage 输出通道
#         self.in_channels = in_channels
#
#         # 统一配置 FDConv 的默认参数（可按需调整/覆盖）
#         self.fdconv_kwargs = dict(
#             kernel_num=8,
#             use_fbm_for_stride=True,
#             fbm_cfg=dict(
#                 k_list=[2, 4, 8], lowfreq_att=False,
#
#
#
#
#
#
#
#
#                 fs_feat='feat', act='sigmoid',
#                 spatial='conv', spatial_group=1,
#                 spatial_kernel=3, init='zero'
#             ),
#             ksm_global_act='sigmoid',
#             ksm_local_act='sigmoid',
#             convert_param=False,
#         )
#
#         # stages
#         self.layer1 = self._make_layer(block, in_channels[0], layers[0], stride=1)
#         self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2,
#                                        dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
#                                        dilate=replace_stride_with_dilation[1])
#         if in_channels[3] > 0:
#             self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
#                                            dilate=replace_stride_with_dilation[2])
#         self.expansion = block.expansion
#
#         # init
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         """只在每个 stage 的第一个 block（通常 stride!=1 或通道变化）启用 FDConv。"""
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         # 入口块：开启 use_fdconv_downsample
#         layers.append(
#             block(self.inplanes, planes, stride, downsample,
#                   self.groups, self.base_width, previous_dilation, norm_layer,
#                   use_fdconv_downsample=True,
#                   fdconv_kwargs=self.fdconv_kwargs)
#         )
#         self.inplanes = planes * block.expansion
#
#         # 其余块：关闭 FDConv
#         for _ in range(1, blocks):
#             layers.append(
#                 block(self.inplanes, planes,
#                       groups=self.groups, base_width=self.base_width,
#                       dilation=self.dilation, norm_layer=norm_layer,
#                       use_fdconv_downsample=False)
#             )
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
#
#         out_layers = []
#         for name in ['layer1', 'layer2', 'layer3', 'layer4']:
#             if not hasattr(self, name):
#                 continue
#             layer = getattr(self, name)
#             x = layer(x)
#             out_layers.append(x)
#         return out_layers
#
#
# # ----------------------- Wrapper -----------------------
# class ResNetWrapper(nn.Module):
#     def __init__(self,
#                  resnet='resnet18',
#                  pretrained=True,
#                  replace_stride_with_dilation=[False, False, False],
#                  out_conv=False,
#                  fea_stride=8,
#                  out_channel=128,
#                  in_channels=[64, 128, 256, 512],
#                  cfg=None):
#         super(ResNetWrapper, self).__init__()
#         self.cfg = cfg
#         self.in_channels = in_channels
#
#         self.model = eval(resnet)(
#             pretrained=pretrained,
#             replace_stride_with_dilation=replace_stride_with_dilation,
#             in_channels=self.in_channels)
#         self.out = None
#         if out_conv:
#             out_channel = 512
#             for chan in reversed(self.in_channels):
#                 if chan < 0:
#                     continue
#                 out_channel = chan
#                 break
#             self.out = conv1x1(out_channel * self.model.expansion,
#                                cfg.featuremap_out_channel)
#
#     def forward(self, x):
#         x = self.model(x)
#         if self.out:
#             x[-1] = self.out(x[-1])
#         return x
#
#
# # ----------------------- Builders -----------------------
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         print('pretrained model: ', model_urls[arch])
#         state_dict = load_state_dict_from_url(model_urls[arch])
#         # 与 FDConv 兼容：strict=False
#         model.load_state_dict(state_dict, strict=False)
#     return model
#
# def resnet18(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
#
# def resnet34(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
#
# def resnet50(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
#
# def resnet101(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
#
# def resnet152(pretrained=False, progress=True, **kwargs):
#     return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)
#
# def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 4
#     return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
#
# def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
#     kwargs['groups'] = 32
#     kwargs['width_per_group'] = 8
#     return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
#
# def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
#
# def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
#     kwargs['width_per_group'] = 64 * 2
#     return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
# # ----------------------- 以下代码为DSconv 以及cbam-zong-----------------------
# -*- coding: utf-8 -*-
import torch

# # ----------------------- 以下代码为主干网络basicblock 2,3层cbam-----------------------
# -*- coding: utf-8 -*-
"""
ResNet(18/34/50/101/152, ResNeXt, Wide-ResNet) backbone with CBAM inserted
*after conv2/bn2 and before the residual add* on specified stages.

- **No DSConv**: reverted to standard Conv2d everywhere.
- CBAM toggled per stage via `use_cbam_stages=(False, True, True, False)` (ON for layer2 & layer3 by default).
- Position: conv2 -> bn2 -> CBAM -> residual add -> ReLU.
- Pretrained ImageNet weights supported via `strict=False`.

Drop-in replacement for your PolarRCNN-style wrapper.
"""
from typing import Sequence, Optional
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

# ------------------------------
# ImageNet 预训练权重 URL
# ------------------------------
model_urls = {
    'resnet18':      'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':      'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':      'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':     'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':     'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':  'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':  'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# ------------------------------
# 基础卷积定义
# ------------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ------------------------------
# CBAM 注意力（Channel + Spatial）
# ------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, pool_types: Sequence[str] = ("avg", "max")):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.pool_types = pool_types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = 0
        if "avg" in self.pool_types:
            attn = attn + self.mlp(F.adaptive_avg_pool2d(x, 1))
        if "max" in self.pool_types:
            attn = attn + self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(attn)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 1 if kernel_size == 3 else 3
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module
    Args:
        channels (int): 输入通道数
        reduction (int): 通道注意力的压缩比
        spatial_kernel (int): 空间注意力卷积核(3或7)
        enable_cam (bool): 是否启用通道注意力
        enable_sam (bool): 是否启用空间注意力
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7,
                 enable_cam: bool = True, enable_sam: bool = True,
                 pool_types: Sequence[str] = ("avg", "max")):
        super().__init__()
        self.enable_cam = enable_cam
        self.enable_sam = enable_sam
        if enable_cam:
            self.ca = ChannelAttention(channels, reduction=reduction, pool_types=pool_types)
        if enable_sam:
            self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.enable_cam:
            out = self.ca(out) * out
        if self.enable_sam:
            out = self.sa(out) * out
        return out

# ------------------------------
# BasicBlock（纯普通卷积）+ 可选 CBAM
# ------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[nn.Module] = None,
                 use_cbam: bool = False, cbam_kwargs: Optional[dict] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        cbam_kwargs = cbam_kwargs or {}

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2   = norm_layer(planes)

        # ⚠️ CBAM 位置：conv2 -> bn2 -> CBAM -> add
        self.cbam = CBAM(planes, **cbam_kwargs) if use_cbam else None
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam is not None:
            out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

# ------------------------------
# Bottleneck（保持卷积结构，增加可选 CBAM）
# ------------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[nn.Module] = None,
                 use_cbam: bool = False, cbam_kwargs: Optional[dict] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        cbam_kwargs = cbam_kwargs or {}
        self.cbam = CBAM(width, **cbam_kwargs) if use_cbam else None  # 对应 conv2 输出的通道数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam is not None:
            out = self.cbam(out)
        out = self.bn3(self.conv3(self.relu(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

# ------------------------------
# ResNet 主体
# ------------------------------
class ResNet(nn.Module):
    def __init__(self, block: nn.Module, layers: Sequence[int], zero_init_residual: bool = False, groups: int = 1,
                 width_per_group: int = 64, replace_stride_with_dilation: Optional[Sequence[bool]] = None,
                 norm_layer: Optional[nn.Module] = None, in_channels: Optional[Sequence[int]] = None,
                 use_cbam_stages: Sequence[bool] = (False, True, True, False),
                 cbam_kwargs: Optional[dict] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        assert len(replace_stride_with_dilation) == 3

        self.groups = groups
        self.base_width = width_per_group
        self.in_channels = in_channels
        cbam_kwargs = cbam_kwargs or {}

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages: layer1..layer4
        self.layer1 = self._make_layer(block, in_channels[0], layers[0], stride=1,
                                       use_cbam=use_cbam_stages[0], cbam_kwargs=cbam_kwargs)
        self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       use_cbam=use_cbam_stages[1], cbam_kwargs=cbam_kwargs)
        self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       use_cbam=use_cbam_stages[2], cbam_kwargs=cbam_kwargs)
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2],
                                           use_cbam=use_cbam_stages[3], cbam_kwargs=cbam_kwargs)
        self.expansion = block.expansion

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: nn.Module, planes: int, blocks: int, stride: int = 1, dilate: bool = False,
                    use_cbam: bool = False, cbam_kwargs: Optional[dict] = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer,
                  use_cbam=use_cbam, cbam_kwargs=cbam_kwargs)
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,
                      use_cbam=use_cbam, cbam_kwargs=cbam_kwargs)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        outs = []
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self, name):
                layer = getattr(self, name)
                x = layer(x)
                outs.append(x)
        return outs

# ------------------------------
# ResNetWrapper 兼容 PolarRCNN
# ------------------------------
class ResNetWrapper(nn.Module):
    def __init__(self, resnet: str = 'resnet18', pretrained: bool = True,
                 replace_stride_with_dilation: Sequence[bool] = (False, False, False),
                 out_conv: bool = False, fea_stride: int = 8, out_channel: int = 128,
                 in_channels: Sequence[int] = (64, 128, 256, 512), cfg=None,
                 use_cbam_stages: Sequence[bool] = (False, True, True, False),
                 cbam_kwargs: Optional[dict] = None):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.model = eval(resnet)(
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            in_channels=self.in_channels,
            use_cbam_stages=use_cbam_stages,
            cbam_kwargs=cbam_kwargs,
        )
        self.out = None
        if out_conv and cfg is not None and hasattr(cfg, 'featuremap_out_channel'):
            self.out = conv1x1(in_channels[-1] * self.model.expansion, cfg.featuremap_out_channel)

    def forward(self, x: torch.Tensor):
        feats = self.model(x)
        if self.out:
            feats[-1] = self.out(feats[-1])
        return feats

# ------------------------------
# 构建接口
# ------------------------------

def _resnet(arch: str, block: nn.Module, layers: Sequence[int], pretrained: bool, progress: bool, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('pretrained model:', model_urls[arch])
        state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


