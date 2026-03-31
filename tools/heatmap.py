import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import cv2
import shutil
import torch
import numpy as np
from tqdm import trange
from PIL import Image

# ========== 关键：确保能 import PolarRCNN 的工程代码 ==========
# 如果你把脚本放在 PolarRCNN-master 根目录，这段可以不用改
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from tools.get_config import get_cfg
from Models.build import build_model

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ------------------------------------------------------------
# 1) PolarRCNN 推理输入预处理：RGB + cut_height裁剪 + resize(320x800)
# ------------------------------------------------------------
def preprocess_polarrcnn(img_path, cfg):
    """
    返回:
        ori_rgb: 原始 RGB 图 (H,W,3)
        crop_rgb: 裁剪后的 RGB 图 (H-cut_height, W, 3)
        net_rgb: resize 后给网络输入的 RGB 图 (img_h, img_w, 3), float32 [0,1]
        tensor:   torch tensor [1,3,img_h,img_w]
    """
    ori_bgr = cv2.imread(img_path)
    if ori_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    ori_rgb = cv2.cvtColor(ori_bgr, cv2.COLOR_BGR2RGB)

    # CULane 测试时会裁剪顶部
    crop_rgb = ori_rgb[cfg.cut_height:, :, :]

    # resize 到网络输入尺寸
    net_rgb = cv2.resize(crop_rgb, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_LINEAR)
    net_rgb_float = np.float32(net_rgb) / 255.0

    tensor = torch.from_numpy(net_rgb_float).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return ori_rgb, crop_rgb, net_rgb_float, tensor


# ------------------------------------------------------------
# 2) CAM forward：为了兼容 grad-cam 的 targets 机制，输出做成 list[per_image_output]
#    同时让输出“可反传”到特征层
# ------------------------------------------------------------
class PolarRCNNForCAM(torch.nn.Module):
    """
    用于 GradCAM 的前向包装：
    直接返回 roi_head 的 pred_dict（包含 cls / end_points 等，可用于构造 scalar loss）
    """
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        # backbone 输出 list -> 取 level3~level5
        y = self.net.backbone(x)[1:]
        feat = self.net.neck(y)
        rpn_dict = self.net.rpn_head(feat)

        # roi_head.forward_function 返回 pred_dict（训练态输出形式），包含 cls / end_points / offsets 等
        feat_list = list(feat)
        feat_list.reverse()
        anchor_embeddings = rpn_dict["anchor_embeddings"]
        anchor_id = rpn_dict["anchor_id"]
        pred_dict = self.net.roi_head.forward_function(anchor_embeddings, feat_list, anchor_id)

        return pred_dict


class PolarRCNN_ActivationsAndGradients:
    """
    这个类模仿你原来 RTDETR 那套写法：
    - 注册 hooks 保存 activation/gradient
    - forward 返回 list 结构，让 grad-cam 的 targets 能正确 zip
    """
    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []

        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, inp, out):
        activation = out
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, inp, out):
        if not hasattr(out, "requires_grad") or not out.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        out.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []

        pred_dict = self.model(x)  # dict

        # CAM 输出结构：每张图输出一个 list，里面放你 target 需要的内容
        # cls: [B,N], end_points: [B,N,2]
        cls = pred_dict["cls_o2o"] if "cls_o2o" in pred_dict else pred_dict["cls"]
        end_points = pred_dict["end_points"]

        # batch=1 场景输出：[[cls[0], end_points[0]]]
        return [[cls[0], end_points[0]]]

    def release(self):
        for h in self.handles:
            h.remove()


# ------------------------------------------------------------
# 3) 目标函数：把 “lane confidence + 回归参数(可选)” 聚合成一个 scalar
# ------------------------------------------------------------
class polarrcnn_target(torch.nn.Module):
    def __init__(self, output_type="class", conf=0.5, ratio=1.0):
        """
        output_type:
            'class' -> 只用置信度 cls
            'box'   -> 这里用 end_points 代替“框回归”（lane 的 start/end）
            'all'   -> cls + end_points
        """
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        cls_scores, end_points = data  # cls_scores: [N], end_points: [N,2]
        cls_scores = cls_scores.view(-1)
        N = cls_scores.shape[0]
        topk = max(1, int(N * self.ratio))

        sorted_score, idx = torch.sort(cls_scores, descending=True)

        result = []
        for i in trange(topk):
            if float(sorted_score[i]) < self.conf:
                break

            if self.output_type in ["class", "all"]:
                result.append(sorted_score[i])

            if self.output_type in ["box", "all"]:
                # 用 lane 的 end_points (start/end) 作为回归项
                result.append(end_points[idx[i], 0])
                result.append(end_points[idx[i], 1])

        if len(result) == 0:
            # 防止 empty 导致报错
            return cls_scores.sum() * 0.0
        return sum(result)


# ------------------------------------------------------------
# 4) 可选：把 CAM 限制在“车道线附近区域”
# ------------------------------------------------------------
def renormalize_cam_in_lane_mask(grayscale_cam, lane_list, cfg, thickness=10):
    """
    grayscale_cam: [H,W] in [0,1]
    lane_list: list of lanes, lane["points"] 是归一化到原图尺寸的点
    返回 renormalized_cam: [H,W], outside mask = 0
    """
    H, W = cfg.img_h, cfg.img_w
    if lane_list is None or len(lane_list) == 0:
        return grayscale_cam

    mask = np.zeros((H, W), dtype=np.uint8)

    crop_h = cfg.ori_img_h - cfg.cut_height
    for lane in lane_list:
        pts = lane["points"]  # (N,2), normalized in original image space
        if pts.shape[0] < 2:
            continue

        xs = pts[:, 0] * cfg.ori_img_w
        ys = pts[:, 1] * cfg.ori_img_h - cfg.cut_height

        xs = xs / cfg.ori_img_w * cfg.img_w
        ys = ys / crop_h * cfg.img_h

        poly = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(mask, [poly], isClosed=False, color=1, thickness=thickness)

    if mask.sum() == 0:
        return grayscale_cam

    cam = grayscale_cam.copy()
    out = np.zeros_like(cam, dtype=np.float32)

    inside = cam[mask > 0]
    mn, mx = inside.min(), inside.max()
    if mx - mn < 1e-8:
        out[mask > 0] = inside
    else:
        out[mask > 0] = (inside - mn) / (mx - mn)

    return out


# ------------------------------------------------------------
# 5) 主类：PolarRCNN heatmap
# ------------------------------------------------------------
class polarrcnn_heatmap:
    def __init__(self, cfg_file, weight, device, method, layer, backward_type,
                 conf_threshold, ratio, show_lane, renormalize):

        # ---- load cfg ----
        from types import SimpleNamespace
        args = SimpleNamespace(cfg=cfg_file, gpu_no=0, is_view=0, is_val=0)
        cfg = get_cfg(args)

        self.cfg = cfg
        self.device = torch.device(device)

        # ---- build net ----
        net = build_model(cfg)
        state = torch.load(weight, map_location="cpu")
        net.load_state_dict(state, strict=True)
        net.to(self.device).eval()

        # ---- CAM wrapper model ----
        cam_model = PolarRCNNForCAM(net).to(self.device).eval()

        # ---- target ----
        target = polarrcnn_target(backward_type, conf_threshold, ratio)

        # ---- choose target layers ----
        # 你可以选：
        # 1) "neck.fpn_convs.2" (默认推荐，最常用)
        # 2) "backbone.model.level5" (更靠后 backbone 特征)
        target_layers = []
        for name in layer:
            target_layers.append(self._get_module_by_name(cam_model, name))

        # ---- CAM method ----
        cam_method = eval(method)(cam_model, target_layers)
        cam_method.activations_and_grads = PolarRCNN_ActivationsAndGradients(cam_model, target_layers, None)

        self.net = net
        self.cam_model = cam_model
        self.method = cam_method
        self.target = target
        self.show_lane = show_lane
        self.renormalize = renormalize

    def _get_module_by_name(self, model, name: str):
        """
        name example:
            "net.neck.fpn_convs.2"  (如果你想从 cam_model 里找)
            "net.backbone.model.level5"
        由于 cam_model 结构是 PolarRCNNForCAM(net)，所以 net 是 cam_model.net
        """
        obj = model
        for k in name.split("."):
            if k.isdigit():
                obj = obj[int(k)]
            else:
                obj = getattr(obj, k)
        return obj

    def process(self, img_path, save_path):
        ori_rgb, crop_rgb, net_rgb_float, tensor = preprocess_polarrcnn(img_path, self.cfg)
        tensor = tensor.to(self.device)

        # --- CAM ---
        try:
            grayscale_cam = self.method(tensor, [self.target])  # [B,H,W]
        except Exception as e:
            print(f"[Warning] CAM failed on {img_path}: {repr(e)}")
            return

        grayscale_cam = grayscale_cam[0]  # [H,W]

        # --- 可选：把 CAM 限制在车道线附近 ---
        if self.renormalize:
            with torch.no_grad():
                # 这里用原始 net 的 eval 输出 lane_list
                out = self.net(tensor)
                lane_list = out.get("lane_list", [[]])[0] if isinstance(out.get("lane_list", []), list) else []
            grayscale_cam = renormalize_cam_in_lane_mask(grayscale_cam, lane_list, self.cfg)

        # --- CAM overlay on resized crop ---
        # cam_on_crop = show_cam_on_image(net_rgb_float, grayscale_cam, use_rgb=True)  # uint8 RGB
        cam_on_crop = show_cam_on_image(
            net_rgb_float,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.7  # ✅ 0.75~0.95 之间自己试，越大越淡
        )

        # --- 拼回原图尺寸（上半部分不动，下半部分替换为 CAM）---
        ori_h, ori_w = ori_rgb.shape[:2]
        crop_h = ori_h - self.cfg.cut_height

        cam_crop_full = cv2.resize(cam_on_crop, (ori_w, crop_h), interpolation=cv2.INTER_LINEAR)

        final = ori_rgb.copy()
        final[self.cfg.cut_height:, :, :] = cam_crop_full

        Image.fromarray(final).save(save_path)

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            exts = [".jpg", ".png", ".jpeg", ".bmp"]
            for fn in os.listdir(img_path):
                if os.path.splitext(fn)[-1].lower() not in exts:
                    continue
                self.process(os.path.join(img_path, fn), os.path.join(save_path, fn))
        else:
            self.process(img_path, os.path.join(save_path, "result.png"))


# ------------------------------------------------------------
# 6) 参数：只需要改 cfg_file 和 weight
#    ✅ 图片输入输出路径不改（你 main 里传的）
# ------------------------------------------------------------
def get_params():
    params = {
        # ✅ 你的配置：polarrcnn_culane_dla34
        "cfg_file": r"Config\polarrcnn_culane_dla34.py",

        # ✅ 你的权重（PolarRCNN 的 .pth）
        "weight": r"D:\Desktop\PolarRCNN-master\work_dir\culane-dla34-cbam theta r\para_31.pth",

        "device": "cuda:0",

        # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
        "method": "XGradCAM",

        # ✅ 推荐 target layer（你可以换）
        # 1) neck.fpn_convs.2：FPN 最后一层 conv（最常用、效果稳定）
        # 2) net.backbone.model.level5：DLA 最深层
        # "layer": ["net.neck.layers.0.out_bu_convs.1.pwconv"],
        # "layer": ["net.roi_head.rcnn_head.cls_layer"],
        # "layer": ["net.roi_head.rcnn_head.end_layer"],
        "layer": ["net.neck.layers.0.laterals.0"],
        # "layer": ["net.neck.lateral_convs.0"],

        # "layer": ["neck.layers.0.out_bu_convs.1.pwconv"],

        # class / box / all
        "backward_type": "all",

        "conf_threshold": 0.0000000000000000000000000000000000000000000000000001,  # 可以用 cfg.conf_thres_nmsfree 附近
        "ratio": 1.0,

        # 是否额外画车道线（这里先不开，你想要我可以加）
        "show_lane": True,

        # ✅ True：把热力图限制在车道线附近区域
        "renormalize": True
    }
    return params


if __name__ == "__main__":
    model = polarrcnn_heatmap(**get_params())

    # ✅ 你说“图片传入和传出地址不需要修改”，这里保持你原来的用法即可
    model(r"D:\Desktop\PolarRCNN-master\heatmap_pic\images_test", "D:\Desktop\PolarRCNN-master\heatmap_pic\heatmap_outtest")

