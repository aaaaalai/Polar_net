import torch
from torch import nn
from typing import Optional, Tuple

class FocalLoss(nn.Module):
    r"""
    Focal Loss with ALWAYS-ON dynamic geometric weights per anchor.

    几何权重始终启用（不可关闭）：
      - 若提供 vp_dist（或 anchor_xy + vp_xy），则启用“消失点距离权重”；
      - 若提供 lane_density，则启用“车道密度权重”；
      - 若某一分量未提供，则该分量按 1 参与（不影响结果）。
      - 最终权重按 combine 方式融合并对每个锚点的 focal 项进行加权。

    Args:
        alpha (Tensor): shape == [2]，alpha[0]=负类权重，alpha[1]=正类权重
        gamma (float): focal γ
        reduction (str): 'none' | 'mean' | 'sum'
        combine (str): 'mul' 或 'sum'，组合方式（默认 'mul'）
        lambda_vp (float): 消失点距离权重的相对权重（'sum' 为线性权重；'mul' 为指数）
        lambda_den (float): 车道密度权重的相对权重（'sum' 为线性权重；'mul' 为指数）
        vp_sigma (float): 距离到权重的高斯衰减尺度：w_vp=exp(-d_norm^2/(2*vp_sigma^2))
        den_gamma (float): 密度映射幂指数：w_den=(density_norm)**den_gamma
        img_size (Tuple[int,int] or None): 若未显式给出 max_dist，用图像对角线归一化 vp_dist

    Forward:
        pred (Tensor): 预测概率∈[0,1]，shape = [...], 与 target 可广播
        target (Tensor): 0/1 标签，shape 同 pred 或可广播
        geo_weight (Tensor, optional): 直接提供最终几何权重（优先级最高）
        vp_dist (Tensor, optional): 每锚点到消失点的距离
        lane_density (Tensor, optional): 每锚点的车道密度（值越大表示越密）
        anchor_xy (Tensor, optional): 若未提供 vp_dist，可给锚点坐标 [...,2]，内部计算距离
        vp_xy (Tensor, optional): 消失点坐标 [2] 或 [...,2]
        max_dist (float or Tensor, optional): vp_dist 归一化尺度；不传则用 img_size 对角线或退化归一化

    组合规则：
        mul: w = (w_vp ** lambda_vp) * (w_den ** lambda_den)
        sum: w = (lambda_vp * w_vp + lambda_den * w_den) / (lambda_vp + lambda_den)
    """
    def __init__(
        self,
        alpha: torch.Tensor,
        gamma: float = 2.0,
        reduction: str = 'none',
        *,
        combine: str = 'mul',           # 'mul' | 'sum'
        lambda_vp: float = 1.0,
        lambda_den: float = 1.0,
        vp_sigma: float = 0.5,
        den_gamma: float = 1.0,
        img_size: Optional[Tuple[int, int]] = None
    ):
        super(FocalLoss, self).__init__()
        self.register_buffer(name='alpha', tensor=alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

        # 几何权重配置（始终启用）
        self.combine = combine
        assert self.combine in ('mul', 'sum'), "combine must be 'mul' or 'sum'"
        self.lambda_vp = float(lambda_vp)
        self.lambda_den = float(lambda_den)
        self.vp_sigma = float(vp_sigma)
        self.den_gamma = float(den_gamma)

        if img_size is not None:
            assert len(img_size) == 2
        self.img_size = img_size

    def _compute_vp_weight(
        self,
        vp_dist: Optional[torch.Tensor] = None,
        anchor_xy: Optional[torch.Tensor] = None,
        vp_xy: Optional[torch.Tensor] = None,
        max_dist: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """计算消失点距离权重（越靠近消失点权重越大）。"""
        if vp_dist is None:
            if (anchor_xy is None) or (vp_xy is None):
                return None
            # 欧氏距离
            vp_dist = torch.linalg.norm(anchor_xy - vp_xy, dim=-1)

        # 归一化距离
        if max_dist is None:
            if self.img_size is not None:
                h, w = self.img_size[1], self.img_size[0]
                diag = (w ** 2 + h ** 2) ** 0.5
                max_dist = vp_dist.new_tensor(diag)
            else:
                # 退化归一化：d / (d + 1)
                d = vp_dist.clamp_min(self.eps)
                d_norm = d / (d + 1.0)
                return torch.exp(- (d_norm ** 2) / (2.0 * (self.vp_sigma ** 2) + self.eps))

        d_norm = vp_dist / (max_dist + self.eps)
        w_vp = torch.exp(- (d_norm ** 2) / (2.0 * (self.vp_sigma ** 2) + self.eps))
        return w_vp

    def _compute_density_weight(self, lane_density: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """密度越大权重越大；对传入的 lane_density 做全局 min-max 归一化。"""
        if lane_density is None:
            return None
        d_min = torch.amin(lane_density, dim=None, keepdim=False)
        d_max = torch.amax(lane_density, dim=None, keepdim=False)
        den_norm = (lane_density - d_min) / (d_max - d_min + self.eps)
        w_den = den_norm.clamp(0.0, 1.0) ** self.den_gamma
        return w_den

    def _combine_geo_weights(
        self,
        w_vp: Optional[torch.Tensor],
        w_den: Optional[torch.Tensor],
        shape_like: torch.Tensor
    ) -> torch.Tensor:
        """融合几何权重并广播到与 `shape_like` 相同的形状。"""
        if w_vp is None:
            w_vp = shape_like.new_ones(())
        if w_den is None:
            w_den = shape_like.new_ones(())

        if self.combine == 'mul':
            w = (w_vp.clamp_min(self.eps) ** self.lambda_vp) * (w_den.clamp_min(self.eps) ** self.lambda_den)
        else:
            denom = (self.lambda_vp + self.lambda_den)
            if denom <= 0:
                w = 0.5 * (w_vp + w_den)
            else:
                w = (self.lambda_vp * w_vp + self.lambda_den * w_den) / (denom + self.eps)

        return w.expand_as(shape_like)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        # 直接给最终权重（优先）
        geo_weight: Optional[torch.Tensor] = None,
        # 或让本类计算
        vp_dist: Optional[torch.Tensor] = None,
        lane_density: Optional[torch.Tensor] = None,
        anchor_xy: Optional[torch.Tensor] = None,
        vp_xy: Optional[torch.Tensor] = None,
        max_dist: Optional[torch.Tensor] = None,
    ):
        # 数值稳定
        pred = pred.clamp(self.eps, 1.0 - self.eps)

        # ---- 原始 focal per-anchor ----
        focal_pos = -self.alpha[1] * torch.pow(1 - pred, self.gamma) * torch.log(pred)
        focal_neg = -self.alpha[0] * torch.pow(pred, self.gamma) * torch.log(1 - pred)
        loss = target * focal_pos + (1 - target) * focal_neg

        # ---- 几何权重（始终启用）----
        if geo_weight is not None:
            w = geo_weight
        else:
            w_vp = self._compute_vp_weight(vp_dist, anchor_xy, vp_xy, max_dist)
            w_den = self._compute_density_weight(lane_density)
            w = self._combine_geo_weights(w_vp, w_den, loss)

        loss = loss * w  # 对每个锚点加权

        # 归约
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")
