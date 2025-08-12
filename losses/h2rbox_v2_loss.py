from jdet.utils.registry import LOSSES, build_from_cfg
import jittor as jt
from jittor import nn
import math

@LOSSES.register_module()
class H2RBoxV2Loss(nn.Module):
    def __init__(self, loss_rot, loss_flp, use_snap_loss,
                 reduction='mean'):
        super(H2RBoxV2Loss, self).__init__()
        self.loss_rot = build_from_cfg(loss_rot, LOSSES)
        self.loss_flp = build_from_cfg(loss_flp, LOSSES)
        self.use_snap_loss = use_snap_loss
        self.reduction = reduction

    def execute(self,
                pred_ori,
                pred_rot,
                pred_flp,
                target_ori,
                target_rot,
                agnostic_mask=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function for Jittor.

        Args:
            pred_ori (jt.Var): Original prediction angles [N,]
            pred_rot (jt.Var): Rotated prediction angles [N,]
            pred_flp (jt.Var): Flipped prediction angles [N,]
            target_ori (jt.Var): Original target angles [N,]
            target_rot (jt.Var): Rotated target angles [N,]
            agnostic_mask (jt.Var, optional): Mask for rotation-agnostic classes. 
            avg_factor (int, optional): Average factor for loss normalization.
            reduction_override (str, optional): Override reduction method.

        Returns:
            jt.Var: Combined consistency loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        # 1. 计算旋转一致性的角度差
        # 原图像预测角度-旋转图像预测角度 与 原图像目标角度-旋转图像目标角度 应该是相等的（都等于旋转角度）
        d_ang_rot = (pred_ori - pred_rot) - (target_ori - target_rot)
        
        # 2. 计算翻转一致性角度差：原图像预测角度+翻转图像预测角度=0
        d_ang_flp = pred_ori + pred_flp
        
        # 3. 应用Snap Loss处理角度周期性
        # 注：这里应用了 Point2RBox 的优化方案：x是角度差， mod(x) = (x+𝜋/2) mod 𝜋 −𝜋/2
        if self.use_snap_loss:
            d_ang_rot = (d_ang_rot + math.pi/2) % math.pi - math.pi/2
            d_ang_flp = (d_ang_flp + math.pi/2) % math.pi - math.pi/2

        # 4. 处理旋转无关类别：通过掩码将旋转无关类别的角度差置零，使其不参与损失计算
        if agnostic_mask is not None:
            d_ang_rot = d_ang_rot * (1 - agnostic_mask.float())
            d_ang_flp = d_ang_flp * (1 - agnostic_mask.float())

        # 5. 计算旋转和翻转损失
        loss_rot = self.loss_rot(
            d_ang_rot,
            jt.zeros_like(d_ang_rot),
            reduction_override=reduction,
            avg_factor=avg_factor)
        
        loss_flp = self.loss_flp(
            d_ang_flp,
            jt.zeros_like(d_ang_flp),
            reduction_override=reduction,
            avg_factor=avg_factor)

        return loss_rot + loss_flp