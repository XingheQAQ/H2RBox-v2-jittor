import jittor as jt
from jittor import nn
import math
import numpy as np
from jdet.models.boxes.box_ops import mintheta_obb, distance2obb, rotated_box_to_poly
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS, LOSSES, build_from_cfg
from jdet.models.utils.weight_init import normal_init, bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.models.boxes.coder import PSCCoder

from jdet.ops.nms_rotated import multiclass_nms_rotated

INF = 1e8

class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = jt.array(scale).float()

    def execute(self, x):
        return x * self.scale

@HEADS.register_module()
class H2RBoxV2Head(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 conv_bias='auto',
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 scale_angle=False,
                 rotation_agnostic_classes=None,
                 agnostic_resize_classes=None,
                 use_circumiou_loss=True,
                 use_standalone_angle=True,
                 use_reweighted_loss_bbox=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_bce=True,
                     loss_weight=1.0),
                 loss_symmetry_ss=dict(type='H2RBoxV2ConsistencyLoss'),
                 norm_cfg=dict(type='GN', num_groups=32, is_train=True),
                 test_cfg=None,
                 conv_cfg=None
                ):
        super(H2RBoxV2Head, self).__init__()
        
        self.angle_version = 'le90'
        self.angle_coder = PSCCoder(self.angle_version)

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.scale_angle = scale_angle

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.bbox_type = 'obb'
        self.reg_dim = 4
        self.stacked_convs = stacked_convs
        self.strides = strides
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        
        self.rotation_agnostic_classes = rotation_agnostic_classes
        self.agnostic_resize_classes = agnostic_resize_classes
        
        self.use_circumiou_loss = use_circumiou_loss
        self.use_standalone_angle = use_standalone_angle
        self.use_reweighted_loss_bbox = use_reweighted_loss_bbox

        self.loss_cls = build_from_cfg(loss_cls, LOSSES)
        self.loss_bbox = build_from_cfg(loss_bbox, LOSSES)
        self.loss_centerness = build_from_cfg(loss_centerness, LOSSES)
        self.loss_symmetry_ss = build_from_cfg(loss_symmetry_ss, LOSSES)
        
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        normal_init(self.conv_angle, std=0.01)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.reg_dim, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        if self.scale_angle:
            self.scale_t = Scale(1.0)

    def obb2xyxy(self, rbboxes):
        """
        计算旋转框的外接水平框
        """
        # 从第2/3/4个维度开始每隔5个取一个值
        w = rbboxes[:, 2::5]
        h = rbboxes[:, 3::5]
        a = rbboxes[:, 4::5].detach()
        cosa = jt.cos(a).abs()
        sina = jt.sin(a).abs()
        # 计算旋转框的外接水平框的宽和高
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        # 外接水平框的中心坐标、宽和高
        dx = rbboxes[..., 0]  # ... 表示所有前面的维度
        dy = rbboxes[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        # 计算外接水平框坐标：左下角(x1,y1)、右上角(x2,y2)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return jt.stack((x1, y1, x2, y2), -1)  # # 沿最后一个维度拼接
    
    
    ####################################################
    # CircumIoU loss中的映射操作
    ####################################################
    def nested_projection(self, pred, target):
        """
        将预测框投影到与目标框同方向，获取投影后的坐标
        """
        # 1. 计算目标框的坐标
        # target_xy1: 目标框的左下角坐标 [x_center - w/2, y_center - h/2]
        target_xy1 = target[..., 0:2] - target[..., 2:4] / 2
        # target_xy2: 目标框的右上角坐标 [x_center + w/2, y_center + h/2]
        target_xy2 = target[..., 0:2] + target[..., 2:4] / 2
        # 合并坐标 [x1, y1, x2, y2]
        target_projected = jt.concat((target_xy1, target_xy2), -1)
        
        # 2. 获取预测框的中心坐标、宽高
        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        
        # 3. 计算预测框相对于目标框方向的外接框(CircumIoU中的 Projected box)
        # 预测框与目标框的角度差
        da = pred[..., 4] - target[..., 4]  
        cosa = jt.cos(da).abs()
        sina = jt.sin(da).abs()
        # 将预测框的宽高乘以投影矩阵
        pred_wh = jt.matmul(
            jt.stack((cosa, sina, sina, cosa), -1).view(*cosa.shape, 2, 2),
            pred_wh[..., None])[..., 0]
        
        # 4. 计算预测框的坐标
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        # 合并坐标 [x1, y1, x2, y2]
        pred_projected = jt.concat((pred_xy1, pred_xy2), -1)
        
        return pred_projected, target_projected

    def _get_rotation_agnostic_mask(self, cls):
        """
        获取旋转无关类别的掩码
        """
        # 初始化为全False的布尔张量
        _rot_agnostic_mask = jt.zeros_like(cls, dtype=jt.bool)
        for c in self.rotation_agnostic_classes:
            # 对每个旋转无关类别 c，检查 cls 中哪些位置等于 c，并通过逻辑或操作合并到掩码中。
            _rot_agnostic_mask = jt.logical_or(_rot_agnostic_mask, cls == c)
        return _rot_agnostic_mask
    
    def execute_train(self, feats, targets):
        
        # Forward
        outs = self.forward(feats)
        
        if self.is_training():
            return self.loss(outs, targets)
        
        else:
            return self.get_bboxes(outs, targets)

    def forward_single(self, x, scale, stride):
        """
        单个特征图（FPN层级）的前向传播
        """
        # 分类、回归、角度三个分支 共享特征
        cls_feat = x
        reg_feat = x
        angle_feat = x
        
        # 分类分支
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        
        # 回归分支
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        bbox_pred = scale(bbox_pred)
        
        # 中心度分支
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        
        if self.norm_on_bbox:
            bbox_pred = nn.relu(bbox_pred)
            if not self.is_training():
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        
        # 角度预测分支
        angle_pred = self.conv_angle(angle_feat)
        if self.scale_angle:
            angle_pred = self.scale_t(angle_pred)

        return cls_score, bbox_pred, angle_pred, centerness

    def forward(self, feats):
        """
        多层级特征图的前向传播
        """
        return multi_apply(self.forward_single, feats, self.scales, self.strides)
    
    def loss(self, outs, targets):
        """ 
        Compute loss of the head
        """
        
        cls_scores, bbox_preds, angle_preds, centernesses = outs

        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)
        
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype)
        # bbox_targets here is in format t,b,l,r
        # angle_targets is not coded here
        labels, bbox_targets, angle_targets, bid_targets = self.get_targets(all_level_points, targets)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds, angle_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        
        flatten_cls_scores = jt.concat(flatten_cls_scores)
        flatten_bbox_preds = jt.concat(flatten_bbox_preds)
        flatten_angle_preds = jt.concat(flatten_angle_preds)
        flatten_centerness = jt.concat(flatten_centerness)
        flatten_labels = jt.concat(labels)
        flatten_bbox_targets = jt.concat(bbox_targets)
        flatten_angle_targets = jt.concat(angle_targets)
        flatten_bid_targets = jt.concat(bid_targets)
        
        # repeat points to align with bbox_preds
        flatten_points = jt.concat([points.repeat(num_imgs, 1) for points in all_level_points])
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        
        num_pos = jt.array(len(pos_inds), dtype='float32')
        
        flatten_labels += 1
        flatten_labels[flatten_labels == bg_class_ind+1] = 0
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs) # avoid num_pos is 0
        
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_centerness = flatten_centerness[pos_inds]
            pos_angle_preds = flatten_angle_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds] 
            pos_angle_targets = flatten_angle_targets[pos_inds]
            pos_bid_targets = flatten_bid_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]
            # centerness weighted iou loss
            centerness_denorm = jt.maximum(jt.mean(pos_centerness_targets.sum().detach()), 1e-6)
            
            ####################################################################
            # 角度解码
            pos_decoded_angle_preds = self.angle_coder.decode(pos_angle_preds, keepdim=True)
            
            # 通过断开梯度回传, 使角度完全由自监督分支学习
            if self.use_standalone_angle:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()

            if self.rotation_agnostic_classes:
                pos_agnostic_mask = self._get_rotation_agnostic_mask(pos_labels)
                pos_decoded_angle_preds[pos_agnostic_mask] = 0
                target_mask = jt.abs(pos_angle_targets[pos_agnostic_mask]) < math.pi / 4
                pos_angle_targets[pos_agnostic_mask] = jt.where(
                    target_mask, 0, -math.pi / 2)
            
            pos_bbox_preds = jt.concat([pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
            pos_bbox_targets = jt.concat([pos_bbox_targets, pos_angle_targets], dim=-1)
            
            pos_decoded_bbox_preds = distance2obb(pos_points, pos_bbox_preds)
            pos_decoded_bbox_targets = distance2obb(pos_points, pos_bbox_targets)
            
            # HBB-supervision
            if self.use_circumiou_loss:
                # Works with random rotation where targets are OBBs
                loss_bbox = self.loss_bbox(
                    *self.nested_projection(pos_decoded_bbox_preds, pos_decoded_bbox_targets),
                    weight=pos_centerness_targets,
                    avg_factor=centerness_denorm)
            else:
                # Targets are supposed to be HBBs
                target_mask = jt.logical_or(
                    pos_decoded_bbox_targets[:, -1] == 0,
                    pos_decoded_bbox_targets[:, -1] == -math.pi / 2)
                loss_bbox = self.loss_bbox(
                    self.obb2xyxy(pos_decoded_bbox_preds[target_mask]),
                    self.obb2xyxy(pos_decoded_bbox_targets[target_mask]),
                    weight=pos_centerness_targets[target_mask],
                    avg_factor=centerness_denorm * target_mask.sum() / target_mask.numel())

            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)

            # Self-supervision
            # Aggregate targets of the same bbox based on their identical bid
            # 将具有相同bid的目标框参数(如位置、尺寸等)进行平均，得到一个更稳定的目标表示
            
            bid, idx = jt.unique(pos_bid_targets, return_inverse=True)
            
            compacted_bid_targets = jt.zeros_like(bid)
            compacted_bid_targets = jt.scatter(compacted_bid_targets, 0, idx, pos_bid_targets, reduce='add')            
            count = jt.zeros_like(bid)
            count = jt.scatter(count, 0, idx, jt.ones_like(pos_bid_targets), reduce='add')
            compacted_bid_targets = compacted_bid_targets / jt.maximum(count, 1)
            
            # Generate a mask to eliminate bboxes without correspondence
            # (bcnt is supposed to be 3, for ori, rot, and flp)
            # 只保留那些在所有三种变换中都存在的目标框，确保自监督的一致性
            _, bidx, bcnt = jt.unique(
                compacted_bid_targets.long(),
                return_inverse=True,
                return_counts=True)
            bmsk = bcnt[bidx] == 3
            
            # The reduce all sample points of each object
            # 角度目标值的聚合
            compacted_angle_targets = jt.zeros_like(bid)
            compacted_angle_targets = jt.scatter(compacted_angle_targets, 0, idx, pos_angle_targets[:, 0], reduce='add')
            compacted_angle_targets = compacted_angle_targets / jt.maximum(count, 1)
            compacted_angle_targets = compacted_angle_targets[bmsk].reshape(-1, 3)
            # 角度预测值的聚合
            compacted_angle_preds = jt.zeros((*bid.shape, pos_angle_preds.shape[-1]), dtype=pos_angle_preds.dtype)
            compacted_angle_preds = jt.scatter(compacted_angle_preds, 0, idx.unsqueeze(-1).expand_as(pos_angle_preds), pos_angle_preds, reduce='add')
            compacted_angle_preds = compacted_angle_preds / jt.maximum(count.unsqueeze(-1), 1)
            compacted_angle_preds = compacted_angle_preds[bmsk].reshape(-1, 3, pos_angle_preds.shape[-1])
            
            compacted_angle_preds = self.angle_coder.decode(
                compacted_angle_preds, keepdim=False)
            
            compacted_agnostic_mask = None
            if self.rotation_agnostic_classes:
                # 聚合标签
                compacted_labels = jt.zeros(bid.shape, dtype=pos_labels.dtype)
                compacted_labels = jt.scatter(compacted_labels, 0, idx, pos_labels, reduce='add')
                compacted_labels = compacted_labels / jt.maximum(count, 1)
                compacted_labels = compacted_labels[bmsk].reshape(-1, 3)[:, 0]
                # 生成旋转无关掩码
                compacted_agnostic_mask = self._get_rotation_agnostic_mask(compacted_labels)
            
            loss_symmetry_ss = self.loss_symmetry_ss(
                compacted_angle_preds[:, 0], compacted_angle_preds[:, 1],
                compacted_angle_preds[:, 2], compacted_angle_targets[:, 0],
                compacted_angle_targets[:, 1], compacted_agnostic_mask)
            
            if self.use_reweighted_loss_bbox:
                loss_bbox = math.exp(-loss_symmetry_ss.item()) * loss_bbox

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_symmetry_ss = pos_angle_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_symmetry_ss=loss_symmetry_ss)

    def get_targets(self, points, targets):
        
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # 将regress_ranges扩展到与points相同形状
        expanded_regress_ranges = [
            jt.array(self.regress_ranges[i], dtype=points[i].dtype)[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # # 合并所有层级的特征点 和 regress ranges
        concat_regress_ranges = jt.concat(expanded_regress_ranges, dim=0)
        concat_points = jt.concat(points, dim=0)
        
        # 每个层级的特征点个数
        num_points = [center.size(0) for center in points]
        
        # 获取目标框信息
        gt_bboxes_list = [t["rboxes"] for t in targets]
        gt_labels_list = [t["labels"] for t in targets]
        gt_bid_list = [t["bid"] for t in targets]
        
        # 获取每张图片的标签和bbox
        labels_list, bbox_targets_list, angle_targets_list, bid_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_bid_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)
        
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0) for bid_targets in bid_targets_list
        ]
        
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_bid_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                jt.concat([labels[i] for labels in labels_list]))
            bbox_targets = jt.concat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = jt.concat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bid_targets = jt.concat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            if self.norm_on_bbox:
                bbox_targets[:, :4] = bbox_targets[:, :4] / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_bid_targets.append(bid_targets)
            
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_bid_targets)

    def regularize_boxes(self, gt_bboxes, angle_version):
        """
        对旋转框的角度进行标准化处理
    
        参数:
            gt_bboxes (jt.Var): 旋转框张量，形状为 [N, 5]，格式为 (x, y, w, h, angle)。
            angle_version (str): 角度定义模式，支持 'le90', 'le135', 'oc'。
    
        返回:
            jt.Var: 标准化后的旋转框，形状仍为 [N, 5]。
        """
        assert angle_version in ['le90', 'le135', 'oc'], \
            f"不支持的 angle_version: {angle_version}，请选择 'le90', 'le135' 或 'oc'"
    
        if gt_bboxes.size(0) == 0:
            return gt_bboxes  # 空张量直接返回

        x, y, w, h, angle = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3], gt_bboxes[:, 4]
    
        if angle_version == 'oc':
            # OpenCV 模式：角度范围 [0, 90°)，长边始终为 width
            angle = angle % (math.pi / 2)  # 约束到 [0, π/2)
            swap_mask = w < h
            w_new = jt.where(swap_mask, h, w)
            h_new = jt.where(swap_mask, w, h)
            angle_new = jt.where(swap_mask, angle + math.pi / 2, angle)
            return jt.stack([x, y, w_new, h_new, angle_new], dim=1)

        elif angle_version == 'le90':
            # 长边基准模式：角度范围 [-45°, 45°)
            angle = angle % math.pi  # 约束到 [0, π)
            angle = jt.where(angle >= math.pi / 2, angle - math.pi, angle)  # [-π/2, π/2)
        
            # 调整到 [-π/4, π/4)
            swap_mask = (angle < -math.pi / 4) | (angle >= math.pi / 4)
            angle_new = jt.where(swap_mask, angle + math.pi / 2, angle)
            angle_new = jt.where(angle_new >= math.pi / 4, angle_new - math.pi, angle_new)  # 二次约束
            
            w_new = jt.where(swap_mask, h, w)
            h_new = jt.where(swap_mask, w, h)
            return jt.stack([x, y, w_new, h_new, angle_new], dim=1)

        elif angle_version == 'le135':
            # 长边基准模式：角度范围 [-67.5°, 67.5°)
            angle = angle % math.pi  # 约束到 [0, π)
            angle = jt.where(angle >= math.pi / 2, angle - math.pi, angle)  # [-π/2, π/2)
        
            # 调整到 [-3π/8, 3π/8)
            swap_mask = (angle < -math.pi / 4) | (angle >= math.pi / 4)
            angle_new = jt.where(swap_mask, angle + math.pi / 2, angle)
            angle_new = jt.where(angle_new >= 3 * math.pi / 8, angle_new - math.pi, angle_new)  # 二次约束
        
            w_new = jt.where(swap_mask, h, w)
            h_new = jt.where(swap_mask, w, h)
            return jt.stack([x, y, w_new, h_new, angle_new], dim=1)

    def _get_target_single(self, gt_bboxes, gt_labels, gt_bid, points, regress_ranges, 
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1)), \
                   gt_bboxes.new_zeros((num_points,))

        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
        gt_bboxes = self.regularize_boxes(gt_bboxes, self.angle_version)
       
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = mintheta_obb(gt_bboxes)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = jt.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = jt.cos(gt_angle), jt.sin(gt_angle)
        rot_matrix = jt.concat([cos_angle, sin_angle, -sin_angle, cos_angle], dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = jt.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = jt.stack((left, top, right, bottom), -1)
        
        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1) > 0
        if self.center_sampling:
            # # inside a `center bbox`
            radius = self.center_sample_radius
            stride = jt.zeros_like(offset)
            
            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
                
            inside_center_bbox_mask = (jt.abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = jt.logical_and(inside_center_bbox_mask, inside_gt_bbox_mask)
        
        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0]) &
            (max_regress_distance <= regress_ranges[..., 1]))
        
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area_inds, min_area  = areas.argmin(dim=1)
        
        # min_area_inds is between 0 and num_gt, for each point
        labels = gt_labels[min_area_inds]-1
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[jt.index((num_points,),dim=0), min_area_inds]
        angle_targets = gt_angle[jt.index((num_points,),dim=0), min_area_inds]
        bid_targets = gt_bid[min_area_inds]

        return labels, bbox_targets, angle_targets, bid_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1) / left_right.max(dim=-1)) * (
                top_bottom.min(dim=-1) / top_bottom.max(dim=-1))
        return jt.sqrt(centerness_targets)

    def get_points(self, featmap_sizes, dtype, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (jt.dtype): Type of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i], dtype))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype):
        h, w = featmap_size
        x_range = jt.arange(w, dtype=dtype)
        y_range = jt.arange(h, dtype=dtype)
        y, x = jt.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        points = jt.stack((x * stride, y * stride), dim=-1) + stride // 2
        return points

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
                   targets,
                   rescale=True):
        """ Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            targets (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype)
        result_list = []
        for img_id in range(len(targets)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = targets[img_id]['img_size']
            scale_factor = targets[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        
        # 遍历每个特征层级的预测结果
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # 处理分类分数、中心度预测
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            
            # 处理角度、边界框预测
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, self.angle_coder.encode_size)
            decoded_angle_pred = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # 合并bbox预测
            bbox_pred = jt.concat([bbox_pred, decoded_angle_pred], dim=-1)
            
            # 从配置中获取NMS预处理参数
            nms_pre = cfg.get('nms_pre', -1)
            # 对中心度进行调整
            centerness = centerness + cfg.get("centerness_factor", 0.)
            
            # NMS处理
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # 计算每个点的最大分数(分类分数×中心度)
                max_scores = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                # 根据topk索引筛选预测结果
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                points = points[topk_inds, :]
                centerness = centerness[topk_inds]
                
            # bbox解码  
            bboxes = distance2obb(points, bbox_pred, max_shape=img_shape)
            
            # 收集各层级结果
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            
        # 合并所有层级的bbox
        mlvl_bboxes = jt.concat(mlvl_bboxes)
        # 缩放bbox（若需要）
        if rescale:
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        # 合并所有层级的分类分数    
        mlvl_scores = jt.concat(mlvl_scores)
        # 添加一列0作为背景类分数
        padding = jt.zeros((mlvl_scores.shape[0], 1), dtype=mlvl_scores.dtype)
        mlvl_centerness = jt.concat(mlvl_centerness)
        mlvl_scores = jt.concat([padding, mlvl_scores], dim=1)
        
        # 旋转NMS处理
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        
        # 提取bbox参数和分数
        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        
        # 将旋转框转换为多边形表示
        polys = rotated_box_to_poly(boxes)
        
        # 旋转无关类别处理
        if self.rotation_agnostic_classes:
            for id in self.rotation_agnostic_classes:
                # 对于旋转无关类别，将其角度设为0
                inds = det_labels == id
                boxes[inds, 4] = 0
            # 需要resize边界框的旋转无关类别
            if self.agnostic_resize_classes:
                for id in self.agnostic_resize_classes:
                    inds = det_labels == id
                    boxes[inds, 2:4] *= 0.85
                    
        return polys, scores, det_labels