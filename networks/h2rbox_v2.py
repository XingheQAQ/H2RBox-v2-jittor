import jittor as jt
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS
import math
from jittor.nn import grid_sample

import copy

@MODELS.register_module()
class H2RBoxV2(nn.Module):
    """H2RBoxV2 implementation for JDet framework"""

    def __init__(self,
                 backbone,
                 neck=None,
                 roi_heads=None,
                 crop_size=(768, 768),
                 view_range=(0.25, 0.75),
                 padding='reflection'):
        super(H2RBoxV2, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(roi_heads, HEADS)

        self.crop_size = crop_size
        self.view_range = view_range
        self.padding = padding

    def rotate_crop(self, img, theta=0., size=(768, 768), gt_bboxes=None, padding='reflection'):
        """Rotate and crop image with optional bounding box transformation"""
        n, c, h, w = img.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
    
        if theta != 0:
            # Create rotation transformation
            cosa, sina = math.cos(theta), math.sin(theta)
            tf = jt.array([[cosa, -sina], [sina, cosa]], dtype=jt.float)
            
            # Create grid for rotation
            x_range = jt.linspace(-1, 1, w)
            y_range = jt.linspace(-1, 1, h)
            y, x = jt.meshgrid(y_range, x_range)
            grid = jt.stack([x, y], -1).unsqueeze(0).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            
            # Apply rotation
            img = grid_sample(img, grid, 'bilinear', padding, align_corners=True)
            
            # Transform bounding boxes if provided
            if gt_bboxes is not None:
                rot_gt_bboxes = []
                for bboxes in gt_bboxes:
                    xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                    ctr = jt.array([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.transpose()) + ctr
                    a = a + theta
                    rot_gt_bboxes.append(jt.concat([xy, wh, a], dim=-1))
                gt_bboxes = rot_gt_bboxes
        
        # Crop image
        img = img[..., crop_h:crop_h + size_h, crop_w:crop_w + size_w]
        
        if gt_bboxes is None:
            return img
        else:
            # Adjust bounding box coordinates after cropping
            crop_gt_bboxes = []
            for bboxes in gt_bboxes:
                xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                xy = xy - jt.array([[crop_w, crop_h]])
                crop_gt_bboxes.append(jt.concat([xy, wh, a], dim=-1))
            gt_bboxes = crop_gt_bboxes
            
            return img, gt_bboxes

    def forward_train(self, images, targets):
        """Training forward pass with symmetric self-supervision"""
        
        #########################################################
        # 处理视图1（无旋转/翻转）
        # 获取边界框标注
        gt_batch = [target for target in targets]
        gt_bboxes_batch = [target['rboxes'] for target in targets]
        # 进行中心裁剪
        images1, gt_bboxes_batch = self.rotate_crop(
            images, 0, self.crop_size, gt_bboxes_batch, self.padding)
    
        # 为每个标注框分配唯一ID (便于计算自监督分支的损失)
        offset = 1  # 初始化ID偏移量，保证跨图像的ID不重复
        for i, gt_bboxes in enumerate(gt_bboxes_batch):
            gt_batch[i]['bid'] = jt.arange(gt_bboxes.shape[0], dtype=jt.float) + offset + 0.2
            # 累加offset保证跨图像唯一性
            offset += gt_bboxes.shape[0]
            # 更新边界框坐标
            gt_batch[i]['rboxes'] = gt_bboxes
        
        #########################################################
        # 处理视图2（旋转） 在视图1的基础上旋转
        # 生成随机旋转角度 (pi/4,3*pi/4) 
        min_angle, max_angle = self.view_range
        rot = (jt.rand(1) * (max_angle - min_angle) + min_angle) * math.pi
        
        rot_gt_batch = [copy.deepcopy(target) for target in gt_batch]
        rot_gt_bboxes_batch = [target['rboxes'] for target in rot_gt_batch]
        images2, rot_gt_bboxes_batch = self.rotate_crop(
            images1, rot, self.crop_size, rot_gt_bboxes_batch, self.padding)
        # ID分配
        offset = 1
        for i, rot_gt_bboxes in enumerate(rot_gt_bboxes_batch):
            rot_gt_batch[i]['bid'] = jt.arange(0, rot_gt_bboxes.shape[0], dtype=jt.float) + offset + 0.4     
            offset += rot_gt_bboxes.shape[0]
            rot_gt_batch[i]['rboxes'] = rot_gt_bboxes
        
        #########################################################
        # 处理视图3（垂直翻转）
        images3 = images1.flip(2)  # 在视图1的基础上垂直翻转
        flip_gt_batch = [copy.deepcopy(target) for target in gt_batch]
        flip_gt_bboxes_batch = [target['rboxes'] for target in flip_gt_batch]
        # ID分配
        offset = 1
        for i, flip_gt_bboxes in enumerate(flip_gt_bboxes_batch):
            H = self.crop_size[0] 
            flip_gt_bboxes[:, 1] = H - flip_gt_bboxes[:, 1]       # y坐标翻转
            flip_gt_bboxes[:, 4] = -flip_gt_bboxes[:, 4]          # 角度取反
            
            flip_gt_batch[i]['bid'] = jt.arange(0, flip_gt_bboxes.shape[0], dtype=jt.float) + offset + 0.6   
            offset += flip_gt_bboxes.shape[0]
            flip_gt_batch[i]['rboxes'] = flip_gt_bboxes
        
        # 合并所有视图的图像和标注信息
        batch_inputs_all = jt.concat([images1, images2, images3])
        
        # 创建合并后的targets列表
        batch_data_samples_all = []
        for target_group in [gt_batch, rot_gt_batch, flip_gt_batch]:
            for target in target_group:
                # 创建一个新的数据样本字典，包含所有必要的信息
                data_sample = {
                    'rboxes': target['rboxes'],
                    'labels': target['labels'],
                    'bid': target['bid']
                }
                batch_data_samples_all.append(data_sample)

        # 提取合并后的特征
        feat = self.backbone(batch_inputs_all)
        if self.neck:
            feat = self.neck(feat)

        # 计算损失
        return self.bbox_head.execute_train(feat, batch_data_samples_all)

    def forward_test(self, images, targets):
        """Testing forward pass"""
        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)
        outs = self.bbox_head.forward(feat)
        return self.bbox_head.get_bboxes(*outs, targets)

    def execute(self, images, targets):
        """
        统一训练和测试的调用接口,自动识别当前模式(train or test)
        """
        
        if 'rboxes' in targets[0]:
            return self.forward_train(images, targets)
        else:
            return self.forward_test(images, targets)