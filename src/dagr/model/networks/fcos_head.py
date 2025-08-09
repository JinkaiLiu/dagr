import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
import numpy as np

class FCOSHead(nn.Module):
    def __init__(self, num_classes=2, in_channels=[256, 512], strides=[8, 16], use_iou_loss=True, use_focal_loss=True, feat_channels=256):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.use_iou_loss = use_iou_loss
        self.use_focal_loss = use_focal_loss
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.centerness_preds = nn.ModuleList()
        
        # 损失权重因子
        self.cls_factor = 0.1
        self.reg_factor = 3.5  # 适度增加回归权重
        self.ctr_factor = 0.5  # 降低centerness的影响
        
        # 控制置信度计算方式
        self.conf_beta = 0.7  # 分类置信度与中心度的权重比例
        
        # 添加调试参数
        self.debug_mode = True
        self.global_step = 0
        self.class_names = {0: 'car', 1: 'person'}
        
        # 添加最小边界框大小约束
        self.min_box_size = 1  # 最小1像素
        
        # 添加置信度校准参数
        self.conf_offset = 0.05  # 轻微正偏置提高整体置信度
        
        # 检测阈值
        self.nms_threshold = 0.5  # 合理的NMS阈值
        self.conf_threshold = 0.05  # 较低的置信度阈值以获得更多检测

        for i, in_c in enumerate(in_channels):
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, feat_channels, 3, padding=1),
                    nn.GroupNorm(32, feat_channels),  # 替换为GroupNorm提高稳定性
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.GroupNorm(32, feat_channels),  # 替换为GroupNorm提高稳定性
                    nn.ReLU(inplace=True),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, feat_channels, 3, padding=1),
                    nn.GroupNorm(32, feat_channels),  # 替换为GroupNorm提高稳定性
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.GroupNorm(32, feat_channels),  # 替换为GroupNorm提高稳定性
                    nn.ReLU(inplace=True),
                )
            )
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 3, padding=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 3, padding=1))
            self.centerness_preds.append(nn.Conv2d(feat_channels, 1, 3, padding=1))

        # 使用IoU Loss进行回归
        self._loss_iou = self.iou_loss
        
        # 正确初始化卷积层权重
        self._init_weights()
        
        # 初始化activation_stats属性
        self._activation_stats = []
    
    def _init_weights(self):
        """初始化网络权重，特别针对预测层"""
        for modules in [self.cls_convs, self.reg_convs]:
            for module in modules:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        # kaiming初始化卷积层
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        # 特别为预测层设置合适的初始化
        for modules in [self.cls_preds, self.reg_preds, self.centerness_preds]:
            for m in modules:
                # 对预测层使用正态分布初始化，方差小一些
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # 为分类层设置先验偏置以提高稳定性
                if m in self.cls_preds:
                    # 使用更平衡的初始化，不再对负样本有强烈偏好
                    nn.init.constant_(m.bias, 0.0)  # 使用零偏置，更平衡
                    # # 设置初始置信度
                    # bias_value = -math.log((1 - 0.25) / 0.25)
                    # nn.init.constant_(m.bias, bias_value)
                elif m in self.reg_preds:
                    # 为回归层设置更大的初始偏置，以预测更大的框
                    nn.init.constant_(m.bias, 1.0)
                else:
                    nn.init.constant_(m.bias, 0)
    
    def iou_loss(self, pred, target, eps=1e-7, reduction='none'):
        """
        计算IoU损失，直接优化IoU值
        pred, target: [N, 4] 格式为 (l, t, r, b)
        """
        # 确保输入是有效的
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 裁剪负值为0，确保有效的边界框
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        
        # 预测框和目标框的左上右下坐标
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        
        # 预测框和目标框的面积
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)
        
        # 计算交集面积
        inter_w = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        inter_h = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
        inter_area = inter_w * inter_h
        
        # 计算并集面积
        union_area = pred_area + target_area - inter_area
        
        # 计算IoU，确保数值稳定
        iou = torch.clamp(inter_area / (union_area + eps), min=0, max=1.0)
        
        # 检查是否有无效的IoU值
        valid_mask = ~torch.isnan(iou) & ~torch.isinf(iou)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 过滤有效值
        iou = iou[valid_mask]
        
        # IoU损失
        loss = 1 - iou
        
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
            
    def giou_loss(self, pred, target, eps=1e-7, reduction='none'):
        """
        计算GIoU损失 (Generalized IoU Loss)，专为FCOS的LTRB格式设计
        
        参数:
            pred: [N, 4] 格式为 (l, t, r, b) 的预测框
            target: [N, 4] 格式为 (l, t, r, b) 的目标框
            eps: 数值稳定性的小常数
            reduction: 'none'|'mean'|'sum' 损失聚合方式
        
        返回:
            损失值，根据reduction方式返回不同形状
        """
        # 确保输入是有效的
        if pred.shape[0] == 0 or target.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 裁剪负值为0，确保有效的边界框
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        
        # 提取左上右下距离
        pred_l, pred_t, pred_r, pred_b = pred.unbind(dim=-1)
        target_l, target_t, target_r, target_b = target.unbind(dim=-1)
        
        # 计算预测框和目标框的宽度和高度
        pred_width = pred_l + pred_r
        pred_height = pred_t + pred_b
        target_width = target_l + target_r
        target_height = target_t + target_b
        
        # 计算框的面积
        pred_area = pred_width * pred_height
        target_area = target_width * target_height
        
        # 计算交集区域的宽度和高度
        intersect_width = torch.min(pred_l, target_l) + torch.min(pred_r, target_r)
        intersect_height = torch.min(pred_t, target_t) + torch.min(pred_b, target_b)
        
        # 计算交集面积
        intersection = intersect_width * intersect_height
        
        # 计算并集面积 = A + B - 交集
        union = pred_area + target_area - intersection
        
        # 计算IoU = 交集/并集
        iou = torch.clamp(intersection / (union + eps), min=0, max=1.0)
        
        # 计算最小包围框的宽度和高度
        enclosing_width = torch.max(pred_l, target_l) + torch.max(pred_r, target_r)
        enclosing_height = torch.max(pred_t, target_t) + torch.max(pred_b, target_b)
        
        # 计算最小包围框的面积
        enclosing_area = enclosing_width * enclosing_height
        
        # 计算GIoU = IoU - (包围框面积 - 并集面积)/包围框面积
        giou = iou - torch.clamp((enclosing_area - union) / (enclosing_area + eps), min=-1.0, max=1.0)
        
        # 检查并处理无效值
        valid_mask = ~torch.isnan(giou) & ~torch.isinf(giou)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        # 过滤有效值
        giou = giou[valid_mask]
        
        # GIoU Loss = 1 - GIoU，范围[0,2]
        loss = 1 - giou
        
        # 根据reduction方式返回
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"不支持的reduction模式: {reduction}")
                    
    def _analyze_activation_stats(self):
        """分析模型激活的统计信息"""
        if not hasattr(self, '_activation_stats') or not self._activation_stats:
            return None
        
        stats = self._activation_stats
        if not stats or len(stats) == 0:
            return None
            
        # 取最近10个样本计算平均值
        result = {
            'cls_mean': np.mean([s.get('cls_mean', 0) for s in stats[-10:]]),
            'cls_pos_mean': np.mean([s.get('cls_pos_mean', 0) for s in stats[-10:]]),
            'cls_neg_mean': np.mean([s.get('cls_neg_mean', 0) for s in stats[-10:]]),
            'ctr_mean': np.mean([s.get('ctr_mean', 0) for s in stats[-10:]]),
            'final_conf_mean': np.mean([s.get('final_conf_mean', 0) for s in stats[-10:]]),
            'pos_ratio': np.mean([s.get('pos_ratio', 0) for s in stats[-10:]])
        }
        return result
        
    def log_activation_stats(self, cls_scores, ctr_scores, labels=None):
        """记录激活统计信息"""
        if not hasattr(self, '_activation_stats'):
            self._activation_stats = []
            
        # 转换为概率
        cls_probs = torch.sigmoid(cls_scores)
        ctr_probs = torch.sigmoid(ctr_scores)
        
        stats = {
            'cls_mean': cls_probs.mean().item(),
            'ctr_mean': ctr_probs.mean().item(),
            'final_conf_mean': self.compute_confidence(cls_probs, ctr_probs).mean().item()
        }
        
        # 如果有标签信息，计算正负样本的统计
        if labels is not None:
            pos_mask = labels > 0
            if pos_mask.sum() > 0:
                stats['cls_pos_mean'] = cls_probs[pos_mask].mean().item()
            if (~pos_mask).sum() > 0:
                stats['cls_neg_mean'] = cls_probs[~pos_mask].mean().item()
            stats['pos_ratio'] = pos_mask.float().mean().item()
            
        self._activation_stats.append(stats)
        
        # 限制缓冲区大小
        if len(self._activation_stats) > 100:
            self._activation_stats = self._activation_stats[-100:]
            
        return stats
        
    def focal_loss(self, logits, targets, alpha=0.5, gamma=1.5, reduction="sum"):
        """
        改进的Focal Loss实现，加入类别权重处理
        """
        # 计算正样本比例，确保数值稳定性
        pos_weight = (targets.sum() + 1e-10) / (targets.numel() + 1e-10)
        neg_weight = 1.0 - pos_weight
        
        # 修复警告：使用推荐的克隆方法而不是torch.tensor()
        pos_weight_value = pos_weight.clone().detach()
        neg_weight_value = neg_weight.clone().detach()
        
        # 限制权重范围，避免极端值
        pos_weight_value = torch.clamp(pos_weight_value, min=0.1, max=0.9).item()
        neg_weight_value = torch.clamp(neg_weight_value, min=0.1, max=0.9).item()
        
        # 调整alpha值以处理类别不平衡
        dynamic_alpha = torch.where(targets > 0, 
                                torch.ones_like(targets) * max(alpha, pos_weight_value), 
                                torch.ones_like(targets) * max(1-alpha, neg_weight_value))
        
        # 计算标准的BCE损失
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # 计算focal loss因子
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_factor = ((1 - p_t) ** gamma)
        
        # 应用focal factor和动态alpha
        loss = dynamic_alpha * focal_factor * ce_loss

        # 处理无效值
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

        if reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss
    
    def compute_confidence(self, cls_scores, centernesses):
        """计算最终置信度，使用加权平均"""
        # 确保维度匹配
        if cls_scores.dim() == 2 and centernesses.dim() == 1:
            centernesses = centernesses.unsqueeze(1).expand(-1, cls_scores.shape[1])
        elif cls_scores.dim() == 2 and centernesses.dim() == 2 and centernesses.shape[1] == 1:
            centernesses = centernesses.expand(-1, cls_scores.shape[1])
        
        # 添加偏置调整，适度提高置信度
        cls_adj = torch.clamp(cls_scores + self.conf_offset, min=0.0, max=1.0)
        ctr_adj = torch.clamp(centernesses, min=0.0, max=1.0)
        
        # 使用加权平均
        confidence = self.conf_beta * cls_adj + (1 - self.conf_beta) * ctr_adj
        
        return confidence

    def compute_centerness(self, left_right, top_bottom, epsilon=1e-6):
        """安全计算centerness，防止NaN"""
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        
        # 确保所有值都是正数
        left_right = torch.clamp(left_right, min=1e-4)
        top_bottom = torch.clamp(top_bottom, min=1e-4)
        
        # 安全计算最小值/最大值比例
        min_left_right = torch.min(left_right, dim=1)[0]
        max_left_right = torch.max(left_right, dim=1)[0]
        min_top_bottom = torch.min(top_bottom, dim=1)[0]
        max_top_bottom = torch.max(top_bottom, dim=1)[0]
        
        # 避免除以零，并确保比值在[0,1]范围内
        left_right_ratio = torch.clamp(min_left_right / max_left_right, min=0.0, max=1.0)
        top_bottom_ratio = torch.clamp(min_top_bottom / max_top_bottom, min=0.0, max=1.0)
        
        # 计算并返回centerness
        centerness = torch.sqrt(left_right_ratio * top_bottom_ratio)
        return centerness

    def forward(self, feats, targets=None, training=False, hw=None):
        # 增加全局步数
        self.global_step += 1
        
        # 确保所有特征尺寸一致性 - 添加特征调整
        feats_adjusted = []
        for i, feat in enumerate(feats):
            if feat.dim() == 4:
                if i == 0:  # 第一个特征作为参考
                    feats_adjusted.append(feat)
                else:  # 后续特征调整到与第一个相同的空间尺寸
                    first_h, first_w = feats[0].shape[2], feats[0].shape[3]
                    feat_resized = F.interpolate(
                        feat, 
                        size=(first_h, first_w), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    feats_adjusted.append(feat_resized)
            else:
                feats_adjusted.append(feat)
        
        # 使用调整后的特征
        feats = feats_adjusted
        
        # 输入检查
        if not feats or len(feats) == 0:
            if training:
                device = next(self.parameters()).device
                return dict(loss_cls=torch.tensor(0.0, device=device), 
                          loss_reg=torch.tensor(0.0, device=device), 
                          loss_ctr=torch.tensor(0.0, device=device), 
                          total_loss=torch.tensor(0.0, device=device))
            else:
                device = next(self.parameters()).device
                return torch.zeros(1, 1, 5 + self.num_classes, device=device)

        cls_scores = []
        reg_preds = []
        centernesses = []
        
        # 为每个特征层创建预测
        for i, x in enumerate(feats):
            if i >= len(self.cls_convs):
                continue
                
            # 检查张量有效性
            if not isinstance(x, torch.Tensor):
                if self.global_step % 200 == 0:
                    print(f"[WARNING] Feature {i} is not a tensor: {type(x)}")
                continue
            
            if x.dim() not in [3, 4]:
                if self.global_step % 200 == 0:
                    print(f"[WARNING] Feature {i} has unexpected dimensions: {x.dim()}, shape={x.shape}")
                continue
            
            if x.dim() == 3:
                x = x.unsqueeze(0)
                
            if x.numel() == 0 or 0 in x.shape:
                if self.global_step % 200 == 0:
                    print(f"[WARNING] Empty tensor at feature level {i}, shape={x.shape}")
                continue
            
            try:
                # 应用卷积层
                cls_feat = self.cls_convs[i](x)
                reg_feat = self.reg_convs[i](x)
                
                # 生成预测
                cls_score = self.cls_preds[i](cls_feat)
                
                # 获取原始回归预测
                raw_reg = self.reg_preds[i](reg_feat)
                
                # 在生成回归预测后添加
                if training and i == 0 and self.global_step % 200 == 0:
                    # 打印原始回归输出统计信息
                    print(f"[LEARN] 回归网络原始输出: min={raw_reg.min().item():.4f}, max={raw_reg.max().item():.4f}, mean={raw_reg.mean().item():.4f}, std={raw_reg.std().item():.4f}")
                
                # 使用exp函数确保回归预测为正，并限制最大值避免数值不稳定
                reg_pred = torch.exp(torch.clamp(raw_reg, max=4.0))
                
                # 在激活后打印统计信息
                if training and i == 0 and self.global_step % 200 == 0:
                    print(f"[LEARN] 激活后: min={reg_pred.min().item():.4f}, max={reg_pred.max().item():.4f}, mean={reg_pred.mean().item():.4f}, std={reg_pred.std().item():.4f}")
                
                # 限制最小尺寸
                reg_pred = torch.max(reg_pred, torch.tensor(self.min_box_size, 
                                                           device=reg_pred.device,
                                                           dtype=reg_pred.dtype))
                
                # 生成centerness预测
                centerness = self.centerness_preds[i](reg_feat)
                
                # 检查输出有效性
                if cls_score.numel() == 0 or reg_pred.numel() == 0 or centerness.numel() == 0:
                    if self.global_step % 200 == 0:
                        print(f"[WARNING] Empty output tensors at level {i}")
                    continue
                
                # 记录激活统计信息 - 减少频率
                if training and i == 0 and self.global_step % 200 == 0:
                    flat_cls = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                    flat_ctr = centerness.permute(0, 2, 3, 1).reshape(-1)
                    stats = self.log_activation_stats(flat_cls, flat_ctr)
                    if self.debug_mode:
                        print(f"[DEBUG] Activation Stats: cls_mean={stats['cls_mean']:.4f}, ctr_mean={stats['ctr_mean']:.4f}, final_conf={stats['final_conf_mean']:.4f}")
                
                # 保存预测结果
                cls_scores.append(cls_score)
                reg_preds.append(reg_pred)
                centernesses.append(centerness)
                
            except Exception as e:
                if self.global_step % 200 == 0:
                    print(f"[ERROR] Failed to process feature {i}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        if len(cls_scores) == 0:
            if self.global_step % 200 == 0:
                print(f"[WARNING] No valid feature outputs generated")
            if training:
                device = next(self.parameters()).device
                return dict(loss_cls=torch.tensor(0.0, device=device), 
                          loss_reg=torch.tensor(0.0, device=device), 
                          loss_ctr=torch.tensor(0.0, device=device), 
                          total_loss=torch.tensor(0.0, device=device))
            else:
                device = next(self.parameters()).device
                return torch.zeros(1, 1, 5 + self.num_classes, device=device)

        if training:
            return self.loss(cls_scores, reg_preds, centernesses, targets)
        else:
            return self.decode_outputs(cls_scores, reg_preds, centernesses)

    def loss(self, cls_scores, reg_preds, centernesses, targets, strides=None):
        """安全的损失计算函数，确保不会产生NaN"""
        # 检查输入是否有效
        if not cls_scores or len(cls_scores) == 0:
            device = next(self.parameters()).device
            dummy_param = next(self.parameters())
            zero_loss = dummy_param.mean() * 0
            return {
                'loss_cls': zero_loss,
                'loss_reg': zero_loss,
                'loss_ctr': zero_loss,
                'total_loss': zero_loss
            }
        
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        strides = strides or self.strides
        
        # 初始化损失
        dummy_param = next(self.parameters())
        cls_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        reg_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        ctr_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        
        # 统计正样本数量
        num_pos = 0
        
        # 处理每个特征层
        for level_idx, stride in enumerate(strides):
            if level_idx >= len(cls_scores):
                continue
            
            # 获取当前特征层预测
            cls_pred = cls_scores[level_idx]
            reg_pred = reg_preds[level_idx]
            centerness_pred = centernesses[level_idx]
            
            # 特征图大小
            h, w = cls_pred.shape[2:]
            
            # 生成网格点坐标
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            locations = torch.stack([
                (grid_x + 0.5) * stride,
                (grid_y + 0.5) * stride
            ], dim=-1).view(-1, 2)  # [h*w, 2]
            
            # 处理每个批次
            for batch_idx in range(batch_size):
                if batch_idx >= len(targets):
                    continue
                
                # 获取当前批次的目标
                batch_targets = targets[batch_idx]
                if batch_targets.numel() == 0:
                    continue
                
                # 获取目标类别和边界框
                gt_cls = batch_targets[:, 0].long()
                gt_bbox = batch_targets[:, 1:5]  # [cx, cy, w, h]
                
                # 将目标框转换为XYXY格式
                gt_x1 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
                gt_y1 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
                gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
                gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
                
                # 计算点到目标框边界的距离
                xs, ys = locations[:, 0].unsqueeze(1), locations[:, 1].unsqueeze(1)
                left = xs - gt_x1.unsqueeze(0)     # [num_locations, num_targets]
                top = ys - gt_y1.unsqueeze(0)      # [num_locations, num_targets]
                right = gt_x2.unsqueeze(0) - xs    # [num_locations, num_targets]
                bottom = gt_y2.unsqueeze(0) - ys   # [num_locations, num_targets]
                
                # 计算回归目标 [num_locations, num_targets, 4]
                reg_targets = torch.stack([left, top, right, bottom], dim=-1)
                
                # 确定哪些点在目标框内部
                inside_gt_mask = reg_targets.min(dim=-1)[0] > 0  # [num_locations, num_targets]
                
                # 如果没有内部点，强制使用最近的点
                if not inside_gt_mask.any():
                    # 计算点到目标中心的距离
                    gt_centers = torch.stack([(gt_x1 + gt_x2) / 2, (gt_y1 + gt_y2) / 2], dim=1)
                    distances = torch.sqrt(
                        (xs - gt_centers[:, 0].unsqueeze(0)) ** 2 + 
                        (ys - gt_centers[:, 1].unsqueeze(0)) ** 2
                    )
                    
                    # 选择每个位置最近的目标
                    min_dist, min_inds = distances.min(dim=1)
                    
                    # 取最近的10%点作为强制正样本
                    k = max(1, int(locations.size(0) * 0.1))
                    _, topk_inds = min_dist.topk(k, largest=False)
                    
                    # 创建强制正样本掩码
                    inside_gt_mask = torch.zeros_like(distances, dtype=torch.bool)
                    for i in range(len(topk_inds)):
                        inside_gt_mask[topk_inds[i], min_inds[topk_inds[i]]] = True
                    
                    print(f"[DEBUG] Forced {inside_gt_mask.sum().item()} positive samples for batch {batch_idx}")
                
                # 跳过没有正样本的情况
                if not inside_gt_mask.any():
                    print(f"[WARNING] Still no positive samples for batch {batch_idx}, level {level_idx}")
                    continue
                
                # 每个位置选择面积最小的目标框
                areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # [num_targets]
                areas = areas.unsqueeze(0).expand(locations.size(0), -1)  # [num_locations, num_targets]
                
                # 复制areas以避免修改原始张量
                areas_for_min = areas.clone()
                areas_for_min[~inside_gt_mask] = float('inf')  # 只考虑inside_flag=True的区域
                min_area_inds = areas_for_min.argmin(dim=1)    # [num_locations]
                
                # 正样本掩码
                pos_mask = inside_gt_mask.any(dim=1)  # [num_locations]
                pos_count = pos_mask.sum().item()
                if pos_count == 0:
                    print(f"[WARNING] No positive samples after area filtering for batch {batch_idx}")
                    continue
                    
                print(f"[DEBUG] Found {pos_count} positive samples for batch {batch_idx}, level {level_idx}")
                num_pos += pos_count
                
                # 获取每个正样本位置对应的目标框
                pos_inds = torch.nonzero(pos_mask).squeeze(1)  # [num_pos]
                target_inds = min_area_inds[pos_mask]  # [num_pos]
                
                # 获取正样本的回归目标
                pos_reg_targets = reg_targets[pos_inds, target_inds]  # [num_pos, 4]
                
                # 安全计算centerness目标
                centerness_targets = self._calculate_centerness(pos_reg_targets)
                
                # 获取预测值
                cls_pred_flat = cls_pred[batch_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred_flat = reg_pred[batch_idx].permute(1, 2, 0).reshape(-1, 4)
                centerness_pred_flat = centerness_pred[batch_idx].permute(1, 2, 0).reshape(-1)
                
                # 获取正样本的预测
                pos_cls_preds = cls_pred_flat[pos_inds]
                pos_reg_preds = reg_pred_flat[pos_inds]
                pos_centerness_preds = centerness_pred_flat[pos_inds]
                
                # 创建分类目标
                pos_cls_targets = torch.zeros_like(pos_cls_preds)
                pos_cls_targets[torch.arange(len(pos_inds)), gt_cls[target_inds]] = 1.0
                
                # 计算分类损失（带有异常值检测）
                try:
                    cls_loss_batch = F.binary_cross_entropy_with_logits(
                        pos_cls_preds, pos_cls_targets, reduction='none'
                    )
                    
                    # 检查分类损失是否有NaN
                    if torch.isnan(cls_loss_batch).any():
                        print(f"[WARNING] NaN in cls_loss_batch, replacing with zeros")
                        cls_loss_batch = torch.where(torch.isnan(cls_loss_batch), torch.zeros_like(cls_loss_batch), cls_loss_batch)
                    
                    cls_loss = cls_loss + cls_loss_batch.sum()
                except Exception as e:
                    print(f"[ERROR] Exception in classification loss: {e}")
                    # 继续处理其他损失
                
                # 计算回归损失
                try:
                    # 将距离转换为边界框
                    pred_boxes = self._distance2bbox(locations[pos_inds], pos_reg_preds)
                    target_boxes = self._distance2bbox(locations[pos_inds], pos_reg_targets)
                    
                    # 打印边界框统计信息
                    print(f"[DEBUG] pred_boxes stats: min={pred_boxes.min().item():.4f}, max={pred_boxes.max().item():.4f}")
                    print(f"[DEBUG] target_boxes stats: min={target_boxes.min().item():.4f}, max={target_boxes.max().item():.4f}")
                    
                    # 使用SmoothL1Loss
                    l1_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='none')
                    
                    # 检查l1_loss是否有NaN
                    if torch.isnan(l1_loss).any() or torch.isinf(l1_loss).any():
                        print(f"[WARNING] NaN or Inf in l1_loss, replacing with zeros")
                        l1_loss = torch.where(torch.isnan(l1_loss) | torch.isinf(l1_loss), torch.zeros_like(l1_loss), l1_loss)
                    
                    # 计算IoU损失
                    iou = self._bbox_overlaps(pred_boxes, target_boxes)
                    iou_loss = 1 - iou
                    
                    # 检查iou_loss是否有NaN
                    if torch.isnan(iou_loss).any():
                        print(f"[WARNING] NaN in iou_loss, replacing with zeros")
                        iou_loss = torch.where(torch.isnan(iou_loss), torch.zeros_like(iou_loss), iou_loss)
                    
                    # 使用centerness加权的损失
                    centerness_weights = centerness_targets.clamp(min=1e-6)  # 确保权重不为零
                    
                    # 混合回归损失
                    l1_weight = 1.0
                    iou_weight = 0.5
                    
                    reg_loss = reg_loss + (l1_loss.sum(dim=1) * centerness_weights).sum() * l1_weight
                    reg_loss = reg_loss + (iou_loss * centerness_weights).sum() * iou_weight
                except Exception as e:
                    print(f"[ERROR] Exception in regression loss: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 计算centerness损失
                try:
                    ctr_loss_batch = F.binary_cross_entropy_with_logits(
                        pos_centerness_preds, centerness_targets, reduction='none'
                    )
                    
                    # 检查centerness损失是否有NaN
                    if torch.isnan(ctr_loss_batch).any():
                        print(f"[WARNING] NaN in ctr_loss_batch, replacing with zeros")
                        ctr_loss_batch = torch.where(torch.isnan(ctr_loss_batch), torch.zeros_like(ctr_loss_batch), ctr_loss_batch)
                    
                    ctr_loss = ctr_loss + ctr_loss_batch.sum()
                except Exception as e:
                    print(f"[ERROR] Exception in centerness loss: {e}")
        
        # 平均损失
        num_pos = max(1, num_pos)
        cls_loss = cls_loss / num_pos
        reg_loss = reg_loss / num_pos
        ctr_loss = ctr_loss / num_pos
        
        # 检查最终损失是否有NaN
        if torch.isnan(cls_loss):
            print(f"[WARNING] Final cls_loss is NaN, replacing with zero")
            cls_loss = torch.zeros_like(cls_loss)
        if torch.isnan(reg_loss):
            print(f"[WARNING] Final reg_loss is NaN, replacing with zero")
            reg_loss = torch.zeros_like(reg_loss)
        if torch.isnan(ctr_loss):
            print(f"[WARNING] Final ctr_loss is NaN, replacing with zero")
            ctr_loss = torch.zeros_like(ctr_loss)
        
        # 确保所有损失都有梯度
        total_loss = cls_loss + reg_loss + ctr_loss
        
        # 打印损失信息
        print(f"[DEBUG] Loss breakdown - cls: {cls_loss.item():.4f}, reg: {reg_loss.item():.4f}, ctr: {ctr_loss.item():.4f}, total: {total_loss.item():.4f}")
        
        return {
            'loss_cls': cls_loss,
            'loss_reg': reg_loss,
            'loss_ctr': ctr_loss,
            'total_loss': total_loss
        }

    def decode_outputs(self, cls_scores, reg_preds, centernesses):
        if len(cls_scores) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
        # 找到所有特征层中的最小批次大小
        min_batch_size = min(cls.shape[0] for cls in cls_scores if hasattr(cls, 'shape') and cls.shape[0] > 0)
        B = min_batch_size
        outputs = []

        # 处理每个批次
        for batch_idx in range(B):
            boxes_all, scores_all, labels_all = [], [], []
            detection_count = 0
            
            # 记录各个特征层的置信度和框尺寸
            level_stats = []

            # 处理每个特征层
            for level_idx in range(len(cls_scores)):
                try:
                    if level_idx >= len(self.strides):
                        break
                    
                    # 获取当前层的预测
                    cls = cls_scores[level_idx]
                    reg = reg_preds[level_idx] 
                    ctr = centernesses[level_idx]
                    
                    # 检查张量有效性
                    if not isinstance(cls, torch.Tensor) or not isinstance(reg, torch.Tensor) or not isinstance(ctr, torch.Tensor):
                        continue
                    
                    if batch_idx >= cls.shape[0] or batch_idx >= reg.shape[0] or batch_idx >= ctr.shape[0]:
                        break
                except Exception as e:
                    continue
                    
                try:
                    # 获取当前批次的预测
                    cls_batch = torch.sigmoid(cls[batch_idx])
                    reg_batch = reg[batch_idx]
                    ctr_batch = torch.sigmoid(ctr[batch_idx])

                    # 处理不同维度的centerness
                    if ctr_batch.dim() == 3 and ctr_batch.shape[0] == 1:
                        ctr_batch = ctr_batch.squeeze(0)
                    elif ctr_batch.dim() == 1:
                        # 如果是1D张量，尝试重塑
                        sqrt_size = int(ctr_batch.shape[0] ** 0.5)
                        if sqrt_size * sqrt_size == ctr_batch.shape[0]:
                            ctr_batch = ctr_batch.view(sqrt_size, sqrt_size)
                        else:
                            continue

                    # 检查张量有效性
                    if cls_batch.numel() == 0 or reg_batch.numel() == 0 or ctr_batch.numel() == 0:
                        continue

                    # 获取形状信息
                    C, H, W = cls_batch.shape
                    stride = self.strides[level_idx]

                    if H <= 0 or W <= 0:
                        continue

                    # 生成网格坐标
                    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=cls_batch.device), 
                                                   torch.arange(W, device=cls_batch.device), 
                                                   indexing='ij')
                    grid_x = grid_x.float() * stride + stride // 2
                    grid_y = grid_y.float() * stride + stride // 2

                    # 获取回归预测值
                    l = reg_batch[0]
                    t = reg_batch[1] 
                    r = reg_batch[2]
                    b_reg = reg_batch[3]

                    # 确保最小边长
                    l = torch.clamp(l, min=self.min_box_size/2)
                    t = torch.clamp(t, min=self.min_box_size/2)
                    r = torch.clamp(r, min=self.min_box_size/2)
                    b_reg = torch.clamp(b_reg, min=self.min_box_size/2)

                    # 计算边界框坐标
                    x1 = grid_x - l
                    y1 = grid_y - t
                    x2 = grid_x + r
                    y2 = grid_y + b_reg
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

                    # 重塑分类和centerness得分
                    cls_reshaped = cls_batch.permute(1, 2, 0).reshape(-1, C)
                    ctr_reshaped = ctr_batch.reshape(-1, 1)
                    
                    # 计算最终置信度
                    scores = self.compute_confidence(cls_reshaped, ctr_reshaped)

                    # 获取每个位置的最大分数和对应类别
                    max_scores, labels = scores.max(dim=1)
                    
                    # 保存特征层统计信息
                    level_stats.append({
                        'level': level_idx,
                        'stride': stride,
                        'cls_mean': cls_reshaped.mean().item(),
                        'ctr_mean': ctr_reshaped.mean().item(),
                        'score_mean': max_scores.mean().item(),
                        'score_high_ratio': (max_scores > 0.5).float().mean().item(),
                        'width_mean': (x2 - x1).mean().item(),
                        'height_mean': (y2 - y1).mean().item()
                    })
                    
                    # 应用置信度阈值过滤
                    mask = max_scores > self.conf_threshold
                    
                    # 如果所有框都被过滤掉，保留少量最高置信度的框
                    if mask.sum() == 0:
                        k = min(5, len(max_scores))
                        if k > 0:
                            topk_scores, topk_indices = torch.topk(max_scores, k)
                            mask = torch.zeros_like(max_scores, dtype=torch.bool)
                            mask[topk_indices] = True

                    if mask.sum() == 0:
                        continue

                    # 获取过滤后的检测结果
                    boxes = boxes[mask]
                    scores = max_scores[mask]
                    labels = labels[mask].long()
                    detection_count += len(boxes)

                    # 记录边界框统计信息
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    if self.debug_mode and self.global_step % 200 == 0 and batch_idx == 0 and level_idx == 0:
                        print(f"[DEBUG] Level {level_idx} Box Stats:")
                        print(f"  Width: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
                        print(f"  Height: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")

                    # 按类别进行NMS
                    for class_id in torch.unique(labels):
                        class_id = int(class_id.item())
                        
                        # 获取当前类别的框
                        cls_mask = labels == class_id
                        cls_boxes = boxes[cls_mask]
                        cls_scores = scores[cls_mask]
                        
                        if len(cls_boxes) == 0:
                            continue
                            
                        # 过滤无效框
                        valid_box_mask = (cls_boxes[:, 2] > cls_boxes[:, 0] + self.min_box_size) & (cls_boxes[:, 3] > cls_boxes[:, 1] + self.min_box_size)
                        if valid_box_mask.sum() == 0:
                            continue
                            
                        cls_boxes = cls_boxes[valid_box_mask]
                        cls_scores = cls_scores[valid_box_mask]
                        
                        if len(cls_boxes) > 0:
                            # 应用NMS
                            keep = ops.nms(cls_boxes, cls_scores, iou_threshold=self.nms_threshold)

                            # 保存结果
                            boxes_all.append(cls_boxes[keep])
                            scores_all.append(cls_scores[keep])
                            labels_all.append(torch.full_like(cls_scores[keep], class_id, dtype=torch.long))

                except Exception as e:
                    if self.global_step % 200 == 0:
                        print(f"[ERROR] Level {level_idx} processing error: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            # 打印特征层统计信息 - 减少频率
            if self.debug_mode and self.global_step % 200 == 0 and batch_idx == 0 and level_stats:
                print(f"[DEBUG] Feature level statistics:")
                for stats in level_stats:
                    print(f"  Level {stats['level']} (stride={stats['stride']}): "
                          f"cls_mean={stats['cls_mean']:.4f}, "
                          f"ctr_mean={stats['ctr_mean']:.4f}, "
                          f"score_mean={stats['score_mean']:.4f}, "
                          f"high_score_ratio={stats['score_high_ratio']:.4f}, "
                          f"size=({stats['width_mean']:.1f}x{stats['height_mean']:.1f})")

            # 处理所有检测结果
            if boxes_all:
                try:
                    # 合并所有检测结果
                    boxes_all = torch.cat(boxes_all)
                    scores_all = torch.cat(scores_all)
                    labels_all = torch.cat(labels_all)

                    # 转换为中心点坐标格式 (XYWH)
                    xywh = torch.zeros_like(boxes_all)
                    xywh[:, 0] = (boxes_all[:, 0] + boxes_all[:, 2]) / 2
                    xywh[:, 1] = (boxes_all[:, 1] + boxes_all[:, 3]) / 2
                    xywh[:, 2] = boxes_all[:, 2] - boxes_all[:, 0]
                    xywh[:, 3] = boxes_all[:, 3] - boxes_all[:, 1]

                    # 过滤极小的框
                    valid_size = (xywh[:, 2] >= self.min_box_size) & (xywh[:, 3] >= self.min_box_size)
                    if valid_size.sum() > 0:
                        xywh = xywh[valid_size]
                        scores_all = scores_all[valid_size]
                        labels_all = labels_all[valid_size]
                    
                    # 放宽长宽比过滤条件
                    aspect_ratio = xywh[:, 2] / (xywh[:, 3] + 1e-6)
                    normal_ratio = (aspect_ratio > 0.2) & (aspect_ratio < 5.0)
                    if normal_ratio.sum() > 0:
                        xywh = xywh[normal_ratio]
                        scores_all = scores_all[normal_ratio]
                        labels_all = labels_all[normal_ratio]

                    # 创建类别的one-hot编码
                    onehot = torch.zeros(len(labels_all), self.num_classes, device=boxes_all.device)
                    indices = torch.arange(len(labels_all), device=boxes_all.device).long()
                    labels_long = labels_all.long()
                    
                    # 确保标签在有效范围内
                    labels_long = torch.clamp(labels_long, 0, self.num_classes - 1)
                    onehot[indices, labels_long] = 1.0

                    # 创建最终输出格式
                    output = torch.cat([xywh, scores_all.unsqueeze(1), onehot], dim=1)
                    
                    # 调试信息 - 减少频率
                    if self.debug_mode and self.global_step % 200 == 0 and batch_idx == 0:
                        print(f"[DEBUG] Final output: {len(output)} boxes")
                        if len(scores_all) > 0:
                            print(f"[DEBUG] Score stats: min={scores_all.min().item():.4f}, max={scores_all.max().item():.4f}, mean={scores_all.mean().item():.4f}")
                            print(f"[DEBUG] Size stats: width={xywh[:, 2].mean().item():.1f}x{xywh[:, 3].mean().item():.1f}")
                
                except Exception as e:
                    if self.global_step % 200 == 0:
                        print(f"[ERROR] Failed to process final detections: {e}")
                    device = cls_scores[0].device
                    # 创建一个伪检测结果
                    fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                    fake_detection[0, 4] = 0.001  # 极低的置信度
                    output = fake_detection
            else:
                # 如果没有检测到任何框，创建一个伪检测
                device = cls_scores[0].device
                fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                fake_detection[0, 4] = 0.001
                output = fake_detection

            outputs.append(output)

        if len(outputs) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
        # 打印置信度统计信息 - 减少频率
        if self.debug_mode and self.global_step % 200 == 0:
            all_scores = []
            for output in outputs:
                if output.numel() > 0:
                    all_scores.append(output[:, 4])
            
            if all_scores and len(all_scores) > 0 and all_scores[0].numel() > 0:
                all_scores = torch.cat(all_scores)
                high_conf = (all_scores > 0.5).sum().item()
                total = len(all_scores)
                high_conf_ratio = high_conf / total if total > 0 else 0
                print(f"[DEBUG] Detection Stats: total={total}, high_conf={high_conf}, ratio={high_conf_ratio:.4f}")
        
        # 在方法末尾的return语句前添加
        if len(outputs) > 0 and self.global_step % 200 == 0:
            all_widths = []
            all_heights = []
            for output in outputs:
                if output.numel() > 0:
                    # 获取预测的XYWH格式
                    xywh = output[:, :4]
                    widths = xywh[:, 2]
                    heights = xywh[:, 3]
                    all_widths.extend(widths.tolist())
                    all_heights.extend(heights.tolist())
            
            if all_widths and all_heights:
                # 计算统计值
                widths_tensor = torch.tensor(all_widths)
                heights_tensor = torch.tensor(all_heights)
                
                mean_width = widths_tensor.mean().item()
                std_width = widths_tensor.std().item()
                mean_height = heights_tensor.mean().item()
                std_height = heights_tensor.std().item()
                
                print(f"[LEARN] 评估预测尺寸: 宽={mean_width:.2f}±{std_width:.2f}, 高={mean_height:.2f}±{std_height:.2f}")
                
        # 格式化输出结果
        try:
            # 将所有输出填充到相同的大小
            max_detections = max(o.shape[0] for o in outputs)
            padded_outputs = []
            
            for output in outputs:
                if output.shape[0] < max_detections:
                    padding = torch.zeros(max_detections - output.shape[0], output.shape[1], device=output.device)
                    output = torch.cat([output, padding], dim=0)
                padded_outputs.append(output)
                
            result = torch.stack(padded_outputs, dim=0)
            return result
            
        except Exception as e:
            if self.global_step % 200 == 0:
                print(f"[ERROR] Failed to format output: {e}")
            device = next(self.parameters()).device
            return torch.zeros(min_batch_size, 1, 5 + self.num_classes, device=device)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.ops as ops

# class FCOSHead(nn.Module):
#     def __init__(self, num_classes=2, in_channels=[256, 512], strides=[8, 16], use_iou_loss=True, use_focal_loss=True, feat_channels=256):
#         super().__init__()
#         self.num_classes = num_classes
#         self.strides = strides
#         self.use_iou_loss = use_iou_loss
#         self.use_focal_loss = use_focal_loss
#         self.cls_convs = nn.ModuleList()
#         self.reg_convs = nn.ModuleList()
#         self.cls_preds = nn.ModuleList()
#         self.reg_preds = nn.ModuleList()
#         self.centerness_preds = nn.ModuleList()
        
#         # 损失权重因子
#         self.cls_factor = 1.0
#         self.reg_factor = 1.0
#         self.ctr_factor = 10.0

#         for i, in_c in enumerate(in_channels):
#             self.cls_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_c, feat_channels, 3, padding=1),
#                     nn.ReLU(),
#                     nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
#                     nn.ReLU(),
#                 )
#             )
#             self.reg_convs.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_c, feat_channels, 3, padding=1),
#                     nn.ReLU(),
#                     nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
#                     nn.ReLU(),
#                 )
#             )
#             self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 3, padding=1))
#             self.reg_preds.append(nn.Conv2d(feat_channels, 4, 3, padding=1))
#             self.centerness_preds.append(nn.Conv2d(feat_channels, 1, 3, padding=1))

#         self._loss_iou = nn.SmoothL1Loss(reduction="none")

#     def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
#         prob = torch.sigmoid(logits)
#         ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
#         p_t = prob * targets + (1 - prob) * (1 - targets)
#         loss = ce_loss * ((1 - p_t) ** gamma)

#         if alpha >= 0:
#             alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#             loss = alpha_t * loss

#         if reduction == "sum":
#             return loss.sum()
#         elif reduction == "mean":
#             return loss.mean()
#         else:
#             return loss

#     def compute_centerness(self, left_right, top_bottom, epsilon=1e-6):
#         """安全计算centerness，防止NaN"""
#         # 确保所有值都是正的
#         left_right = F.relu(left_right) + epsilon
#         top_bottom = F.relu(top_bottom) + epsilon
        
#         # 计算最小最大比
#         lr_ratio = torch.min(left_right, dim=1)[0] / torch.max(left_right, dim=1)[0]
#         tb_ratio = torch.min(top_bottom, dim=1)[0] / torch.max(top_bottom, dim=1)[0]
        
#         # 确保比值在有效范围内
#         lr_ratio = torch.clamp(lr_ratio, min=epsilon, max=1.0)
#         tb_ratio = torch.clamp(tb_ratio, min=epsilon, max=1.0)
        
#         # 计算centerness
#         centerness = torch.sqrt(lr_ratio * tb_ratio)
        
#         # 最后一道防线：替换任何可能的NaN
#         centerness = torch.nan_to_num(centerness, nan=0.5)
        
#         return centerness.clamp(min=0.0, max=1.0)

#     def forward(self, feats, targets=None, training=False, hw=None):
#         if not feats or len(feats) == 0:
#             if training:
#                 return dict(loss_cls=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
#                           loss_reg=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
#                           loss_ctr=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
#                           total_loss=torch.tensor(0.0, device=self.cls_preds[0].weight.device))
#             else:
#                 device = next(self.parameters()).device
#                 return torch.zeros(1, 1, 5 + self.num_classes, device=device)

#         cls_scores = []
#         reg_preds = []
#         centernesses = []
        
#         for i, x in enumerate(feats):
#             if i >= len(self.cls_convs):
#                 break
                
#             if not isinstance(x, torch.Tensor):
#                 raise ValueError(f"Expected tensor, got {type(x)}")
            
#             if x.dim() not in [3, 4]:
#                 raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor with shape {x.shape}")
            
#             if x.dim() == 3:
#                 x = x.unsqueeze(0)
                
#             if x.numel() == 0:
#                 print(f"[WARNING] Empty tensor at feature level {i}")
#                 continue
            
#             try:
#                 cls_feat = self.cls_convs[i](x)
#                 reg_feat = self.reg_convs[i](x)
#                 cls_score = self.cls_preds[i](cls_feat)
#                 reg_pred = F.relu(self.reg_preds[i](reg_feat))
#                 centerness = self.centerness_preds[i](reg_feat)
                
#                 if cls_score.numel() == 0 or reg_pred.numel() == 0 or centerness.numel() == 0:
#                     print(f"[WARNING] Empty output tensors at level {i}")
#                     continue
                    
#                 cls_scores.append(cls_score)
#                 reg_preds.append(reg_pred)
#                 centernesses.append(centerness)
                
#             except Exception as e:
#                 print(f"[ERROR] Failed to process feature {i}: {e}")
#                 continue

#         if len(cls_scores) == 0:
#             print(f"[WARNING] No valid feature outputs generated")
#             if training:
#                 device = next(self.parameters()).device
#                 return dict(loss_cls=torch.tensor(0.0, device=device), 
#                           loss_reg=torch.tensor(0.0, device=device), 
#                           loss_ctr=torch.tensor(0.0, device=device), 
#                           total_loss=torch.tensor(0.0, device=device))
#             else:
#                 device = next(self.parameters()).device
#                 return torch.zeros(1, 1, 5 + self.num_classes, device=device)

#         if training:
#             return self.loss(cls_scores, reg_preds, centernesses, targets)
#         else:
#             return self.decode_outputs(cls_scores, reg_preds, centernesses)

#     def loss(self, cls_scores, reg_preds, centernesses, targets):
#         device = cls_scores[0].device
        
#         min_batch_size = min(cls.shape[0] for cls in cls_scores)
        
#         feature_info = []
#         flat_cls_scores = []
#         flat_reg_preds = []
#         flat_centernesses = []
        
#         for i, (cls, reg, ctr) in enumerate(zip(cls_scores, reg_preds, centernesses)):
#             cls = cls[:min_batch_size]
#             reg = reg[:min_batch_size] 
#             ctr = ctr[:min_batch_size]
            
#             B, C, H, W = cls.shape
#             stride = self.strides[i] if i < len(self.strides) else 8
            
#             cls_flat = cls.permute(0, 2, 3, 1).reshape(B, H*W, C)
#             reg_flat = reg.permute(0, 2, 3, 1).reshape(B, H*W, 4)
#             ctr_flat = ctr.permute(0, 2, 3, 1).reshape(B, H*W, 1)
            
#             flat_cls_scores.append(cls_flat)
#             flat_reg_preds.append(reg_flat)
#             flat_centernesses.append(ctr_flat)
            
#             feature_info.append((H, W, stride))

#         all_cls_scores = torch.cat(flat_cls_scores, dim=1)
#         all_reg_preds = torch.cat(flat_reg_preds, dim=1)
#         all_centernesses = torch.cat(flat_centernesses, dim=1)

#         losses = dict(
#             loss_cls=torch.tensor(0.0, device=device),
#             loss_reg=torch.tensor(0.0, device=device),
#             loss_ctr=torch.tensor(0.0, device=device)
#         )
#         num_pos_total = 0
#         num_batches_with_pos = 0
#         B = min_batch_size

#         for batch_idx in range(B):
#             if batch_idx >= len(targets):
#                 continue
                
#             gt = targets[batch_idx]
            
#             if gt.numel() == 0:
#                 continue
                
#             gt_areas = gt[:, 3] * gt[:, 4]
#             valid_mask = gt_areas > 0
            
#             coord_valid = (gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 3] > 0) & (gt[:, 4] > 0)
#             valid_mask = valid_mask & coord_valid
            
#             gt_valid = gt[valid_mask]
            
#             if gt_valid.numel() == 0:
#                 continue

#             gt_cls = gt_valid[:, 0].long()
#             gt_boxes = gt_valid[:, 1:]

#             points = []
#             strides_all = []
#             for level_idx, (H, W, stride) in enumerate(feature_info):
#                 y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#                 x = x.flatten().float() * stride + stride // 2
#                 y = y.flatten().float() * stride + stride // 2
#                 p = torch.stack([x, y], dim=1).to(device)
#                 points.append(p)
#                 strides_all.append(torch.full((len(p),), stride, device=device))
            
#             if len(points) == 0:
#                 continue
                
#             points = torch.cat(points, dim=0)
#             strides_all = torch.cat(strides_all, dim=0)

#             num_points = points.shape[0]
#             num_gts = gt_boxes.shape[0]

#             gt_xy = gt_boxes[:, :2]
#             gt_wh = gt_boxes[:, 2:]
#             x1y1 = gt_xy - gt_wh / 2
#             x2y2 = gt_xy + gt_wh / 2
#             gt_area = (x2y2[:, 0] - x1y1[:, 0]) * (x2y2[:, 1] - x1y1[:, 1])

#             px = points[:, 0].unsqueeze(1)
#             py = points[:, 1].unsqueeze(1)
#             l = px - x1y1[:, 0]
#             t = py - x1y1[:, 1]
#             r = x2y2[:, 0] - px
#             b = x2y2[:, 1] - py
#             reg_targets = torch.stack([l, t, r, b], dim=2)

#             inside_box = reg_targets.min(dim=2)[0] > 0

#             center = gt_xy
#             radius_factor = 1.5
#             radius = strides_all.unsqueeze(1) * radius_factor
#             center_x = torch.abs(px - center[:, 0])
#             center_y = torch.abs(py - center[:, 1])
#             center_dist = (center_x < radius) & (center_y < radius)
            
#             is_pos = inside_box & center_dist
            
#             if is_pos.sum().item() == 0:
#                 is_pos = inside_box | center_dist

#             gt_area_expanded = gt_area.unsqueeze(0).repeat(num_points, 1)
#             gt_area_expanded[~is_pos] = float('inf')
#             min_area, min_inds = gt_area_expanded.min(dim=1)
#             pos_mask = min_area < float('inf')
#             num_pos = pos_mask.sum().item()
            
#             if num_pos == 0:
#                 continue

#             num_batches_with_pos += 1
#             matched_inds = min_inds[pos_mask]
#             matched_boxes = gt_boxes[matched_inds]
#             matched_cls = gt_cls[matched_inds]

#             cls_target = torch.zeros_like(all_cls_scores[batch_idx])
#             cls_target[pos_mask, matched_cls] = 1.0
            
#             # 计算分类损失
#             try:
#                 loss_cls = self.focal_loss(all_cls_scores[batch_idx], cls_target, alpha=0.25, gamma=2.0) / max(num_pos, 1)
#                 if torch.isnan(loss_cls).any():
#                     print(f"[WARNING] NaN in loss_cls, setting to zero")
#                     loss_cls = torch.tensor(0.0, device=device)
#             except Exception as e:
#                 print(f"[ERROR] Failed to compute cls loss: {e}")
#                 loss_cls = torch.tensor(0.0, device=device)
            
#             # 计算回归损失
#             try:
#                 matched_x1y1 = matched_boxes[:, :2] - matched_boxes[:, 2:] / 2
#                 matched_x2y2 = matched_boxes[:, :2] + matched_boxes[:, 2:] / 2
#                 matched_l = points[pos_mask][:, 0] - matched_x1y1[:, 0]
#                 matched_t = points[pos_mask][:, 1] - matched_x1y1[:, 1]
#                 matched_r = matched_x2y2[:, 0] - points[pos_mask][:, 0]
#                 matched_b = matched_x2y2[:, 1] - points[pos_mask][:, 1]
#                 reg_target = torch.stack([matched_l, matched_t, matched_r, matched_b], dim=1)
                
#                 reg_pred = all_reg_preds[batch_idx][pos_mask]
                
#                 # 过滤无效回归目标
#                 valid_reg_mask = (reg_target > 0).all(dim=1)
#                 if valid_reg_mask.sum() > 0:
#                     iou_loss = self._loss_iou(
#                         reg_pred[valid_reg_mask], 
#                         reg_target[valid_reg_mask]
#                     ).mean()
                    
#                     if torch.isnan(iou_loss).any():
#                         print(f"[WARNING] NaN in iou_loss, setting to zero")
#                         iou_loss = torch.tensor(0.0, device=device)
#                 else:
#                     iou_loss = torch.tensor(0.0, device=device)
#             except Exception as e:
#                 print(f"[ERROR] Failed to compute reg loss: {e}")
#                 iou_loss = torch.tensor(0.0, device=device)

#             # 计算centerness损失
#             try:
#                 left_right = reg_target[:, [0, 2]]
#                 top_bottom = reg_target[:, [1, 3]]
                
#                 # 使用安全的centerness计算函数
#                 centerness = self.compute_centerness(left_right, top_bottom)
#                 pred_ctr = all_centernesses[batch_idx][pos_mask].squeeze(-1)
                
#                 if pred_ctr.numel() > 0:
#                     loss_ctr = F.binary_cross_entropy_with_logits(
#                         pred_ctr, centerness, reduction='mean')
                    
#                     if torch.isnan(loss_ctr).any():
#                         print(f"[WARNING] NaN in loss_ctr, setting to zero")
#                         loss_ctr = torch.tensor(0.0, device=device)
#                 else:
#                     loss_ctr = torch.tensor(0.0, device=device)
#             except Exception as e:
#                 print(f"[ERROR] Failed to compute ctr loss: {e}")
#                 loss_ctr = torch.tensor(0.0, device=device)

#             losses['loss_cls'] += loss_cls
#             losses['loss_reg'] += iou_loss
#             losses['loss_ctr'] += loss_ctr
#             num_pos_total += num_pos

#         if num_pos_total > 0:
#             print(f"[TRAIN] Found {num_pos_total} positive samples across batch")
        
#         # 平均损失
#         for k in losses:
#             losses[k] = losses[k] / max(num_batches_with_pos, 1)
#             # 确保没有NaN
#             if torch.isnan(losses[k]).any():
#                 print(f"[WARNING] Final {k} is NaN, replacing with zero")
#                 losses[k] = torch.tensor(0.0, device=device)
        
#         # 计算总损失 - 应用权重系数
#         weighted_cls_loss = self.cls_factor * losses['loss_cls']
#         weighted_reg_loss = self.reg_factor * losses['loss_reg'] 
#         weighted_ctr_loss = self.ctr_factor * losses['loss_ctr']
        
#         # 使用加权损失计算总损失
#         losses['total_loss'] = weighted_cls_loss + weighted_reg_loss + weighted_ctr_loss
        
#         # 记录原始损失和加权损失，用于调试
#         losses['weighted_cls_loss'] = weighted_cls_loss
#         losses['weighted_reg_loss'] = weighted_reg_loss 
#         losses['weighted_ctr_loss'] = weighted_ctr_loss
        
#         # 打印损失权重信息
#         print(f"[LOSS WEIGHTS] cls:{self.cls_factor}, reg:{self.reg_factor}, ctr:{self.ctr_factor}")
#         print(f"[WEIGHTED LOSS] cls:{weighted_cls_loss.item():.4f}, reg:{weighted_reg_loss.item():.4f}, ctr:{weighted_ctr_loss.item():.4f}")
        
#         return losses

#     def decode_outputs(self, cls_scores, reg_preds, centernesses):
#         if len(cls_scores) == 0:
#             device = next(self.parameters()).device
#             return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
#         # 创建深拷贝避免修改原始数据
#         cls_scores_copy = [cls.clone() for cls in cls_scores]
#         reg_preds_copy = [reg.clone() for reg in reg_preds]  
#         centernesses_copy = [ctr.clone() for ctr in centernesses]
        
#         try:
#             min_batch_size = min(cls.shape[0] for cls in cls_scores_copy if hasattr(cls, 'shape') and cls.shape[0] > 0)
#         except Exception as e:
#             print(f"[ERROR] Failed to compute min batch size: {e}")
#             device = next(self.parameters()).device
#             return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
#         B = min_batch_size
#         outputs = []

#         for batch_idx in range(B):
#             boxes_all, scores_all, labels_all = [], [], []
#             detection_count = 0

#             for level_idx in range(len(cls_scores_copy)):
#                 try:
#                     if level_idx >= len(self.strides):
#                         break
                    
#                     cls = cls_scores_copy[level_idx]
#                     reg = reg_preds_copy[level_idx] 
#                     ctr = centernesses_copy[level_idx]
                    
#                     if not isinstance(cls, torch.Tensor) or not isinstance(reg, torch.Tensor) or not isinstance(ctr, torch.Tensor):
#                         continue
                    
#                     if len(cls.shape) < 1 or len(reg.shape) < 1 or len(ctr.shape) < 1:
#                         continue
                        
#                     if batch_idx >= cls.shape[0] or batch_idx >= reg.shape[0] or batch_idx >= ctr.shape[0]:
#                         break
#                 except Exception as e:
#                     continue
                    
#                 try:
#                     cls_batch = torch.sigmoid(cls[batch_idx])
#                     reg_batch = reg[batch_idx]
#                     ctr_batch = torch.sigmoid(ctr[batch_idx])

#                     if ctr_batch.dim() == 3:
#                         ctr_batch = ctr_batch[0]
#                     elif ctr_batch.dim() == 2:
#                         pass
#                     elif ctr_batch.dim() == 1:
#                         sqrt_size = int(ctr_batch.shape[0] ** 0.5)
#                         if sqrt_size * sqrt_size == ctr_batch.shape[0]:
#                             ctr_batch = ctr_batch.view(sqrt_size, sqrt_size)
#                         else:
#                             continue

#                     if cls_batch.numel() == 0 or reg_batch.numel() == 0 or ctr_batch.numel() == 0:
#                         continue

#                     C, H, W = cls_batch.shape
#                     stride = self.strides[level_idx]

#                     if H <= 0 or W <= 0:
#                         continue

#                     grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#                     grid_x = grid_x.to(cls_batch.device).float() * stride + stride // 2
#                     grid_y = grid_y.to(cls_batch.device).float() * stride + stride // 2

#                     l = F.relu(reg_batch[0])
#                     t = F.relu(reg_batch[1]) 
#                     r = F.relu(reg_batch[2])
#                     b_reg = F.relu(reg_batch[3])

#                     x1 = grid_x - l
#                     y1 = grid_y - t
#                     x2 = grid_x + r
#                     y2 = grid_y + b_reg
#                     boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

#                     cls_reshaped = cls_batch.permute(1, 2, 0).reshape(-1, C)
#                     ctr_reshaped = ctr_batch.reshape(-1, 1)
#                     scores = torch.sqrt(torch.clamp(cls_reshaped * ctr_reshaped, min=0))

#                     max_scores, labels = scores.max(dim=1)
                    
#                     score_threshold = 0.01
#                     mask = max_scores > score_threshold
                    
#                     valid_detections = mask.sum().item()
                    
#                     if mask.sum() == 0:
#                         k = min(10, len(max_scores))
#                         if k > 0:
#                             topk_scores, topk_indices = torch.topk(max_scores, k)
#                             mask = torch.zeros_like(max_scores, dtype=torch.bool)
#                             mask[topk_indices] = True

#                     if mask.sum() == 0:
#                         continue

#                     boxes = boxes[mask]
#                     scores = max_scores[mask]
#                     labels = labels[mask].long()
#                     detection_count += len(boxes)

#                     # 记录边界框统计信息（宽高）
#                     widths = boxes[:, 2] - boxes[:, 0]
#                     heights = boxes[:, 3] - boxes[:, 1]
#                     if batch_idx == 0 and level_idx == 0:
#                         print(f"[BOX STATS] Width: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
#                         print(f"[BOX STATS] Height: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")

#                     for class_id in torch.unique(labels):
#                         class_id = int(class_id.item())
                        
#                         cls_mask = labels == class_id
#                         cls_boxes = boxes[cls_mask]
#                         cls_scores = scores[cls_mask]
                        
#                         if len(cls_boxes) == 0:
#                             continue
                            
#                         valid_box_mask = (cls_boxes[:, 2] > cls_boxes[:, 0]) & (cls_boxes[:, 3] > cls_boxes[:, 1])
#                         cls_boxes = cls_boxes[valid_box_mask]
#                         cls_scores = cls_scores[valid_box_mask]
                        
#                         if len(cls_boxes) > 0:
#                             keep = ops.nms(cls_boxes, cls_scores, iou_threshold=0.6)

#                             boxes_all.append(cls_boxes[keep])
#                             scores_all.append(cls_scores[keep])
#                             labels_all.append(torch.full_like(cls_scores[keep], class_id))

#                 except Exception as e:
#                     continue

#             if boxes_all:
#                 try:
#                     boxes_all = torch.cat(boxes_all)
#                     scores_all = torch.cat(scores_all)
#                     labels_all = torch.cat(labels_all)

#                     xywh = torch.zeros_like(boxes_all)
#                     xywh[:, 0] = (boxes_all[:, 0] + boxes_all[:, 2]) / 2
#                     xywh[:, 1] = (boxes_all[:, 1] + boxes_all[:, 3]) / 2
#                     xywh[:, 2] = boxes_all[:, 2] - boxes_all[:, 0]
#                     xywh[:, 3] = boxes_all[:, 3] - boxes_all[:, 1]

#                     onehot = torch.zeros(len(labels_all), self.num_classes, device=boxes_all.device)
#                     indices = torch.arange(len(labels_all), device=boxes_all.device).long()
#                     labels_long = labels_all.long()
                    
#                     labels_long = torch.clamp(labels_long, 0, self.num_classes - 1)
#                     onehot[indices, labels_long] = 1.0

#                     output = torch.cat([xywh, scores_all.unsqueeze(1), onehot], dim=1)
                
#                 except Exception as e:
#                     device = cls_scores[0].device
#                     fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
#                     fake_detection[0, 4] = 0.001
#                     output = fake_detection
#             else:
#                 device = cls_scores[0].device
#                 fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
#                 fake_detection[0, 4] = 0.001
#                 output = fake_detection

#             outputs.append(output)

#         if len(outputs) == 0:
#             device = next(self.parameters()).device
#             return torch.zeros(1, 1, 5 + self.num_classes, device=device)
            
#         try:
#             max_detections = max(o.shape[0] for o in outputs)
#             padded_outputs = []
            
#             for i, output in enumerate(outputs):
#                 if output.shape[0] < max_detections:
#                     padding = torch.zeros(max_detections - output.shape[0], output.shape[1], device=output.device)
#                     output = torch.cat([output, padding], dim=0)
#                 padded_outputs.append(output)
                
#             result = torch.stack(padded_outputs, dim=0)
#             return result
            
#         except Exception as e:
#             device = next(self.parameters()).device
#             return torch.zeros(min_batch_size, 1, 5 + self.num_classes, device=device)
