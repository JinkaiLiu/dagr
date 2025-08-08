import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    
    def forward(self, x):
        return torch.exp(torch.clamp(x * self.scale, max=10.0))


class FCOSHead(nn.Module):
    def __init__(self, num_classes=2, in_channels=[256, 256], strides=[8, 16], use_gn=True, 
                 feat_channels=256, init_prior=0.01, scale_exp_init=1.0, center_sampling=True, 
                 center_radius=3.5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.centerness_preds = nn.ModuleList()
        self.scale_exps = nn.ModuleList()
        
        self.center_sampling = center_sampling
        self.center_radius = center_radius
        
        self.debug_logger = None

        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if len(in_channels) < len(strides):
            in_channels = in_channels * len(strides)
            
        in_channels = [256] * len(in_channels)
            
        print(f"[INFO] FCOSHead input channels: {in_channels}")
        
        for i, in_c in enumerate(in_channels):
            if i >= len(strides):
                break
                
            if isinstance(in_c, list):
                in_c = in_c[0]
                
            print(f"[INFO] FCOSHead layer {i} input channels: {in_c}")
            
            cls_convs = []
            reg_convs = []
            
            for j in range(4):
                layer_in_c = 256 if j == 0 else feat_channels
                
                print(f"[INFO] FCOSHead layer {i}, conv {j} - in: {layer_in_c}, out: {feat_channels}")
                
                cls_convs.append(nn.Conv2d(layer_in_c, feat_channels, 3, padding=1))
                if use_gn:
                    cls_convs.append(nn.GroupNorm(32, feat_channels))
                cls_convs.append(nn.ReLU(inplace=True))
                
                reg_convs.append(nn.Conv2d(layer_in_c, feat_channels, 3, padding=1))
                if use_gn:
                    reg_convs.append(nn.GroupNorm(32, feat_channels))
                reg_convs.append(nn.ReLU(inplace=True))
            
            self.cls_convs.append(nn.Sequential(*cls_convs))
            self.reg_convs.append(nn.Sequential(*reg_convs))
            
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 3, padding=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 3, padding=1))
            self.centerness_preds.append(nn.Conv2d(feat_channels, 1, 3, padding=1))
            self.scale_exps.append(ScaleExp(scale_exp_init))
        
        self._init_weights(init_prior)
    
    def _init_weights(self, init_prior=0.01):
        for modules in [self.cls_convs, self.reg_convs]:
            for module in modules:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        
        for modules in [self.cls_preds, self.reg_preds, self.centerness_preds]:
            for m in modules:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m in self.cls_preds:
                    bias_value = -math.log((1 - init_prior) / init_prior)
                    
                    if m.bias.shape[0] > 1:
                        bias_data = torch.zeros_like(m.bias.data)
                        bias_data[0] = -bias_value
                        bias_data[1] = 1.0
                        m.bias.data.copy_(bias_data)
                    else:
                        nn.init.constant_(m.bias, -bias_value)
                else:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feats):
        if not feats or len(feats) == 0:
            print("[WARNING] Empty features input to FCOSHead.forward")
            return [], [], []
            
        verbose = self.debug_logger is None or self.debug_logger.iteration <= 1
        if verbose:
            print(f"[DEBUG] FCOSHead input: {len(feats)} features, shapes: {[f.shape for f in feats]}")
            
        cls_scores = []
        reg_preds = []
        centernesses = []
        
        for i, feat in enumerate(feats):
            if i >= len(self.cls_convs):
                continue
                
            if not isinstance(feat, torch.Tensor):
                print(f"[WARNING] Feature {i} is not a tensor: {type(feat)}")
                continue
                
            if feat.dim() != 4:
                print(f"[WARNING] Feature {i} has wrong dimension: {feat.dim()}, shape={feat.shape}")
                continue
                
            try:
                cls_feat = self.cls_convs[i](feat)
                reg_feat = self.reg_convs[i](feat)
                
                cls_score = self.cls_preds[i](cls_feat)
                
                reg_pred_raw = self.reg_preds[i](reg_feat)
                
                reg_pred = self.scale_exps[i](reg_pred_raw)
                
                if self.debug_logger is not None:
                    self.debug_logger.log_reg_pred(reg_pred_raw, reg_pred)
                elif verbose:
                    print(f"[DEBUG] Reg pred raw stats: min={reg_pred_raw.min().item():.4f}, max={reg_pred_raw.max().item():.4f}, mean={reg_pred_raw.mean().item():.4f}")
                    print(f"[DEBUG] Reg pred after scale exp: min={reg_pred.min().item():.4f}, max={reg_pred.max().item():.4f}, mean={reg_pred.mean().item():.4f}")
                
                reg_pred = torch.clamp(reg_pred, min=0.0, max=100.0)
                
                centerness = self.centerness_preds[i](reg_feat)
                
                cls_scores.append(cls_score)
                reg_preds.append(reg_pred)
                centernesses.append(centerness)
                
                if verbose:
                    print(f"[DEBUG] Level {i} outputs - cls: {cls_score.shape}, reg: {reg_pred.shape}, ctr: {centerness.shape}")
                
            except Exception as e:
                print(f"[ERROR] Error processing feature {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return cls_scores, reg_preds, centernesses
    
    def loss(self, cls_scores, reg_preds, centernesses, targets, strides=None):
        if not cls_scores or len(cls_scores) == 0:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
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
        
        dummy_param = next(self.parameters())
        cls_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        reg_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        ctr_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        
        num_pos = 0
        total_locations = 0
        
        pos_samples_by_class = torch.zeros(self.num_classes, device=device)
        
        for level_idx, stride in enumerate(strides):
            if level_idx >= len(cls_scores):
                continue
                
            cls_pred = cls_scores[level_idx]
            reg_pred = reg_preds[level_idx]
            centerness_pred = centernesses[level_idx]
            
            h, w = cls_pred.shape[2:]
            
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing='ij'
            )
            locations = torch.stack([
                (grid_x + 0.5) * stride,
                (grid_y + 0.5) * stride
            ], dim=-1)
            
            locations = locations.view(-1, 2)
            total_locations += locations.shape[0]
            
            for batch_idx in range(batch_size):
                if batch_idx >= len(targets):
                    continue
                
                batch_targets = targets[batch_idx]
                if batch_targets.numel() == 0:
                    continue
                
                gt_cls = batch_targets[:, 0].long()
                gt_bbox = batch_targets[:, 1:5]
                
                verbose = self.debug_logger is None or self.debug_logger.iteration <= 1
                if verbose:
                    print(f"[DEBUG] GT bbox for batch {batch_idx}: {gt_bbox}")
                
                gt_x1 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
                gt_y1 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
                gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
                gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
                
                xs, ys = locations[:, 0].unsqueeze(1), locations[:, 1].unsqueeze(1)
                left = xs - gt_x1.unsqueeze(0)
                top = ys - gt_y1.unsqueeze(0)
                right = gt_x2.unsqueeze(0) - xs
                bottom = gt_y2.unsqueeze(0) - ys
                
                reg_targets = torch.stack([left, top, right, bottom], dim=-1)
                
                inside_gt_mask = reg_targets.min(dim=-1)[0] > 0
                
                if self.center_sampling:
                    gt_centers = batch_targets[:, 1:3]
                    
                    radius_x = torch.ones_like(gt_bbox[:, 2]) * self.center_radius
                    radius_y = torch.ones_like(gt_bbox[:, 3]) * self.center_radius
                    
                    cls1_mask = gt_cls == 1
                    if cls1_mask.any():
                        radius_x[cls1_mask] = self.center_radius * 1.5
                        radius_y[cls1_mask] = self.center_radius * 1.5
                    
                    radius_x = radius_x * gt_bbox[:, 2] / 2
                    radius_y = radius_y * gt_bbox[:, 3] / 2
                    
                    distances_x = (locations[:, 0].unsqueeze(1) - gt_centers[:, 0].unsqueeze(0)).abs()
                    distances_y = (locations[:, 1].unsqueeze(1) - gt_centers[:, 1].unsqueeze(0)).abs()
                    
                    inside_center_x = distances_x < radius_x.unsqueeze(0)
                    inside_center_y = distances_y < radius_y.unsqueeze(0)
                    inside_center = inside_center_x & inside_center_y
                    
                    inside_gt_mask = inside_gt_mask & inside_center
                
                if not inside_gt_mask.any():
                    continue
                
                areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                areas = areas.unsqueeze(0).expand(locations.size(0), -1)
                
                areas_for_min = areas.clone()
                areas_for_min[~inside_gt_mask] = 1e9
                min_area_inds = areas_for_min.argmin(dim=1)
                
                pos_mask = inside_gt_mask.any(dim=1)
                num_pos_level = pos_mask.sum().item()
                if num_pos_level == 0:
                    continue
                
                if self.debug_logger is not None:
                    self.debug_logger.log_pos_samples(num_pos_level)
                    
                num_pos += num_pos_level
                
                pos_inds = torch.nonzero(pos_mask).squeeze(1)
                target_inds = min_area_inds[pos_mask]
                
                for cls_id in range(self.num_classes):
                    pos_samples_by_class[cls_id] += (gt_cls[target_inds] == cls_id).sum().item()
                
                if self.debug_logger is not None:
                    self.debug_logger.log_cls_distribution(gt_cls[target_inds])
                
                pos_reg_targets = reg_targets[pos_inds, min_area_inds[pos_mask]]
                
                left_right = pos_reg_targets[:, [0, 2]]
                top_bottom = pos_reg_targets[:, [1, 3]]
                
                max_left_right = torch.max(left_right, dim=1)[0].clamp(min=1e-6)
                max_top_bottom = torch.max(top_bottom, dim=1)[0].clamp(min=1e-6)
                
                centerness_targets = torch.sqrt(
                    (torch.min(left_right, dim=1)[0] / max_left_right) *
                    (torch.min(top_bottom, dim=1)[0] / max_top_bottom)
                )
                
                cls_pred_flat = cls_pred[batch_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
                reg_pred_flat = reg_pred[batch_idx].permute(1, 2, 0).reshape(-1, 4)
                centerness_pred_flat = centerness_pred[batch_idx].permute(1, 2, 0).reshape(-1)
                
                pos_cls_preds = cls_pred_flat[pos_inds]
                pos_reg_preds = reg_pred_flat[pos_inds]
                pos_centerness_preds = centerness_pred_flat[pos_inds]
                
                pos_cls_targets = torch.zeros_like(pos_cls_preds)
                pos_cls_targets[torch.arange(len(pos_inds)), gt_cls[target_inds]] = 1.0
                
                if verbose and batch_idx == 0 and level_idx == 0:
                    print(f"[CLS-STATS] Level {level_idx}, Batch {batch_idx}:")
                    print(f"  Total locations: {cls_pred_flat.shape[0]}, Positive samples: {pos_inds.shape[0]}")
                    print(f"  Positive ratio: {pos_inds.shape[0] / cls_pred_flat.shape[0] * 100:.2f}%")
                    print(f"  Class distribution: {torch.bincount(gt_cls[target_inds], minlength=self.num_classes)}")
                
                if num_pos_level > 0:
                    cls_counts = torch.bincount(gt_cls[target_inds], minlength=self.num_classes)
                    
                    if cls_counts[1] < 3:
                        class_weights = torch.tensor([1.0, 100.0], device=device)
                    else:
                        cls_ratio = max(1.0, cls_counts[0].float() / max(cls_counts[1].float(), 1))
                        class_weights = torch.tensor([1.0, min(50.0, max(30.0, cls_ratio * 5))], device=device)
                    
                    print(f"[WEIGHT] Class distribution: {cls_counts.tolist()}, weights: {class_weights.tolist()}")
                else:
                    class_weights = torch.tensor([1.0, 50.0], device=device)
                
                gamma = 4.0
                alpha = 0.25
                
                pred_sigmoid = torch.sigmoid(pos_cls_preds)
                pt = pred_sigmoid * pos_cls_targets + (1 - pred_sigmoid) * (1 - pos_cls_targets)
                focal_weight = (alpha * pos_cls_targets + (1 - alpha) * (1 - pos_cls_targets)) * (1 - pt).pow(gamma)
                
                cls1_mask = gt_cls[target_inds] == 1
                cls1_indices = torch.nonzero(cls1_mask).squeeze(1)
                if len(cls1_indices) > 0:
                    sample_cls1_mask = torch.zeros_like(pos_cls_targets, dtype=torch.bool)
                    for i in range(len(cls1_indices)):
                        sample_cls1_mask[cls1_indices[i], 1] = True
                    
                    focal_weight[sample_cls1_mask] = focal_weight[sample_cls1_mask] * 1.5
                
                cls_loss_weight = class_weights.unsqueeze(0).expand_as(pos_cls_targets)
                
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    pos_cls_preds, pos_cls_targets, reduction='none', 
                    pos_weight=cls_loss_weight
                ).mul(focal_weight).sum()
                
                pred_boxes = self._distance2bbox(locations[pos_inds], pos_reg_preds)
                target_boxes = self._distance2bbox(locations[pos_inds], pos_reg_targets)
                
                if self.debug_logger is not None:
                    self.debug_logger.log_boxes(pred_boxes, target_boxes)
                elif verbose:
                    print(f"[DEBUG] Pred boxes stats: min={pred_boxes.min().item():.4f}, max={pred_boxes.max().item():.4f}")
                    print(f"[DEBUG] Target boxes stats: min={target_boxes.min().item():.4f}, max={target_boxes.max().item():.4f}")
                
                giou_loss = self._calculate_giou_loss(pred_boxes, target_boxes)
                
                if torch.isnan(giou_loss).any():
                    print(f"[WARNING] NaN in giou_loss, replacing with zeros")
                    giou_loss = torch.where(torch.isnan(giou_loss), torch.zeros_like(giou_loss), giou_loss)
                
                reg_loss = reg_loss + (giou_loss * centerness_targets).sum()
                
                ctr_loss = ctr_loss + F.binary_cross_entropy_with_logits(
                    pos_centerness_preds, centerness_targets, reduction='sum'
                )
                
                neg_mask = ~pos_mask
                neg_inds = torch.nonzero(neg_mask).squeeze(1)
                
                if len(neg_inds) > 0 and len(pos_inds) > 0:
                    neg_cls_preds = cls_pred_flat[neg_inds]
                    neg_cls_targets = torch.zeros_like(neg_cls_preds)
                    
                    neg_loss = F.binary_cross_entropy_with_logits(
                        neg_cls_preds, neg_cls_targets, reduction='none'
                    )
                    
                    neg_loss_sum = neg_loss.sum(dim=1)
                    
                    num_hard_negs = min(len(neg_inds), len(pos_inds) * 3)
                    
                    _, hard_neg_indices = neg_loss_sum.topk(num_hard_negs)
                    hard_neg_inds = neg_inds[hard_neg_indices]
                    
                    hard_neg_cls_preds = cls_pred_flat[hard_neg_inds]
                    hard_neg_cls_targets = torch.zeros_like(hard_neg_cls_preds)
                    
                    neg_cls_loss = F.binary_cross_entropy_with_logits(
                        hard_neg_cls_preds, hard_neg_cls_targets, reduction='sum'
                    )
                    
                    cls_loss = cls_loss + neg_cls_loss
        
        print(f"[LOSS-STATS] Total positive samples: {num_pos}, Total locations: {total_locations}")
        print(f"[LOSS-STATS] Positive ratio: {num_pos / max(1, total_locations) * 100:.4f}%")
        
        print(f"[CLASS-STATS] Class 0: {pos_samples_by_class[0]}, Class 1: {pos_samples_by_class[1]}")
        
        num_pos = max(1, num_pos)
        cls_loss = cls_loss / num_pos
        reg_loss = reg_loss / num_pos
        ctr_loss = ctr_loss / num_pos
        
        if torch.isnan(cls_loss):
            print(f"[WARNING] cls_loss is NaN, replacing with zero")
            cls_loss = torch.zeros_like(cls_loss)
        if torch.isnan(reg_loss):
            print(f"[WARNING] reg_loss is NaN, replacing with zero")
            reg_loss = torch.zeros_like(reg_loss)
        if torch.isnan(ctr_loss):
            print(f"[WARNING] ctr_loss is NaN, replacing with zero")
            ctr_loss = torch.zeros_like(ctr_loss)
        
        total_loss = cls_loss + reg_loss + ctr_loss
        
        return {
            'loss_cls': cls_loss,
            'loss_reg': reg_loss,
            'loss_ctr': ctr_loss,
            'total_loss': total_loss
        }
    
    def _calculate_giou_loss(self, pred_boxes, target_boxes):
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        
        union = pred_area + target_area - intersection
        
        iou = intersection / union.clamp(min=1e-6)
        
        enclose_lt = torch.min(pred_boxes[:, :2], target_boxes[:, :2])
        enclose_rb = torch.max(pred_boxes[:, 2:], target_boxes[:, 2:])
        
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
        
        giou = iou - (enclose_area - union) / enclose_area.clamp(min=1e-6)
        
        return 1 - giou
    
    def _distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        x1 = torch.clamp(x1, min=0.0)
        y1 = torch.clamp(y1, min=0.0)
        x2 = torch.clamp(x2, min=x1 + 1e-5)
        y2 = torch.clamp(y2, min=y1 + 1e-5)
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _bbox_overlaps(self, pred_boxes, target_boxes):
        lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
        rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        union = (area1 + area2 - overlap).clamp(min=1e-6)
        iou = overlap / union
        
        return iou
    
    def get_bboxes(self, cls_scores, reg_preds, centernesses, score_thr=0.001, nms_thr=0.6, max_num=100, 
                  score_thr_cls1=None, nms_thr_cls1=None):
        try:
            # 类别特定的阈值
            score_thr_cls0 = 0.2  # 为类别0添加特定阈值
            score_thr_cls1 = 0.1 if score_thr_cls1 is None else score_thr_cls1
            nms_thr_cls0 = nms_thr + 0.05  # 为类别0也提供稍宽松的NMS阈值
            nms_thr_cls1 = nms_thr + 0.1 if nms_thr_cls1 is None else nms_thr_cls1
            
            # 检查输入是否有效
            if not cls_scores or len(cls_scores) == 0:
                device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return [{
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'labels': torch.zeros(0, dtype=torch.long, device=device)
                }]
                
            batch_size = cls_scores[0].shape[0]
            device = cls_scores[0].device
            
            # 检查并打印分类分数的分布
            print(f"[INFERENCE] Classification score distribution:")
            for level_idx, cls_score in enumerate(cls_scores):
                # 确保张量非空
                if cls_score.numel() == 0:
                    print(f"  Level {level_idx}: 空张量，跳过")
                    continue
                    
                sigmoid_scores = torch.sigmoid(cls_score)
                print(f"  Level {level_idx}: mean={sigmoid_scores.mean().item():.4f}, "
                    f"max={sigmoid_scores.max().item():.4f}, "
                    f"min={sigmoid_scores.min().item():.4f}")
                print(f"  Level {level_idx}: # scores > 0.5: {(sigmoid_scores > 0.5).sum().item()}")
                print(f"  Level {level_idx}: # scores > 0.25: {(sigmoid_scores > 0.25).sum().item()}")
                print(f"  Level {level_idx}: # scores > 0.1: {(sigmoid_scores > 0.1).sum().item()}")
                
                # 打印每个类别的分数分布
                for cls_idx in range(sigmoid_scores.shape[1]):
                    cls_scores_i = sigmoid_scores[:, cls_idx]
                    print(f"    Class {cls_idx}: mean={cls_scores_i.mean().item():.4f}, "
                        f"max={cls_scores_i.max().item():.4f}, "
                        f"# > 0.5: {(cls_scores_i > 0.5).sum().item()}, "
                        f"# > 0.3: {(cls_scores_i > 0.3).sum().item()}, "
                        f"# > 0.1: {(cls_scores_i > 0.1).sum().item()}")
            
            result_list = []
            
            for batch_idx in range(batch_size):
                # 收集所有特征层的预测
                multi_level_bboxes = []
                multi_level_scores = []
                multi_level_centerns = []
                
                for level_idx, (cls_score, reg_pred, centerness) in enumerate(zip(cls_scores, reg_preds, centernesses)):
                    # 只处理有效的层级
                    if level_idx >= len(self.strides):
                        continue
                    
                    # 修复: 确保cls_score是有效张量并且batch_idx有效
                    if not isinstance(cls_score, torch.Tensor) or cls_score.dim() == 0:
                        print(f"[WARNING] cls_score is not a valid tensor: {type(cls_score)}")
                        continue
                        
                    if batch_idx >= cls_score.shape[0]:
                        print(f"[WARNING] batch_idx {batch_idx} is out of bounds for cls_score with shape {cls_score.shape}")
                        continue
                    
                    # 提取当前批次的预测
                    cls_score_batch = cls_score[batch_idx]
                    reg_pred_batch = reg_pred[batch_idx]
                    centerness_batch = centerness[batch_idx]
                    
                    # 特征图大小
                    _, h, w = cls_score_batch.shape
                    stride = self.strides[level_idx]
                    
                    # 生成网格点坐标
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(h, device=device),
                        torch.arange(w, device=device),
                        indexing='ij'
                    )
                    # 网格点在原图上的坐标
                    points = torch.stack([
                        (grid_x + 0.5) * stride,
                        (grid_y + 0.5) * stride
                    ], dim=-1).reshape(-1, 2)
                    
                    # 展平预测
                    cls_score_flat = cls_score_batch.permute(1, 2, 0).reshape(-1, self.num_classes)
                    reg_pred_flat = reg_pred_batch.permute(1, 2, 0).reshape(-1, 4)
                    centerness_flat = centerness_batch.reshape(-1)
                    
                    # 转换回归预测为边界框
                    bboxes = self._distance2bbox(points, reg_pred_flat)
                    
                    # 修改: 分类分数计算 - 为类别1提供适度的分数提升
                    cls_sigmoid = torch.sigmoid(cls_score_flat)
                    ctr_sigmoid = torch.sigmoid(centerness_flat).unsqueeze(1)
                    
                    # 为类别1提供更适中的分数提升
                    cls_boost = torch.ones_like(cls_sigmoid)
                    cls_boost[:, 1] = 1.2  # 类别1的分数适度提升
                    
                    # 最终得分计算
                    scores = cls_sigmoid * ctr_sigmoid * cls_boost
                    
                    # 收集该层级的预测
                    multi_level_bboxes.append(bboxes)
                    multi_level_scores.append(scores)
                    multi_level_centerns.append(torch.sigmoid(centerness_flat))
                
                # 如果没有有效的预测，返回空结果
                if len(multi_level_bboxes) == 0:
                    result_list.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device)
                    })
                    continue
                
                # 合并所有层级的预测
                bboxes = torch.cat(multi_level_bboxes)
                scores = torch.cat(multi_level_scores)
                
                # 确保框在有效范围内
                bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0)
                bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0)
                
                # 过滤无效框
                valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                bboxes = bboxes[valid_mask]
                scores = scores[valid_mask]
                
                # 打印框统计信息
                print(f"[INFERENCE] Batch {batch_idx}: {bboxes.shape[0]} valid boxes")
                
                if bboxes.numel() == 0:
                    result_list.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device)
                    })
                    continue
                
                # 获取每个位置最高分数和对应类别
                max_scores, labels = scores.max(dim=1)
                
                # 修改: 使用类别特定的分数阈值
                keep_mask = torch.zeros_like(max_scores, dtype=torch.bool)
                
                # 类别0使用特定阈值
                cls0_mask = (labels == 0)
                if cls0_mask.any():
                    keep_mask[cls0_mask] = max_scores[cls0_mask] > score_thr_cls0
                
                # 类别1使用更低的阈值 - 添加安全检查
                cls1_mask = (labels == 1) if labels.numel() > 0 else torch.zeros_like(max_scores, dtype=torch.bool)
                if cls1_mask.any():
                    keep_mask[cls1_mask] = max_scores[cls1_mask] > score_thr_cls1
                
                # 基于掩码过滤
                bboxes = bboxes[keep_mask]
                scores = max_scores[keep_mask]
                labels = labels[keep_mask]
                
                # 打印阈值过滤后的框统计
                print(f"[INFERENCE] Batch {batch_idx}: {bboxes.shape[0]} boxes after score threshold")
                if bboxes.shape[0] > 0:
                    print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                    print(f"  Class distribution: {torch.bincount(labels, minlength=self.num_classes)}")
                    
                    # 打印每个类别的分数统计
                    for cls_id in range(self.num_classes):
                        cls_mask = labels == cls_id
                        if cls_mask.sum() > 0:
                            cls_scores = scores[cls_mask]
                            print(f"  Class {cls_id} scores: min={cls_scores.min().item():.4f}, "
                                f"max={cls_scores.max().item():.4f}, "
                                f"mean={cls_scores.mean().item():.4f}, "
                                f"count={cls_mask.sum().item()}")
                
                # 如果没有通过分数阈值的框，返回空结果
                if bboxes.numel() == 0:
                    result_list.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device)
                    })
                    continue
                
                # 应用类别特定的NMS
                keep = self._nms(bboxes, scores, labels, nms_thr, nms_thr_cls0, nms_thr_cls1)
                
                # 打印NMS后的框统计
                print(f"[INFERENCE] Batch {batch_idx}: {len(keep)} boxes after NMS")
                
                # 如果NMS后没有保留的框，返回空结果
                if len(keep) == 0:
                    result_list.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device)
                    })
                    continue
                
                # 限制返回的边界框数量
                if len(keep) > max_num:
                    keep = keep[:max_num]
                
                # 保存最终保留的框
                final_boxes = bboxes[keep]
                final_scores = scores[keep]
                final_labels = labels[keep]
                
                # 打印详细的预测框信息
                print(f"[PRED-BOXES] Batch {batch_idx}: {len(keep)} final predictions")
                if len(keep) > 0:
                    # 按分数排序
                    _, sorted_indices = final_scores.sort(descending=True)
                    
                    # 打印前10个或全部（如果少于10个）预测框的详细信息
                    num_to_print = min(10, len(keep))
                    print(f"  Top {num_to_print} predictions:")
                    
                    for i in range(num_to_print):
                        idx = sorted_indices[i]
                        x1, y1, x2, y2 = final_boxes[idx].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        cls_id = final_labels[idx].item()
                        score = final_scores[idx].item()
                        
                        print(f"    Box #{i}: class={cls_id}, score={score:.4f}, "
                            f"coords=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], "
                            f"size=[{w:.1f}x{h:.1f}], center=[{(x1+x2)/2:.1f},{(y1+y2)/2:.1f}]")
                
                # 保存结果
                result_list.append({
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels
                })
            
            # 打印整体检测统计
            total_detections = sum(len(result['boxes']) for result in result_list)
            print(f"[DETECTION-SUMMARY] Total detections: {total_detections} across {batch_size} batches")
            if total_detections > 0:
                # 收集所有检测结果的标签和分数
                all_labels_list = [result['labels'] for result in result_list if len(result['labels']) > 0]
                all_scores_list = [result['scores'] for result in result_list if len(result['scores']) > 0]
                
                # 修复：检查列表是否为空
                if len(all_labels_list) > 0 and len(all_scores_list) > 0:
                    all_labels = torch.cat(all_labels_list)
                    all_scores = torch.cat(all_scores_list)
                    
                    # 类别分布
                    label_counts = torch.bincount(all_labels, minlength=self.num_classes)
                    print(f"  Class distribution: {label_counts.tolist()}")
                    
                    # 分数统计
                    for cls_id in range(self.num_classes):
                        cls_mask = all_labels == cls_id
                        if cls_mask.sum() > 0:
                            cls_scores = all_scores[cls_mask]
                            print(f"  Class {cls_id} scores: min={cls_scores.min().item():.4f}, "
                                f"max={cls_scores.max().item():.4f}, "
                                f"mean={cls_scores.mean().item():.4f}, "
                                f"count={cls_scores.shape[0]}")
            
            return result_list
        
        except Exception as e:
            # 全局异常处理
            print(f"[CRITICAL ERROR] Error in get_bboxes: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回空结果
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            empty_results = []
            for _ in range(batch_size if 'batch_size' in locals() else 1):
                empty_results.append({
                    'boxes': torch.zeros((0, 4), device=device),
                    'scores': torch.zeros(0, device=device),
                    'labels': torch.zeros(0, dtype=torch.long, device=device)
                })
            return empty_results
    
    def _nms(self, bboxes, scores, labels, threshold, threshold_cls0=None, threshold_cls1=None):
        """类别特定的NMS实现"""
        try:
            if bboxes.numel() == 0:
                return torch.tensor([], device=bboxes.device, dtype=torch.long)
            
            threshold_cls0 = threshold + 0.05 if threshold_cls0 is None else threshold_cls0
            threshold_cls1 = threshold + 0.1 if threshold_cls1 is None else threshold_cls1
            
            keep = []
            
            # 添加安全检查：如果labels为空，直接返回空列表
            if labels.numel() == 0:
                return torch.tensor([], device=bboxes.device, dtype=torch.long)
                
            # 获取唯一类别
            unique_labels = labels.unique()
            if unique_labels.numel() == 0:
                return torch.tensor([], device=bboxes.device, dtype=torch.long)
            
            # 对每个类别单独处理
            for cls in unique_labels:
                # 获取当前类别的框
                cls_mask = labels == cls
                cls_bboxes = bboxes[cls_mask]
                cls_scores = scores[cls_mask]
                
                # 添加安全检查：如果当前类别没有框，跳过
                if cls_bboxes.shape[0] == 0:
                    continue
                
                # 类别特定的NMS阈值
                if cls == 0:
                    cls_threshold = threshold_cls0
                elif cls == 1:
                    cls_threshold = threshold_cls1
                else:
                    cls_threshold = threshold
                
                # 原始索引
                cls_inds = torch.nonzero(cls_mask).squeeze(1)
                
                # 按分数排序
                if cls_scores.numel() > 0:  # 安全检查
                    _, sort_idx = cls_scores.sort(descending=True)
                    cls_bboxes = cls_bboxes[sort_idx]
                    cls_inds = cls_inds[sort_idx]
                else:
                    continue
                
                # 执行NMS
                x1, y1, x2, y2 = cls_bboxes.unbind(1)
                areas = (x2 - x1) * (y2 - y1)
                
                order = torch.arange(len(cls_bboxes), device=cls_bboxes.device)
                
                keep_cls = []
                while order.numel() > 0:
                    i = order[0].item()
                    keep_cls.append(i)
                    
                    if order.numel() == 1:
                        break
                    
                    # 计算交集坐标
                    xx1 = torch.max(x1[i], x1[order[1:]])
                    yy1 = torch.max(y1[i], y1[order[1:]])
                    xx2 = torch.min(x2[i], x2[order[1:]])
                    yy2 = torch.min(y2[i], y2[order[1:]])
                    
                    # 计算交集面积
                    w = (xx2 - xx1).clamp(min=0)
                    h = (yy2 - yy1).clamp(min=0)
                    inter = w * h
                    
                    # 计算IoU，添加小的常数避免除零
                    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                    
                    # 保留IoU小于阈值的框 - 使用类别特定的阈值
                    inds = (iou <= cls_threshold).nonzero().squeeze(1)
                    if inds.numel() == 0:
                        break
                    order = order[inds + 1]
                
                # 保存当前类别的结果
                for idx in keep_cls:
                    keep.append(cls_inds[idx].item())
            
            # 处理空结果的情况
            if not keep:
                return torch.tensor([], device=bboxes.device, dtype=torch.long)
                
            return torch.tensor(keep, device=bboxes.device, dtype=torch.long)
            
        except Exception as e:
            print(f"[ERROR] NMS failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor([], device=bboxes.device, dtype=torch.long)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class ScaleExp(nn.Module):
#     def __init__(self, init_value=1.0):
#         super().__init__()
#         self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    
#     def forward(self, x):
#         # 使用clamp防止数值过大导致指数爆炸
#         return torch.exp(torch.clamp(x * self.scale, max=10.0))


# class FCOSHead(nn.Module):
#     def __init__(self, num_classes=2, in_channels=[256, 256], strides=[8, 16], use_gn=True, feat_channels=256):
#         super().__init__()
#         self.num_classes = num_classes
#         self.strides = strides
#         self.cls_convs = nn.ModuleList()
#         self.reg_convs = nn.ModuleList()
#         self.cls_preds = nn.ModuleList()
#         self.reg_preds = nn.ModuleList()
#         self.centerness_preds = nn.ModuleList()
#         self.scale_exps = nn.ModuleList()
        
#         # 调试记录器会在DAGR_FCOS类中设置
#         self.debug_logger = None

#         # 确保in_channels是列表并包含足够的元素
#         if not isinstance(in_channels, list):
#             in_channels = [in_channels]
#         if len(in_channels) < len(strides):
#             in_channels = in_channels * len(strides)
            
#         # 强制使用256通道
#         in_channels = [256] * len(in_channels)
            
#         # 打印输入通道信息
#         print(f"[INFO] FCOSHead input channels: {in_channels}")
        
#         for i, in_c in enumerate(in_channels):
#             if i >= len(strides):
#                 break
                
#             # 确保in_c是一个整数而不是列表
#             if isinstance(in_c, list):
#                 in_c = in_c[0]  # 取第一个元素
                
#             # 打印当前层的输入通道数
#             print(f"[INFO] FCOSHead layer {i} input channels: {in_c}")
            
#             # 创建分类卷积分支
#             cls_convs = []
#             # 创建回归卷积分支
#             reg_convs = []
            
#             for j in range(4):
#                 # 第一层的输入通道数始终使用256，后续层使用feat_channels
#                 layer_in_c = 256 if j == 0 else feat_channels
                
#                 # 打印每个卷积层的通道数
#                 print(f"[INFO] FCOSHead layer {i}, conv {j} - in: {layer_in_c}, out: {feat_channels}")
                
#                 # 分类分支卷积层
#                 cls_convs.append(nn.Conv2d(layer_in_c, feat_channels, 3, padding=1))
#                 if use_gn:
#                     cls_convs.append(nn.GroupNorm(32, feat_channels))
#                 cls_convs.append(nn.ReLU(inplace=True))
                
#                 # 回归分支卷积层
#                 reg_convs.append(nn.Conv2d(layer_in_c, feat_channels, 3, padding=1))
#                 if use_gn:
#                     reg_convs.append(nn.GroupNorm(32, feat_channels))
#                 reg_convs.append(nn.ReLU(inplace=True))
            
#             self.cls_convs.append(nn.Sequential(*cls_convs))
#             self.reg_convs.append(nn.Sequential(*reg_convs))
            
#             # 预测层
#             self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 3, padding=1))
#             self.reg_preds.append(nn.Conv2d(feat_channels, 4, 3, padding=1))
#             self.centerness_preds.append(nn.Conv2d(feat_channels, 1, 3, padding=1))
#             self.scale_exps.append(ScaleExp(1.0))
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for modules in [self.cls_convs, self.reg_convs]:
#             for module in modules:
#                 for m in module.modules():
#                     if isinstance(m, nn.Conv2d):
#                         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                         if m.bias is not None:
#                             nn.init.constant_(m.bias, 0)
        
#         # 特别为预测层设置合适的初始化
#         for modules in [self.cls_preds, self.reg_preds, self.centerness_preds]:
#             for m in modules:
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#                 if m in self.cls_preds:
#                     # 使用负偏置以提高训练初期的稳定性
#                     bias_value = -math.log((1 - 0.01) / 0.01)
#                     nn.init.constant_(m.bias, bias_value)
#                 else:
#                     nn.init.constant_(m.bias, 0)
    
#     def forward(self, feats):
#         # 检查输入是否有效
#         if not feats or len(feats) == 0:
#             print("[WARNING] Empty features input to FCOSHead.forward")
#             return [], [], []
            
#         # 输出调试信息
#         verbose = self.debug_logger is None or self.debug_logger.iteration <= 1
#         if verbose:
#             print(f"[DEBUG] FCOSHead input: {len(feats)} features, shapes: {[f.shape for f in feats]}")
            
#         cls_scores = []
#         reg_preds = []
#         centernesses = []
        
#         for i, feat in enumerate(feats):
#             # 确保特征索引在有效范围内
#             if i >= len(self.cls_convs):
#                 continue
                
#             # 检查特征有效性
#             if not isinstance(feat, torch.Tensor):
#                 print(f"[WARNING] Feature {i} is not a tensor: {type(feat)}")
#                 continue
                
#             if feat.dim() != 4:
#                 print(f"[WARNING] Feature {i} has wrong dimension: {feat.dim()}, shape={feat.shape}")
#                 continue
                
#             # 特征通过分类和回归分支
#             try:
#                 cls_feat = self.cls_convs[i](feat)
#                 reg_feat = self.reg_convs[i](feat)
                
#                 # 生成预测
#                 cls_score = self.cls_preds[i](cls_feat)
                
#                 # 回归预测经过指数缩放，并添加统计信息
#                 reg_pred_raw = self.reg_preds[i](reg_feat)
                
#                 # 使用缩放指数，并添加clamp以避免数值过大
#                 reg_pred = self.scale_exps[i](reg_pred_raw)
                
#                 # 记录回归预测的统计信息
#                 if self.debug_logger is not None:
#                     self.debug_logger.log_reg_pred(reg_pred_raw, reg_pred)
#                 elif verbose:
#                     print(f"[DEBUG] Reg pred raw stats: min={reg_pred_raw.min().item():.4f}, max={reg_pred_raw.max().item():.4f}, mean={reg_pred_raw.mean().item():.4f}")
#                     print(f"[DEBUG] Reg pred after scale exp: min={reg_pred.min().item():.4f}, max={reg_pred.max().item():.4f}, mean={reg_pred.mean().item():.4f}")
                
#                 # 限制回归预测的范围，防止过大的值
#                 reg_pred = torch.clamp(reg_pred, min=0.0, max=1000.0)
                
#                 # 中心度预测
#                 centerness = self.centerness_preds[i](reg_feat)
                
#                 # 收集预测结果
#                 cls_scores.append(cls_score)
#                 reg_preds.append(reg_pred)
#                 centernesses.append(centerness)
                
#                 # 只在第一次迭代时输出调试信息
#                 if verbose:
#                     print(f"[DEBUG] Level {i} outputs - cls: {cls_score.shape}, reg: {reg_pred.shape}, ctr: {centerness.shape}")
                
#             except Exception as e:
#                 print(f"[ERROR] Error processing feature {i}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         return cls_scores, reg_preds, centernesses
    
#     def loss(self, cls_scores, reg_preds, centernesses, targets, strides=None):
#         # 检查输入是否有效
#         if not cls_scores or len(cls_scores) == 0:
#             # 获取设备，默认使用CPU
#             device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
#             # 创建一个需要梯度的零损失
#             # 使用模型参数确保损失有梯度
#             dummy_param = next(self.parameters())
#             # 使用参数的均值乘以0，这样可以保持梯度流
#             zero_loss = dummy_param.mean() * 0
            
#             # 返回零损失
#             return {
#                 'loss_cls': zero_loss,
#                 'loss_reg': zero_loss,
#                 'loss_ctr': zero_loss,
#                 'total_loss': zero_loss
#             }
            
#         device = cls_scores[0].device
#         batch_size = cls_scores[0].shape[0]
#         strides = strides or self.strides
        
#         # 初始化损失 - 确保损失有梯度
#         dummy_param = next(self.parameters())
#         cls_loss = dummy_param.new_tensor(0.0, requires_grad=True)
#         reg_loss = dummy_param.new_tensor(0.0, requires_grad=True)
#         ctr_loss = dummy_param.new_tensor(0.0, requires_grad=True)
        
#         # 统计正样本数量
#         num_pos = 0
        
#         for level_idx, stride in enumerate(strides):
#             if level_idx >= len(cls_scores):
#                 continue
                
#             # 获取当前特征层的预测
#             cls_pred = cls_scores[level_idx]
#             reg_pred = reg_preds[level_idx]
#             centerness_pred = centernesses[level_idx]
            
#             # 特征图大小
#             h, w = cls_pred.shape[2:]
            
#             # 生成网格点坐标
#             grid_y, grid_x = torch.meshgrid(
#                 torch.arange(h, device=device),
#                 torch.arange(w, device=device),
#                 indexing='ij'
#             )
#             # 将网格点坐标转换为原图坐标
#             locations = torch.stack([
#                 (grid_x + 0.5) * stride,
#                 (grid_y + 0.5) * stride
#             ], dim=-1)  # [h, w, 2]
            
#             # 展平网格坐标
#             locations = locations.view(-1, 2)  # [h*w, 2]
            
#             # 处理每个批次
#             for batch_idx in range(batch_size):
#                 if batch_idx >= len(targets):
#                     continue
                
#                 # 获取当前批次的目标
#                 batch_targets = targets[batch_idx]
#                 if batch_targets.numel() == 0:
#                     continue
                
#                 # 获取目标的类别和边界框
#                 gt_cls = batch_targets[:, 0].long()
#                 gt_bbox = batch_targets[:, 1:5]  # [cx, cy, w, h]
                
#                 # 只在第一次迭代时打印目标边界框信息
#                 verbose = self.debug_logger is None or self.debug_logger.iteration <= 1
#                 if verbose:
#                     print(f"[DEBUG] GT bbox for batch {batch_idx}: {gt_bbox}")
                
#                 # 将目标框转换为XYXY格式
#                 gt_x1 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2
#                 gt_y1 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2
#                 gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2
#                 gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2
                
#                 # 计算每个网格点到目标框四边的距离
#                 xs, ys = locations[:, 0].unsqueeze(1), locations[:, 1].unsqueeze(1)
#                 left = xs - gt_x1.unsqueeze(0)     # 左边距离
#                 top = ys - gt_y1.unsqueeze(0)      # 上边距离
#                 right = gt_x2.unsqueeze(0) - xs    # 右边距离
#                 bottom = gt_y2.unsqueeze(0) - ys   # 下边距离
                
#                 # 计算回归目标，形状为[num_locations, num_gts, 4]
#                 reg_targets = torch.stack([left, top, right, bottom], dim=-1)
                
#                 # 确定哪些点在目标框内部
#                 inside_gt_mask = reg_targets.min(dim=-1)[0] > 0  # [num_locations, num_gts]
                
#                 # 跳过没有正样本的情况
#                 if not inside_gt_mask.any():
#                     continue
                
#                 # 对于每个位置，选择面积最小的目标框
#                 areas = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # [num_gts]
#                 areas = areas.unsqueeze(0).expand(locations.size(0), -1)  # [num_locations, num_gts]
                
#                 # 复制areas以避免修改原始张量
#                 areas_for_min = areas.clone()
#                 areas_for_min[~inside_gt_mask] = 1e9  # 使用较大的值替代inf，避免NaN
#                 min_area_inds = areas_for_min.argmin(dim=1)    # [num_locations]
                
#                 # 正样本掩码
#                 pos_mask = inside_gt_mask.any(dim=1)   # [num_locations]
#                 num_pos_level = pos_mask.sum().item()
#                 if num_pos_level == 0:
#                     continue
                    
#                 num_pos += num_pos_level
                
#                 # 获取每个正样本对应的GT
#                 pos_inds = torch.nonzero(pos_mask).squeeze(1)
#                 target_inds = min_area_inds[pos_mask]
                
#                 # 获取正样本的回归目标
#                 pos_reg_targets = reg_targets[pos_inds, min_area_inds[pos_mask]]  # [num_pos, 4]
                
#                 # 计算centerness目标，添加小的常数避免除零
#                 left_right = pos_reg_targets[:, [0, 2]]
#                 top_bottom = pos_reg_targets[:, [1, 3]]
                
#                 # 确保分母不为零
#                 max_left_right = torch.max(left_right, dim=1)[0].clamp(min=1e-6)
#                 max_top_bottom = torch.max(top_bottom, dim=1)[0].clamp(min=1e-6)
                
#                 centerness_targets = torch.sqrt(
#                     (torch.min(left_right, dim=1)[0] / max_left_right) *
#                     (torch.min(top_bottom, dim=1)[0] / max_top_bottom)
#                 )
                
#                 # 获取预测值
#                 cls_pred_flat = cls_pred[batch_idx].permute(1, 2, 0).reshape(-1, self.num_classes)
#                 reg_pred_flat = reg_pred[batch_idx].permute(1, 2, 0).reshape(-1, 4)
#                 centerness_pred_flat = centerness_pred[batch_idx].permute(1, 2, 0).reshape(-1)
                
#                 # 获取正样本的预测
#                 pos_cls_preds = cls_pred_flat[pos_inds]
#                 pos_reg_preds = reg_pred_flat[pos_inds]
#                 pos_centerness_preds = centerness_pred_flat[pos_inds]
                
#                 # 创建分类目标
#                 pos_cls_targets = torch.zeros_like(pos_cls_preds)
#                 pos_cls_targets[torch.arange(len(pos_inds)), gt_cls[target_inds]] = 1.0
                
#                 # 计算分类损失
#                 cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
#                     pos_cls_preds, pos_cls_targets, reduction='sum'
#                 )
                
#                 # 计算回归损失
#                 # 使用IoU损失
#                 pred_boxes = self._distance2bbox(locations[pos_inds], pos_reg_preds)
#                 target_boxes = self._distance2bbox(locations[pos_inds], pos_reg_targets)
                
#                 # 记录边界框统计信息
#                 if self.debug_logger is not None:
#                     self.debug_logger.log_boxes(pred_boxes, target_boxes)
#                 elif verbose:
#                     print(f"[DEBUG] Pred boxes stats: min={pred_boxes.min().item():.4f}, max={pred_boxes.max().item():.4f}")
#                     print(f"[DEBUG] Target boxes stats: min={target_boxes.min().item():.4f}, max={target_boxes.max().item():.4f}")
                
#                 iou_loss = 1 - self._bbox_overlaps(pred_boxes, target_boxes)
                
#                 # 检查iou_loss是否包含NaN
#                 if torch.isnan(iou_loss).any():
#                     print(f"[WARNING] NaN in iou_loss, replacing with zeros")
#                     iou_loss = torch.where(torch.isnan(iou_loss), torch.zeros_like(iou_loss), iou_loss)
                
#                 reg_loss = reg_loss + (iou_loss * centerness_targets).sum()
                
#                 # 计算centerness损失
#                 ctr_loss = ctr_loss + F.binary_cross_entropy_with_logits(
#                     pos_centerness_preds, centerness_targets, reduction='sum'
#                 )
        
#         # 平均损失
#         num_pos = max(1, num_pos)
#         cls_loss = cls_loss / num_pos
#         reg_loss = reg_loss / num_pos
#         ctr_loss = ctr_loss / num_pos
        
#         # 检查损失是否为NaN
#         if torch.isnan(cls_loss):
#             print(f"[WARNING] cls_loss is NaN, replacing with zero")
#             cls_loss = torch.zeros_like(cls_loss)
#         if torch.isnan(reg_loss):
#             print(f"[WARNING] reg_loss is NaN, replacing with zero")
#             reg_loss = torch.zeros_like(reg_loss)
#         if torch.isnan(ctr_loss):
#             print(f"[WARNING] ctr_loss is NaN, replacing with zero")
#             ctr_loss = torch.zeros_like(ctr_loss)
        
#         # 确保所有损失都有梯度
#         total_loss = cls_loss + reg_loss + ctr_loss
        
#         return {
#             'loss_cls': cls_loss,
#             'loss_reg': reg_loss,
#             'loss_ctr': ctr_loss,
#             'total_loss': total_loss
#         }
    
#     def _distance2bbox(self, points, distance):
#         x1 = points[:, 0] - distance[:, 0]
#         y1 = points[:, 1] - distance[:, 1]
#         x2 = points[:, 0] + distance[:, 2]
#         y2 = points[:, 1] + distance[:, 3]
        
#         # 确保坐标有效，防止无效框
#         x1 = torch.clamp(x1, min=0.0)
#         y1 = torch.clamp(y1, min=0.0)
#         x2 = torch.clamp(x2, min=x1 + 1e-5)  # 确保x2 > x1
#         y2 = torch.clamp(y2, min=y1 + 1e-5)  # 确保y2 > y1
        
#         return torch.stack([x1, y1, x2, y2], dim=-1)
    
#     def _bbox_overlaps(self, pred_boxes, target_boxes):
#         # 计算交集
#         lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
#         rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
#         wh = (rb - lt).clamp(min=0)
#         overlap = wh[:, 0] * wh[:, 1]
        
#         # 计算面积
#         area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
#         area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
#         # 添加小的常数避免除零，计算IoU
#         union = (area1 + area2 - overlap).clamp(min=1e-6)
#         iou = overlap / union
        
#         return iou
    
#     def get_bboxes(self, cls_scores, reg_preds, centernesses, score_thr=0.05, nms_thr=0.6, max_num=100):
#         # 检查输入是否有效
#         if not cls_scores or len(cls_scores) == 0:
#             device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             return [{
#                 'boxes': torch.zeros((0, 4), device=device),
#                 'scores': torch.zeros(0, device=device),
#                 'labels': torch.zeros(0, dtype=torch.long, device=device)
#             }]
            
#         batch_size = cls_scores[0].shape[0]
#         device = cls_scores[0].device
        
#         result_list = []
#         for batch_idx in range(batch_size):
#             # 收集所有特征层的预测
#             multi_level_bboxes = []
#             multi_level_scores = []
#             multi_level_centerns = []
            
#             for level_idx, (cls_score, reg_pred, centerness) in enumerate(zip(cls_scores, reg_preds, centernesses)):
#                 # 只处理有效的层级
#                 if level_idx >= len(self.strides):
#                     continue
                    
#                 # 提取当前批次的预测
#                 cls_score_batch = cls_score[batch_idx]
#                 reg_pred_batch = reg_pred[batch_idx]
#                 centerness_batch = centerness[batch_idx]
                
#                 # 特征图大小
#                 _, h, w = cls_score_batch.shape
#                 stride = self.strides[level_idx]
                
#                 # 生成网格点坐标
#                 grid_y, grid_x = torch.meshgrid(
#                     torch.arange(h, device=device),
#                     torch.arange(w, device=device),
#                     indexing='ij'
#                 )
#                 # 网格点在原图上的坐标
#                 points = torch.stack([
#                     (grid_x + 0.5) * stride,
#                     (grid_y + 0.5) * stride
#                 ], dim=-1).reshape(-1, 2)
                
#                 # 展平预测
#                 cls_score_flat = cls_score_batch.permute(1, 2, 0).reshape(-1, self.num_classes)
#                 reg_pred_flat = reg_pred_batch.permute(1, 2, 0).reshape(-1, 4)
#                 centerness_flat = centerness_batch.reshape(-1)
                
#                 # 转换回归预测为边界框
#                 bboxes = self._distance2bbox(points, reg_pred_flat)
                
#                 # 将分类分数和中心度结合
#                 scores = torch.sigmoid(cls_score_flat) * torch.sigmoid(centerness_flat).unsqueeze(1)
                
#                 # 收集该层级的预测
#                 multi_level_bboxes.append(bboxes)
#                 multi_level_scores.append(scores)
#                 multi_level_centerns.append(torch.sigmoid(centerness_flat))
            
#             # 如果没有有效的预测，返回空结果
#             if len(multi_level_bboxes) == 0:
#                 result_list.append({
#                     'boxes': torch.zeros((0, 4), device=device),
#                     'scores': torch.zeros(0, device=device),
#                     'labels': torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue
            
#             # 合并所有层级的预测
#             bboxes = torch.cat(multi_level_bboxes)
#             scores = torch.cat(multi_level_scores)
            
#             # 确保框在有效范围内
#             bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0)
#             bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0)
            
#             # 过滤无效框
#             valid_mask = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
#             bboxes = bboxes[valid_mask]
#             scores = scores[valid_mask]
            
#             if bboxes.numel() == 0:
#                 result_list.append({
#                     'boxes': torch.zeros((0, 4), device=device),
#                     'scores': torch.zeros(0, device=device),
#                     'labels': torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue
            
#             # 获取每个位置最高分数和对应类别
#             max_scores, labels = scores.max(dim=1)
            
#             # 过滤低分数预测
#             keep = max_scores > score_thr
#             bboxes = bboxes[keep]
#             scores = max_scores[keep]
#             labels = labels[keep]
            
#             # 如果没有通过分数阈值的框，返回空结果
#             if bboxes.numel() == 0:
#                 result_list.append({
#                     'boxes': torch.zeros((0, 4), device=device),
#                     'scores': torch.zeros(0, device=device),
#                     'labels': torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue
            
#             # 应用NMS
#             keep = self._nms(bboxes, scores, labels, nms_thr)
            
#             # 如果NMS后没有保留的框，返回空结果
#             if len(keep) == 0:
#                 result_list.append({
#                     'boxes': torch.zeros((0, 4), device=device),
#                     'scores': torch.zeros(0, device=device),
#                     'labels': torch.zeros(0, dtype=torch.long, device=device)
#                 })
#                 continue
            
#             # 限制返回的边界框数量
#             if len(keep) > max_num:
#                 keep = keep[:max_num]
            
#             # 保存结果
#             result_list.append({
#                 'boxes': bboxes[keep],
#                 'scores': scores[keep],
#                 'labels': labels[keep]
#             })
        
#         return result_list
    
#     def _nms(self, bboxes, scores, labels, threshold):
#         if bboxes.numel() == 0:
#             return torch.tensor([], device=bboxes.device, dtype=torch.long)
        
#         keep = []
#         for cls in labels.unique():
#             # 获取当前类别的框
#             cls_mask = labels == cls
#             cls_bboxes = bboxes[cls_mask]
#             cls_scores = scores[cls_mask]
            
#             # 原始索引
#             cls_inds = torch.nonzero(cls_mask).squeeze(1)
            
#             # 按分数排序
#             _, sort_idx = cls_scores.sort(descending=True)
#             cls_bboxes = cls_bboxes[sort_idx]
#             cls_inds = cls_inds[sort_idx]
            
#             # 执行NMS
#             x1, y1, x2, y2 = cls_bboxes.unbind(1)
#             areas = (x2 - x1) * (y2 - y1)
            
#             order = torch.arange(len(cls_bboxes), device=cls_bboxes.device)
            
#             keep_cls = []
#             while order.numel() > 0:
#                 i = order[0].item()
#                 keep_cls.append(i)
                
#                 if order.numel() == 1:
#                     break
                
#                 # 计算交集坐标
#                 xx1 = torch.max(x1[i], x1[order[1:]])
#                 yy1 = torch.max(y1[i], y1[order[1:]])
#                 xx2 = torch.min(x2[i], x2[order[1:]])
#                 yy2 = torch.min(y2[i], y2[order[1:]])
                
#                 # 计算交集面积
#                 w = (xx2 - xx1).clamp(min=0)
#                 h = (yy2 - yy1).clamp(min=0)
#                 inter = w * h
                
#                 # 计算IoU，添加小的常数避免除零
#                 iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                
#                 # 保留IoU小于阈值的框
#                 inds = (iou <= threshold).nonzero().squeeze(1)
#                 if inds.numel() == 0:
#                     break
#                 order = order[inds + 1]
            
#             # 保存当前类别的结果
#             keep.extend(cls_inds[keep_cls].tolist())
        
#         return torch.tensor(keep, device=bboxes.device, dtype=torch.long)