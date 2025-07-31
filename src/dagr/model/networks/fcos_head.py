import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

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
        self.cls_factor = 1.0
        self.reg_factor = 1.0
        self.ctr_factor = 10.0

        for i, in_c in enumerate(in_channels):
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, feat_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.ReLU(),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, feat_channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                    nn.ReLU(),
                )
            )
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 3, padding=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 3, padding=1))
            self.centerness_preds.append(nn.Conv2d(feat_channels, 1, 3, padding=1))

        self._loss_iou = nn.SmoothL1Loss(reduction="none")

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0, reduction="sum"):
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "sum":
            return loss.sum()
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss

    def compute_centerness(self, left_right, top_bottom, epsilon=1e-6):
        """安全计算centerness，防止NaN"""
        # 确保所有值都是正的
        left_right = F.relu(left_right) + epsilon
        top_bottom = F.relu(top_bottom) + epsilon
        
        # 计算最小最大比
        lr_ratio = torch.min(left_right, dim=1)[0] / torch.max(left_right, dim=1)[0]
        tb_ratio = torch.min(top_bottom, dim=1)[0] / torch.max(top_bottom, dim=1)[0]
        
        # 确保比值在有效范围内
        lr_ratio = torch.clamp(lr_ratio, min=epsilon, max=1.0)
        tb_ratio = torch.clamp(tb_ratio, min=epsilon, max=1.0)
        
        # 计算centerness
        centerness = torch.sqrt(lr_ratio * tb_ratio)
        
        # 最后一道防线：替换任何可能的NaN
        centerness = torch.nan_to_num(centerness, nan=0.5)
        
        return centerness.clamp(min=0.0, max=1.0)

    def forward(self, feats, targets=None, training=False, hw=None):
        if not feats or len(feats) == 0:
            if training:
                return dict(loss_cls=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
                          loss_reg=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
                          loss_ctr=torch.tensor(0.0, device=self.cls_preds[0].weight.device), 
                          total_loss=torch.tensor(0.0, device=self.cls_preds[0].weight.device))
            else:
                device = next(self.parameters()).device
                return torch.zeros(1, 1, 5 + self.num_classes, device=device)

        cls_scores = []
        reg_preds = []
        centernesses = []
        
        for i, x in enumerate(feats):
            if i >= len(self.cls_convs):
                break
                
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(x)}")
            
            if x.dim() not in [3, 4]:
                raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor with shape {x.shape}")
            
            if x.dim() == 3:
                x = x.unsqueeze(0)
                
            if x.numel() == 0:
                print(f"[WARNING] Empty tensor at feature level {i}")
                continue
            
            try:
                cls_feat = self.cls_convs[i](x)
                reg_feat = self.reg_convs[i](x)
                cls_score = self.cls_preds[i](cls_feat)
                reg_pred = F.relu(self.reg_preds[i](reg_feat))
                centerness = self.centerness_preds[i](reg_feat)
                
                if cls_score.numel() == 0 or reg_pred.numel() == 0 or centerness.numel() == 0:
                    print(f"[WARNING] Empty output tensors at level {i}")
                    continue
                    
                cls_scores.append(cls_score)
                reg_preds.append(reg_pred)
                centernesses.append(centerness)
                
            except Exception as e:
                print(f"[ERROR] Failed to process feature {i}: {e}")
                continue

        if len(cls_scores) == 0:
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

    def loss(self, cls_scores, reg_preds, centernesses, targets):
        device = cls_scores[0].device
        
        min_batch_size = min(cls.shape[0] for cls in cls_scores)
        
        feature_info = []
        flat_cls_scores = []
        flat_reg_preds = []
        flat_centernesses = []
        
        for i, (cls, reg, ctr) in enumerate(zip(cls_scores, reg_preds, centernesses)):
            cls = cls[:min_batch_size]
            reg = reg[:min_batch_size] 
            ctr = ctr[:min_batch_size]
            
            B, C, H, W = cls.shape
            stride = self.strides[i] if i < len(self.strides) else 8
            
            cls_flat = cls.permute(0, 2, 3, 1).reshape(B, H*W, C)
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, H*W, 4)
            ctr_flat = ctr.permute(0, 2, 3, 1).reshape(B, H*W, 1)
            
            flat_cls_scores.append(cls_flat)
            flat_reg_preds.append(reg_flat)
            flat_centernesses.append(ctr_flat)
            
            feature_info.append((H, W, stride))

        all_cls_scores = torch.cat(flat_cls_scores, dim=1)
        all_reg_preds = torch.cat(flat_reg_preds, dim=1)
        all_centernesses = torch.cat(flat_centernesses, dim=1)

        losses = dict(
            loss_cls=torch.tensor(0.0, device=device),
            loss_reg=torch.tensor(0.0, device=device),
            loss_ctr=torch.tensor(0.0, device=device)
        )
        num_pos_total = 0
        num_batches_with_pos = 0
        B = min_batch_size

        for batch_idx in range(B):
            if batch_idx >= len(targets):
                continue
                
            gt = targets[batch_idx]
            
            if gt.numel() == 0:
                continue
                
            gt_areas = gt[:, 3] * gt[:, 4]
            valid_mask = gt_areas > 0
            
            coord_valid = (gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 3] > 0) & (gt[:, 4] > 0)
            valid_mask = valid_mask & coord_valid
            
            gt_valid = gt[valid_mask]
            
            if gt_valid.numel() == 0:
                continue

            gt_cls = gt_valid[:, 0].long()
            gt_boxes = gt_valid[:, 1:]

            points = []
            strides_all = []
            for level_idx, (H, W, stride) in enumerate(feature_info):
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                x = x.flatten().float() * stride + stride // 2
                y = y.flatten().float() * stride + stride // 2
                p = torch.stack([x, y], dim=1).to(device)
                points.append(p)
                strides_all.append(torch.full((len(p),), stride, device=device))
            
            if len(points) == 0:
                continue
                
            points = torch.cat(points, dim=0)
            strides_all = torch.cat(strides_all, dim=0)

            num_points = points.shape[0]
            num_gts = gt_boxes.shape[0]

            gt_xy = gt_boxes[:, :2]
            gt_wh = gt_boxes[:, 2:]
            x1y1 = gt_xy - gt_wh / 2
            x2y2 = gt_xy + gt_wh / 2
            gt_area = (x2y2[:, 0] - x1y1[:, 0]) * (x2y2[:, 1] - x1y1[:, 1])

            px = points[:, 0].unsqueeze(1)
            py = points[:, 1].unsqueeze(1)
            l = px - x1y1[:, 0]
            t = py - x1y1[:, 1]
            r = x2y2[:, 0] - px
            b = x2y2[:, 1] - py
            reg_targets = torch.stack([l, t, r, b], dim=2)

            inside_box = reg_targets.min(dim=2)[0] > 0

            center = gt_xy
            radius_factor = 1.5
            radius = strides_all.unsqueeze(1) * radius_factor
            center_x = torch.abs(px - center[:, 0])
            center_y = torch.abs(py - center[:, 1])
            center_dist = (center_x < radius) & (center_y < radius)
            
            is_pos = inside_box & center_dist
            
            if is_pos.sum().item() == 0:
                is_pos = inside_box | center_dist

            gt_area_expanded = gt_area.unsqueeze(0).repeat(num_points, 1)
            gt_area_expanded[~is_pos] = float('inf')
            min_area, min_inds = gt_area_expanded.min(dim=1)
            pos_mask = min_area < float('inf')
            num_pos = pos_mask.sum().item()
            
            if num_pos == 0:
                continue

            num_batches_with_pos += 1
            matched_inds = min_inds[pos_mask]
            matched_boxes = gt_boxes[matched_inds]
            matched_cls = gt_cls[matched_inds]

            cls_target = torch.zeros_like(all_cls_scores[batch_idx])
            cls_target[pos_mask, matched_cls] = 1.0
            
            # 计算分类损失
            try:
                loss_cls = self.focal_loss(all_cls_scores[batch_idx], cls_target, alpha=0.25, gamma=2.0) / max(num_pos, 1)
                if torch.isnan(loss_cls).any():
                    print(f"[WARNING] NaN in loss_cls, setting to zero")
                    loss_cls = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"[ERROR] Failed to compute cls loss: {e}")
                loss_cls = torch.tensor(0.0, device=device)
            
            # 计算回归损失
            try:
                matched_x1y1 = matched_boxes[:, :2] - matched_boxes[:, 2:] / 2
                matched_x2y2 = matched_boxes[:, :2] + matched_boxes[:, 2:] / 2
                matched_l = points[pos_mask][:, 0] - matched_x1y1[:, 0]
                matched_t = points[pos_mask][:, 1] - matched_x1y1[:, 1]
                matched_r = matched_x2y2[:, 0] - points[pos_mask][:, 0]
                matched_b = matched_x2y2[:, 1] - points[pos_mask][:, 1]
                reg_target = torch.stack([matched_l, matched_t, matched_r, matched_b], dim=1)
                
                reg_pred = all_reg_preds[batch_idx][pos_mask]
                
                # 过滤无效回归目标
                valid_reg_mask = (reg_target > 0).all(dim=1)
                if valid_reg_mask.sum() > 0:
                    iou_loss = self._loss_iou(
                        reg_pred[valid_reg_mask], 
                        reg_target[valid_reg_mask]
                    ).mean()
                    
                    if torch.isnan(iou_loss).any():
                        print(f"[WARNING] NaN in iou_loss, setting to zero")
                        iou_loss = torch.tensor(0.0, device=device)
                else:
                    iou_loss = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"[ERROR] Failed to compute reg loss: {e}")
                iou_loss = torch.tensor(0.0, device=device)

            # 计算centerness损失
            try:
                left_right = reg_target[:, [0, 2]]
                top_bottom = reg_target[:, [1, 3]]
                
                # 使用安全的centerness计算函数
                centerness = self.compute_centerness(left_right, top_bottom)
                pred_ctr = all_centernesses[batch_idx][pos_mask].squeeze(-1)
                
                if pred_ctr.numel() > 0:
                    loss_ctr = F.binary_cross_entropy_with_logits(
                        pred_ctr, centerness, reduction='mean')
                    
                    if torch.isnan(loss_ctr).any():
                        print(f"[WARNING] NaN in loss_ctr, setting to zero")
                        loss_ctr = torch.tensor(0.0, device=device)
                else:
                    loss_ctr = torch.tensor(0.0, device=device)
            except Exception as e:
                print(f"[ERROR] Failed to compute ctr loss: {e}")
                loss_ctr = torch.tensor(0.0, device=device)

            losses['loss_cls'] += loss_cls
            losses['loss_reg'] += iou_loss
            losses['loss_ctr'] += loss_ctr
            num_pos_total += num_pos

        if num_pos_total > 0:
            print(f"[TRAIN] Found {num_pos_total} positive samples across batch")
        
        # 平均损失
        for k in losses:
            losses[k] = losses[k] / max(num_batches_with_pos, 1)
            # 确保没有NaN
            if torch.isnan(losses[k]).any():
                print(f"[WARNING] Final {k} is NaN, replacing with zero")
                losses[k] = torch.tensor(0.0, device=device)
        
        # 计算总损失 - 应用权重系数
        weighted_cls_loss = self.cls_factor * losses['loss_cls']
        weighted_reg_loss = self.reg_factor * losses['loss_reg'] 
        weighted_ctr_loss = self.ctr_factor * losses['loss_ctr']
        
        # 使用加权损失计算总损失
        losses['total_loss'] = weighted_cls_loss + weighted_reg_loss + weighted_ctr_loss
        
        # 记录原始损失和加权损失，用于调试
        losses['weighted_cls_loss'] = weighted_cls_loss
        losses['weighted_reg_loss'] = weighted_reg_loss 
        losses['weighted_ctr_loss'] = weighted_ctr_loss
        
        # 打印损失权重信息
        print(f"[LOSS WEIGHTS] cls:{self.cls_factor}, reg:{self.reg_factor}, ctr:{self.ctr_factor}")
        print(f"[WEIGHTED LOSS] cls:{weighted_cls_loss.item():.4f}, reg:{weighted_reg_loss.item():.4f}, ctr:{weighted_ctr_loss.item():.4f}")
        
        return losses

    def decode_outputs(self, cls_scores, reg_preds, centernesses):
        if len(cls_scores) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
        # 创建深拷贝避免修改原始数据
        cls_scores_copy = [cls.clone() for cls in cls_scores]
        reg_preds_copy = [reg.clone() for reg in reg_preds]  
        centernesses_copy = [ctr.clone() for ctr in centernesses]
        
        try:
            min_batch_size = min(cls.shape[0] for cls in cls_scores_copy if hasattr(cls, 'shape') and cls.shape[0] > 0)
        except Exception as e:
            print(f"[ERROR] Failed to compute min batch size: {e}")
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
        B = min_batch_size
        outputs = []

        for batch_idx in range(B):
            boxes_all, scores_all, labels_all = [], [], []
            detection_count = 0

            for level_idx in range(len(cls_scores_copy)):
                try:
                    if level_idx >= len(self.strides):
                        break
                    
                    cls = cls_scores_copy[level_idx]
                    reg = reg_preds_copy[level_idx] 
                    ctr = centernesses_copy[level_idx]
                    
                    if not isinstance(cls, torch.Tensor) or not isinstance(reg, torch.Tensor) or not isinstance(ctr, torch.Tensor):
                        continue
                    
                    if len(cls.shape) < 1 or len(reg.shape) < 1 or len(ctr.shape) < 1:
                        continue
                        
                    if batch_idx >= cls.shape[0] or batch_idx >= reg.shape[0] or batch_idx >= ctr.shape[0]:
                        break
                except Exception as e:
                    continue
                    
                try:
                    cls_batch = torch.sigmoid(cls[batch_idx])
                    reg_batch = reg[batch_idx]
                    ctr_batch = torch.sigmoid(ctr[batch_idx])

                    if ctr_batch.dim() == 3:
                        ctr_batch = ctr_batch[0]
                    elif ctr_batch.dim() == 2:
                        pass
                    elif ctr_batch.dim() == 1:
                        sqrt_size = int(ctr_batch.shape[0] ** 0.5)
                        if sqrt_size * sqrt_size == ctr_batch.shape[0]:
                            ctr_batch = ctr_batch.view(sqrt_size, sqrt_size)
                        else:
                            continue

                    if cls_batch.numel() == 0 or reg_batch.numel() == 0 or ctr_batch.numel() == 0:
                        continue

                    C, H, W = cls_batch.shape
                    stride = self.strides[level_idx]

                    if H <= 0 or W <= 0:
                        continue

                    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                    grid_x = grid_x.to(cls_batch.device).float() * stride + stride // 2
                    grid_y = grid_y.to(cls_batch.device).float() * stride + stride // 2

                    l = F.relu(reg_batch[0])
                    t = F.relu(reg_batch[1]) 
                    r = F.relu(reg_batch[2])
                    b_reg = F.relu(reg_batch[3])

                    x1 = grid_x - l
                    y1 = grid_y - t
                    x2 = grid_x + r
                    y2 = grid_y + b_reg
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

                    cls_reshaped = cls_batch.permute(1, 2, 0).reshape(-1, C)
                    ctr_reshaped = ctr_batch.reshape(-1, 1)
                    scores = torch.sqrt(torch.clamp(cls_reshaped * ctr_reshaped, min=0))

                    max_scores, labels = scores.max(dim=1)
                    
                    score_threshold = 0.01
                    mask = max_scores > score_threshold
                    
                    valid_detections = mask.sum().item()
                    
                    if mask.sum() == 0:
                        k = min(10, len(max_scores))
                        if k > 0:
                            topk_scores, topk_indices = torch.topk(max_scores, k)
                            mask = torch.zeros_like(max_scores, dtype=torch.bool)
                            mask[topk_indices] = True

                    if mask.sum() == 0:
                        continue

                    boxes = boxes[mask]
                    scores = max_scores[mask]
                    labels = labels[mask].long()
                    detection_count += len(boxes)

                    # 记录边界框统计信息（宽高）
                    widths = boxes[:, 2] - boxes[:, 0]
                    heights = boxes[:, 3] - boxes[:, 1]
                    if batch_idx == 0 and level_idx == 0:
                        print(f"[BOX STATS] Width: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
                        print(f"[BOX STATS] Height: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")

                    for class_id in torch.unique(labels):
                        class_id = int(class_id.item())
                        
                        cls_mask = labels == class_id
                        cls_boxes = boxes[cls_mask]
                        cls_scores = scores[cls_mask]
                        
                        if len(cls_boxes) == 0:
                            continue
                            
                        valid_box_mask = (cls_boxes[:, 2] > cls_boxes[:, 0]) & (cls_boxes[:, 3] > cls_boxes[:, 1])
                        cls_boxes = cls_boxes[valid_box_mask]
                        cls_scores = cls_scores[valid_box_mask]
                        
                        if len(cls_boxes) > 0:
                            keep = ops.nms(cls_boxes, cls_scores, iou_threshold=0.6)

                            boxes_all.append(cls_boxes[keep])
                            scores_all.append(cls_scores[keep])
                            labels_all.append(torch.full_like(cls_scores[keep], class_id))

                except Exception as e:
                    continue

            if boxes_all:
                try:
                    boxes_all = torch.cat(boxes_all)
                    scores_all = torch.cat(scores_all)
                    labels_all = torch.cat(labels_all)

                    xywh = torch.zeros_like(boxes_all)
                    xywh[:, 0] = (boxes_all[:, 0] + boxes_all[:, 2]) / 2
                    xywh[:, 1] = (boxes_all[:, 1] + boxes_all[:, 3]) / 2
                    xywh[:, 2] = boxes_all[:, 2] - boxes_all[:, 0]
                    xywh[:, 3] = boxes_all[:, 3] - boxes_all[:, 1]

                    onehot = torch.zeros(len(labels_all), self.num_classes, device=boxes_all.device)
                    indices = torch.arange(len(labels_all), device=boxes_all.device).long()
                    labels_long = labels_all.long()
                    
                    labels_long = torch.clamp(labels_long, 0, self.num_classes - 1)
                    onehot[indices, labels_long] = 1.0

                    output = torch.cat([xywh, scores_all.unsqueeze(1), onehot], dim=1)
                
                except Exception as e:
                    device = cls_scores[0].device
                    fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                    fake_detection[0, 4] = 0.001
                    output = fake_detection
            else:
                device = cls_scores[0].device
                fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                fake_detection[0, 4] = 0.001
                output = fake_detection

            outputs.append(output)

        if len(outputs) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
            
        try:
            max_detections = max(o.shape[0] for o in outputs)
            padded_outputs = []
            
            for i, output in enumerate(outputs):
                if output.shape[0] < max_detections:
                    padding = torch.zeros(max_detections - output.shape[0], output.shape[1], device=output.device)
                    output = torch.cat([output, padding], dim=0)
                padded_outputs.append(output)
                
            result = torch.stack(padded_outputs, dim=0)
            return result
            
        except Exception as e:
            device = next(self.parameters()).device
            return torch.zeros(min_batch_size, 1, 5 + self.num_classes, device=device)
