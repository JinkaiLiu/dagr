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

    def forward(self, feats, targets=None, training=False, hw=None):
        if not feats or len(feats) == 0:
            if training:
                return dict(loss_cls=torch.tensor(0.0), loss_reg=torch.tensor(0.0), 
                          loss_ctr=torch.tensor(0.0), total_loss=torch.tensor(0.0))
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
                
            # 检查tensor是否为空
            if x.numel() == 0:
                print(f"[WARNING] Empty tensor at feature level {i}")
                continue
            
            print(f"[DEBUG] Processing feature {i} with shape: {x.shape}")
            
            try:
                cls_feat = self.cls_convs[i](x)
                reg_feat = self.reg_convs[i](x)
                cls_score = self.cls_preds[i](cls_feat)
                reg_pred = F.relu(self.reg_preds[i](reg_feat))
                centerness = self.centerness_preds[i](reg_feat)
                
                # 验证输出不为空
                if cls_score.numel() == 0 or reg_pred.numel() == 0 or centerness.numel() == 0:
                    print(f"[WARNING] Empty output tensors at level {i}")
                    continue
                    
                cls_scores.append(cls_score)
                reg_preds.append(reg_pred)
                centernesses.append(centerness)
                
            except Exception as e:
                print(f"[ERROR] Failed to process feature {i}: {e}")
                print(f"[DEBUG] Feature shape: {x.shape}, expected channels: {self.cls_convs[i][0].in_channels}")
                continue

        if len(cls_scores) == 0:
            print(f"[WARNING] No valid feature outputs generated")
            if training:
                return dict(loss_cls=torch.tensor(0.0), loss_reg=torch.tensor(0.0), 
                          loss_ctr=torch.tensor(0.0), total_loss=torch.tensor(0.0))
            else:
                device = next(self.parameters()).device
                return torch.zeros(1, 1, 5 + self.num_classes, device=device)

        if training:
            return self.loss(cls_scores, reg_preds, centernesses, targets)
        else:
            return self.decode_outputs(cls_scores, reg_preds, centernesses)

    def loss(self, cls_scores, reg_preds, centernesses, targets):
        device = cls_scores[0].device
        
        print(f"[DEBUG] Input shapes: cls={[s.shape for s in cls_scores]}")
        
        # 修复：保持batch维度
        feature_info = []
        flat_cls_scores = []
        flat_reg_preds = []
        flat_centernesses = []
        
        for i, (cls, reg, ctr) in enumerate(zip(cls_scores, reg_preds, centernesses)):
            B, C, H, W = cls.shape
            stride = self.strides[i] if i < len(self.strides) else 8
            
            print(f"[DEBUG] Level {i}: {B}x{C}x{H}x{W}, stride={stride}")
            
            # Flatten spatial dimensions but keep batch dimension
            cls_flat = cls.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, H*W, num_classes]
            reg_flat = reg.permute(0, 2, 3, 1).reshape(B, H*W, 4)   # [B, H*W, 4] 
            ctr_flat = ctr.permute(0, 2, 3, 1).reshape(B, H*W, 1)   # [B, H*W, 1]
            
            flat_cls_scores.append(cls_flat)
            flat_reg_preds.append(reg_flat)
            flat_centernesses.append(ctr_flat)
            
            # Store feature map info for point generation
            feature_info.append((H, W, stride))

        # Concatenate across feature levels
        all_cls_scores = torch.cat(flat_cls_scores, dim=1)  # [B, total_points, num_classes]
        all_reg_preds = torch.cat(flat_reg_preds, dim=1)    # [B, total_points, 4]
        all_centernesses = torch.cat(flat_centernesses, dim=1)  # [B, total_points, 1]

        losses = dict(loss_cls=0.0, loss_reg=0.0, loss_ctr=0.0)
        num_pos_total = 0

        B = all_cls_scores.shape[0]
        print(f"[DEBUG] Processing batch size: {B}")
        print(f"[DEBUG] Total points per batch: {all_cls_scores.shape[1]}")

        for batch_idx in range(B):
            gt = targets[batch_idx]
            
            # 过滤掉无效的GT boxes (全0的boxes)
            if gt.numel() == 0:
                print(f"[DEBUG] Batch {batch_idx}: No ground truth")
                continue
                
            # 检查GT boxes的有效性
            # 方法1: 过滤掉面积为0的boxes
            gt_areas = gt[:, 3] * gt[:, 4]  # width * height
            valid_mask = gt_areas > 0
            
            # 方法2: 也检查坐标是否合理 (假设图像尺寸大概在几百像素)
            coord_valid = (gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 3] > 0) & (gt[:, 4] > 0)
            valid_mask = valid_mask & coord_valid
            
            gt_valid = gt[valid_mask]
            
            print(f"[DEBUG] Batch {batch_idx}: Total GT: {gt.shape[0]}, Valid GT: {gt_valid.shape[0]}")
            
            if gt_valid.numel() == 0:
                print(f"[DEBUG] Batch {batch_idx}: No valid ground truth")
                print(f"[DEBUG] Sample GT boxes: {gt[:5]}")  # 打印前5个GT看看格式
                continue

            gt_cls = gt_valid[:, 0].long()
            gt_boxes = gt_valid[:, 1:]
            
            print(f"[DEBUG] Batch {batch_idx}: Valid GT boxes: {gt_boxes.shape[0]}")
            print(f"[DEBUG] Batch {batch_idx}: GT box coords range: x[{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 0].max():.1f}] y[{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 1].max():.1f}]")
            print(f"[DEBUG] Batch {batch_idx}: GT box sizes: w={gt_boxes[:, 2].mean():.1f}, h={gt_boxes[:, 3].mean():.1f}")

            # Generate feature points
            points = []
            strides_all = []
            for H, W, stride in feature_info:
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                x = x.flatten().float() * stride + stride // 2  # Center of pixels
                y = y.flatten().float() * stride + stride // 2
                p = torch.stack([x, y], dim=1).to(device)
                points.append(p)
                strides_all.append(torch.full((len(p),), stride, device=device))
                
                print(f"[DEBUG] Level: Feature map {H}x{W}, stride {stride}")
                print(f"[DEBUG] Level: Points range x[{x.min():.1f}, {x.max():.1f}] y[{y.min():.1f}, {y.max():.1f}]")
            
            if len(points) == 0:
                print(f"[DEBUG] Batch {batch_idx}: No feature points generated")
                continue
                
            points = torch.cat(points, dim=0)
            strides_all = torch.cat(strides_all, dim=0)
            
            print(f"[DEBUG] Batch {batch_idx}: Total feature points: {points.shape[0]}")

            num_points = points.shape[0]
            num_gts = gt_boxes.shape[0]

            # Convert center-size to corner format for point-in-box check
            gt_xy = gt_boxes[:, :2]
            gt_wh = gt_boxes[:, 2:]
            x1y1 = gt_xy - gt_wh / 2
            x2y2 = gt_xy + gt_wh / 2
            gt_area = (x2y2[:, 0] - x1y1[:, 0]) * (x2y2[:, 1] - x1y1[:, 1])

            # Check which points are inside GT boxes
            px = points[:, 0].unsqueeze(1)  # [num_points, 1]
            py = points[:, 1].unsqueeze(1)  # [num_points, 1]
            l = px - x1y1[:, 0]  # [num_points, num_gts]
            t = py - x1y1[:, 1]
            r = x2y2[:, 0] - px
            b = x2y2[:, 1] - py
            reg_targets = torch.stack([l, t, r, b], dim=2)  # [num_points, num_gts, 4]

            inside_box = reg_targets.min(dim=2)[0] > 0  # [num_points, num_gts]
            
            # Center region constraint
            center = gt_xy
            radius = strides_all.unsqueeze(1) * 1.5  # [num_points, 1]
            center_x = torch.abs(px - center[:, 0])  # [num_points, num_gts]
            center_y = torch.abs(py - center[:, 1])
            center_dist = (center_x < radius) & (center_y < radius)
            
            is_pos = inside_box & center_dist  # [num_points, num_gts]
            
            print(f"[DEBUG] Batch {batch_idx}: Points inside boxes: {inside_box.sum().item()}")
            print(f"[DEBUG] Batch {batch_idx}: Points in center regions: {center_dist.sum().item()}")
            print(f"[DEBUG] Batch {batch_idx}: Final positive points: {is_pos.sum().item()}")

            # Assign points to GT boxes based on minimum area
            gt_area_expanded = gt_area.unsqueeze(0).repeat(num_points, 1)  # [num_points, num_gts]
            gt_area_expanded[~is_pos] = float('inf')
            min_area, min_inds = gt_area_expanded.min(dim=1)  # [num_points]
            pos_mask = min_area < float('inf')  # [num_points]
            num_pos = pos_mask.sum().item()
            
            print(f"[DEBUG] Batch {batch_idx}: Final assigned positive samples: {num_pos}")
            
            if num_pos == 0:
                print(f"[DEBUG] Batch {batch_idx}: NO POSITIVE SAMPLES FOUND!")
                continue

            # Get matched targets
            matched_inds = min_inds[pos_mask]
            matched_boxes = gt_boxes[matched_inds]
            matched_cls = gt_cls[matched_inds]

            # Classification loss
            cls_target = torch.zeros_like(all_cls_scores[batch_idx])  # [total_points, num_classes]
            cls_target[pos_mask, matched_cls] = 1.0
            loss_cls = self.focal_loss(all_cls_scores[batch_idx], cls_target, alpha=0.25, gamma=2.0) / max(num_pos, 1)

            # Regression loss
            matched_x1y1 = matched_boxes[:, :2] - matched_boxes[:, 2:] / 2
            matched_x2y2 = matched_boxes[:, :2] + matched_boxes[:, 2:] / 2
            matched_l = points[pos_mask][:, 0] - matched_x1y1[:, 0]
            matched_t = points[pos_mask][:, 1] - matched_x1y1[:, 1]
            matched_r = matched_x2y2[:, 0] - points[pos_mask][:, 0]
            matched_b = matched_x2y2[:, 1] - points[pos_mask][:, 1]
            reg_target = torch.stack([matched_l, matched_t, matched_r, matched_b], dim=1)

            reg_pred = all_reg_preds[batch_idx][pos_mask]  # [num_pos, 4]
            iou_loss = self._loss_iou(reg_pred, reg_target).mean()

            # Centerness loss
            left_right = reg_target[:, [0, 2]]
            top_bottom = reg_target[:, [1, 3]]
            centerness = (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
                         (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])
            centerness = torch.sqrt(centerness).clamp(0, 1).detach()

            pred_ctr = all_centernesses[batch_idx][pos_mask].squeeze(-1)  # [num_pos]
            
            if pred_ctr.numel() > 0 and centerness.numel() > 0:
                loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr, centerness, reduction='mean')
            else:
                loss_ctr = torch.tensor(0.0, device=device)

            losses['loss_cls'] += loss_cls
            losses['loss_reg'] += iou_loss
            losses['loss_ctr'] += loss_ctr
            num_pos_total += num_pos

        print(f"[DEBUG] Total positive samples across batch: {num_pos_total}")
        
        # Average losses
        for k in losses:
            losses[k] = losses[k] / max(B, 1)
        losses['total_loss'] = losses['loss_cls'] + losses['loss_reg'] + losses['loss_ctr']
        return losses

    def decode_outputs(self, cls_scores, reg_preds, centernesses):
        if len(cls_scores) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
            
        B = cls_scores[0].shape[0]
        outputs = []

        for batch_idx in range(B):
            boxes_all, scores_all, labels_all = [], [], []

            for i, (cls, reg, ctr) in enumerate(zip(cls_scores, reg_preds, centernesses)):
                if i >= len(self.strides):
                    break
                    
                cls = torch.sigmoid(cls[batch_idx])
                reg = reg[batch_idx]
                ctr = torch.sigmoid(ctr[batch_idx])

                if ctr.dim() == 3:
                    ctr = ctr[0]
                elif ctr.dim() == 2:
                    pass
                elif ctr.dim() == 1:
                    sqrt_size = int(ctr.shape[0] ** 0.5)
                    if sqrt_size * sqrt_size == ctr.shape[0]:
                        ctr = ctr.view(sqrt_size, sqrt_size)
                    else:
                        print(f"[WARNING] Cannot reshape centerness {ctr.shape} to 2D")
                        continue

                C, H, W = cls.shape
                stride = self.strides[i]

                if H <= 0 or W <= 0:
                    print(f"[WARNING] Invalid spatial dimensions: H={H}, W={W}")
                    continue

                grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                grid_x = grid_x.to(cls.device).float() * stride + stride // 2
                grid_y = grid_y.to(cls.device).float() * stride + stride // 2

                l = F.relu(reg[0])
                t = F.relu(reg[1]) 
                r = F.relu(reg[2])
                b_reg = F.relu(reg[3])

                x1 = grid_x - l
                y1 = grid_y - t
                x2 = grid_x + r
                y2 = grid_y + b_reg
                boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)

                cls = cls.permute(1, 2, 0).reshape(-1, C)
                ctr = ctr.reshape(-1, 1)
                scores = torch.sqrt(torch.clamp(cls * ctr, min=0))

                max_scores, labels = scores.max(dim=1)
                mask = max_scores > 0.05
                if mask.sum() == 0:
                    continue

                boxes = boxes[mask]
                scores = max_scores[mask]
                labels = labels[mask].long()

                for class_id in torch.unique(labels):
                    class_id = int(class_id.item())
                    
                    cls_mask = labels == class_id
                    cls_boxes = boxes[cls_mask]
                    cls_scores = scores[cls_mask]
                    
                    if len(cls_boxes) == 0:
                        continue
                        
                    keep = ops.nms(cls_boxes, cls_scores, iou_threshold=0.6)

                    boxes_all.append(cls_boxes[keep])
                    scores_all.append(cls_scores[keep])
                    labels_all.append(torch.full_like(cls_scores[keep], class_id))

            if boxes_all:
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
                onehot[indices, labels_long] = 1.0

                output = torch.cat([xywh, scores_all.unsqueeze(1), onehot], dim=1)
            else:
                output = torch.zeros(1, 5 + self.num_classes, device=cls_scores[0].device)

            outputs.append(output)

        return torch.stack(outputs, dim=0)
