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
        
        min_batch_size = min(cls.shape[0] for cls in cls_scores)
        print(f"[DEBUG] Min batch size across levels: {min_batch_size}")
        
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
            
            print(f"[DEBUG] Level {i}: {B}x{C}x{H}x{W}, stride={stride}")
            
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

        losses = dict(loss_cls=0.0, loss_reg=0.0, loss_ctr=0.0)
        num_pos_total = 0

        B = min_batch_size
        print(f"[DEBUG] Processing batch size: {B}")
        print(f"[DEBUG] Total points per batch: {all_cls_scores.shape[1]}")

        for batch_idx in range(B):
            if batch_idx >= len(targets):
                print(f"[DEBUG] Batch {batch_idx}: No target available")
                continue
                
            gt = targets[batch_idx]
            
            if gt.numel() == 0:
                print(f"[DEBUG] Batch {batch_idx}: No ground truth")
                continue
                
            gt_areas = gt[:, 3] * gt[:, 4]
            valid_mask = gt_areas > 0
            
            coord_valid = (gt[:, 1] >= 0) & (gt[:, 2] >= 0) & (gt[:, 3] > 0) & (gt[:, 4] > 0)
            valid_mask = valid_mask & coord_valid
            
            gt_valid = gt[valid_mask]
            
            print(f"[DEBUG] Batch {batch_idx}: Total GT: {gt.shape[0]}, Valid GT: {gt_valid.shape[0]}")
            
            if gt_valid.numel() == 0:
                print(f"[DEBUG] Batch {batch_idx}: No valid ground truth")
                print(f"[DEBUG] Sample GT boxes: {gt[:3]}")
                continue

            gt_cls = gt_valid[:, 0].long()
            gt_boxes = gt_valid[:, 1:]
            
            print(f"[DEBUG] Batch {batch_idx}: Valid GT boxes: {gt_boxes.shape[0]}")
            print(f"[DEBUG] Batch {batch_idx}: GT box coords range: x[{gt_boxes[:, 0].min():.1f}, {gt_boxes[:, 0].max():.1f}] y[{gt_boxes[:, 1].min():.1f}, {gt_boxes[:, 1].max():.1f}]")
            print(f"[DEBUG] Batch {batch_idx}: GT box sizes: w={gt_boxes[:, 2].mean():.1f}, h={gt_boxes[:, 3].mean():.1f}")

            points = []
            strides_all = []
            for level_idx, (H, W, stride) in enumerate(feature_info):
                y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                x = x.flatten().float() * stride + stride // 2
                y = y.flatten().float() * stride + stride // 2
                p = torch.stack([x, y], dim=1).to(device)
                points.append(p)
                strides_all.append(torch.full((len(p),), stride, device=device))
                
                print(f"[DEBUG] Level {level_idx}: Feature map {H}x{W}, stride {stride}")
                print(f"[DEBUG] Level {level_idx}: Points range x[{x.min():.1f}, {x.max():.1f}] y[{y.min():.1f}, {y.max():.1f}]")
                
                gt_center_x = gt_boxes[:, 0]
                gt_center_y = gt_boxes[:, 1]
                print(f"[DEBUG] Level {level_idx}: GT centers x[{gt_center_x.min():.1f}, {gt_center_x.max():.1f}] y[{gt_center_y.min():.1f}, {gt_center_y.max():.1f}]")
            
            if len(points) == 0:
                print(f"[DEBUG] Batch {batch_idx}: No feature points generated")
                continue
                
            points = torch.cat(points, dim=0)
            strides_all = torch.cat(strides_all, dim=0)
            
            print(f"[DEBUG] Batch {batch_idx}: Total feature points: {points.shape[0]}")

            num_points = points.shape[0]
            num_gts = gt_boxes.shape[0]

            gt_xy = gt_boxes[:, :2]
            gt_wh = gt_boxes[:, 2:]
            x1y1 = gt_xy - gt_wh / 2
            x2y2 = gt_xy + gt_wh / 2
            gt_area = (x2y2[:, 0] - x1y1[:, 0]) * (x2y2[:, 1] - x1y1[:, 1])

            print(f"[DEBUG] Batch {batch_idx}: GT boxes in xyxy format: x1y1={x1y1.min(0)[0]} to x2y2={x2y2.max(0)[0]}")

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
            
            print(f"[DEBUG] Batch {batch_idx}: Points inside boxes: {inside_box.sum().item()}")
            print(f"[DEBUG] Batch {batch_idx}: Points in center regions (radius_factor={radius_factor}): {center_dist.sum().item()}")
            print(f"[DEBUG] Batch {batch_idx}: Final positive points: {is_pos.sum().item()}")
            
            if inside_box.sum().item() == 0:
                print(f"[DEBUG] Batch {batch_idx}: Debugging point-box overlap...")
                for gt_idx in range(num_gts):
                    gt_box = gt_boxes[gt_idx]
                    print(f"[DEBUG]   GT {gt_idx}: center=({gt_box[0]:.1f}, {gt_box[1]:.1f}), size=({gt_box[2]:.1f}, {gt_box[3]:.1f})")
                    gt_x1y1 = gt_xy[gt_idx] - gt_wh[gt_idx] / 2
                    gt_x2y2 = gt_xy[gt_idx] + gt_wh[gt_idx] / 2
                    print(f"[DEBUG]   GT {gt_idx}: xyxy=({gt_x1y1[0]:.1f}, {gt_x1y1[1]:.1f}, {gt_x2y2[0]:.1f}, {gt_x2y2[1]:.1f})")
                    
                    closest_points_idx = torch.argmin(
                        torch.sqrt((px.squeeze() - gt_xy[gt_idx, 0])**2 + (py.squeeze() - gt_xy[gt_idx, 1])**2)
                    )
                    closest_point = points[closest_points_idx]
                    closest_stride = strides_all[closest_points_idx]
                    print(f"[DEBUG]   Closest point to GT {gt_idx}: ({closest_point[0]:.1f}, {closest_point[1]:.1f}), stride={closest_stride:.0f}")
                    
                    distances_to_center = torch.sqrt((px.squeeze() - gt_xy[gt_idx, 0])**2 + (py.squeeze() - gt_xy[gt_idx, 1])**2)
                    within_1_stride = distances_to_center < strides_all
                    within_2_stride = distances_to_center < (strides_all * 2)
                    print(f"[DEBUG]   Points within 1*stride: {within_1_stride.sum().item()}, within 2*stride: {within_2_stride.sum().item()}")

            if is_pos.sum().item() == 0:
                print(f"[DEBUG] Batch {batch_idx}: Trying more relaxed assignment...")
                
                relaxed_radius = strides_all.unsqueeze(1) * 2.5
                relaxed_center_dist = (center_x < relaxed_radius) & (center_y < relaxed_radius)
                is_pos = inside_box | relaxed_center_dist
                
                print(f"[DEBUG] Batch {batch_idx}: Relaxed positive points: {is_pos.sum().item()}")

            gt_area_expanded = gt_area.unsqueeze(0).repeat(num_points, 1)
            gt_area_expanded[~is_pos] = float('inf')
            min_area, min_inds = gt_area_expanded.min(dim=1)
            pos_mask = min_area < float('inf')
            num_pos = pos_mask.sum().item()
            
            print(f"[DEBUG] Batch {batch_idx}: Final assigned positive samples: {num_pos}")
            
            if num_pos == 0:
                print(f"[DEBUG] Batch {batch_idx}: NO POSITIVE SAMPLES FOUND!")
                continue

            matched_inds = min_inds[pos_mask]
            matched_boxes = gt_boxes[matched_inds]
            matched_cls = gt_cls[matched_inds]

            cls_target = torch.zeros_like(all_cls_scores[batch_idx])
            cls_target[pos_mask, matched_cls] = 1.0
            loss_cls = self.focal_loss(all_cls_scores[batch_idx], cls_target, alpha=0.25, gamma=2.0) / max(num_pos, 1)

            matched_x1y1 = matched_boxes[:, :2] - matched_boxes[:, 2:] / 2
            matched_x2y2 = matched_boxes[:, :2] + matched_boxes[:, 2:] / 2
            matched_l = points[pos_mask][:, 0] - matched_x1y1[:, 0]
            matched_t = points[pos_mask][:, 1] - matched_x1y1[:, 1]
            matched_r = matched_x2y2[:, 0] - points[pos_mask][:, 0]
            matched_b = matched_x2y2[:, 1] - points[pos_mask][:, 1]
            reg_target = torch.stack([matched_l, matched_t, matched_r, matched_b], dim=1)

            reg_pred = all_reg_preds[batch_idx][pos_mask]
            iou_loss = self._loss_iou(reg_pred, reg_target).mean()

            left_right = reg_target[:, [0, 2]]
            top_bottom = reg_target[:, [1, 3]]
            centerness = (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
                         (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])
            centerness = torch.sqrt(centerness).clamp(0, 1).detach()

            pred_ctr = all_centernesses[batch_idx][pos_mask].squeeze(-1)
            
            if pred_ctr.numel() > 0 and centerness.numel() > 0:
                loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr, centerness, reduction='mean')
            else:
                loss_ctr = torch.tensor(0.0, device=device)

            losses['loss_cls'] += loss_cls
            losses['loss_reg'] += iou_loss
            losses['loss_ctr'] += loss_ctr
            num_pos_total += num_pos

        print(f"[DEBUG] Total positive samples across batch: {num_pos_total}")
        
        for k in losses:
            losses[k] = losses[k] / max(B, 1)
        losses['total_loss'] = losses['loss_cls'] + losses['loss_reg'] + losses['loss_ctr']
        return losses

    def decode_outputs(self, cls_scores, reg_preds, centernesses):
        print(f"[DEBUG] decode_outputs called with cls_scores type: {type(cls_scores)}")
        print(f"[DEBUG] cls_scores length: {len(cls_scores) if isinstance(cls_scores, (list, tuple)) else 'not list/tuple'}")
        
        if len(cls_scores) == 0:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, 5 + self.num_classes, device=device)
        
        min_batch_size = min(cls.shape[0] for cls in cls_scores)
        print(f"[DEBUG] Decode - Min batch size: {min_batch_size}")
        
        cls_scores = [cls[:min_batch_size] for cls in cls_scores]
        reg_preds = [reg[:min_batch_size] for reg in reg_preds]
        centernesses = [ctr[:min_batch_size] for ctr in centernesses]
            
        B = min_batch_size
        outputs = []

        for batch_idx in range(B):
            print(f"[DEBUG] Processing batch {batch_idx}/{B}")
            boxes_all, scores_all, labels_all = [], [], []
            detection_count = 0

            for level_idx, (cls, reg, ctr) in enumerate(zip(cls_scores, reg_preds, centernesses)):
                if level_idx >= len(self.strides):
                    print(f"[DEBUG] Skipping level {level_idx}, no stride available")
                    break
                
                if batch_idx >= cls.shape[0]:
                    print(f"[WARNING] Batch index {batch_idx} exceeds cls tensor size {cls.shape[0]} at level {level_idx}")
                    break
                    
                try:
                    print(f"[DEBUG] Level {level_idx}, Batch {batch_idx}: cls.shape={cls.shape}, reg.shape={reg.shape}, ctr.shape={ctr.shape}")
                    
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
                            print(f"[WARNING] Cannot reshape centerness {ctr_batch.shape} to 2D at level {level_idx}, batch {batch_idx}")
                            continue

                    if cls_batch.numel() == 0 or reg_batch.numel() == 0 or ctr_batch.numel() == 0:
                        print(f"[WARNING] Empty tensors at level {level_idx}, batch {batch_idx}")
                        continue

                    C, H, W = cls_batch.shape
                    stride = self.strides[level_idx]

                    if H <= 0 or W <= 0:
                        print(f"[WARNING] Invalid spatial dimensions: H={H}, W={W} at level {level_idx}")
                        continue

                    print(f"[DEBUG] Level {level_idx}, Batch {batch_idx}: Processing {C}x{H}x{W} with stride {stride}")

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

                    print(f"[DEBUG] Level {level_idx}, Batch {batch_idx}: Score stats - min={scores.min().item():.4f}, max={scores.max().item():.4f}, mean={scores.mean().item():.4f}")

                    max_scores, labels = scores.max(dim=1)
                    
                    score_threshold = 0.01
                    mask = max_scores > score_threshold
                    
                    valid_detections = mask.sum().item()
                    print(f"[DEBUG] Level {level_idx}, Batch {batch_idx}: Valid detections above {score_threshold}: {valid_detections}")
                    
                    if mask.sum() == 0:
                        k = min(10, len(max_scores))
                        if k > 0:
                            topk_scores, topk_indices = torch.topk(max_scores, k)
                            mask = torch.zeros_like(max_scores, dtype=torch.bool)
                            mask[topk_indices] = True
                            print(f"[DEBUG] Level {level_idx}, Batch {batch_idx}: Using top-{k} detections, min_score={topk_scores.min().item():.4f}")

                    if mask.sum() == 0:
                        continue

                    boxes = boxes[mask]
                    scores = max_scores[mask]
                    labels = labels[mask].long()
                    detection_count += len(boxes)

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
                    print(f"[ERROR] Error in decode at level {level_idx}, batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"[DEBUG] Batch {batch_idx}: Total detections before NMS: {detection_count}")

            if boxes_all:
                try:
                    boxes_all = torch.cat(boxes_all)
                    scores_all = torch.cat(scores_all)
                    labels_all = torch.cat(labels_all)

                    print(f"[DEBUG] Batch {batch_idx}: Final detections after NMS: {len(boxes_all)}")

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
                    
                    print(f"[DEBUG] Batch {batch_idx}: Output tensor shape: {output.shape}")
                    print(f"[DEBUG] Batch {batch_idx}: Score range: {scores_all.min().item():.4f} to {scores_all.max().item():.4f}")
                
                except Exception as e:
                    print(f"[ERROR] Failed to concatenate results for batch {batch_idx}: {e}")
                    device = cls_scores[0].device
                    fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                    fake_detection[0, 4] = 0.001
                    output = fake_detection
            else:
                print(f"[DEBUG] Batch {batch_idx}: No detections, creating fake output")
                device = cls_scores[0].device
                fake_detection = torch.zeros(1, 5 + self.num_classes, device=device)
                fake_detection[0, 4] = 0.001
                output = fake_detection

            outputs.append(output)

        if len(outputs) == 0:
            device = next(self.parameters()).device
            print(f"[WARNING] No outputs generated, creating default output")
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
            print(f"[DEBUG] Final stacked output shape: {result.shape}")
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to stack outputs: {e}")
            print(f"[DEBUG] Output shapes: {[o.shape for o in outputs]}")
            device = next(self.parameters()).device
            return torch.zeros(min_batch_size, 1, 5 + self.num_classes, device=device)
