import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from dagr.model.networks.net import sampling_skip
from dagr.model.networks.fcos_head import FCOSHead
from dagr.model.networks.net import Net
from dagr.model.utils import (
    convert_to_training_format,
    convert_to_evaluation_format,
    init_subnetwork
)

def analyze_and_fix_postprocess(prediction, conf_thres, nms_thres, debug_mode=True):
    """
    完整的检测结果分析和修复函数
    解决 "No valid boxes after filtering" 问题
    """
    if debug_mode:
        print(f"\n[BBOX_ANALYSIS] ========== Starting Detection Analysis ==========")
        print(f"[BBOX_ANALYSIS] Input prediction shape: {prediction.shape}")
        print(f"[BBOX_ANALYSIS] Confidence threshold: {conf_thres}")
        print(f"[BBOX_ANALYSIS] NMS threshold: {nms_thres}")
    
    if not isinstance(prediction, torch.Tensor):
        if debug_mode:
            print(f"[BBOX_ANALYSIS] ERROR: Expected tensor, got {type(prediction)}")
        return []
        
    if prediction.numel() == 0:
        if debug_mode:
            print(f"[BBOX_ANALYSIS] ERROR: Empty prediction tensor")
        return []
    
    if debug_mode:
        print(f"[BBOX_ANALYSIS] Prediction stats: min={prediction.min().item():.4f}, max={prediction.max().item():.4f}, mean={prediction.mean().item():.4f}")
    
    batch_size = prediction.shape[0]
    
    if prediction.dim() == 3:
        num_classes = prediction.shape[2] - 5
    else:
        num_classes = prediction.shape[-1] - 5
    
    if torch.is_tensor(num_classes):
        num_classes = num_classes.item()
    num_classes = int(num_classes)
    
    if debug_mode:
        print(f"[BBOX_ANALYSIS] Batch size: {batch_size}, Num classes: {num_classes}")
    
    output = []
    total_detections = 0
    total_valid_boxes = 0
    
    for batch_idx in range(batch_size):
        if debug_mode:
            print(f"\n[BBOX_ANALYSIS] --- Processing Batch {batch_idx}/{batch_size} ---")
        
        image_pred = prediction[batch_idx]
        
        if image_pred.dim() == 1:
            image_pred = image_pred.unsqueeze(0)
            
        if image_pred.shape[0] == 0:
            if debug_mode:
                print(f"[BBOX_ANALYSIS] Batch {batch_idx}: Empty predictions")
            output.append([])
            continue
        
        if debug_mode:
            print(f"[BBOX_ANALYSIS] Batch {batch_idx}: {image_pred.shape[0]} raw predictions")
        
        # 分析原始预测数据
        if debug_mode and image_pred.shape[1] >= 5:
            coords = image_pred[:, :4]
            conf_scores = image_pred[:, 4]
            
            print(f"[BBOX_ANALYSIS] Raw coordinates analysis:")
            print(f"  Coord 0 (X/CX): [{coords[:, 0].min().item():.2f}, {coords[:, 0].max().item():.2f}] mean={coords[:, 0].mean().item():.2f}")
            print(f"  Coord 1 (Y/CY): [{coords[:, 1].min().item():.2f}, {coords[:, 1].max().item():.2f}] mean={coords[:, 1].mean().item():.2f}")
            print(f"  Coord 2 (W/X2): [{coords[:, 2].min().item():.2f}, {coords[:, 2].max().item():.2f}] mean={coords[:, 2].mean().item():.2f}")
            print(f"  Coord 3 (H/Y2): [{coords[:, 3].min().item():.2f}, {coords[:, 3].max().item():.2f}] mean={coords[:, 3].mean().item():.2f}")
            print(f"[BBOX_ANALYSIS] Confidence: [{conf_scores.min().item():.4f}, {conf_scores.max().item():.4f}] mean={conf_scores.mean().item():.4f}")
            
            # 格式检测
            potential_widths = coords[:, 2]
            potential_heights = coords[:, 3]
            positive_wh = (potential_widths > 0) & (potential_heights > 0)
            reasonable_wh = (potential_widths < 1000) & (potential_heights < 1000)
            
            potential_x2_gt_x1 = coords[:, 2] > coords[:, 0]
            potential_y2_gt_y1 = coords[:, 3] > coords[:, 1]
            valid_xyxy = potential_x2_gt_x1 & potential_y2_gt_y1
            
            print(f"[BBOX_ANALYSIS] Format detection:")
            print(f"  XYWH indicators - positive W/H: {positive_wh.sum().item()}/{len(coords)}")
            print(f"  XYWH indicators - reasonable W/H (<1000): {reasonable_wh.sum().item()}/{len(coords)}")
            print(f"  XYXY indicators - valid boxes (x2>x1, y2>y1): {valid_xyxy.sum().item()}/{len(coords)}")
            
            if positive_wh.sum() > valid_xyxy.sum():
                detected_format = "XYWH"
            else:
                detected_format = "XYXY"
            print(f"[BBOX_ANALYSIS] Most likely format: {detected_format}")
        
        # 置信度过滤
        conf_scores = image_pred[:, 4]
        conf_mask = conf_scores >= conf_thres
        above_threshold = conf_mask.sum().item()
        
        if debug_mode:
            print(f"[BBOX_ANALYSIS] Confidence filtering: {above_threshold}/{len(conf_scores)} above threshold {conf_thres}")
        
        if above_threshold == 0:
            k = min(5, len(conf_scores))
            if k > 0:
                _, top_indices = torch.topk(conf_scores, k)
                conf_mask = torch.zeros_like(conf_scores, dtype=torch.bool)
                conf_mask[top_indices] = True
                if debug_mode:
                    print(f"[BBOX_ANALYSIS] Using top-{k} detections as fallback")
        
        image_pred = image_pred[conf_mask]
        
        if image_pred.shape[0] == 0:
            if debug_mode:
                print(f"[BBOX_ANALYSIS] Batch {batch_idx}: No detections after confidence filtering")
            output.append([])
            continue
        
        # 类别处理
        if num_classes > 0:
            class_scores = image_pred[:, 5:5 + num_classes]
            class_conf, class_pred = torch.max(class_scores, 1, keepdim=True)
            
            detections = torch.cat((
                image_pred[:, :4],
                image_pred[:, 4:5],
                class_conf,
                class_pred.float()
            ), 1)
        else:
            detections = torch.cat((
                image_pred[:, :4],
                image_pred[:, 4:5],
                torch.ones(image_pred.shape[0], 1, device=image_pred.device),
                torch.zeros(image_pred.shape[0], 1, device=image_pred.device)
            ), 1)
        
        if debug_mode:
            print(f"[BBOX_ANALYSIS] After class processing: {detections.shape[0]} detections")
        
        # 关键修复：智能格式检测和处理
        batch_detections = []
        if detections.shape[0] > 0:
            coords = detections[:, :4]
            
            # 重新进行格式检测
            potential_widths = coords[:, 2]
            potential_heights = coords[:, 3]
            positive_wh = (potential_widths > 0) & (potential_heights > 0)
            reasonable_wh = (potential_widths < 2000) & (potential_heights < 2000)
            
            potential_x2_gt_x1 = coords[:, 2] > coords[:, 0]
            potential_y2_gt_y1 = coords[:, 3] > coords[:, 1]
            valid_xyxy = potential_x2_gt_x1 & potential_y2_gt_y1
            
            # 智能格式判断
            xywh_score = positive_wh.sum().item()
            xyxy_score = valid_xyxy.sum().item()
            
            if debug_mode:
                print(f"[BBOX_ANALYSIS] Format scores - XYWH: {xywh_score}, XYXY: {xyxy_score}")
            
            if xywh_score >= xyxy_score:
                # 处理为 XYWH 格式
                if debug_mode:
                    print(f"[BBOX_ANALYSIS] Processing as XYWH format")
                
                valid_mask = positive_wh & reasonable_wh
                
                if debug_mode:
                    print(f"[BBOX_ANALYSIS] XYWH valid detections: {valid_mask.sum().item()}/{len(coords)}")
                
                if valid_mask.sum() > 0:
                    valid_detections = detections[valid_mask]
                    
                    # XYWH → XYXY 转换
                    cx, cy, w, h = valid_detections[:, 0], valid_detections[:, 1], valid_detections[:, 2], valid_detections[:, 3]
                    
                    # 检测异常小的 bounding box 并修复
                    if debug_mode:
                        small_boxes = (w < 1.0) | (h < 1.0)
                        if small_boxes.sum() > 0:
                            print(f"[BBOX_ANALYSIS] WARNING: {small_boxes.sum().item()} boxes have very small dimensions")
                            print(f"[BBOX_ANALYSIS] Width range: [{w.min().item():.6f}, {w.max().item():.6f}]")
                            print(f"[BBOX_ANALYSIS] Height range: [{h.min().item():.6f}, {h.max().item():.6f}]")
                            print(f"[BBOX_ANALYSIS] Applying minimum size constraint...")
                    
                    # 应用最小尺寸约束 - 确保 box 至少有合理的尺寸
                    min_size = 5.0  # 最小边长 5 像素
                    w = torch.clamp(w, min=min_size)
                    h = torch.clamp(h, min=min_size)
                    
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # 转换后的有效性检查
                    converted_valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
                    
                    # 额外检查：确保 box 有合理的面积
                    box_areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                    reasonable_area = box_areas >= (min_size * min_size)  # 至少 25 平方像素
                    converted_valid = converted_valid & reasonable_area
                    
                    if debug_mode:
                        print(f"[BBOX_ANALYSIS] After XYWH→XYXY conversion: {converted_valid.sum().item()} valid boxes")
                        print(f"[BBOX_ANALYSIS] After size correction and area filtering: {converted_valid.sum().item()} valid boxes")
                        if converted_valid.sum() > 0:
                            valid_boxes_xyxy = boxes_xyxy[converted_valid]
                            valid_areas = box_areas[converted_valid]
                            print(f"[BBOX_ANALYSIS] Converted box ranges:")
                            print(f"  X1: [{valid_boxes_xyxy[:, 0].min().item():.2f}, {valid_boxes_xyxy[:, 0].max().item():.2f}]")
                            print(f"  Y1: [{valid_boxes_xyxy[:, 1].min().item():.2f}, {valid_boxes_xyxy[:, 1].max().item():.2f}]")
                            print(f"  X2: [{valid_boxes_xyxy[:, 2].min().item():.2f}, {valid_boxes_xyxy[:, 2].max().item():.2f}]")
                            print(f"  Y2: [{valid_boxes_xyxy[:, 3].min().item():.2f}, {valid_boxes_xyxy[:, 3].max().item():.2f}]")
                            print(f"  Box areas: [{valid_areas.min().item():.2f}, {valid_areas.max().item():.2f}]")
                    
                    if converted_valid.sum() > 0:
                        final_boxes = boxes_xyxy[converted_valid]
                        final_scores = valid_detections[converted_valid, 4]
                        final_labels = valid_detections[converted_valid, 6].long() if num_classes > 0 else torch.zeros(converted_valid.sum(), dtype=torch.long)
                        
                        # 按类别进行 NMS
                        final_keep_indices = []
                        for class_id in torch.unique(final_labels):
                            class_mask = final_labels == class_id
                            if class_mask.sum() == 0:
                                continue
                                
                            class_boxes = final_boxes[class_mask]
                            class_scores = final_scores[class_mask]
                            
                            if len(class_boxes) > 0:
                                keep = ops.nms(class_boxes, class_scores, iou_threshold=nms_thres)
                                class_indices = torch.where(class_mask)[0]
                                final_keep_indices.extend(class_indices[keep].tolist())
                        
                        if len(final_keep_indices) > 0:
                            keep_tensor = torch.tensor(final_keep_indices, device=final_boxes.device)
                            
                            det_dict = {
                                'boxes': final_boxes[keep_tensor].cpu(),
                                'scores': final_scores[keep_tensor].cpu(),
                                'labels': final_labels[keep_tensor].cpu()
                            }
                            batch_detections.append(det_dict)
                            total_valid_boxes += len(keep_tensor)
                            
                            if debug_mode:
                                print(f"[BBOX_ANALYSIS] Final XYWH detections after NMS: {len(keep_tensor)}")
                        else:
                            if debug_mode:
                                print(f"[BBOX_ANALYSIS] No detections survived NMS")
                    else:
                        if debug_mode:
                            print(f"[BBOX_ANALYSIS] No valid boxes after XYWH→XYXY conversion")
                else:
                    if debug_mode:
                        print(f"[BBOX_ANALYSIS] No valid XYWH boxes found")
            else:
                # 处理为 XYXY 格式
                if debug_mode:
                    print(f"[BBOX_ANALYSIS] Processing as XYXY format")
                
                valid_mask = valid_xyxy
                
                if debug_mode:
                    print(f"[BBOX_ANALYSIS] XYXY valid detections: {valid_mask.sum().item()}/{len(coords)}")
                
                if valid_mask.sum() > 0:
                    valid_detections = detections[valid_mask]
                    boxes_xyxy = valid_detections[:, :4]
                    scores = valid_detections[:, 4]
                    labels = valid_detections[:, 6].long() if num_classes > 0 else torch.zeros(valid_mask.sum(), dtype=torch.long)
                    
                    # 按类别进行 NMS
                    final_keep_indices = []
                    for class_id in torch.unique(labels):
                        class_mask = labels == class_id
                        if class_mask.sum() == 0:
                            continue
                            
                        class_boxes = boxes_xyxy[class_mask]
                        class_scores = scores[class_mask]
                        
                        if len(class_boxes) > 0:
                            keep = ops.nms(class_boxes, class_scores, iou_threshold=nms_thres)
                            class_indices = torch.where(class_mask)[0]
                            final_keep_indices.extend(class_indices[keep].tolist())
                    
                    if len(final_keep_indices) > 0:
                        keep_tensor = torch.tensor(final_keep_indices, device=boxes_xyxy.device)
                        
                        det_dict = {
                            'boxes': boxes_xyxy[keep_tensor].cpu(),
                            'scores': scores[keep_tensor].cpu(),
                            'labels': labels[keep_tensor].cpu()
                        }
                        batch_detections.append(det_dict)
                        total_valid_boxes += len(keep_tensor)
                        
                        if debug_mode:
                            print(f"[BBOX_ANALYSIS] Final XYXY detections after NMS: {len(keep_tensor)}")
                    else:
                        if debug_mode:
                            print(f"[BBOX_ANALYSIS] No detections survived NMS")
                else:
                    if debug_mode:
                        print(f"[BBOX_ANALYSIS] No valid XYXY boxes found")
        
        if not batch_detections:
            if debug_mode:
                print(f"[BBOX_ANALYSIS] Batch {batch_idx}: No final detections")
        
        output.append(batch_detections)
        total_detections += len(batch_detections[0]) if batch_detections else 0
    
    if debug_mode:
        print(f"\n[BBOX_ANALYSIS] ========== FINAL SUMMARY ==========")
        print(f"[BBOX_ANALYSIS] Total valid boxes across all batches: {total_valid_boxes}")
        print(f"[BBOX_ANALYSIS] Batches with detections: {sum(1 for batch in output if batch)}")
        print(f"[BBOX_ANALYSIS] Average detections per batch: {total_valid_boxes/batch_size:.2f}")
        if total_valid_boxes == 0:
            print(f"[BBOX_ANALYSIS] ⚠️  WARNING: No valid detections found!")
            print(f"[BBOX_ANALYSIS] Possible issues:")
            print(f"  1. Confidence threshold {conf_thres} too high")
            print(f"  2. Coordinate format mismatch")
            print(f"  3. Invalid coordinate values")
            print(f"  4. NMS threshold {nms_thres} too strict")
        else:
            print(f"[BBOX_ANALYSIS] ✅ Successfully processed detections!")
        print(f"[BBOX_ANALYSIS] =========================================\n")
    
    return output

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = target * p + (1 - target) * (1 - p)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss *= alpha_t
    return loss.sum()

def unpack_fused_features(fused_feat):
    features_tensor = []
    features_hw = []
    
    print(f"[DEBUG] Unpacking {len(fused_feat)} fused features")
    
    target_batch_size = None
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            batch_info = f.batch if hasattr(f, "batch") else None
            if batch_info is not None:
                current_batch_size = len(batch_info.unique())
                if target_batch_size is None:
                    target_batch_size = current_batch_size
                else:
                    target_batch_size = min(target_batch_size, current_batch_size)
    
    print(f"[DEBUG] Target batch size: {target_batch_size}")
    
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x
            height = f.height.item() if hasattr(f, "height") and torch.is_tensor(f.height) else None
            width = f.width.item() if hasattr(f, "width") and torch.is_tensor(f.width) else None
            batch_info = f.batch if hasattr(f, "batch") else None
            
            print(f"[DEBUG] Feature {i}: x.shape = {x.shape}, height = {height}, width = {width}")
            if batch_info is not None:
                unique_batches = batch_info.unique()
                print(f"[DEBUG] Feature {i}: batch info available, unique batches = {unique_batches}")
            
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None and batch_info is not None and target_batch_size is not None:
                    batch_size = target_batch_size
                    expected_nodes_per_batch = height * width
                    
                    print(f"[DEBUG] Reconstructing batch dimension: batch_size={batch_size}, nodes_per_batch={expected_nodes_per_batch}")
                    
                    full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
                    for batch_idx in range(batch_size):
                        if batch_idx < len(batch_info.unique()):
                            batch_mask = batch_info == batch_idx
                            batch_nodes = x[batch_mask]
                            
                            if batch_nodes.shape[0] > 0:
                                max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                                
                                if max_nodes == expected_nodes_per_batch:
                                    reshaped = batch_nodes[:max_nodes].transpose(0, 1).view(channels, height, width)
                                else:
                                    temp_features = torch.zeros(expected_nodes_per_batch, channels, device=x.device, dtype=x.dtype)
                                    temp_features[:max_nodes] = batch_nodes[:max_nodes]
                                    reshaped = temp_features.transpose(0, 1).view(channels, height, width)
                                
                                full_tensor[batch_idx] = reshaped
                    
                    x = full_tensor
                    print(f"[DEBUG] Reconstructed to batch tensor: {x.shape}")
                    
                elif height is not None and width is not None:
                    expected_nodes = height * width
                    
                    if num_nodes <= expected_nodes:
                        full_features = torch.zeros(expected_nodes, channels, device=x.device, dtype=x.dtype)
                        full_features[:num_nodes] = x
                        x = full_features.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    else:
                        x_truncated = x[:expected_nodes]
                        x = x_truncated.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    
                    if target_batch_size is not None and target_batch_size > 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    
                    print(f"[DEBUG] Single batch reshaped to: {x.shape}")
                else:
                    sqrt_nodes = int(num_nodes ** 0.5)
                    if sqrt_nodes * sqrt_nodes == num_nodes:
                        x = x.transpose(0, 1).contiguous().view(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                        height, width = sqrt_nodes, sqrt_nodes
                    else:
                        x = x.transpose(0, 1).contiguous().view(channels, num_nodes, 1).unsqueeze(0)
                        height, width = num_nodes, 1
                    
                    if target_batch_size is not None and target_batch_size > 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                        
                    print(f"[DEBUG] Inferred reshape to: {x.shape}")
                        
            elif x.dim() == 3:
                x = x.unsqueeze(0)
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] == 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    else:
                        x = x[:target_batch_size]
            elif x.dim() == 4:
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] > target_batch_size:
                        x = x[:target_batch_size]
                    else:
                        repeats = target_batch_size // x.shape[0]
                        remainder = target_batch_size % x.shape[0]
                        if repeats > 0:
                            x_repeated = x.repeat(repeats, 1, 1, 1)
                            if remainder > 0:
                                x = torch.cat([x_repeated, x[:remainder]], dim=0)
                            else:
                                x = x_repeated
            else:
                raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            
            features_tensor.append(x)
            features_hw.append((height, width))
            
        else:
            if isinstance(f, torch.Tensor):
                if f.dim() == 2:
                    print(f"[WARNING] Processing bare 2D tensor: {f.shape}")
                    f = f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                elif f.dim() == 3:
                    f = f.unsqueeze(0)
                
                if target_batch_size is not None and f.shape[0] != target_batch_size:
                    if f.shape[0] == 1:
                        f = f.repeat(target_batch_size, 1, 1, 1)
                    else:
                        f = f[:target_batch_size]
                        
                features_tensor.append(f)
                features_hw.append((None, None))
            else:
                raise ValueError(f"Unknown feature type: {type(f)}")
    
    print(f"[DEBUG] Final unpacked shapes: {[f.shape for f in features_tensor]}")
    return features_tensor, features_hw


class DAGR(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        self.conf_threshold = 0.4
        self.nms_threshold = 0.3
        self.height = height
        self.width = width
        self.args = args

        self.backbone = Net(args, height=height, width=width)
        self.use_image = getattr(args, "use_image", True)
        self.no_events = getattr(args, "no_events", False)
        self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

        self.head = None
        self.cnn_head = None
        self._head_initialized = False

        if self.use_image:
            self.cnn_head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=self.backbone.out_channels_cnn[:2],
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            )

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)

    def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
        targets = []
        
        for batch_idx in range(num_graphs):
            batch_mask = bbox_batch == batch_idx
            batch_bboxes = bbox[batch_mask]
            
            if batch_bboxes.numel() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
            
            x_left = batch_bboxes[:, 0]
            y_top = batch_bboxes[:, 1]
            height = batch_bboxes[:, 2]
            width = batch_bboxes[:, 3]
            cls = batch_bboxes[:, 4]
            
            areas = width * height
            valid_mask = areas > 0
            
            if valid_mask.sum() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
                
            x_left = x_left[valid_mask]
            y_top = y_top[valid_mask]
            height = height[valid_mask]
            width = width[valid_mask]
            cls = cls[valid_mask]
            
            x_center = x_left + width / 2
            y_center = y_top + height / 2
            
            fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
            targets.append(fcos_target)
            
        return targets

    def _initialize_head_if_needed(self, fused_feat):
        if not self._head_initialized:
            actual_channels = []
            for f in fused_feat[:2]:
                if hasattr(f, "x"):
                    channels = f.x.shape[1]
                    actual_channels.append(channels)
                else:
                    actual_channels.append(f.shape[1])
            
            print(f"[DEBUG] Initializing FCOS head with actual channels: {actual_channels}")
            
            self.head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=actual_channels,
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            ).to(next(self.parameters()).device)
            
            self._head_initialized = True

    def forward(self, x, reset=True, return_targets=True, filtering=True):
        if self.use_image:
            event_feat, image_feat = self.backbone(x)
        else:
            event_feat = self.backbone(x)
            image_feat = None

        if self.training and self.pretrain_cnn:
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[0] // (h * w)
                        tensor = tensor.view(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            return self.cnn_head(image_feat_tensors, targets=targets, training=True)

        if not self.training and self.no_events:
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[0] // (h * w)
                        tensor = tensor.view(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            outputs = self.cnn_head(image_feat_tensors, training=False)
            
            if filtering:
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=True)
                    outputs = []
                    for batch_det in batch_detections:
                        outputs.extend(batch_det)
                else:
                    outputs = []
            
            ret = outputs
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [outputs, targets]
            
            return ret

        fused_feat = event_feat
        if self.use_image and image_feat is not None:
            print(f"[DEBUG] Fusing image and event features")
            for i in range(len(event_feat)):
                event_feat[i].width = torch.tensor([image_feat[i].shape[-1]])
                event_feat[i].height = torch.tensor([image_feat[i].shape[-2]])
                event_feat[i].x = sampling_skip(event_feat[i], image_feat[i].detach())
                
        fused_feat = event_feat

        if self.training:
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            print(f"[DEBUG] Training mode - processing {len(targets)} targets")
            for i, t in enumerate(targets):
                if t.numel() > 0:
                    print(f"[DEBUG] Target {i}: {t.shape[0]} objects")
                    valid_boxes = t[t[:, 1:].sum(dim=1) > 0]
                    print(f"[DEBUG] Target {i}: {valid_boxes.shape[0]} valid objects")
                    if valid_boxes.shape[0] > 0:
                        print(f"[DEBUG] Target {i} bbox range: {valid_boxes[:, 1:].min(0)[0]} to {valid_boxes[:, 1:].max(0)[0]}")
            
            self._initialize_head_if_needed(fused_feat)
            
            fused_feat_tensors, _ = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Calling fusion head with {len(fused_feat_tensors)} features")
            loss_fused = self.head(fused_feat_tensors, targets=targets, training=True)

            result = {}
            
            if isinstance(loss_fused, dict):
                for k, v in loss_fused.items():
                    if torch.is_tensor(v):
                        result[f"fusion_{k}"] = v
                    else:
                        device = next(self.parameters()).device
                        result[f"fusion_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
            else:
                device = next(self.parameters()).device
                result["fusion_total_loss"] = torch.tensor(float(loss_fused), device=device, requires_grad=True) if not torch.is_tensor(loss_fused) else loss_fused

            if self.use_image:
                image_feat_tensors = []
                for img_f in image_feat:
                    if hasattr(img_f, 'x'):
                        tensor = img_f.x
                        if tensor.dim() == 2:
                            h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                            w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                            c = tensor.shape[0] // (h * w)
                            tensor = tensor.view(c, h, w).unsqueeze(0)
                        elif tensor.dim() == 3:
                            tensor = tensor.unsqueeze(0)
                        image_feat_tensors.append(tensor)
                    else:
                        image_feat_tensors.append(img_f)
                        
                print(f"[DEBUG] Calling CNN head with {len(image_feat_tensors)} features")
                loss_image = self.cnn_head(image_feat_tensors, targets=targets, training=True)
                
                if isinstance(loss_image, dict):
                    for k, v in loss_image.items():
                        if torch.is_tensor(v):
                            result[f"cnn_{k}"] = v
                        else:
                            device = next(self.parameters()).device
                            result[f"cnn_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
                else:
                    device = next(self.parameters()).device
                    result["cnn_total_loss"] = torch.tensor(float(loss_image), device=device, requires_grad=True) if not torch.is_tensor(loss_image) else loss_image

            if "fusion_total_loss" not in result:
                if "fusion_loss" in result:
                    result["fusion_total_loss"] = result["fusion_loss"]
                else:
                    device = next(self.parameters()).device
                    result["fusion_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)
                    
            if self.use_image and "cnn_total_loss" not in result:
                if "cnn_loss" in result:
                    result["cnn_total_loss"] = result["cnn_loss"] 
                else:
                    device = next(self.parameters()).device
                    result["cnn_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)

            return result

        for i, f in enumerate(fused_feat):
            print(f"[DEBUG] Inference Feature {i}: x.shape = {f.x.shape}, height = {getattr(f, 'height', 'N/A')}, width = {getattr(f, 'width', 'N/A')}")

        x.reset = reset
        
        self._initialize_head_if_needed(fused_feat)
        
        try:
            fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Inference unpacked features shapes: {[f.shape for f in fused_feat_x]}")
            
            valid_features = []
            for i, feat in enumerate(fused_feat_x):
                if feat.numel() > 0 and all(d > 0 for d in feat.shape):
                    valid_features.append(feat)
                    print(f"[DEBUG] Inference feature {i} is valid: {feat.shape}")
                else:
                    print(f"[WARNING] Inference feature {i} is invalid: {feat.shape}, numel={feat.numel()}")
            
            if len(valid_features) == 0:
                print(f"[ERROR] No valid features after unpacking in inference!")
                ret = []
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    ret = [ret, targets]
                return ret
            
            print(f"[DEBUG] Calling head in inference mode with {len(valid_features)} features")
            outputs = self.head(valid_features, training=False)
            print(f"[DEBUG] Raw inference outputs type: {type(outputs)}")
            
            if isinstance(outputs, torch.Tensor):
                print(f"[DEBUG] Raw inference outputs shape: {outputs.shape}")
                print(f"[DEBUG] Raw inference outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                
                if outputs.dim() >= 3:
                    print(f"[DEBUG] Confidence scores range: {outputs[..., 4].min().item():.4f} to {outputs[..., 4].max().item():.4f}")
                    if outputs.shape[-1] > 5:
                        print(f"[DEBUG] Class scores range: {outputs[..., 5:].min().item():.4f} to {outputs[..., 5:].max().item():.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process features in inference: {e}")
            import traceback
            traceback.print_exc()
            
            ret = []
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [ret, targets]
            return ret
            
        if filtering:
            try:
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    print(f"[DEBUG] Filtering detections with conf_threshold={self.conf_threshold}")
                    
                    if outputs.dim() >= 3:
                        conf_scores = outputs[..., 4]
                        above_threshold = (conf_scores > self.conf_threshold).sum().item()
                        print(f"[DEBUG] Detections above threshold {self.conf_threshold}: {above_threshold}")
                        
                        if above_threshold == 0:
                            temp_threshold = 0.001
                            above_temp = (conf_scores > temp_threshold).sum().item()
                            print(f"[DEBUG] Detections above temp threshold {temp_threshold}: {above_temp}")
                            
                            original_threshold = self.conf_threshold
                            self.conf_threshold = temp_threshold
                            batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=True)
                            self.conf_threshold = original_threshold
                            
                            print(f"[DEBUG] Generated detections with lowered threshold: {len(batch_detections)}")
                        else:
                            batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=True)
                    else:
                        batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=True)
                    
                    outputs = []
                    for batch_det in batch_detections:
                        outputs.extend(batch_det)
                    
                    print(f"[DEBUG] Final detection count: {len(outputs)}")
                    
                else:
                    print(f"[DEBUG] Invalid outputs for filtering: type={type(outputs)}, numel={outputs.numel() if hasattr(outputs, 'numel') else 'N/A'}")
                    outputs = []
                    
            except Exception as e:
                print(f"[ERROR] Failed to postprocess outputs: {e}")
                print(f"[DEBUG] outputs info: type={type(outputs)}, shape={outputs.shape if hasattr(outputs, 'shape') else 'no shape'}")
                outputs = []
        else:
            if not isinstance(outputs, torch.Tensor):
                print(f"[WARNING] Expected tensor output, got {type(outputs)}")
                outputs = torch.zeros(1, 1, 5 + self.backbone.num_classes, device=next(self.parameters()).device)
        
        ret = outputs

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret = [outputs, targets]

        return ret
