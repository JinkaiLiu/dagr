import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import math
from dagr.model.networks.net import sampling_skip
from dagr.model.networks.fcos_head import FCOSHead
from dagr.model.networks.net import Net
from dagr.model.utils import (
    convert_to_training_format,
    convert_to_evaluation_format,
    init_subnetwork
)

def analyze_and_fix_postprocess(prediction, conf_thres, nms_thres, debug_mode=False):
    """
    完整的检测结果分析和修复函数
    解决 "No valid boxes after filtering" 问题
    """
    if debug_mode:
        print(f"\n[BBOX] ========== Detection Analysis ==========")
        print(f"[BBOX] Input shape: {prediction.shape}, Conf threshold: {conf_thres}")
    
    if not isinstance(prediction, torch.Tensor):
        if debug_mode:
            print(f"[BBOX] ERROR: Expected tensor, got {type(prediction)}")
        return []
        
    if prediction.numel() == 0:
        if debug_mode:
            print(f"[BBOX] ERROR: Empty prediction tensor")
        return []
    
    batch_size = prediction.shape[0]
    
    if prediction.dim() == 3:
        num_classes = prediction.shape[2] - 5
    else:
        num_classes = prediction.shape[-1] - 5
    
    if torch.is_tensor(num_classes):
        num_classes = num_classes.item()
    num_classes = int(num_classes)
    
    output = []
    total_detections = 0
    total_valid_boxes = 0
    
    for batch_idx in range(batch_size):
        if debug_mode and batch_idx == 0:
            print(f"[BBOX] Processing batch {batch_idx}: {prediction[batch_idx].shape[0]} predictions")
        
        image_pred = prediction[batch_idx]
        
        if image_pred.dim() == 1:
            image_pred = image_pred.unsqueeze(0)
            
        if image_pred.shape[0] == 0:
            output.append([])
            continue
        
        # 置信度过滤 - 使用较低的阈值
        conf_scores = image_pred[:, 4]
        conf_mask = conf_scores >= conf_thres
        above_threshold = conf_mask.sum().item()
        
        if debug_mode and batch_idx == 0:
            print(f"[BBOX] Detections above threshold {conf_thres}: {above_threshold}")
        
        # 如果没有框超过阈值，保留最高置信度的几个框
        if above_threshold == 0:
            k = min(5, len(conf_scores))  # 保留前5个
            if k > 0:
                _, top_indices = torch.topk(conf_scores, k)
                conf_mask = torch.zeros_like(conf_scores, dtype=torch.bool)
                conf_mask[top_indices] = True
                if debug_mode and batch_idx == 0:
                    print(f"[BBOX] Using top-{k} detections as fallback")
        
        # 应用置信度过滤
        image_pred = image_pred[conf_mask]
        
        if image_pred.shape[0] == 0:
            output.append([])
            continue
        
        # 类别处理
        if num_classes > 0:
            # 获取每个位置的最大类别分数
            class_scores = image_pred[:, 5:5 + num_classes]
            class_conf, class_pred = torch.max(class_scores, 1, keepdim=True)
            
            # 合并边界框、置信度和类别信息
            detections = torch.cat((
                image_pred[:, :4],  # 边界框坐标
                image_pred[:, 4:5],  # 置信度
                class_conf,          # 类别分数
                class_pred.float()   # 类别索引
            ), 1)
        else:
            # 如果没有类别信息，默认全部为第0类
            detections = torch.cat((
                image_pred[:, :4],
                image_pred[:, 4:5],
                torch.ones(image_pred.shape[0], 1, device=image_pred.device),
                torch.zeros(image_pred.shape[0], 1, device=image_pred.device)
            ), 1)
        
        # 关键修复：智能格式检测和处理
        batch_detections = []
        if detections.shape[0] > 0:
            coords = detections[:, :4]
            
            # 检测边界框格式 - 区分XYWH和XYXY格式
            # XYWH格式：宽度和高度应该为正
            # XYXY格式：x2>x1且y2>y1
            
            # 假设是XYWH格式
            potential_widths = coords[:, 2]
            potential_heights = coords[:, 3]
            positive_wh = (potential_widths > 0) & (potential_heights > 0)
            
            # 假设是XYXY格式
            potential_x2_gt_x1 = coords[:, 2] > coords[:, 0]
            potential_y2_gt_y1 = coords[:, 3] > coords[:, 1]
            valid_xyxy = potential_x2_gt_x1 & potential_y2_gt_y1
            
            # 智能判断格式
            xywh_score = positive_wh.sum().item()
            xyxy_score = valid_xyxy.sum().item()
            
            if xywh_score >= xyxy_score:
                # 处理为 XYWH 格式 (xcenter, ycenter, width, height)
                if debug_mode and batch_idx == 0:
                    print(f"[BBOX] Processing as XYWH format: {positive_wh.sum().item()} valid boxes")
                
                valid_mask = positive_wh
                
                if valid_mask.sum() > 0:
                    valid_detections = detections[valid_mask]
                    
                    # XYWH → XYXY 转换
                    cx, cy, w, h = valid_detections[:, 0], valid_detections[:, 1], valid_detections[:, 2], valid_detections[:, 3]
                    
                    # 应用最小尺寸约束
                    min_size = 2.0  # 最小边长
                    w = torch.clamp(w, min=min_size)
                    h = torch.clamp(h, min=min_size)
                    
                    # 计算XYXY坐标
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    # 检查转换后的有效性
                    converted_valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
                    
                    # 确保框有合理的面积
                    box_areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                    reasonable_area = box_areas >= (min_size * min_size)
                    converted_valid = converted_valid & reasonable_area
                    
                    if debug_mode and batch_idx == 0:
                        print(f"[BBOX] After correction: {converted_valid.sum().item()} valid boxes")
                    
                    if converted_valid.sum() > 0:
                        final_boxes = boxes_xyxy[converted_valid]
                        final_scores = valid_detections[converted_valid, 4]
                        final_labels = valid_detections[converted_valid, 6].long() if num_classes > 0 else torch.zeros(converted_valid.sum(), dtype=torch.long, device=valid_detections.device)
                        
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
                            
                            # 创建检测结果字典
                            det_dict = {
                                'boxes': final_boxes[keep_tensor].cpu(),
                                'scores': final_scores[keep_tensor].cpu(),
                                'labels': final_labels[keep_tensor].cpu()
                            }
                            batch_detections.append(det_dict)
                            total_valid_boxes += len(keep_tensor)
            else:
                # 处理为 XYXY 格式 (x1, y1, x2, y2)
                if debug_mode and batch_idx == 0:
                    print(f"[BBOX] Processing as XYXY format: {valid_xyxy.sum().item()} valid boxes")
                
                valid_mask = valid_xyxy
                
                if valid_mask.sum() > 0:
                    valid_detections = detections[valid_mask]
                    boxes_xyxy = valid_detections[:, :4]
                    scores = valid_detections[:, 4]
                    labels = valid_detections[:, 6].long() if num_classes > 0 else torch.zeros(valid_mask.sum(), dtype=torch.long, device=valid_detections.device)
                    
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
                        
                        # 创建检测结果字典
                        det_dict = {
                            'boxes': boxes_xyxy[keep_tensor].cpu(),
                            'scores': scores[keep_tensor].cpu(),
                            'labels': labels[keep_tensor].cpu()
                        }
                        batch_detections.append(det_dict)
                        total_valid_boxes += len(keep_tensor)
        
        output.append(batch_detections)
        if batch_detections:
            total_detections += len(batch_detections[0]['boxes']) if 'boxes' in batch_detections[0] else 0
    
    if debug_mode:
        print(f"[BBOX] Total valid boxes: {total_valid_boxes}, Batches with detections: {sum(1 for batch in output if batch)}")
    
    return output

def unpack_fused_features(fused_feat):
    """
    改进的特征解包函数，更健壮地处理不同形状的特征
    """
    features_tensor = []
    features_hw = []
    
    #if len(fused_feat) > 0:
        #print(f"[INFO] Unpacking {len(fused_feat)} fused features")
    
    target_batch_size = None
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            batch_info = f.batch if hasattr(f, "batch") else None
            if batch_info is not None:
                current_batch_size = len(torch.unique(batch_info))
                if target_batch_size is None:
                    target_batch_size = current_batch_size
                else:
                    target_batch_size = min(target_batch_size, current_batch_size)
    
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x
            
            # 安全获取高度和宽度
            height = None
            width = None
            
            if hasattr(f, "height") and torch.is_tensor(f.height):
                # 处理高度张量，确保它是标量或能转换为标量
                try:
                    if f.height.numel() == 1:
                        height = max(1, f.height.item())  # 确保高度至少为1
                    else:
                        # 如果不是单元素张量，使用第一个元素
                        height = max(1, f.height[0].item() if f.height.numel() > 0 else 1)
                except Exception as e:
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[WARNING] Failed to convert height to scalar: {e}")
                    height = max(1, int(f.height.float().mean().item()) if f.height.numel() > 0 else 1)
            
            if hasattr(f, "width") and torch.is_tensor(f.width):
                # 处理宽度张量，确保它是标量或能转换为标量
                try:
                    if f.width.numel() == 1:
                        width = max(1, f.width.item())  # 确保宽度至少为1
                    else:
                        # 如果不是单元素张量，使用第一个元素
                        width = max(1, f.width[0].item() if f.width.numel() > 0 else 1)
                except Exception as e:
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[WARNING] Failed to convert width to scalar: {e}")
                    width = max(1, int(f.width.float().mean().item()) if f.width.numel() > 0 else 1)
            
            batch_info = f.batch if hasattr(f, "batch") else None
            
            #if i == 0:
                #print(f"[INFO] Feature {i}: x.shape = {x.shape}, h={height}, w={width}")
                #if batch_info is not None and getattr(f, 'global_step', 0) % 200 == 0:
                    #print(f"[DEBUG] Batch info: {torch.unique(batch_info).tolist() if len(torch.unique(batch_info)) < 10 else f'{len(torch.unique(batch_info))} unique values'}")
            
            # 如果未能获取高度宽度，尝试从数据推断
            if height is None or width is None:
                if x.dim() == 2:
                    # 从节点数推断合理的高度和宽度
                    nodes = x.shape[0]
                    aspect_ratio = 1.5  # 默认宽高比
                    
                    # 尝试计算高度宽度
                    height = max(1, int(math.sqrt(nodes / aspect_ratio)))
                    width = max(1, int(nodes / height))
                    
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[INFO] Inferred dimensions from node count: h={height}, w={width}")
                elif x.dim() >= 3:
                    # 如果是3D或4D张量，可以直接从形状获取高度宽度
                    if x.dim() == 3:  # [C, H, W]
                        height, width = max(1, x.shape[1]), max(1, x.shape[2])
                    elif x.dim() == 4:  # [B, C, H, W]
                        height, width = max(1, x.shape[2]), max(1, x.shape[3])
                
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None and batch_info is not None and target_batch_size is not None:
                    batch_size = target_batch_size
                    expected_nodes_per_batch = height * width
                    
                    full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
                    for batch_idx in range(batch_size):
                        # 安全获取batch掩码
                        batch_mask = None
                        try:
                            batch_mask = batch_info == batch_idx
                        except Exception as e:
                            if getattr(f, 'global_step', 0) % 200 == 0:
                                print(f"[WARNING] Failed to create batch mask for idx {batch_idx}: {e}")
                            continue
                        
                        if batch_mask is None or batch_mask.sum() == 0:
                            continue
                            
                        batch_nodes = x[batch_mask]
                        
                        if batch_nodes.shape[0] > 0:
                            try:
                                # 处理节点不足的情况
                                if batch_nodes.shape[0] < expected_nodes_per_batch:
                                    # 直接将节点reshape为可能的最佳形状
                                    nodes_count = batch_nodes.shape[0]
                                    # 安全计算高宽比，避免除零错误
                                    hw_ratio = max(0.01, width / max(1, height))
                                    
                                    # 安全计算新高度，避免除零错误
                                    new_h = int(max(1, (nodes_count / max(0.01, hw_ratio)) ** 0.5))
                                    new_w = int(max(1, nodes_count / max(1, new_h)))
                                    
                                    # 确保高度和宽度至少为1
                                    new_h = max(1, new_h)
                                    new_w = max(1, new_w)
                                    
                                    # 调整节点数量以匹配新的高度和宽度
                                    actual_nodes = min(nodes_count, new_h * new_w)
                                    batch_nodes = batch_nodes[:actual_nodes]
                                    
                                    # 安全检查: 确保在reshape前尺寸合适
                                    if batch_nodes.shape[0] < new_h * new_w:
                                        # 填充到所需大小
                                        padding = torch.zeros(
                                            new_h * new_w - batch_nodes.shape[0], 
                                            channels, 
                                            device=batch_nodes.device, 
                                            dtype=batch_nodes.dtype
                                        )
                                        batch_nodes = torch.cat([batch_nodes, padding], dim=0)
                                    
                                    # 重塑为新的形状
                                    reshaped = batch_nodes.transpose(0, 1).reshape(channels, new_h, new_w)
                                    
                                    # 使用插值调整到目标大小
                                    import torch.nn.functional as F
                                    reshaped = F.interpolate(
                                        reshaped.unsqueeze(0), 
                                        size=(height, width), 
                                        mode='bilinear', 
                                        align_corners=False
                                    ).squeeze(0)
                                else:
                                    # 如果节点数量足够或过多，取前H*W个节点
                                    max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                                    reshaped = batch_nodes[:max_nodes].transpose(0, 1).reshape(channels, height, width)
                                
                                full_tensor[batch_idx] = reshaped
                            except Exception as e:
                                print(f"[ERROR] Failed to reshape batch {batch_idx}: {e}")
                                # 使用全零张量作为备选
                                full_tensor[batch_idx] = torch.zeros(channels, height, width, device=x.device, dtype=x.dtype)
                    
                    x = full_tensor
                    
                elif height is not None and width is not None:
                    # 处理无batch信息的情况
                    try:
                        # 自适应处理节点数不匹配的情况
                        expected_nodes = height * width
                        if num_nodes != expected_nodes:
                            # 安全计算高宽比，避免除零错误
                            hw_ratio = max(0.01, width / max(1, height))
                            new_h = int(max(1, (num_nodes / max(0.01, hw_ratio)) ** 0.5))
                            new_w = int(max(1, num_nodes / max(1, new_h)))
                            
                            # 确保新的高度和宽度至少为1
                            new_h = max(1, new_h)
                            new_w = max(1, new_w)
                            
                            # 截取或填充节点以匹配新的形状
                            if num_nodes < new_h * new_w:
                                # 填充
                                padded_x = torch.zeros(new_h * new_w, channels, device=x.device, dtype=x.dtype)
                                padded_x[:num_nodes] = x
                                x_reshaped = padded_x.transpose(0, 1).reshape(channels, new_h, new_w)
                            else:
                                # 截取
                                x_reshaped = x[:new_h * new_w].transpose(0, 1).reshape(channels, new_h, new_w)
                            
                            # 插值到目标大小
                            import torch.nn.functional as F
                            x = F.interpolate(
                                x_reshaped.unsqueeze(0),
                                size=(height, width),
                                mode='bilinear',
                                align_corners=False
                            )
                        else:
                            x = x.transpose(0, 1).reshape(channels, height, width).unsqueeze(0)
                        
                        if target_batch_size is not None and target_batch_size > 1:
                            x = x.repeat(target_batch_size, 1, 1, 1)
                    except Exception as e:
                        if getattr(f, 'global_step', 0) % 200 == 0:
                            print(f"[ERROR] Failed to reshape feature: {e}")
                        # 创建备选张量
                        x = torch.zeros(target_batch_size or 1, channels, height, width, device=x.device, dtype=x.dtype)
                else:
                    # 无height/width信息时尝试自动推断
                    try:
                        sqrt_nodes = int(num_nodes ** 0.5)
                        if sqrt_nodes * sqrt_nodes == num_nodes:
                            # 完美平方数
                            x = x.transpose(0, 1).reshape(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                            height, width = sqrt_nodes, sqrt_nodes
                        else:
                            # 寻找最接近的长宽比
                            from math import sqrt
                            aspect_ratio = 1.5  # 默认长宽比
                            h = max(1, int(sqrt(num_nodes / aspect_ratio)))
                            w = max(1, int(num_nodes / max(1, h)))
                            
                            # 确保h*w不大于num_nodes
                            while h * w > num_nodes:
                                h -= 1
                                w = max(1, int(num_nodes / max(1, h)))
                            
                            # 确保h和w至少为1
                            h = max(1, h)
                            w = max(1, w)
                            
                            # 如果节点数不足，填充到h*w
                            if num_nodes < h * w:
                                padded_x = torch.zeros(h * w, channels, device=x.device, dtype=x.dtype)
                                padded_x[:num_nodes] = x
                                x = padded_x
                            else:
                                x = x[:h * w]
                                
                            x = x.transpose(0, 1).reshape(channels, h, w).unsqueeze(0)
                            height, width = h, w
                        
                        if target_batch_size is not None and target_batch_size > 1:
                            x = x.repeat(target_batch_size, 1, 1, 1)
                    except Exception as e:
                        if getattr(f, 'global_step', 0) % 200 == 0:
                            print(f"[ERROR] Failed to auto-reshape feature: {e}")
                        # 创建备选张量
                        height = width = max(1, int(num_nodes**0.5) + 1)
                        x = torch.zeros(target_batch_size or 1, channels, height, width, device=x.device, dtype=x.dtype)
                        
            elif x.dim() == 3:
                # 3D张量 [C, H, W] -> [1, C, H, W]
                x = x.unsqueeze(0)
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] == 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    else:
                        x = x[:target_batch_size]
                
                if height is None or width is None:
                    height, width = max(1, x.shape[2]), max(1, x.shape[3])
                
            elif x.dim() == 4:
                # 已经是 [B, C, H, W] 格式，只需要调整batch size
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] > target_batch_size:
                        x = x[:target_batch_size]
                    else:
                        repeats = target_batch_size // max(1, x.shape[0])
                        remainder = target_batch_size % max(1, x.shape[0])
                        if repeats > 0:
                            x_repeated = x.repeat(repeats, 1, 1, 1)
                            if remainder > 0:
                                x = torch.cat([x_repeated, x[:remainder]], dim=0)
                            else:
                                x = x_repeated
                
                if height is None or width is None:
                    height, width = max(1, x.shape[2]), max(1, x.shape[3])
            else:
                raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape={x.shape}")
            
            # 最终检查
            if x.numel() == 0 or any(s == 0 for s in x.shape):
                if getattr(f, 'global_step', 0) % 200 == 0:
                    print(f"[WARNING] Empty tensor for feature {i}, creating placeholder")
                x = torch.zeros(target_batch_size or 1, channels, max(1, height or 1), max(1, width or 1), device=x.device, dtype=x.dtype)
            
            features_tensor.append(x)
            features_hw.append((height, width))
            
        else:
            if isinstance(f, torch.Tensor):
                if f.dim() == 2:
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
    
    #if len(features_tensor) > 0:
        #print(f"[INFO] Final unpacked shapes: {[f.shape for f in features_tensor]}")
    return features_tensor, features_hw
    """
    改进的特征解包函数，更健壮地处理不同形状的特征
    """
    features_tensor = []
    features_hw = []
    
    if len(fused_feat) > 0:
        print(f"[INFO] Unpacking {len(fused_feat)} fused features")
    
    target_batch_size = None
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            batch_info = f.batch if hasattr(f, "batch") else None
            if batch_info is not None:
                current_batch_size = len(torch.unique(batch_info))
                if target_batch_size is None:
                    target_batch_size = current_batch_size
                else:
                    target_batch_size = min(target_batch_size, current_batch_size)
    
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x
            
            # 安全获取高度和宽度
            height = None
            width = None
            
            if hasattr(f, "height") and torch.is_tensor(f.height):
                # 处理高度张量，确保它是标量或能转换为标量
                try:
                    if f.height.numel() == 1:
                        height = f.height.item()
                    else:
                        # 如果不是单元素张量，使用第一个元素
                        height = f.height[0].item() if f.height.numel() > 0 else None
                except Exception as e:
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[WARNING] Failed to convert height to scalar: {e}")
                    height = int(f.height.float().mean().item()) if f.height.numel() > 0 else None
            
            if hasattr(f, "width") and torch.is_tensor(f.width):
                # 处理宽度张量，确保它是标量或能转换为标量
                try:
                    if f.width.numel() == 1:
                        width = f.width.item()
                    else:
                        # 如果不是单元素张量，使用第一个元素
                        width = f.width[0].item() if f.width.numel() > 0 else None
                except Exception as e:
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[WARNING] Failed to convert width to scalar: {e}")
                    width = int(f.width.float().mean().item()) if f.width.numel() > 0 else None
            
            batch_info = f.batch if hasattr(f, "batch") else None
            
            #if i == 0:
                #print(f"[INFO] Feature {i}: x.shape = {x.shape}, h={height}, w={width}")
                #if batch_info is not None and getattr(f, 'global_step', 0) % 200 == 0:
                    #print(f"[DEBUG] Batch info: {torch.unique(batch_info).tolist() if len(torch.unique(batch_info)) < 10 else f'{len(torch.unique(batch_info))} unique values'}")
            
            # 如果未能获取高度宽度，尝试从数据推断
            if height is None or width is None:
                if x.dim() == 2:
                    # 从节点数推断合理的高度和宽度
                    nodes = x.shape[0]
                    aspect_ratio = 1.5  # 默认宽高比
                    
                    # 尝试计算高度宽度
                    height = int(math.sqrt(nodes / aspect_ratio))
                    width = int(nodes / height)
                    
                    # 确保高度宽度至少为1
                    height = max(1, height)
                    width = max(1, width)
                    if getattr(f, 'global_step', 0) % 200 == 0:
                        print(f"[INFO] Inferred dimensions from node count: h={height}, w={width}")
                elif x.dim() >= 3:
                    # 如果是3D或4D张量，可以直接从形状获取高度宽度
                    if x.dim() == 3:  # [C, H, W]
                        height, width = x.shape[1], x.shape[2]
                    elif x.dim() == 4:  # [B, C, H, W]
                        height, width = x.shape[2], x.shape[3]
                
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None and batch_info is not None and target_batch_size is not None:
                    batch_size = target_batch_size
                    expected_nodes_per_batch = height * width
                    
                    full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
                    for batch_idx in range(batch_size):
                        # 安全获取batch掩码
                        batch_mask = None
                        try:
                            batch_mask = batch_info == batch_idx
                        except Exception as e:
                            if getattr(f, 'global_step', 0) % 200 == 0:
                                print(f"[WARNING] Failed to create batch mask for idx {batch_idx}: {e}")
                            continue
                        
                        if batch_mask is None or batch_mask.sum() == 0:
                            continue
                            
                        batch_nodes = x[batch_mask]
                        
                        if batch_nodes.shape[0] > 0:
                            try:
                                # 处理节点不足的情况
                                if batch_nodes.shape[0] < expected_nodes_per_batch:
                                    # 直接将节点reshape为可能的最佳形状
                                    nodes_count = batch_nodes.shape[0]
                                    hw_ratio = width / height
                                    
                                    # 计算适合当前节点数的新高度和宽度
                                    new_h = int((nodes_count / hw_ratio) ** 0.5)
                                    new_w = int(nodes_count / new_h)
                                    
                                    # 确保高度和宽度至少为1
                                    new_h = max(1, new_h)
                                    new_w = max(1, new_w)
                                    
                                    # 调整节点数量以匹配新的高度和宽度
                                    actual_nodes = min(nodes_count, new_h * new_w)
                                    batch_nodes = batch_nodes[:actual_nodes]
                                    
                                    # 重塑为新的形状
                                    reshaped = batch_nodes.transpose(0, 1).reshape(channels, new_h, new_w)
                                    
                                    # 使用插值调整到目标大小
                                    import torch.nn.functional as F
                                    reshaped = F.interpolate(
                                        reshaped.unsqueeze(0), 
                                        size=(height, width), 
                                        mode='bilinear', 
                                        align_corners=False
                                    ).squeeze(0)
                                else:
                                    # 如果节点数量足够或过多，取前H*W个节点
                                    max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                                    reshaped = batch_nodes[:max_nodes].transpose(0, 1).reshape(channels, height, width)
                                
                                full_tensor[batch_idx] = reshaped
                            except Exception as e:
                                if getattr(f, 'global_step', 0) % 200 == 0:
                                    print(f"[ERROR] Failed to reshape batch {batch_idx}: {e}")
                                # 使用全零张量作为备选
                                full_tensor[batch_idx] = torch.zeros(channels, height, width, device=x.device, dtype=x.dtype)
                    
                    x = full_tensor
                    
                elif height is not None and width is not None:
                    # 处理无batch信息的情况
                    try:
                        # 自适应处理节点数不匹配的情况
                        expected_nodes = height * width
                        if num_nodes != expected_nodes:
                            hw_ratio = width / height
                            new_h = int((num_nodes / hw_ratio) ** 0.5)
                            new_w = int(num_nodes / new_h)
                            
                            # 确保新的高度和宽度至少为1
                            new_h = max(1, new_h)
                            new_w = max(1, new_w)
                            
                            # 截取或填充节点以匹配新的形状
                            if num_nodes < new_h * new_w:
                                # 填充
                                padded_x = torch.zeros(new_h * new_w, channels, device=x.device, dtype=x.dtype)
                                padded_x[:num_nodes] = x
                                x_reshaped = padded_x.transpose(0, 1).reshape(channels, new_h, new_w)
                            else:
                                # 截取
                                x_reshaped = x[:new_h * new_w].transpose(0, 1).reshape(channels, new_h, new_w)
                            
                            # 插值到目标大小
                            import torch.nn.functional as F
                            x = F.interpolate(
                                x_reshaped.unsqueeze(0),
                                size=(height, width),
                                mode='bilinear',
                                align_corners=False
                            )
                        else:
                            x = x.transpose(0, 1).reshape(channels, height, width).unsqueeze(0)
                        
                        if target_batch_size is not None and target_batch_size > 1:
                            x = x.repeat(target_batch_size, 1, 1, 1)
                    except Exception as e:
                        if getattr(f, 'global_step', 0) % 200 == 0:
                            print(f"[ERROR] Failed to reshape feature: {e}")
                        # 创建备选张量
                        x = torch.zeros(target_batch_size or 1, channels, height, width, device=x.device, dtype=x.dtype)
                else:
                    # 无height/width信息时尝试自动推断
                    try:
                        sqrt_nodes = int(num_nodes ** 0.5)
                        if sqrt_nodes * sqrt_nodes == num_nodes:
                            # 完美平方数
                            x = x.transpose(0, 1).reshape(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                            height, width = sqrt_nodes, sqrt_nodes
                        else:
                            # 寻找最接近的长宽比
                            from math import sqrt
                            aspect_ratio = 1.5  # 默认长宽比
                            h = int(sqrt(num_nodes / aspect_ratio))
                            w = int(num_nodes / h)
                            
                            # 确保h*w不大于num_nodes
                            while h * w > num_nodes:
                                h -= 1
                                w = int(num_nodes / h)
                            
                            # 确保h和w至少为1
                            h = max(1, h)
                            w = max(1, w)
                            
                            # 如果节点数不足，填充到h*w
                            if num_nodes < h * w:
                                padded_x = torch.zeros(h * w, channels, device=x.device, dtype=x.dtype)
                                padded_x[:num_nodes] = x
                                x = padded_x
                            else:
                                x = x[:h * w]
                                
                            x = x.transpose(0, 1).reshape(channels, h, w).unsqueeze(0)
                            height, width = h, w
                        
                        if target_batch_size is not None and target_batch_size > 1:
                            x = x.repeat(target_batch_size, 1, 1, 1)
                    except Exception as e:
                        if getattr(f, 'global_step', 0) % 200 == 0:
                            print(f"[ERROR] Failed to auto-reshape feature: {e}")
                        # 创建备选张量
                        height = width = int(num_nodes**0.5) + 1
                        x = torch.zeros(target_batch_size or 1, channels, height, width, device=x.device, dtype=x.dtype)
                        
            elif x.dim() == 3:
                # 3D张量 [C, H, W] -> [1, C, H, W]
                x = x.unsqueeze(0)
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] == 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    else:
                        x = x[:target_batch_size]
                
                if height is None or width is None:
                    height, width = x.shape[2], x.shape[3]
                
            elif x.dim() == 4:
                # 已经是 [B, C, H, W] 格式，只需要调整batch size
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
                
                if height is None or width is None:
                    height, width = x.shape[2], x.shape[3]
            else:
                raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            
            # 最终检查
            if x.numel() == 0 or any(s == 0 for s in x.shape):
                if getattr(f, 'global_step', 0) % 200 == 0:
                    print(f"[WARNING] Empty tensor for feature {i}, creating placeholder")
                x = torch.zeros(target_batch_size or 1, channels, height or 1, width or 1, device=x.device, dtype=x.dtype)
            
            features_tensor.append(x)
            features_hw.append((height, width))
            
        else:
            if isinstance(f, torch.Tensor):
                if f.dim() == 2:
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
    
    #if len(features_tensor) > 0:
        #print(f"[INFO] Final unpacked shapes: {[f.shape for f in features_tensor]}")
    return features_tensor, features_hw

class DAGR(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        # 降低检测阈值
        self.conf_threshold = 0.2  # 从0.4降低到0.2
        self.nms_threshold = 0.45  # 从0.3提高到0.45
        self.height = height
        self.width = width
        self.args = args
        self.global_step = 0  # 添加全局步数计数器

        self.backbone = Net(args, height=height, width=width)
        self.use_image = getattr(args, "use_image", True)
        self.no_events = getattr(args, "no_events", False)
        self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

        self.head = None
        self.cnn_head = None
        self._head_initialized = False

        if self.use_image:
            # 确保cnn_head使用正确的通道数
            out_channels = getattr(self.backbone, "out_channels_cnn", [256, 512])
            if isinstance(out_channels, list) and len(out_channels) >= 2:
                cnn_channels = out_channels[:2]
            else:
                if self.global_step % 200 == 0:
                    print(f"[WARNING] Using default channels for CNN head")
                cnn_channels = [256, 512]
                
            self.cnn_head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=cnn_channels,
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            )

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)

    def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
        """
        修改后的边界框转换函数，始终使用左上角格式 (x_tl, y_tl, h, w)
        """
        targets = []
        
        # 检查输入有效性
        if bbox is None or bbox.numel() == 0 or bbox_batch is None or bbox_batch.numel() == 0:
            # 创建空目标
            for _ in range(num_graphs):
                targets.append(torch.zeros(0, 5, device=bbox.device if bbox is not None else self.head.cls_preds[0].weight.device))
            return targets
        
        # 打印边界框统计信息
        if self.global_step % 200 == 0:
            box_stats = {}
            if bbox.shape[1] >= 5:  # 确保至少有5列：x,y,h,w,class
                box_stats["x_range"] = (bbox[:, 0].min().item(), bbox[:, 0].max().item())
                box_stats["y_range"] = (bbox[:, 1].min().item(), bbox[:, 1].max().item())
                box_stats["height_range"] = (bbox[:, 2].min().item(), bbox[:, 2].max().item())
                box_stats["width_range"] = (bbox[:, 3].min().item(), bbox[:, 3].max().item())
                print(f"[BBOX] Stats: x={box_stats['x_range']}, y={box_stats['y_range']}, "
                    f"h={box_stats['height_range']}, w={box_stats['width_range']}")
        
        # 为每个批次处理边界框
        for batch_idx in range(num_graphs):
            # 获取当前批次的边界框
            batch_mask = bbox_batch == batch_idx
            batch_bboxes = bbox[batch_mask]
            
            # 如果当前批次没有边界框，添加空目标
            if batch_bboxes.numel() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
            
            # 解析边界框信息
            if batch_bboxes.shape[1] >= 5:  # 确保有足够的列
                x_tl = batch_bboxes[:, 0]    # 左上角x坐标
                y_tl = batch_bboxes[:, 1]    # 左上角y坐标
                height = batch_bboxes[:, 2]  # 高度
                width = batch_bboxes[:, 3]   # 宽度
                cls = batch_bboxes[:, 4]     # 类别
                
                # 检查是否为归一化坐标
                max_x = x_tl.max().item()
                max_y = y_tl.max().item()
                min_x = x_tl.min().item()
                min_y = y_tl.min().item()
                
                # 如果坐标明显小于图像尺寸，可能是归一化坐标，放大到绝对尺寸
                if max_x <= 1.0 and max_y <= 1.0 and min_x >= 0.0 and min_y >= 0.0:
                    image_width = self.width
                    image_height = self.height
                    x_tl = x_tl * image_width
                    y_tl = y_tl * image_height
                    width = width * image_width
                    height = height * image_height
                    if self.global_step % 200 == 0:
                        print(f"[BBOX] Detected normalized coordinates, scaled to image size: {image_width}x{image_height}")
                
                # 过滤异常大的边界框
                max_reasonable_size = max(self.width, self.height) * 2  # 合理的最大尺寸
                valid_size_mask = (width < max_reasonable_size) & (height < max_reasonable_size)
                
                if valid_size_mask.sum() < len(valid_size_mask) and self.global_step % 200 == 0:
                    print(f"[WARNING] Filtered {len(valid_size_mask) - valid_size_mask.sum().item()} boxes with abnormal size")
                
                # 应用有效性过滤
                x_tl = x_tl[valid_size_mask]
                y_tl = y_tl[valid_size_mask]
                height = height[valid_size_mask]
                width = width[valid_size_mask]
                cls = cls[valid_size_mask]
                
                # 计算面积以检查有效性
                areas = width * height
                valid_mask = areas > 0
                
                if valid_mask.sum() == 0:
                    targets.append(torch.zeros(0, 5, device=bbox.device))
                    continue
                    
                # 应用面积有效性过滤
                x_tl = x_tl[valid_mask]
                y_tl = y_tl[valid_mask]
                height = height[valid_mask]
                width = width[valid_mask]
                cls = cls[valid_mask]
                
                # 从左上角转换为中心点（FCOS需要中心点格式）
                x_center = x_tl + width / 2
                y_center = y_tl + height / 2
                
                # 检查坐标是否在合理范围内
                x_center = torch.clamp(x_center, 0, self.width)
                y_center = torch.clamp(y_center, 0, self.height)
                
                # 创建FCOS格式的目标
                fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
                
                # 打印统计信息
                if fcos_target.shape[0] > 0 and self.global_step % 200 == 0:
                    print(f"[BBOX] Batch {batch_idx}: {fcos_target.shape[0]} valid boxes")
                    print(f"[BBOX] Format: left-top corner (converted to center)")
                    print(f"[BBOX] Center stats: x=({x_center.min().item():.1f}, {x_center.max().item():.1f}), "
                        f"y=({y_center.min().item():.1f}, {y_center.max().item():.1f})")
                    print(f"[BBOX] Size stats: width=({width.min().item():.1f}, {width.max().item():.1f}), "
                        f"height=({height.min().item():.1f}, {height.max().item():.1f})")
                
                targets.append(fcos_target)
            else:
                # 边界框格式不正确，添加空目标
                if self.global_step % 200 == 0:
                    print(f"[WARNING] Invalid bbox format for batch {batch_idx}: shape={batch_bboxes.shape}")
                targets.append(torch.zeros(0, 5, device=bbox.device))
                
        return targets

    # def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
    #     """
    #     改进的边界框转换函数，更健壮地处理不同格式的边界框
    #     """
    #     targets = []
        
    #     # 检查输入有效性
    #     if bbox is None or bbox.numel() == 0 or bbox_batch is None or bbox_batch.numel() == 0:
    #         # 创建空目标
    #         for _ in range(num_graphs):
    #             targets.append(torch.zeros(0, 5, device=bbox.device if bbox is not None else self.head.cls_preds[0].weight.device))
    #         return targets
        
    #     # 打印边界框统计信息，帮助调试（减少频率）
    #     if self.global_step % 200 == 0:
    #         box_stats = {}
    #         if bbox.shape[1] >= 5:  # 确保至少有5列：x,y,h,w,class
    #             box_stats["x_range"] = (bbox[:, 0].min().item(), bbox[:, 0].max().item())
    #             box_stats["y_range"] = (bbox[:, 1].min().item(), bbox[:, 1].max().item())
    #             box_stats["height_range"] = (bbox[:, 2].min().item(), bbox[:, 2].max().item())
    #             box_stats["width_range"] = (bbox[:, 3].min().item(), bbox[:, 3].max().item())
    #             print(f"[BBOX] Stats: x={box_stats['x_range']}, y={box_stats['y_range']}, "
    #                 f"h={box_stats['height_range']}, w={box_stats['width_range']}")
        
    #     # 为每个批次处理边界框
    #     for batch_idx in range(num_graphs):
    #         # 获取当前批次的边界框
    #         batch_mask = bbox_batch == batch_idx
    #         batch_bboxes = bbox[batch_mask]
            
    #         # 如果当前批次没有边界框，添加空目标
    #         if batch_bboxes.numel() == 0:
    #             targets.append(torch.zeros(0, 5, device=bbox.device))
    #             continue
            
    #         # 解析边界框信息
    #         if batch_bboxes.shape[1] >= 5:  # 确保有足够的列
    #             x_left = batch_bboxes[:, 0]    # x坐标（左上角或中心）
    #             y_top = batch_bboxes[:, 1]     # y坐标（左上角或中心）
    #             height = batch_bboxes[:, 2]    # 高度
    #             width = batch_bboxes[:, 3]     # 宽度
    #             cls = batch_bboxes[:, 4]       # 类别
                
    #             # 检测是否为左上角格式 (x_tl, y_tl, h, w)，还是中心点格式 (x_center, y_center, h, w)
    #             # 边界框的格式应该是一致的，所以我们根据统计特征判断
    #             box_count = batch_bboxes.shape[0]
    #             if box_count > 0:
    #                 # 计算并检查坐标范围
    #                 max_x = x_left.max().item()
    #                 max_y = y_top.max().item()
    #                 min_x = x_left.min().item()
    #                 min_y = y_top.min().item()
                    
    #                 # 计算宽高统计信息
    #                 max_w = width.max().item()
    #                 max_h = height.max().item()
                    
    #                 # 如果坐标明显小于图像尺寸，可能是归一化坐标，放大到绝对尺寸
    #                 if max_x <= 1.0 and max_y <= 1.0 and min_x >= 0.0 and min_y >= 0.0:
    #                     image_width = self.width
    #                     image_height = self.height
    #                     x_left = x_left * image_width
    #                     y_top = y_top * image_height
    #                     width = width * image_width
    #                     height = height * image_height
    #                     if self.global_step % 200 == 0:
    #                         print(f"[BBOX] Detected normalized coordinates, scaled to image size: {image_width}x{image_height}")
                    
    #                 # 如果x+width > width或y+height > height，很可能是中心点格式
    #                 # 否则可能是左上角格式
    #                 test_points = 5  # 检查前几个点
    #                 center_format = True
                    
    #                 # 如果边界框数量很少，我们假定是中心点格式（最常见）
    #                 if box_count <= 3:
    #                     center_format = True
    #                     if self.global_step % 200 == 0:
    #                         print(f"[BBOX] Few boxes ({box_count}), assuming center format")
                    
    #                 # 对于更多的边界框，我们进行简单的检测
    #                 # 这种检测并不完美，但提供了一个启发式方法
    #                 if max_w < self.width/4 and max_h < self.height/4:
    #                     # 边界框尺寸合理，假设是中心点格式
    #                     center_format = True
    #             else:
    #                 # 没有足够的边界框来判断，默认使用中心点格式
    #                 center_format = True
                
    #             # 计算面积以检查有效性
    #             areas = width * height
    #             valid_mask = areas > 0
                
    #             if valid_mask.sum() == 0:
    #                 targets.append(torch.zeros(0, 5, device=bbox.device))
    #                 continue
                    
    #             # 应用有效性过滤
    #             x_left = x_left[valid_mask]
    #             y_top = y_top[valid_mask]
    #             height = height[valid_mask]
    #             width = width[valid_mask]
    #             cls = cls[valid_mask]
                
    #             # 根据检测到的格式计算中心点坐标
    #             if center_format:
    #                 # 已经是中心点格式，直接使用
    #                 x_center = x_left
    #                 y_center = y_top
    #             else:
    #                 # 从左上角转换为中心点
    #                 x_center = x_left + width / 2
    #                 y_center = y_top + height / 2
                
    #             # 创建FCOS格式的目标
    #             fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
                
    #             # 打印统计信息（减少频率）
    #             if fcos_target.shape[0] > 0 and self.global_step % 200 == 0:
    #                 print(f"[BBOX] Batch {batch_idx}: {fcos_target.shape[0]} valid boxes")
    #                 print(f"[BBOX] Format: {'center' if center_format else 'corner'}")
    #                 print(f"[BBOX] Center stats: x=({x_center.min().item():.1f}, {x_center.max().item():.1f}), "
    #                     f"y=({y_center.min().item():.1f}, {y_center.max().item():.1f})")
    #                 print(f"[BBOX] Size stats: width=({width.min().item():.1f}, {width.max().item():.1f}), "
    #                     f"height=({height.min().item():.1f}, {height.max().item():.1f})")
                
    #             targets.append(fcos_target)
    #         else:
    #             # 边界框格式不正确，添加空目标
    #             if self.global_step % 200 == 0:
    #                 print(f"[WARNING] Invalid bbox format for batch {batch_idx}: shape={batch_bboxes.shape}")
    #             targets.append(torch.zeros(0, 5, device=bbox.device))
                
    #     return targets

    def _initialize_head_if_needed(self, fused_feat):
        """按需初始化FCOS头部"""
        if not self._head_initialized:
            actual_channels = []
            for f in fused_feat[:2]:
                if hasattr(f, "x"):
                    channels = f.x.shape[1]
                    actual_channels.append(channels)
                elif isinstance(f, torch.Tensor):
                    if f.dim() == 4:  # [B,C,H,W]
                        channels = f.shape[1]
                    elif f.dim() == 3:  # [C,H,W]
                        channels = f.shape[0]
                    elif f.dim() == 2:  # [N,C] or [C,N]
                        channels = min(f.shape[0], f.shape[1])  # 保守估计
                    else:
                        channels = 256  # 默认值
                    actual_channels.append(channels)
                else:
                    actual_channels.append(320)  # 默认通道数
            
            if len(actual_channels) < 2:
                # 确保至少有两个通道数
                while len(actual_channels) < 2:
                    actual_channels.append(actual_channels[-1] if actual_channels else 320)
            
            print(f"[INFO] Initializing FCOS head with actual channels: {actual_channels}")
            
            self.head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=actual_channels,
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            ).to(next(self.parameters()).device)
            
            self._head_initialized = True

    def forward(self, x, reset=True, return_targets=True, filtering=True):
        """
        改进的DAGR模型前向传播函数
        """
        self.global_step += 1  # 增加全局步数
        
        # 特征提取
        if self.use_image:
            event_feat, image_feat = self.backbone(x)
        else:
            event_feat = self.backbone(x)
            image_feat = None

        # 处理预训练CNN模式
        if self.training and self.pretrain_cnn:
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[1]  # 注意这里的维度修改，应该是shape[1]而不是shape[0]//(h*w)
                        tensor = tensor.permute(1, 0).reshape(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            return self.cnn_head(image_feat_tensors, targets=targets, training=True)

        # 处理纯图像模式
        if not self.training and self.no_events:
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[1]  # 同样修改这里的维度
                        tensor = tensor.permute(1, 0).reshape(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            outputs = self.cnn_head(image_feat_tensors, training=False)
            
            # 后处理
            if filtering:
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=False)
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

        # 处理常规模式
        fused_feat = event_feat
        if self.use_image and image_feat is not None:
            # 只在全局步数为200的倍数时打印信息
            if self.global_step % 200 == 0:
                print(f"[INFO] Fusing image and event features")
            for i in range(min(len(event_feat), len(image_feat))):
                if i >= len(image_feat):
                    break
                    
                if not hasattr(event_feat[i], 'width') or not hasattr(event_feat[i], 'height'):
                    event_feat[i].width = torch.tensor([image_feat[i].shape[-1]])
                    event_feat[i].height = torch.tensor([image_feat[i].shape[-2]])
                    
                # 使用detach减少梯度流，避免内存泄漏
                try:
                    # 先确保融合前的特征形状是兼容的
                    if hasattr(event_feat[i], 'x') and isinstance(event_feat[i].x, torch.Tensor):
                        # 获取图像特征的形状信息
                        img_feat = image_feat[i]
                        img_c = img_feat.shape[1] if img_feat.dim() == 4 else img_feat.shape[0]
                        img_h = img_feat.shape[-2]
                        img_w = img_feat.shape[-1]
                        
                        # 记录原始事件特征
                        event_x = event_feat[i].x
                        
                        # 调整事件特征的形状以匹配图像特征
                        if event_x.dim() == 2:  # 如果是2D张量 [N, C]
                            event_c = event_x.shape[1]
                            
                            # 创建融合后的特征张量
                            fused_x = torch.zeros(event_x.shape[0], event_c + img_c, device=event_x.device)
                            
                            # 填充事件特征
                            fused_x[:, :event_c] = event_x
                            
                            # 重塑并采样图像特征以匹配事件特征
                            event_batch = event_feat[i].batch if hasattr(event_feat[i], 'batch') else None
                            if event_batch is not None:
                                # 为每个批次节点采样对应的图像特征
                                for b_idx in torch.unique(event_batch):
                                    batch_mask = event_batch == b_idx
                                    batch_nodes = event_x[batch_mask]
                                    
                                    # 确保图像特征批次索引有效
                                    img_b_idx = min(b_idx.item(), img_feat.shape[0]-1) if img_feat.dim() == 4 else 0
                                    
                                    # 创建图像特征网格采样点
                                    h_idx = torch.linspace(0, img_h-1, steps=int(batch_nodes.shape[0]**0.5), device=event_x.device).long()
                                    w_idx = torch.linspace(0, img_w-1, steps=int(batch_nodes.shape[0]**0.5), device=event_x.device).long()
                                    
                                    # 创建网格点
                                    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')
                                    flat_h = grid_h.reshape(-1)
                                    flat_w = grid_w.reshape(-1)
                                    
                                    # 限制采样点数量
                                    max_pts = min(batch_nodes.shape[0], len(flat_h))
                                    flat_h = flat_h[:max_pts]
                                    flat_w = flat_w[:max_pts]
                                    
                                    # 采样图像特征
                                    if img_feat.dim() == 4:
                                        sampled_img_feat = img_feat[img_b_idx, :, flat_h, flat_w].t()
                                    else:
                                        sampled_img_feat = img_feat[:, flat_h, flat_w].t()
                                    
                                    # 处理形状不匹配
                                    if sampled_img_feat.shape[0] < batch_nodes.shape[0]:
                                        # 填充不足的部分
                                        padding = torch.zeros(batch_nodes.shape[0] - sampled_img_feat.shape[0], 
                                                            sampled_img_feat.shape[1], 
                                                            device=sampled_img_feat.device)
                                        sampled_img_feat = torch.cat([sampled_img_feat, padding], dim=0)
                                    elif sampled_img_feat.shape[0] > batch_nodes.shape[0]:
                                        # 裁剪多余的部分
                                        sampled_img_feat = sampled_img_feat[:batch_nodes.shape[0]]
                                    
                                    # 合并到融合特征中
                                    fused_x[batch_mask, event_c:] = sampled_img_feat
                            
                            # 更新事件特征
                            event_feat[i].x = fused_x
                        else:
                            # 对于其他维度的特征，使用sampling_skip函数
                            event_feat[i].x = sampling_skip(event_feat[i], img_feat.detach())
                except Exception as e:
                    if self.global_step % 200 == 0:
                        print(f"[WARNING] Failed to fuse features at level {i}: {e}")
                    # 保持原始特征不变
                    pass
                    
            fused_feat = event_feat

        # 训练模式
        if self.training:
            # 转换目标格式
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            # 打印目标信息（减少频率）
            if self.global_step % 200 == 0:
                print(f"[INFO] Training mode - processing {len(targets)} targets")
                valid_targets = 0
                for i, t in enumerate(targets):
                    if t.numel() > 0:
                        valid_targets += 1
                        valid_boxes = t[t[:, 1:].sum(dim=1) > 0]
                        if valid_boxes.shape[0] > 0 and valid_boxes.shape[0] <= 5:  # 限制打印数量
                            print(f"[INFO] Target {i}: {valid_boxes.shape[0]} valid objects")
                            for j, box in enumerate(valid_boxes):
                                cls_id = box[0].item()
                                cls_name = getattr(self, 'class_names', {}).get(cls_id, f"class_{cls_id}")
                                print(f"       Box {j}: {cls_name}, center=({box[1].item():.1f}, {box[2].item():.1f}), size=({box[3].item():.1f}, {box[4].item():.1f})")
                print(f"[INFO] Total valid targets: {valid_targets}")
            
            # 初始化检测头
            self._initialize_head_if_needed(fused_feat)
            
            # 解包特征
            fused_feat_tensors, _ = unpack_fused_features(fused_feat)
            
            # 前向传播计算损失
            loss_fused = self.head(fused_feat_tensors, targets=targets, training=True)

            # 整理损失结果
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

            # 处理图像分支
            if self.use_image:
                image_feat_tensors = []
                for img_f in image_feat:
                    if hasattr(img_f, 'x'):
                        tensor = img_f.x
                        if tensor.dim() == 2:
                            h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                            w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                            c = tensor.shape[1]  # 修改维度
                            tensor = tensor.permute(1, 0).reshape(c, h, w).unsqueeze(0)
                        elif tensor.dim() == 3:
                            tensor = tensor.unsqueeze(0)
                        image_feat_tensors.append(tensor)
                    else:
                        image_feat_tensors.append(img_f)
                        
                # 计算CNN分支损失
                loss_image = self.cnn_head(image_feat_tensors, targets=targets, training=True)
                
                # 整理CNN损失
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

            # 确保损失键存在
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

        # 推理模式
        # 打印特征信息（减少频率）
        if self.global_step % 200 == 0:
            for i, f in enumerate(fused_feat[:1]):  # 只打印第一个特征层
                print(f"[INFO] Inference Feature {i}: shape={f.x.shape}, h={f.height.item() if hasattr(f, 'height') else 'N/A'}, w={f.width.item() if hasattr(f, 'width') else 'N/A'}")

        x.reset = reset
        
        # 初始化检测头
        self._initialize_head_if_needed(fused_feat)
        
        try:
            # 解包特征
            fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            
            # 验证特征有效性
            valid_features = []
            for i, feat in enumerate(fused_feat_x):
                if feat.numel() > 0 and all(d > 0 for d in feat.shape):
                    valid_features.append(feat)
                elif self.global_step % 200 == 0:
                    print(f"[WARNING] Inference feature {i} is invalid: {feat.shape}, numel={feat.numel()}")
            
            if len(valid_features) == 0:
                if self.global_step % 200 == 0:
                    print(f"[ERROR] No valid features after unpacking in inference!")
                ret = []
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    ret = [ret, targets]
                return ret
            
            # 进行推理
            outputs = self.head(valid_features, training=False)
            
        except Exception as e:
            if self.global_step % 200 == 0:
                print(f"[ERROR] Failed to process features in inference: {e}")
                import traceback
                traceback.print_exc()
            
            ret = []
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [ret, targets]
            return ret
            
        # 后处理
        if filtering:
            try:
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    # 使用改进的后处理函数
                    batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=(self.global_step % 200 == 0))
                    outputs = []
                    for batch_det in batch_detections:
                        outputs.extend(batch_det)
                    
                    if self.global_step % 200 == 0:
                        print(f"[INFO] Final detection count: {len(outputs)}")
                    
                else:
                    outputs = []
                    
            except Exception as e:
                if self.global_step % 200 == 0:
                    print(f"[ERROR] Failed to postprocess outputs: {e}")
                outputs = []
        else:
            if not isinstance(outputs, torch.Tensor) and self.global_step % 200 == 0:
                print(f"[WARNING] Expected tensor output, got {type(outputs)}")
                outputs = torch.zeros(1, 1, 5 + self.backbone.num_classes, device=next(self.parameters()).device)
        
        ret = outputs

        # 添加目标信息（如果需要）
        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret = [outputs, targets]

        return ret





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.ops as ops
# from dagr.model.networks.net import sampling_skip
# from dagr.model.networks.fcos_head import FCOSHead
# from dagr.model.networks.net import Net
# from dagr.model.utils import (
#     convert_to_training_format,
#     convert_to_evaluation_format,
#     init_subnetwork
# )

# def analyze_and_fix_postprocess(prediction, conf_thres, nms_thres, debug_mode=False):
#     """
#     完整的检测结果分析和修复函数
#     解决 "No valid boxes after filtering" 问题
#     """
#     if debug_mode:
#         print(f"\n[BBOX] ========== Detection Analysis ==========")
#         print(f"[BBOX] Input shape: {prediction.shape}, Conf threshold: {conf_thres}")
    
#     if not isinstance(prediction, torch.Tensor):
#         if debug_mode:
#             print(f"[BBOX] ERROR: Expected tensor, got {type(prediction)}")
#         return []
        
#     if prediction.numel() == 0:
#         if debug_mode:
#             print(f"[BBOX] ERROR: Empty prediction tensor")
#         return []
    
#     batch_size = prediction.shape[0]
    
#     if prediction.dim() == 3:
#         num_classes = prediction.shape[2] - 5
#     else:
#         num_classes = prediction.shape[-1] - 5
    
#     if torch.is_tensor(num_classes):
#         num_classes = num_classes.item()
#     num_classes = int(num_classes)
    
#     output = []
#     total_detections = 0
#     total_valid_boxes = 0
    
#     for batch_idx in range(batch_size):
#         if debug_mode and batch_idx == 0:
#             print(f"[BBOX] Processing batch {batch_idx}: {prediction[batch_idx].shape[0]} predictions")
        
#         image_pred = prediction[batch_idx]
        
#         if image_pred.dim() == 1:
#             image_pred = image_pred.unsqueeze(0)
            
#         if image_pred.shape[0] == 0:
#             output.append([])
#             continue
        
#         # 置信度过滤
#         conf_scores = image_pred[:, 4]
#         conf_mask = conf_scores >= conf_thres
#         above_threshold = conf_mask.sum().item()
        
#         if debug_mode and batch_idx == 0:
#             print(f"[BBOX] Detections above threshold {conf_thres}: {above_threshold}")
        
#         if above_threshold == 0:
#             k = min(5, len(conf_scores))
#             if k > 0:
#                 _, top_indices = torch.topk(conf_scores, k)
#                 conf_mask = torch.zeros_like(conf_scores, dtype=torch.bool)
#                 conf_mask[top_indices] = True
#                 if debug_mode and batch_idx == 0:
#                     print(f"[BBOX] Using top-{k} detections as fallback")
        
#         image_pred = image_pred[conf_mask]
        
#         if image_pred.shape[0] == 0:
#             output.append([])
#             continue
        
#         # 类别处理
#         if num_classes > 0:
#             class_scores = image_pred[:, 5:5 + num_classes]
#             class_conf, class_pred = torch.max(class_scores, 1, keepdim=True)
            
#             detections = torch.cat((
#                 image_pred[:, :4],
#                 image_pred[:, 4:5],
#                 class_conf,
#                 class_pred.float()
#             ), 1)
#         else:
#             detections = torch.cat((
#                 image_pred[:, :4],
#                 image_pred[:, 4:5],
#                 torch.ones(image_pred.shape[0], 1, device=image_pred.device),
#                 torch.zeros(image_pred.shape[0], 1, device=image_pred.device)
#             ), 1)
        
#         # 关键修复：智能格式检测和处理
#         batch_detections = []
#         if detections.shape[0] > 0:
#             coords = detections[:, :4]
            
#             # 重新进行格式检测
#             potential_widths = coords[:, 2]
#             potential_heights = coords[:, 3]
#             positive_wh = (potential_widths > 0) & (potential_heights > 0)
            
#             potential_x2_gt_x1 = coords[:, 2] > coords[:, 0]
#             potential_y2_gt_y1 = coords[:, 3] > coords[:, 1]
#             valid_xyxy = potential_x2_gt_x1 & potential_y2_gt_y1
            
#             # 智能格式判断
#             xywh_score = positive_wh.sum().item()
#             xyxy_score = valid_xyxy.sum().item()
            
#             if xywh_score >= xyxy_score:
#                 # 处理为 XYWH 格式
#                 if debug_mode and batch_idx == 0:
#                     print(f"[BBOX] Processing as XYWH format: {positive_wh.sum().item()} valid boxes")
                
#                 valid_mask = positive_wh
                
#                 if valid_mask.sum() > 0:
#                     valid_detections = detections[valid_mask]
                    
#                     # XYWH → XYXY 转换
#                     cx, cy, w, h = valid_detections[:, 0], valid_detections[:, 1], valid_detections[:, 2], valid_detections[:, 3]
                    
#                     # 应用最小尺寸约束 - 确保 box 至少有合理的尺寸
#                     min_size = 5.0  # 最小边长 5 像素
#                     w = torch.clamp(w, min=min_size)
#                     h = torch.clamp(h, min=min_size)
                    
#                     x1 = cx - w / 2
#                     y1 = cy - h / 2
#                     x2 = cx + w / 2
#                     y2 = cy + h / 2
#                     boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                    
#                     # 转换后的有效性检查
#                     converted_valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
                    
#                     # 额外检查：确保 box 有合理的面积
#                     box_areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
#                     reasonable_area = box_areas >= (min_size * min_size)  # 至少 25 平方像素
#                     converted_valid = converted_valid & reasonable_area
                    
#                     if debug_mode and batch_idx == 0:
#                         print(f"[BBOX] After correction: {converted_valid.sum().item()} valid boxes")
                    
#                     if converted_valid.sum() > 0:
#                         final_boxes = boxes_xyxy[converted_valid]
#                         final_scores = valid_detections[converted_valid, 4]
#                         final_labels = valid_detections[converted_valid, 6].long() if num_classes > 0 else torch.zeros(converted_valid.sum(), dtype=torch.long)
                        
#                         # 按类别进行 NMS
#                         final_keep_indices = []
#                         for class_id in torch.unique(final_labels):
#                             class_mask = final_labels == class_id
#                             if class_mask.sum() == 0:
#                                 continue
                                
#                             class_boxes = final_boxes[class_mask]
#                             class_scores = final_scores[class_mask]
                            
#                             if len(class_boxes) > 0:
#                                 keep = ops.nms(class_boxes, class_scores, iou_threshold=nms_thres)
#                                 class_indices = torch.where(class_mask)[0]
#                                 final_keep_indices.extend(class_indices[keep].tolist())
                        
#                         if len(final_keep_indices) > 0:
#                             keep_tensor = torch.tensor(final_keep_indices, device=final_boxes.device)
                            
#                             det_dict = {
#                                 'boxes': final_boxes[keep_tensor].cpu(),
#                                 'scores': final_scores[keep_tensor].cpu(),
#                                 'labels': final_labels[keep_tensor].cpu()
#                             }
#                             batch_detections.append(det_dict)
#                             total_valid_boxes += len(keep_tensor)
#             else:
#                 # 处理为 XYXY 格式
#                 if debug_mode and batch_idx == 0:
#                     print(f"[BBOX] Processing as XYXY format: {valid_xyxy.sum().item()} valid boxes")
                
#                 valid_mask = valid_xyxy
                
#                 if valid_mask.sum() > 0:
#                     valid_detections = detections[valid_mask]
#                     boxes_xyxy = valid_detections[:, :4]
#                     scores = valid_detections[:, 4]
#                     labels = valid_detections[:, 6].long() if num_classes > 0 else torch.zeros(valid_mask.sum(), dtype=torch.long)
                    
#                     # 按类别进行 NMS
#                     final_keep_indices = []
#                     for class_id in torch.unique(labels):
#                         class_mask = labels == class_id
#                         if class_mask.sum() == 0:
#                             continue
                            
#                         class_boxes = boxes_xyxy[class_mask]
#                         class_scores = scores[class_mask]
                        
#                         if len(class_boxes) > 0:
#                             keep = ops.nms(class_boxes, class_scores, iou_threshold=nms_thres)
#                             class_indices = torch.where(class_mask)[0]
#                             final_keep_indices.extend(class_indices[keep].tolist())
                    
#                     if len(final_keep_indices) > 0:
#                         keep_tensor = torch.tensor(final_keep_indices, device=boxes_xyxy.device)
                        
#                         det_dict = {
#                             'boxes': boxes_xyxy[keep_tensor].cpu(),
#                             'scores': scores[keep_tensor].cpu(),
#                             'labels': labels[keep_tensor].cpu()
#                         }
#                         batch_detections.append(det_dict)
#                         total_valid_boxes += len(keep_tensor)
        
#         output.append(batch_detections)
#         total_detections += len(batch_detections[0]) if batch_detections else 0
    
#     if debug_mode:
#         print(f"[BBOX] Total valid boxes: {total_valid_boxes}, Batches with detections: {sum(1 for batch in output if batch)}")
    
#     return output

# def focal_loss(pred, target, alpha=0.25, gamma=2.0):
#     p = torch.sigmoid(pred)
#     ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
#     p_t = target * p + (1 - target) * (1 - p)
#     loss = ce_loss * ((1 - p_t) ** gamma)
#     if alpha >= 0:
#         alpha_t = target * alpha + (1 - target) * (1 - alpha)
#         loss *= alpha_t
#     return loss.sum()

# def unpack_fused_features(fused_feat):
#     features_tensor = []
#     features_hw = []
    
#     if len(fused_feat) > 0:
#         print(f"[INFO] Unpacking {len(fused_feat)} fused features")
    
#     target_batch_size = None
#     for i, f in enumerate(fused_feat):
#         if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
#             batch_info = f.batch if hasattr(f, "batch") else None
#             if batch_info is not None:
#                 current_batch_size = len(batch_info.unique())
#                 if target_batch_size is None:
#                     target_batch_size = current_batch_size
#                 else:
#                     target_batch_size = min(target_batch_size, current_batch_size)
    
#     for i, f in enumerate(fused_feat):
#         if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
#             x = f.x
#             height = f.height.item() if hasattr(f, "height") and torch.is_tensor(f.height) else None
#             width = f.width.item() if hasattr(f, "width") and torch.is_tensor(f.width) else None
#             batch_info = f.batch if hasattr(f, "batch") else None
            
#             if i == 0:
#                 print(f"[INFO] Feature {i}: x.shape = {x.shape}, h={height}, w={width}")
            
#             if x.dim() == 2:
#                 num_nodes, channels = x.shape
                
#                 if height is not None and width is not None and batch_info is not None and target_batch_size is not None:
#                     batch_size = target_batch_size
#                     expected_nodes_per_batch = height * width
                    
#                     full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
#                     for batch_idx in range(batch_size):
#                         if batch_idx < len(batch_info.unique()):
#                             batch_mask = batch_info == batch_idx
#                             batch_nodes = x[batch_mask]
                            
#                             if batch_nodes.shape[0] > 0:
#                                 max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                                
#                                 if max_nodes == expected_nodes_per_batch:
#                                     reshaped = batch_nodes[:max_nodes].transpose(0, 1).view(channels, height, width)
#                                 else:
#                                     temp_features = torch.zeros(expected_nodes_per_batch, channels, device=x.device, dtype=x.dtype)
#                                     temp_features[:max_nodes] = batch_nodes[:max_nodes]
#                                     reshaped = temp_features.transpose(0, 1).view(channels, height, width)
                                
#                                 full_tensor[batch_idx] = reshaped
                    
#                     x = full_tensor
                    
#                 elif height is not None and width is not None:
#                     expected_nodes = height * width
                    
#                     if num_nodes <= expected_nodes:
#                         full_features = torch.zeros(expected_nodes, channels, device=x.device, dtype=x.dtype)
#                         full_features[:num_nodes] = x
#                         x = full_features.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
#                     else:
#                         x_truncated = x[:expected_nodes]
#                         x = x_truncated.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    
#                     if target_batch_size is not None and target_batch_size > 1:
#                         x = x.repeat(target_batch_size, 1, 1, 1)
#                 else:
#                     sqrt_nodes = int(num_nodes ** 0.5)
#                     if sqrt_nodes * sqrt_nodes == num_nodes:
#                         x = x.transpose(0, 1).contiguous().view(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
#                         height, width = sqrt_nodes, sqrt_nodes
#                     else:
#                         x = x.transpose(0, 1).contiguous().view(channels, num_nodes, 1).unsqueeze(0)
#                         height, width = num_nodes, 1
                    
#                     if target_batch_size is not None and target_batch_size > 1:
#                         x = x.repeat(target_batch_size, 1, 1, 1)
                        
#             elif x.dim() == 3:
#                 x = x.unsqueeze(0)
#                 if target_batch_size is not None and x.shape[0] != target_batch_size:
#                     if x.shape[0] == 1:
#                         x = x.repeat(target_batch_size, 1, 1, 1)
#                     else:
#                         x = x[:target_batch_size]
#             elif x.dim() == 4:
#                 if target_batch_size is not None and x.shape[0] != target_batch_size:
#                     if x.shape[0] > target_batch_size:
#                         x = x[:target_batch_size]
#                     else:
#                         repeats = target_batch_size // x.shape[0]
#                         remainder = target_batch_size % x.shape[0]
#                         if repeats > 0:
#                             x_repeated = x.repeat(repeats, 1, 1, 1)
#                             if remainder > 0:
#                                 x = torch.cat([x_repeated, x[:remainder]], dim=0)
#                             else:
#                                 x = x_repeated
#             else:
#                 raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            
#             features_tensor.append(x)
#             features_hw.append((height, width))
            
#         else:
#             if isinstance(f, torch.Tensor):
#                 if f.dim() == 2:
#                     f = f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
#                 elif f.dim() == 3:
#                     f = f.unsqueeze(0)
                
#                 if target_batch_size is not None and f.shape[0] != target_batch_size:
#                     if f.shape[0] == 1:
#                         f = f.repeat(target_batch_size, 1, 1, 1)
#                     else:
#                         f = f[:target_batch_size]
                        
#                 features_tensor.append(f)
#                 features_hw.append((None, None))
#             else:
#                 raise ValueError(f"Unknown feature type: {type(f)}")
    
#     if len(features_tensor) > 0:
#         print(f"[INFO] Final unpacked shapes: {[f.shape for f in features_tensor]}")
#     return features_tensor, features_hw


# class DAGR(nn.Module):
#     def __init__(self, args, height, width):
#         super().__init__()
#         self.conf_threshold = 0.4
#         self.nms_threshold = 0.3
#         self.height = height
#         self.width = width
#         self.args = args

#         self.backbone = Net(args, height=height, width=width)
#         self.use_image = getattr(args, "use_image", True)
#         self.no_events = getattr(args, "no_events", False)
#         self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

#         self.head = None
#         self.cnn_head = None
#         self._head_initialized = False

#         if self.use_image:
#             self.cnn_head = FCOSHead(
#                 num_classes=self.backbone.num_classes,
#                 in_channels=self.backbone.out_channels_cnn[:2],
#                 strides=self.backbone.strides[:2],
#                 use_focal_loss=True,
#                 use_iou_loss=True
#             )

#         if "img_net_checkpoint" in args:
#             state_dict = torch.load(args.img_net_checkpoint)
#             init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)

#     def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
#         targets = []
        
#         for batch_idx in range(num_graphs):
#             batch_mask = bbox_batch == batch_idx
#             batch_bboxes = bbox[batch_mask]
            
#             if batch_bboxes.numel() == 0:
#                 targets.append(torch.zeros(0, 5, device=bbox.device))
#                 continue
            
#             x_left = batch_bboxes[:, 0]
#             y_top = batch_bboxes[:, 1]
#             height = batch_bboxes[:, 2]
#             width = batch_bboxes[:, 3]
#             cls = batch_bboxes[:, 4]
            
#             areas = width * height
#             valid_mask = areas > 0
            
#             if valid_mask.sum() == 0:
#                 targets.append(torch.zeros(0, 5, device=bbox.device))
#                 continue
                
#             x_left = x_left[valid_mask]
#             y_top = y_top[valid_mask]
#             height = height[valid_mask]
#             width = width[valid_mask]
#             cls = cls[valid_mask]
            
#             x_center = x_left + width / 2
#             y_center = y_top + height / 2
            
#             fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
#             targets.append(fcos_target)
            
#         return targets

#     def _initialize_head_if_needed(self, fused_feat):
#         if not self._head_initialized:
#             actual_channels = []
#             for f in fused_feat[:2]:
#                 if hasattr(f, "x"):
#                     channels = f.x.shape[1]
#                     actual_channels.append(channels)
#                 else:
#                     actual_channels.append(f.shape[1])
            
#             print(f"[INFO] Initializing FCOS head with actual channels: {actual_channels}")
            
#             self.head = FCOSHead(
#                 num_classes=self.backbone.num_classes,
#                 in_channels=actual_channels,
#                 strides=self.backbone.strides[:2],
#                 use_focal_loss=True,
#                 use_iou_loss=True
#             ).to(next(self.parameters()).device)
            
#             self._head_initialized = True

#     def forward(self, x, reset=True, return_targets=True, filtering=True):
#         if self.use_image:
#             event_feat, image_feat = self.backbone(x)
#         else:
#             event_feat = self.backbone(x)
#             image_feat = None

#         if self.training and self.pretrain_cnn:
#             targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
#             image_feat_tensors = []
#             for img_f in image_feat:
#                 if hasattr(img_f, 'x'):
#                     tensor = img_f.x
#                     if tensor.dim() == 2:
#                         h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
#                         w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
#                         c = tensor.shape[0] // (h * w)
#                         tensor = tensor.view(c, h, w).unsqueeze(0)
#                     elif tensor.dim() == 3:
#                         tensor = tensor.unsqueeze(0)
#                     image_feat_tensors.append(tensor)
#                 else:
#                     image_feat_tensors.append(img_f)
#             return self.cnn_head(image_feat_tensors, targets=targets, training=True)

#         if not self.training and self.no_events:
#             image_feat_tensors = []
#             for img_f in image_feat:
#                 if hasattr(img_f, 'x'):
#                     tensor = img_f.x
#                     if tensor.dim() == 2:
#                         h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
#                         w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
#                         c = tensor.shape[0] // (h * w)
#                         tensor = tensor.view(c, h, w).unsqueeze(0)
#                     elif tensor.dim() == 3:
#                         tensor = tensor.unsqueeze(0)
#                     image_feat_tensors.append(tensor)
#                 else:
#                     image_feat_tensors.append(img_f)
#             outputs = self.cnn_head(image_feat_tensors, training=False)
            
#             if filtering:
#                 if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
#                     batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=False)
#                     outputs = []
#                     for batch_det in batch_detections:
#                         outputs.extend(batch_det)
#                 else:
#                     outputs = []
            
#             ret = outputs
#             if return_targets and hasattr(x, 'bbox'):
#                 targets = convert_to_evaluation_format(x)
#                 ret = [outputs, targets]
            
#             return ret

#         fused_feat = event_feat
#         if self.use_image and image_feat is not None:
#             print(f"[INFO] Fusing image and event features")
#             for i in range(len(event_feat)):
#                 event_feat[i].width = torch.tensor([image_feat[i].shape[-1]])
#                 event_feat[i].height = torch.tensor([image_feat[i].shape[-2]])
#                 event_feat[i].x = sampling_skip(event_feat[i], image_feat[i].detach())
                
#         fused_feat = event_feat

#         if self.training:
#             targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
#             print(f"[INFO] Training mode - processing {len(targets)} targets")
#             valid_targets = 0
#             for i, t in enumerate(targets):
#                 if t.numel() > 0:
#                     valid_targets += 1
#                     valid_boxes = t[t[:, 1:].sum(dim=1) > 0]
#                     if valid_boxes.shape[0] > 0 and valid_boxes.shape[0] <= 3:  # 只打印少量目标的情况
#                         print(f"[INFO] Target {i}: {valid_boxes.shape[0]} valid objects, bbox ranges: {valid_boxes[:, 1:].min(0)[0]} to {valid_boxes[:, 1:].max(0)[0]}")
#             print(f"[INFO] Total valid targets: {valid_targets}")
            
#             self._initialize_head_if_needed(fused_feat)
            
#             fused_feat_tensors, _ = unpack_fused_features(fused_feat)
#             loss_fused = self.head(fused_feat_tensors, targets=targets, training=True)

#             result = {}
            
#             if isinstance(loss_fused, dict):
#                 for k, v in loss_fused.items():
#                     if torch.is_tensor(v):
#                         result[f"fusion_{k}"] = v
#                     else:
#                         device = next(self.parameters()).device
#                         result[f"fusion_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
#             else:
#                 device = next(self.parameters()).device
#                 result["fusion_total_loss"] = torch.tensor(float(loss_fused), device=device, requires_grad=True) if not torch.is_tensor(loss_fused) else loss_fused

#             if self.use_image:
#                 image_feat_tensors = []
#                 for img_f in image_feat:
#                     if hasattr(img_f, 'x'):
#                         tensor = img_f.x
#                         if tensor.dim() == 2:
#                             h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
#                             w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
#                             c = tensor.shape[0] // (h * w)
#                             tensor = tensor.view(c, h, w).unsqueeze(0)
#                         elif tensor.dim() == 3:
#                             tensor = tensor.unsqueeze(0)
#                         image_feat_tensors.append(tensor)
#                     else:
#                         image_feat_tensors.append(img_f)
                        
#                 loss_image = self.cnn_head(image_feat_tensors, targets=targets, training=True)
                
#                 if isinstance(loss_image, dict):
#                     for k, v in loss_image.items():
#                         if torch.is_tensor(v):
#                             result[f"cnn_{k}"] = v
#                         else:
#                             device = next(self.parameters()).device
#                             result[f"cnn_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
#                 else:
#                     device = next(self.parameters()).device
#                     result["cnn_total_loss"] = torch.tensor(float(loss_image), device=device, requires_grad=True) if not torch.is_tensor(loss_image) else loss_image

#             if "fusion_total_loss" not in result:
#                 if "fusion_loss" in result:
#                     result["fusion_total_loss"] = result["fusion_loss"]
#                 else:
#                     device = next(self.parameters()).device
#                     result["fusion_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)
                    
#             if self.use_image and "cnn_total_loss" not in result:
#                 if "cnn_loss" in result:
#                     result["cnn_total_loss"] = result["cnn_loss"] 
#                 else:
#                     device = next(self.parameters()).device
#                     result["cnn_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)

#             return result

#         for i, f in enumerate(fused_feat[:1]):  # 只打印第一个特征层
#             print(f"[INFO] Inference Feature {i}: x.shape = {f.x.shape}, h={getattr(f, 'height', 'N/A')}, w={getattr(f, 'width', 'N/A')}")

#         x.reset = reset
        
#         self._initialize_head_if_needed(fused_feat)
        
#         try:
#             fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            
#             valid_features = []
#             for i, feat in enumerate(fused_feat_x):
#                 if feat.numel() > 0 and all(d > 0 for d in feat.shape):
#                     valid_features.append(feat)
#                 else:
#                     print(f"[WARNING] Inference feature {i} is invalid: {feat.shape}, numel={feat.numel()}")
            
#             if len(valid_features) == 0:
#                 print(f"[ERROR] No valid features after unpacking in inference!")
#                 ret = []
#                 if return_targets and hasattr(x, 'bbox'):
#                     targets = convert_to_evaluation_format(x)
#                     ret = [ret, targets]
#                 return ret
            
#             outputs = self.head(valid_features, training=False)
            
#         except Exception as e:
#             print(f"[ERROR] Failed to process features in inference: {e}")
#             import traceback
#             traceback.print_exc()
            
#             ret = []
#             if return_targets and hasattr(x, 'bbox'):
#                 targets = convert_to_evaluation_format(x)
#                 ret = [ret, targets]
#             return ret
            
#         if filtering:
#             try:
#                 if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
#                     batch_detections = analyze_and_fix_postprocess(outputs, self.conf_threshold, self.nms_threshold, debug_mode=True)
#                     outputs = []
#                     for batch_det in batch_detections:
#                         outputs.extend(batch_det)
                    
#                     print(f"[INFO] Final detection count: {len(outputs)}")
                    
#                 else:
#                     outputs = []
                    
#             except Exception as e:
#                 print(f"[ERROR] Failed to postprocess outputs: {e}")
#                 outputs = []
#         else:
#             if not isinstance(outputs, torch.Tensor):
#                 print(f"[WARNING] Expected tensor output, got {type(outputs)}")
#                 outputs = torch.zeros(1, 1, 5 + self.backbone.num_classes, device=next(self.parameters()).device)
        
#         ret = outputs

#         if return_targets and hasattr(x, 'bbox'):
#             targets = convert_to_evaluation_format(x)
#             ret = [outputs, targets]

#         return ret
