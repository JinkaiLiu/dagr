import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import os.path as osp
from pathlib import Path

# IoU计算辅助函数
def box_iou(boxes1, boxes2):
    """
    计算两组边界框之间的IoU
    boxes1, boxes2: [N, 4], [M, 4], 格式为 (x1, y1, x2, y2)
    返回: [N, M] IoU矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union.clamp(min=1e-10)
    return iou

def xywh_to_xyxy(boxes):
    """将(x,y,w,h)格式转换为(x1,y1,x2,y2)格式"""
    if boxes.size(0) == 0:
        return boxes
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def xyxy_to_xywh(boxes):
    """将(x1,y1,x2,y2)格式转换为(x,y,w,h)格式"""
    if boxes.size(0) == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([x, y, w, h], dim=1)

# 可视化函数
def visualize_detections(image, gt_boxes, pred_boxes, gt_labels=None, pred_labels=None, 
                         pred_scores=None, iou_matrix=None, filename='detection_vis.jpg'):
    """
    可视化检测结果和GT框
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 创建一个空白图像
    if image.shape[0] == 3:  # CHW -> HWC
        image = image.transpose(1, 2, 0)
    
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]
    
    # 绘制GT框
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = [int(val * width) if j % 2 == 0 else int(val * height) 
                          for j, val in enumerate(box)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色表示GT
        
        if gt_labels is not None:
            label = f"GT:{gt_labels[i]}"
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制预测框
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = [int(val * width) if j % 2 == 0 else int(val * height) 
                          for j, val in enumerate(box)]
        
        # 找到与当前预测框IoU最大的GT框
        max_iou = 0
        max_gt_idx = -1
        if iou_matrix is not None and i < iou_matrix.shape[0]:
            max_iou, max_gt_idx = iou_matrix[i].max().item(), iou_matrix[i].argmax().item()
        
        # 根据IoU值调整颜色
        if max_iou > 0.5:
            color = (0, 0, 255)  # 红色表示高IoU匹配
        else:
            color = (255, 0, 0)  # 蓝色表示低IoU匹配
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label_text = ""
        if pred_labels is not None:
            label_text += f"Pred:{pred_labels[i]} "
        if pred_scores is not None:
            label_text += f"Conf:{pred_scores[i]:.2f} "
        if max_iou > 0:
            label_text += f"IoU:{max_iou:.2f}"
            
        cv2.putText(image, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(filename, image)
    print(f"[VIS] Detection visualization saved to {filename}")
    return image

def analyze_precision_recall(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold=0.5):
    """详细分析精度和召回率"""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return {
            'TP': 0, 'FP': len(pred_boxes), 'FN': len(gt_boxes),
            'precision': 0.0, 'recall': 0.0, 'details': []
        }
    
    # 计算IoU矩阵
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    
    # 对每个预测框，找到最匹配的GT框
    max_ious, matched_gt_indices = iou_matrix.max(dim=1)
    
    # 标记每个GT框是否被匹配
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
    
    # 按照置信度排序处理预测框
    sorted_indices = torch.argsort(pred_scores, descending=True)
    
    true_positives = 0
    false_positives = 0
    details = []
    
    for idx in sorted_indices:
        pred_label = pred_labels[idx]
        gt_idx = matched_gt_indices[idx]
        iou = max_ious[idx]
        
        # 检查类别匹配和IoU阈值
        if iou >= iou_threshold and pred_label == gt_labels[gt_idx] and not gt_matched[gt_idx]:
            true_positives += 1
            gt_matched[gt_idx] = True
            match_type = "TP"
        else:
            false_positives += 1
            if iou >= iou_threshold:
                if pred_label != gt_labels[gt_idx]:
                    match_type = "FP (Wrong Class)"
                else:
                    match_type = "FP (Already Matched)"
            else:
                match_type = "FP (Low IoU)"
                
        details.append({
            'pred_idx': idx.item(),
            'pred_score': pred_scores[idx].item(),
            'pred_label': pred_label.item(),
            'best_gt_idx': gt_idx.item(),
            'iou': iou.item(),
            'type': match_type
        })
    
    false_negatives = (gt_matched == False).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    result = {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'precision': precision,
        'recall': recall,
        'details': details,
        'iou_matrix': iou_matrix
    }
    
    return result

def analyze_confidence_distribution(pred_scores, pred_labels, gt_boxes, gt_labels, iou_matrix):
    """分析置信度分布与真实性能的关系"""
    if len(pred_scores) == 0:
        return {}
    
    # 按照置信度阈值统计TP/FP
    thresholds = torch.linspace(0, 1, 20)
    stats = []
    
    # 如果有GT框，确定每个预测是TP还是FP
    if len(gt_boxes) > 0:
        # 获取每个预测框最匹配的GT框
        max_ious, matched_gt_indices = iou_matrix.max(dim=1)
        
        for threshold in thresholds:
            # 筛选高于阈值的预测
            mask = pred_scores >= threshold
            if mask.sum() == 0:
                stats.append({'threshold': threshold.item(), 'tp': 0, 'fp': 0, 'precision': 0})
                continue
                
            filtered_ious = max_ious[mask]
            filtered_pred_labels = pred_labels[mask]
            filtered_gt_indices = matched_gt_indices[mask]
            
            # 计算TP和FP
            tp = ((filtered_ious >= 0.5) & 
                  (filtered_pred_labels == gt_labels[filtered_gt_indices])).sum().item()
            fp = len(filtered_ious) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            stats.append({
                'threshold': threshold.item(),
                'tp': tp,
                'fp': fp,
                'precision': precision
            })
    else:
        # 没有GT框，所有预测都是FP
        for threshold in thresholds:
            fp = (pred_scores >= threshold).sum().item()
            stats.append({
                'threshold': threshold.item(),
                'tp': 0,
                'fp': fp,
                'precision': 0
            })
    
    return stats

def debug_gradient_flow(named_parameters):
    """
    可视化梯度流动情况，帮助调试
    """
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
            else:
                print(f"Parameter {n} has no gradient")
    
    if not ave_grads:
        print("No gradients to analyze")
        return
        
    plt.figure(figsize=(10, 7))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('gradient_flow.png')
    plt.close()
    print("Gradient flow visualization saved to gradient_flow.png")

def inspect_model_stats(model, prefix=''):
    """
    检查模型的权重统计信息，帮助识别问题
    """
    stats = {}
    
    for name, param in model.named_parameters():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if param.requires_grad:
            stats[full_name] = {
                'shape': list(param.shape),
                'mean': float(param.data.mean().item()),
                'std': float(param.data.std().item()),
                'min': float(param.data.min().item()),
                'max': float(param.data.max().item()),
                'has_nan': bool(torch.isnan(param.data).any().item()),
                'has_inf': bool(torch.isinf(param.data).any().item()),
            }
            
            if param.grad is not None:
                stats[full_name].update({
                    'grad_mean': float(param.grad.mean().item()),
                    'grad_std': float(param.grad.std().item()),
                    'grad_min': float(param.grad.min().item()),
                    'grad_max': float(param.grad.max().item()),
                    'grad_has_nan': bool(torch.isnan(param.grad).any().item()),
                    'grad_has_inf': bool(torch.isinf(param.grad).any().item()),
                })
            else:
                stats[full_name].update({
                    'grad': None
                })
    
    # 找出可能有问题的层
    problematic_layers = []
    for name, layer_stats in stats.items():
        if layer_stats.get('has_nan', False) or layer_stats.get('has_inf', False):
            problematic_layers.append((name, 'weights contain NaN/Inf'))
        elif layer_stats.get('grad_has_nan', False) or layer_stats.get('grad_has_inf', False):
            problematic_layers.append((name, 'gradients contain NaN/Inf'))
        elif abs(layer_stats.get('mean', 0)) > 100 or layer_stats.get('std', 0) > 100:
            problematic_layers.append((name, 'unusual weight statistics'))
    
    # 打印问题层
    if problematic_layers:
        print(f"[WARNING] Found {len(problematic_layers)} problematic layers:")
        for name, issue in problematic_layers:
            print(f"  - {name}: {issue}")
            layer_stats = stats[name]
            print(f"    Stats: mean={layer_stats['mean']:.4f}, std={layer_stats['std']:.4f}, min={layer_stats['min']:.4f}, max={layer_stats['max']:.4f}")
            if 'grad_mean' in layer_stats:
                print(f"    Grad stats: mean={layer_stats['grad_mean']:.4f}, std={layer_stats['grad_std']:.4f}, min={layer_stats['grad_min']:.4f}, max={layer_stats['grad_max']:.4f}")
    
    return stats, problematic_layers

def debug_data_format(data, targets=None):
    """调试数据和标签格式"""
    print(f"\n[DATA INFO] Batch info: graphs={data.num_graphs}, features={data.x.shape}")
    
    if hasattr(data, 'bbox'):
        print(f"[DATA INFO] BBox shape: {data.bbox.shape}")
        non_zero_boxes = (data.bbox.sum(dim=1) != 0).sum().item()
        print(f"[DATA INFO] Valid boxes: {non_zero_boxes} / {data.bbox.shape[0]}")
        
        if targets is not None:
            print(f"[DATA INFO] Targets count: {len(targets)}")
            valid_targets = sum(1 for t in targets if t.numel() > 0)
            print(f"[DATA INFO] Valid targets: {valid_targets}")

def gradients_broken(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)

class mAPCalculator:
    """增强的mAP计算器，用于分析mAP为0的原因"""
    def __init__(self, iou_threshold=0.5, num_classes=2):
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.class_names = {0: 'car', 1: 'person'}  # 添加类别名称映射
        self.reset()
        
    def reset(self):
        self.all_gt_boxes = []
        self.all_gt_labels = []
        self.all_pred_boxes = []
        self.all_pred_scores = []
        self.all_pred_labels = []
        self.num_images = 0
        
    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        self.all_gt_boxes.append(gt_boxes)
        self.all_gt_labels.append(gt_labels)
        self.all_pred_boxes.append(pred_boxes)
        self.all_pred_scores.append(pred_scores)
        self.all_pred_labels.append(pred_labels)
        self.num_images += 1
        
    def compute_per_image(self, image_idx, verbose=False):
        """为单张图像计算AP和详细指标"""
        gt_boxes = self.all_gt_boxes[image_idx]
        gt_labels = self.all_gt_labels[image_idx]
        pred_boxes = self.all_pred_boxes[image_idx]
        pred_scores = self.all_pred_scores[image_idx]
        pred_labels = self.all_pred_labels[image_idx]
        
        # 检查是否有预测框
        if len(pred_boxes) == 0:
            if verbose:
                print(f"[mAP] Image {image_idx}: No predictions")
            return {
                'AP': 0.0,
                'TP': 0,
                'FP': 0,
                'FN': len(gt_boxes),
                'details': 'No predictions'
            }
        
        # 检查是否有GT框
        if len(gt_boxes) == 0:
            if verbose:
                print(f"[mAP] Image {image_idx}: No ground truth boxes")
            return {
                'AP': 0.0, 
                'TP': 0,
                'FP': len(pred_boxes),
                'FN': 0,
                'details': 'No ground truth'
            }
        
        # 计算每个类别的AP
        ap_per_class = []
        
        for class_id in range(self.num_classes):
            # 筛选当前类别的GT和预测
            gt_class_mask = gt_labels == class_id
            pred_class_mask = pred_labels == class_id
            
            gt_class_boxes = gt_boxes[gt_class_mask]
            pred_class_boxes = pred_boxes[pred_class_mask]
            pred_class_scores = pred_scores[pred_class_mask]
            
            if len(gt_class_boxes) == 0 and len(pred_class_boxes) == 0:
                # 没有这个类别的GT和预测，跳过
                continue
                
            if len(gt_class_boxes) == 0:
                # 没有这个类别的GT，所有预测都是假阳性
                if verbose:
                    print(f"[mAP] Image {image_idx}, Class {class_id} ({self.class_names.get(class_id, 'unknown')}): No ground truth")
                ap_per_class.append(0.0)
                continue
                
            if len(pred_class_boxes) == 0:
                # 没有这个类别的预测，但有GT
                if verbose:
                    print(f"[mAP] Image {image_idx}, Class {class_id} ({self.class_names.get(class_id, 'unknown')}): No predictions")
                ap_per_class.append(0.0)
                continue
            
            # 计算IoU矩阵
            iou_matrix = box_iou(pred_class_boxes, gt_class_boxes)
            
            # 按置信度排序处理预测
            sorted_indices = torch.argsort(pred_class_scores, descending=True)
            
            # 记录每个GT框是否被检测到
            gt_detected = torch.zeros(len(gt_class_boxes), dtype=torch.bool)
            
            # 记录每个预测是TP还是FP
            true_positives = torch.zeros(len(pred_class_boxes))
            false_positives = torch.zeros(len(pred_class_boxes))
            
            # 详细分析信息
            details = []
            
            for i, pred_idx in enumerate(sorted_indices):
                # 找到与当前预测IoU最大的GT框
                max_iou, max_gt_idx = iou_matrix[pred_idx].max(dim=0)
                
                if max_iou >= self.iou_threshold and not gt_detected[max_gt_idx]:
                    true_positives[i] = 1
                    gt_detected[max_gt_idx] = True
                    match_status = "TP"
                else:
                    false_positives[i] = 1
                    if max_iou >= self.iou_threshold:
                        match_status = "FP (Already Matched)"
                    else:
                        match_status = "FP (Low IoU)"
                
                details.append({
                    'pred_idx': pred_idx.item(),
                    'pred_score': pred_class_scores[pred_idx].item(),
                    'max_iou': max_iou.item(),
                    'best_gt_idx': max_gt_idx.item(),
                    'status': match_status
                })
            
            # 累积TP和FP
            cum_tp = torch.cumsum(true_positives, dim=0)
            cum_fp = torch.cumsum(false_positives, dim=0)
            
            # 计算精度和召回率
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / len(gt_class_boxes)
            
            # 11点插值计算AP
            ap = 0.0
            for t in torch.linspace(0, 1, 11):
                if not (recall >= t).any():
                    continue
                ap += torch.max(precision[recall >= t]).item() / 11
            
            ap_per_class.append(ap)
            
            if verbose:
                tp = true_positives.sum().item()
                fp = false_positives.sum().item()
                fn = (gt_detected == False).sum().item()
                
                print(f"[mAP] Image {image_idx}, Class {class_id} ({self.class_names.get(class_id, 'unknown')}):")
                print(f"  - AP: {ap:.6f}")
                print(f"  - TP: {tp}, FP: {fp}, FN: {fn}")
                print(f"  - Precision: {precision[-1].item() if len(precision) > 0 else 0:.6f}")
                print(f"  - Recall: {recall[-1].item() if len(recall) > 0 else 0:.6f}")
                
                if len(details) > 0:
                    print("  - Detection Details:")
                    for i, det in enumerate(details[:min(5, len(details))]):
                        print(f"    #{i}: Score={det['pred_score']:.4f}, IoU={det['max_iou']:.4f}, Status={det['status']}")
                    
                    if len(details) > 5:
                        print(f"    ... and {len(details) - 5} more")
        
        # 计算所有类别的平均AP
        mAP = sum(ap_per_class) / len(ap_per_class) if len(ap_per_class) > 0 else 0.0
        
        # 计算整体指标
        pr_analysis = analyze_precision_recall(
            gt_boxes, gt_labels, 
            pred_boxes, pred_labels, 
            pred_scores, iou_threshold=self.iou_threshold
        )
        
        result = {
            'AP': mAP,
            'precision': pr_analysis['precision'],
            'recall': pr_analysis['recall'],
            'TP': pr_analysis['TP'],
            'FP': pr_analysis['FP'],
            'FN': pr_analysis['FN']
        }
        
        return result
        
    def compute(self, verbose=True):
        """计算整体mAP和详细分析信息"""
        if self.num_images == 0:
            return {'mAP': 0.0, 'details': 'No images'}
        
        # 计算每张图像的AP
        ap_per_image = []
        precision_per_image = []
        recall_per_image = []
        
        for i in range(self.num_images):
            result = self.compute_per_image(i, verbose=False)
            ap_per_image.append(result['AP'])
            if 'precision' in result:
                precision_per_image.append(result['precision'])
            if 'recall' in result:
                recall_per_image.append(result['recall'])
        
        # 计算平均AP
        mAP = sum(ap_per_image) / len(ap_per_image)
        
        # 计算平均精确度和召回率
        precision = sum(precision_per_image) / len(precision_per_image) if precision_per_image else 0
        recall = sum(recall_per_image) / len(recall_per_image) if recall_per_image else 0
        
        if verbose:
            print(f"\n[mAP] Overall mAP: {mAP:.6f}")
            print(f"[mAP] Per-image AP: {', '.join([f'{ap:.6f}' for ap in ap_per_image])}")
            print(f"[mAP] Overall Precision: {precision:.6f}")
            print(f"[mAP] Overall Recall: {recall:.6f}")
            
            # 分析mAP为0的情况
            if mAP == 0:
                print("\n[mAP] Analyzing why mAP = 0:")
                zero_ap_count = sum(1 for ap in ap_per_image if ap == 0)
                print(f"[mAP] {zero_ap_count}/{len(ap_per_image)} images have AP = 0")
                
                # 分析每个类别的召回率
                for class_id in range(self.num_classes):
                    class_gt_count = sum(sum(1 for label in gt_labels if label == class_id) for gt_labels in self.all_gt_labels)
                    class_pred_count = sum(sum(1 for label in pred_labels if label == class_id) for pred_labels in self.all_pred_labels)
                    
                    print(f"[mAP] Class {class_id} ({self.class_names.get(class_id, 'unknown')}):")
                    print(f"  - Total GT: {class_gt_count}")
                    print(f"  - Total Predictions: {class_pred_count}")
                    
                    if class_gt_count > 0:
                        # 分析IoU
                        all_ious = []
                        for img_idx in range(self.num_images):
                            gt_boxes = self.all_gt_boxes[img_idx]
                            gt_labels = self.all_gt_labels[img_idx]
                            pred_boxes = self.all_pred_boxes[img_idx]
                            pred_labels = self.all_pred_labels[img_idx]
                            
                            # 筛选当前类别
                            gt_mask = gt_labels == class_id
                            pred_mask = pred_labels == class_id
                            
                            if gt_mask.sum() > 0 and pred_mask.sum() > 0:
                                iou = box_iou(pred_boxes[pred_mask], gt_boxes[gt_mask])
                                all_ious.append(iou.max(dim=0)[0])
                        
                        if all_ious:
                            all_ious = torch.cat(all_ious)
                            print(f"  - IoU stats: min={all_ious.min().item():.4f}, max={all_ious.max().item():.4f}, mean={all_ious.mean().item():.4f}")
                            print(f"  - IoU >= {self.iou_threshold} ratio: {(all_ious >= self.iou_threshold).float().mean().item():.4f}")
                
                # 分析置信度分布
                all_pred_scores = torch.cat([scores for scores in self.all_pred_scores if len(scores) > 0], dim=0) if any(len(scores) > 0 for scores in self.all_pred_scores) else torch.tensor([])
                
                if len(all_pred_scores) > 0:
                    print("\n[mAP] Confidence Score Distribution:")
                    print(f"  - Min: {all_pred_scores.min().item():.4f}")
                    print(f"  - Max: {all_pred_scores.max().item():.4f}")
                    print(f"  - Mean: {all_pred_scores.mean().item():.4f}")
                    print(f"  - Scores > 0.5: {(all_pred_scores > 0.5).float().mean().item():.4f}")
                    print(f"  - Scores > 0.75: {(all_pred_scores > 0.75).float().mean().item():.4f}")
                    
                # 分析bbox尺寸
                all_gt_sizes = []
                all_pred_sizes = []
                
                for img_idx in range(self.num_images):
                    gt_boxes = self.all_gt_boxes[img_idx]
                    pred_boxes = self.all_pred_boxes[img_idx]
                    
                    if len(gt_boxes) > 0:
                        if gt_boxes.shape[1] == 4:  # xyxy格式
                            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
                            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
                        else:  # xywh格式
                            gt_widths = gt_boxes[:, 2]
                            gt_heights = gt_boxes[:, 3]
                        all_gt_sizes.append(torch.stack([gt_widths, gt_heights], dim=1))
                    
                    if len(pred_boxes) > 0:
                        if pred_boxes.shape[1] == 4:  # xyxy格式
                            pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                            pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                        else:  # xywh格式
                            pred_widths = pred_boxes[:, 2]
                            pred_heights = pred_boxes[:, 3]
                        all_pred_sizes.append(torch.stack([pred_widths, pred_heights], dim=1))
                
                if all_gt_sizes:
                    all_gt_sizes = torch.cat(all_gt_sizes, dim=0)
                    print("\n[mAP] GT Box Size Analysis:")
                    print(f"  - Width: min={all_gt_sizes[:, 0].min().item():.6f}, max={all_gt_sizes[:, 0].max().item():.6f}, mean={all_gt_sizes[:, 0].mean().item():.6f}")
                    print(f"  - Height: min={all_gt_sizes[:, 1].min().item():.6f}, max={all_gt_sizes[:, 1].max().item():.6f}, mean={all_gt_sizes[:, 1].mean().item():.6f}")
                
                if all_pred_sizes:
                    all_pred_sizes = torch.cat(all_pred_sizes, dim=0)
                    print("\n[mAP] Predicted Box Size Analysis:")
                    print(f"  - Width: min={all_pred_sizes[:, 0].min().item():.6f}, max={all_pred_sizes[:, 0].max().item():.6f}, mean={all_pred_sizes[:, 0].mean().item():.6f}")
                    print(f"  - Height: min={all_pred_sizes[:, 1].min().item():.6f}, max={all_pred_sizes[:, 1].max().item():.6f}, mean={all_pred_sizes[:, 1].mean().item():.6f}")
                    
                # 提供可能的修复建议
                print("\n[mAP] Possible Issues and Fixes:")
                print("  1. IoU threshold too high - Try lowering the IoU threshold for evaluation")
                print("  2. Poor localization - Check if predicted boxes align with GT boxes")
                print("  3. Confidence miscalibration - Adjust confidence threshold or loss weights")
                print("  4. Class imbalance - Check if certain classes have few training samples")
                print("  5. Scale mismatch - Verify if model can detect objects at the appropriate scales")
        
        # 构建返回结果
        metrics = {
            'mAP': mAP,
            'precision': precision,
            'recall': recall
        }
        
        # 添加每个类别的AP
        for cls in range(self.num_classes):
            cls_ap = []
            for i in range(self.num_images):
                # 筛选当前类别
                gt_labels = self.all_gt_labels[i]
                pred_labels = self.all_pred_labels[i]
                
                gt_mask = gt_labels == cls
                pred_mask = pred_labels == cls
                
                if gt_mask.sum() > 0 or pred_mask.sum() > 0:
                    # 只考虑有当前类别的图像
                    gt_boxes_cls = self.all_gt_boxes[i][gt_mask]
                    gt_labels_cls = gt_labels[gt_mask]
                    pred_boxes_cls = self.all_pred_boxes[i][pred_mask]
                    pred_scores_cls = self.all_pred_scores[i][pred_mask]
                    pred_labels_cls = pred_labels[pred_mask]
                    
                    # 创建一个临时的mAPCalculator计算单类别AP
                    temp_calc = mAPCalculator(iou_threshold=self.iou_threshold, num_classes=1)
                    temp_calc.update(pred_boxes_cls, pred_scores_cls, torch.zeros_like(pred_labels_cls), 
                                    gt_boxes_cls, torch.zeros_like(gt_labels_cls))
                    
                    result = temp_calc.compute_per_image(0, verbose=False)
                    cls_ap.append(result['AP'])
            
            if cls_ap:
                metrics[f'AP_{cls}'] = sum(cls_ap) / len(cls_ap)
        
        return metrics
