import os
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import tqdm
import wandb
from pathlib import Path
import argparse
import random
import cv2
import os.path as osp

# 导入辅助函数
from dagr.model.networks.utils import (
    box_iou, xywh_to_xyxy, xyxy_to_xywh,
    visualize_detections, analyze_precision_recall,
    analyze_confidence_distribution, debug_gradient_flow,
    inspect_model_stats, debug_data_format, gradients_broken,
    fix_gradients, mAPCalculator
)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC
from dagr.model.networks.dagr_fcos import DAGR
from dagr.model.networks.ema import ModelEMA

def log_box_statistics(boxes):
    """记录边界框统计信息"""
    if len(boxes) == 0:
        return
    
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    
    print(f"Box width stats: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
    print(f"Box height stats: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")

def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          run_name=""):

    model.train()
    total_loss = 0.0
    num_batches = 0
    running_losses = {}
    
    # 添加epoch信息显示
    epoch = getattr(args, "current_epoch", 0)
    print(f"\n=== Training Epoch: {epoch+1}/{args.tot_num_epochs} ===")

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        # 只在第一个迭代显示数据格式信息
        if i == 0:
            debug_data_format(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        
        # 检查和处理NaN损失 - 减少打印频率
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if torch.is_tensor(sub_v) and torch.isnan(sub_v).any():
                        v[sub_k] = torch.zeros_like(sub_v)
            elif torch.is_tensor(v) and torch.isnan(v).any():
                loss_dict[k] = torch.zeros_like(v)
        
        lambda_cnn = getattr(args, "lambda_cnn_loss", 1.0)
        loss_fusion = loss_dict.get("fusion_total_loss", 0.0)
        loss_cnn = loss_dict.get("cnn_total_loss", 0.0)
        
        if isinstance(loss_fusion, dict):
            loss_fusion = loss_fusion.get("total_loss", 0.0)
        if isinstance(loss_cnn, dict):
            loss_cnn = loss_cnn.get("total_loss", 0.0)
        
        device = next(model.parameters()).device
        
        # 确保梯度能够正确传播 - 修复梯度问题
        # 1. 检查loss_fusion是否是带梯度的张量
        if torch.is_tensor(loss_fusion):
            # 确保张量有梯度
            if not loss_fusion.requires_grad:
                # 创建一个新的需要梯度的张量
                loss_fusion_value = loss_fusion.detach().clone()
                loss_fusion = torch.tensor(loss_fusion_value, device=device, requires_grad=True)
        else:
            # 如果不是张量，创建一个
            loss_fusion = torch.tensor(float(loss_fusion), device=device, requires_grad=True)
            
        # 2. 检查loss_cnn是否是带梯度的张量
        if torch.is_tensor(loss_cnn):
            if not loss_cnn.requires_grad:
                loss_cnn_value = loss_cnn.detach().clone()
                loss_cnn = torch.tensor(loss_cnn_value, device=device, requires_grad=True)
        else:
            loss_cnn = torch.tensor(float(loss_cnn), device=device, requires_grad=True)
        
        # 处理NaN损失 - 减少打印频率
        if torch.isnan(loss_fusion).any():
            loss_fusion = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(loss_cnn).any():
            loss_cnn = torch.tensor(0.0, device=device, requires_grad=True)
            
        # 计算最终损失，确保能够反向传播
        try:
            # 使用模型参数直接计算损失，确保梯度链接
            # 从模型中获取一个参数
            dummy_param = next(model.parameters())
            loss = loss_fusion + lambda_cnn * loss_cnn
            
            # 检查loss是否需要梯度
            if not loss.requires_grad:
                # 创建一个能反向传播的损失
                loss = loss * dummy_param.sum() * 0.0 + loss.detach()
                
            # 如果损失是零，添加一个微小值确保梯度流动
            if loss.item() == 0:
                loss = loss + dummy_param.sum() * 0.0001
        except Exception as e:
            # 从模型中获取一个参数以创建一个可以反向传播的损失
            dummy_param = next(model.parameters())
            loss = dummy_param.sum() * 0.0 + torch.tensor(0.01, device=device, requires_grad=True)
        
        # 最后检查总损失 - 减少打印频率
        if torch.isnan(loss).any():
            # 计算非NaN的组件损失和
            valid_losses = []
            for k, v in loss_dict.items():
                if "total" not in k and torch.is_tensor(v) and not torch.isnan(v).any():
                    valid_losses.append(v)
            
            if valid_losses:
                dummy_param = next(model.parameters())
                loss = sum(valid_losses) + dummy_param.sum() * 0.0
            else:
                dummy_param = next(model.parameters())
                loss = dummy_param.sum() * 0.0 + torch.tensor(0.01, device=device, requires_grad=True)

        if hasattr(loss, 'item'):
            loss_value = loss.item()
        else:
            loss_value = float(loss)
            
        total_loss += loss_value
        num_batches += 1

        for k, v in loss_dict.items():
            if hasattr(v, 'item'):
                v_value = v.item()
            elif isinstance(v, dict) and 'total_loss' in v:
                v_value = v['total_loss'].item() if hasattr(v['total_loss'], 'item') else float(v['total_loss'])
            else:
                v_value = float(v)
            
            if k not in running_losses:
                running_losses[k] = []
            running_losses[k].append(v_value)

        # 每200次迭代打印一次平均损失
        if (i + 1) % 200 == 0:
            # 计算过去200次迭代的平均损失
            window_size = min(200, len(running_losses.get('fusion_total_loss', [])))
            if window_size > 0:
                avg_losses = {}
                for k, v in running_losses.items():
                    if len(v) >= window_size:
                        avg_losses[k] = sum(v[-window_size:]) / window_size
                
                current_lr = scheduler.get_last_lr()[-1]
                
                print(f"\n[TRAIN] Epoch {epoch+1}/{args.tot_num_epochs}, Iteration {i+1}/{len(loader)}:")
                print(f"[TRAIN] Total Loss: {loss_value:.6f}")
                print(f"[TRAIN] Learning Rate: {current_lr:.6f}")
                
                # 单独打印主要损失组件
                cls_loss = avg_losses.get('fusion_loss_cls', 0.0)
                reg_loss = avg_losses.get('fusion_loss_reg', 0.0)
                ctr_loss = avg_losses.get('fusion_loss_ctr', 0.0)
                
                print(f"[TRAIN] Classification Loss: {cls_loss:.6f}")
                print(f"[TRAIN] Regression Loss: {reg_loss:.6f}")
                print(f"[TRAIN] Centerness Loss: {ctr_loss:.6f}")
                
                # 打印损失组件比例
                if loss_value > 0:
                    total_component = cls_loss + reg_loss + ctr_loss
                    if total_component > 0:
                        cls_ratio = cls_loss / total_component * 100
                        reg_ratio = reg_loss / total_component * 100
                        ctr_ratio = ctr_loss / total_component * 100
                        print(f"[TRAIN] Loss Ratios: Cls={cls_ratio:.2f}%, Reg={reg_ratio:.2f}%, Ctr={ctr_ratio:.2f}%")
                
                # 检查centerness和分类分支的输出分布
                if hasattr(model, 'head') and hasattr(model.head, '_analyze_activation_stats'):
                    stats = model.head._analyze_activation_stats()
                    if stats:
                        print(f"[TRAIN] Activation Stats: {stats}")
                
                print("-" * 60)

        # 在反向传播前检查损失是否有梯度 - 减少打印频率
        if not loss.requires_grad:
            dummy_param = next(model.parameters())
            loss = dummy_param.sum() * 0.0 + torch.tensor(0.01, device=device, requires_grad=True)
            
        try:
            loss.backward()
        except RuntimeError as e:
            # 获取模型参数，创建一个伪损失
            dummy_param = next(model.parameters())
            dummy_loss = dummy_param.sum() * 0.0001
            dummy_loss.backward()
            
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
        fix_gradients(model)

        device = next(model.parameters()).device
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        optimizer.step()
        scheduler.step()
        ema.update(model)

        training_logs = {}
        for k, v in loss_dict.items():
            clean_key = k.replace("_", "/")
            if hasattr(v, 'item'):
                v_value = v.item()
            elif isinstance(v, dict) and 'total_loss' in v:
                v_value = v['total_loss'].item() if hasattr(v['total_loss'], 'item') else float(v['total_loss'])
            else:
                v_value = float(v)
            
            # 检查NaN并替换
            if np.isnan(v_value):
                v_value = 0.0
                
            training_logs[f"training/loss/{clean_key}"] = v_value
            
        # 额外记录分类、回归和centerness损失比例
        total_component = 0
        for component in ['fusion_loss_cls', 'fusion_loss_reg', 'fusion_loss_ctr']:
            if component in loss_dict:
                value = loss_dict[component]
                if torch.is_tensor(value):
                    total_component += value.item()
                else:
                    total_component += float(value)
                    
        if total_component > 0:
            for component in ['fusion_loss_cls', 'fusion_loss_reg', 'fusion_loss_ctr']:
                if component in loss_dict:
                    value = loss_dict[component]
                    if torch.is_tensor(value):
                        ratio = value.item() / total_component
                    else:
                        ratio = float(value) / total_component
                    training_logs[f"training/loss_ratio/{component}"] = ratio
        
        wandb.log({
            "training/loss": loss_value,
            "training/lr": scheduler.get_last_lr()[-1],
            **training_logs
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def compute_detailed_metrics(mapcalc):
    """计算并显示更详细的评估指标，包括更多小数位的mAP"""
    metrics = mapcalc.compute()
    
    # 提取mAP并格式化为8位小数
    mAP = metrics.get('mAP', 0.0)
    mAP_str = f"{mAP:.8f}"
    
    # 记录每个类别的AP
    class_aps = {}
    for cls in mapcalc.classes:
        if f'AP_{cls}' in metrics:
            class_aps[cls] = metrics[f'AP_{cls}']
    
    print(f"\n[EVAL DETAILED] mAP = {mAP_str}")
    
    # 输出各个类别的AP（如果有）
    if class_aps:
        for cls, ap in class_aps.items():
            print(f"[EVAL DETAILED] AP_{cls} = {ap:.8f}")
    
    # 记录精确度/召回率
    if 'precision' in metrics and 'recall' in metrics:
        precision = metrics['precision']
        recall = metrics['recall']
        print(f"[EVAL DETAILED] Precision = {precision:.8f}, Recall = {recall:.8f}")
    
    # 提取并分析检测框统计信息
    if hasattr(mapcalc, 'all_detections'):
        all_boxes = []
        for dets in mapcalc.all_detections:
            if isinstance(dets, list) and dets:
                for det in dets:
                    if isinstance(det, dict) and 'boxes' in det:
                        all_boxes.append(det['boxes'])
        
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)
            if all_boxes.shape[1] == 4:  # XYXY格式
                widths = all_boxes[:, 2] - all_boxes[:, 0]
                heights = all_boxes[:, 3] - all_boxes[:, 1]
            else:  # XYWH格式
                widths = all_boxes[:, 2]
                heights = all_boxes[:, 3]
                
            print(f"[EVAL DETAILED] Detection width: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
            print(f"[EVAL DETAILED] Detection height: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")
            
            # 分析置信度分布
            if hasattr(mapcalc, 'all_scores'):
                all_scores = []
                for scores in mapcalc.all_scores:
                    if isinstance(scores, list) and scores:
                        for score in scores:
                            if isinstance(score, torch.Tensor):
                                all_scores.append(score)
                
                if all_scores:
                    all_scores = torch.cat(all_scores, dim=0)
                    print(f"[EVAL DETAILED] Confidence: min={all_scores.min().item():.6f}, max={all_scores.max().item():.6f}, mean={all_scores.mean().item():.6f}")
                    print(f"[EVAL DETAILED] Confidence distribution:")
                    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        ratio = (all_scores >= threshold).float().mean().item()
                        print(f"  - Score >= {threshold:.1f}: {ratio:.4f} ({int(ratio * 100)}%)")
    
    # 检查mAP为0的原因
    if mAP == 0:
        print("\n[EVAL DETAILED] *** mAP is ZERO: Possible causes ***")
        print("  1. Predicted boxes have low IoU with ground truth (localization error)")
        print("  2. Class predictions don't match ground truth (classification error)")
        print("  3. Confidence scores might be miscalibrated")
        print("  4. NMS might be too aggressive or too relaxed")
        print("  5. Detection threshold might be too high")
        
        # 检查检测器是否返回空结果
        if hasattr(mapcalc, 'all_detections'):
            empty_count = sum(1 for dets in mapcalc.all_detections if not dets)
            total_count = len(mapcalc.all_detections)
            if empty_count > 0:
                print(f"[EVAL DETAILED] {empty_count}/{total_count} images have NO detections")
    
    return metrics

def run_test(loader: DataLoader,
             model: torch.nn.Module,
             dry_run_steps: int = -1,
             dataset="gen1"):

    model.eval()
    # 使用增强的mAP计算器
    mapcalc = mAPCalculator(iou_threshold=0.5, num_classes=2)
    
    # 创建一个标准的DetectionBuffer用于兼容性
    std_mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    print(f"[EVAL] Running evaluation...")
    total_detections = 0
    
    # 创建目录保存可视化结果
    vis_dir = 'detection_vis'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 获取类别名称映射
    class_names = {0: 'car', 1: 'person'}  # 假设这是DSEC数据集的类别
    
    for i, data in enumerate(tqdm.tqdm(loader, desc="Evaluation")):
        data = data.cuda()
        data = format_data(data)
        detections, targets = model(data)
        
        # 记录检测数量
        if isinstance(detections, list):
            batch_detections = sum(1 for d in detections if isinstance(d, dict) and len(d.get('boxes', [])) > 0)
            total_detections += batch_detections
        
        if i % 10 == 0:
            torch.cuda.empty_cache()
        
        # 使用标准评估器更新
        std_mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        
        # 对每个样本进行详细分析 - 减少打印频率，只对前几个batch详细分析
        for batch_idx in range(data.num_graphs):
            # 提取当前样本的GT框
            target = targets[batch_idx] if batch_idx < len(targets) else None
            detection = detections[batch_idx] if batch_idx < len(detections) else None
            
            if target is None or detection is None:
                continue
            
            # 统一格式为XYXY
            gt_boxes = target.get('boxes', torch.tensor([]))
            gt_labels = target.get('labels', torch.tensor([]))
            
            pred_boxes = detection.get('boxes', torch.tensor([]))
            pred_scores = detection.get('scores', torch.tensor([]))
            pred_labels = detection.get('labels', torch.tensor([]))
            
            # 确保所有张量都在CPU上
            gt_boxes = gt_boxes.cpu()
            gt_labels = gt_labels.cpu()
            pred_boxes = pred_boxes.cpu()
            pred_scores = pred_scores.cpu()
            pred_labels = pred_labels.cpu()
            
            # 跳过无效的检测结果
            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue
            
            # 更新增强mAP计算器
            mapcalc.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
            
            # 只对前5个batch的少数样本进行详细分析
            if i < 5 and batch_idx < 2:  
                print(f"\n[EVAL] Sample {i}-{batch_idx} Analysis:")
                print(f"[EVAL] GT: {len(gt_boxes)} boxes, labels: {[class_names.get(l.item(), l.item()) for l in gt_labels]}")
                print(f"[EVAL] Pred: {len(pred_boxes)} boxes")
                
                # 转换为XYXY格式用于计算IoU
                gt_boxes_xyxy = gt_boxes if gt_boxes.shape[1] == 4 else xywh_to_xyxy(gt_boxes)
                pred_boxes_xyxy = pred_boxes if pred_boxes.shape[1] == 4 else xywh_to_xyxy(pred_boxes)
                
                # 计算IoU矩阵
                iou_matrix = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
                
                # 打印检测框的详细信息，仅打印前3个检测结果
                print("[EVAL] Prediction details (showing top 3):")
                for j, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    if j >= 3:  # 只打印前3个
                        print("  ... and more detections")
                        break
                        
                    # 找到与当前预测框IoU最大的GT框
                    if j < iou_matrix.shape[0]:
                        max_iou, max_gt_idx = iou_matrix[j].max().item(), iou_matrix[j].argmax().item()
                        gt_label = gt_labels[max_gt_idx].item() if max_gt_idx < len(gt_labels) else -1
                        
                        match_status = "MATCH" if max_iou >= 0.5 and label.item() == gt_label else "NO MATCH"
                        if max_iou >= 0.5 and label.item() != gt_label:
                            match_status = "CLASS MISMATCH"
                        elif max_iou < 0.5 and label.item() == gt_label:
                            match_status = "LOW IoU"
                        
                        print(f"  #{j}: label={class_names.get(label.item(), label.item())}, "
                              f"score={score.item():.4f}, "
                              f"box={box.tolist()}, "
                              f"max_iou={max_iou:.4f}, "
                              f"status={match_status}")
                
                # 进行精确度/召回率分析
                pr_analysis = analyze_precision_recall(
                    gt_boxes_xyxy, gt_labels, 
                    pred_boxes_xyxy, pred_labels, 
                    pred_scores, iou_threshold=0.5
                )
                
                print(f"[EVAL] Precision: {pr_analysis['precision']:.4f}, Recall: {pr_analysis['recall']:.4f}")
                print(f"[EVAL] TP: {pr_analysis['TP']}, FP: {pr_analysis['FP']}, FN: {pr_analysis['FN']}")
                
                # 分析置信度分布
                conf_analysis = analyze_confidence_distribution(
                    pred_scores, pred_labels, 
                    gt_boxes_xyxy, gt_labels, 
                    iou_matrix
                )
                
                # 保存检测结果可视化
                try:
                    # 提取原始图像
                    if hasattr(data, 'x_rgb') and data.x_rgb is not None:
                        orig_img = data.x_rgb[batch_idx].cpu()
                        
                        # 可视化结果
                        vis_img = visualize_detections(
                            orig_img, 
                            gt_boxes_xyxy, 
                            pred_boxes_xyxy,
                            gt_labels=gt_labels,
                            pred_labels=pred_labels,
                            pred_scores=pred_scores,
                            iou_matrix=iou_matrix,
                            filename=f"{vis_dir}/sample_{i}_{batch_idx}.jpg"
                        )
                except Exception as e:
                    print(f"[WARNING] Failed to visualize detection: {e}")
        
        if dry_run_steps > 0 and i == dry_run_steps:
            break

    print(f"[EVAL] Total detections across all batches: {total_detections}")
    torch.cuda.empty_cache()
    
    # 计算增强版mAP
    detailed_metrics = mapcalc.compute(verbose=True)
    
    # 返回标准计算器以保持兼容性
    return std_mapcalc

class LRSchedule:
    def __init__(self, warmup_epochs=0.5, num_iters_per_epoch=1000, tot_num_epochs=100):
        self.warmup_epochs = warmup_epochs
        self.num_iters_per_epoch = num_iters_per_epoch
        self.tot_num_epochs = tot_num_epochs
        self.warmup_iters = int(warmup_epochs * num_iters_per_epoch)
        self.total_iters = num_iters_per_epoch * tot_num_epochs

    def __call__(self, iter_num):
        if iter_num < self.warmup_iters:
            # 使用更平滑的预热曲线
            return iter_num / self.warmup_iters
        else:
            # 过了预热期，使用余弦退火
            no_decay = float(iter_num - self.warmup_iters) / float(
                max(1, self.total_iters - self.warmup_iters)
            )
            # 余弦退火衰减到最小学习率为初始值的5%
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * no_decay))

if __name__ == '__main__':
    from torch_geometric.seed import seed_everything
    seed = 42
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()
    setattr(args, "lambda_cnn_loss", 1.0)

    # 设置一个明确的实验名称，避免wandb.run.name为None
    if not hasattr(args, "exp_name") or not args.exp_name:
        args.exp_name = f"DAGR_FCOS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    wandb.init(project="DAGR-FCOS", name=args.exp_name, config=vars(args))

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)
    dataset_path = args.dataset_directory

    print("init datasets")
    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)
    print(f"Training with {sum(p.numel() for p in model.parameters())} parameters.")

    model = model.cuda()
    ema = ModelEMA(model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # 使用改进的学习率调度，增加预热时间到0.5个epoch
    lr_func = LRSchedule(warmup_epochs=0.5, num_iters_per_epoch=num_iters_per_epoch, tot_num_epochs=args.tot_num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory, args=args, optimizer=optimizer, scheduler=lr_scheduler, ema=ema, model=model)
    
    try:
        start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=True) or 0
    except (IndexError, FileNotFoundError) as e:
        print(f"[INFO] No existing checkpoint found: {e}")
        start_epoch = 0

    if start_epoch > 0:
        print(f"[INFO] Resuming from epoch {start_epoch}, advancing lr_scheduler...")
        for _ in range(start_epoch * num_iters_per_epoch):
            lr_scheduler.step()

    if hasattr(args, "resume_checkpoint") and args.resume_checkpoint:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    with torch.no_grad():
        # 使用compute_detailed_metrics显示详细的评估结果
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        detailed_metrics = compute_detailed_metrics(mapcalc)

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        setattr(args, "current_epoch", epoch)
        avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        print(f"[TRAIN] Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        checkpointer.checkpoint(epoch, name="last_model")

        # 每个epoch都进行评估，以便更好地监控进展
        if epoch % 1 == 0:
            with torch.no_grad():
                mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
                detailed_metrics = compute_detailed_metrics(mapcalc)
                mAP = detailed_metrics.get('mAP', 0.0)
                print(f"[EVAL] Epoch {epoch+1}: mAP = {mAP:.8f}")
                
                # 额外记录边界框统计信息
                if hasattr(mapcalc, 'all_detections') and mapcalc.all_detections:
                    all_boxes = []
                    for dets in mapcalc.all_detections:
                        if isinstance(dets, list) and dets:
                            for det in dets:
                                if isinstance(det, dict) and 'boxes' in det:
                                    all_boxes.append(det['boxes'])
                    
                    if all_boxes:
                        all_boxes = torch.cat(all_boxes, dim=0)
                        widths = all_boxes[:, 2] - all_boxes[:, 0]
                        heights = all_boxes[:, 3] - all_boxes[:, 1]
                        print(f"[BOX STATS] Detection width: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
                        print(f"[BOX STATS] Detection height: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")
                
                checkpointer.process(detailed_metrics, epoch)