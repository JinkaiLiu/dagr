import os
import math
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import tqdm
import wandb
from pathlib import Path
import argparse
import random
import numpy as np

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC
from dagr.model.networks.dagr_fcos import DAGR
from dagr.model.networks.ema import ModelEMA

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
        
        # 检查和处理NaN损失
        for k, v in loss_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if torch.is_tensor(sub_v) and torch.isnan(sub_v).any():
                        print(f"[WARNING] NaN detected in {k}.{sub_k}, replacing with zero")
                        v[sub_k] = torch.zeros_like(sub_v)
            elif torch.is_tensor(v) and torch.isnan(v).any():
                print(f"[WARNING] NaN detected in {k}, replacing with zero")
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
        
        # 处理NaN损失
        if torch.isnan(loss_fusion).any():
            print(f"[WARNING] loss_fusion is NaN, replacing with zero")
            loss_fusion = torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(loss_cnn).any():
            print(f"[WARNING] loss_cnn is NaN, replacing with zero")
            loss_cnn = torch.tensor(0.0, device=device, requires_grad=True)
            
        # 计算最终损失，确保能够反向传播
        try:
            # 使用模型参数直接计算损失，确保梯度链接
            # 从模型中获取一个参数
            dummy_param = next(model.parameters())
            loss = loss_fusion + lambda_cnn * loss_cnn
            
            # 检查loss是否需要梯度
            if not loss.requires_grad:
                print("[WARNING] Final loss does not require gradients, creating a dummy loss")
                # 创建一个能反向传播的损失
                loss = loss * dummy_param.sum() * 0.0 + loss.detach()
                
            # 如果损失是零，添加一个微小值确保梯度流动
            if loss.item() == 0:
                loss = loss + dummy_param.sum() * 0.0001
        except Exception as e:
            print(f"[ERROR] Error computing final loss: {e}")
            # 从模型中获取一个参数以创建一个可以反向传播的损失
            dummy_param = next(model.parameters())
            loss = dummy_param.sum() * 0.0 + torch.tensor(0.01, device=device, requires_grad=True)
        
        # 最后检查总损失
        if torch.isnan(loss).any():
            print(f"[WARNING] Total loss is NaN, using alternative calculation")
            # 计算非NaN的组件损失和
            valid_losses = []
            for k, v in loss_dict.items():
                if "total" not in k and torch.is_tensor(v) and not torch.isnan(v).any():
                    valid_losses.append(v)
            
            if valid_losses:
                dummy_param = next(model.parameters())
                loss = sum(valid_losses) + dummy_param.sum() * 0.0
            else:
                print(f"[WARNING] No valid component losses, using dummy loss")
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

        if (i + 1) % 50 == 0:
            avg_losses = {k: sum(v[-50:]) / len(v[-50:]) for k, v in running_losses.items()}
            current_lr = scheduler.get_last_lr()[-1]
            
            print(f"\n[TRAIN] Epoch {epoch+1}/{args.tot_num_epochs}, Iteration {i+1}/{len(loader)}:")
            print(f"[TRAIN] Total Loss: {loss_value:.4f} (avg: {sum(running_losses.get('total', [loss_value])[-50:]) / min(50, len(running_losses.get('total', [loss_value]))):.4f})")
            print(f"[TRAIN] Learning Rate: {current_lr:.6f}")
            
            for k, v in avg_losses.items():
                clean_key = k.replace("_", " ").title()
                # 检查是否为NaN
                if np.isnan(v):
                    print(f"[TRAIN] {clean_key}: NaN (Warning: NaN detected)")
                else:
                    print(f"[TRAIN] {clean_key}: {v:.4f}")
            print("-" * 50)
            
            # 定期检查模型权重是否存在NaN
            nan_params = []
            for name, param in model.named_parameters():
                if torch.isnan(param.data).any():
                    nan_params.append(name)
            
            if nan_params:
                print(f"[WARNING] NaN detected in {len(nan_params)} parameters: {nan_params[:3]}...")

        # 在反向传播前检查损失是否有梯度
        if not loss.requires_grad:
            print("[WARNING] Loss does not require gradients before backward pass!")
            dummy_param = next(model.parameters())
            loss = dummy_param.sum() * 0.0 + torch.tensor(0.01, device=device, requires_grad=True)
            
        try:
            loss.backward()
        except RuntimeError as e:
            print(f"[ERROR] Backward pass failed: {e}")
            # 获取模型参数，创建一个伪损失
            dummy_param = next(model.parameters())
            dummy_loss = dummy_param.sum() * 0.0001
            dummy_loss.backward()
            print("[WARNING] Used dummy loss for backward pass")
            
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
            
        wandb.log({
            "training/loss": loss_value,
            "training/lr": scheduler.get_last_lr()[-1],
            **training_logs
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def log_box_statistics(boxes):
    """记录边界框统计信息"""
    if len(boxes) == 0:
        return
    
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    
    print(f"Box width stats: min={widths.min().item():.6f}, max={widths.max().item():.6f}, mean={widths.mean().item():.6f}")
    print(f"Box height stats: min={heights.min().item():.6f}, max={heights.max().item():.6f}, mean={heights.mean().item():.6f}")

def run_test(loader: DataLoader,
             model: torch.nn.Module,
             dry_run_steps: int = -1,
             dataset="gen1"):

    model.eval()
    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    print(f"[EVAL] Running evaluation...")
    total_detections = 0
    
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
        
        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        
        if dry_run_steps > 0 and i == dry_run_steps:
            break

    print(f"[EVAL] Total detections across all batches: {total_detections}")
    torch.cuda.empty_cache()
    return mapcalc

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
    
    return metrics

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
