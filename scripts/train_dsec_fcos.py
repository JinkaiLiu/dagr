import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

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
from dagr.utils.learning_rate_scheduler import LRSchedule
from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC
from dagr.model.networks.dagr_fcos import DAGR
from dagr.model.networks.ema import ModelEMA

def debug_data_format(data, targets=None):
    """调试数据和标签格式"""
    print(f"\n[DATA DEBUG] ========== Data Format Debug ==========")
    print(f"[DATA DEBUG] Data batch info:")
    print(f"  - data.x.shape: {data.x.shape if hasattr(data, 'x') else 'No x'}")
    print(f"  - data.pos.shape: {data.pos.shape if hasattr(data, 'pos') else 'No pos'}")
    print(f"  - data.batch: {data.batch.shape if hasattr(data, 'batch') else 'No batch'}")
    print(f"  - data.num_graphs: {data.num_graphs if hasattr(data, 'num_graphs') else 'No num_graphs'}")
    
    if hasattr(data, 'bbox'):
        print(f"  - data.bbox.shape: {data.bbox.shape}")
        print(f"  - data.bbox_batch.shape: {data.bbox_batch.shape if hasattr(data, 'bbox_batch') else 'No bbox_batch'}")
        
        bbox_stats = {
            'min': data.bbox.min(0)[0],
            'max': data.bbox.max(0)[0],
            'mean': data.bbox.mean(0),
            'std': data.bbox.std(0)
        }
        print(f"  - BBox statistics:")
        for k, v in bbox_stats.items():
            print(f"    {k}: {v}")
        
        non_zero_boxes = (data.bbox.sum(dim=1) != 0).sum().item()
        print(f"  - Non-zero boxes: {non_zero_boxes} / {data.bbox.shape[0]}")
        
        print(f"  - Sample bboxes:")
        for i in range(min(10, data.bbox.shape[0])):
            print(f"    [{i}]: {data.bbox[i]}")
    
    if targets is not None:
        print(f"[DATA DEBUG] Targets info:")
        for i, target in enumerate(targets):
            if target.numel() > 0:
                print(f"  - Target {i}: shape={target.shape}")
                print(f"    Sample: {target[0] if len(target) > 0 else 'empty'}")
                non_zero = (target[:, 1:].sum(dim=1) != 0).sum().item()
                print(f"    Non-zero: {non_zero} / {target.shape[0]}")
    
    print(f"[DATA DEBUG] =====================================\n")

def gradients_broken(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)

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

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        if i == 0:
            debug_data_format(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        lambda_cnn = getattr(args, "lambda_cnn_loss", 1.0)
        loss_fusion = loss_dict.get("fusion_total_loss", 0.0)
        loss_cnn = loss_dict.get("cnn_total_loss", 0.0)
        
        if isinstance(loss_fusion, dict):
            loss_fusion = loss_fusion.get("total_loss", 0.0)
        if isinstance(loss_cnn, dict):
            loss_cnn = loss_cnn.get("total_loss", 0.0)
        
        device = next(model.parameters()).device
        if not torch.is_tensor(loss_fusion):
            loss_fusion = torch.tensor(float(loss_fusion), device=device, requires_grad=True)
        if not torch.is_tensor(loss_cnn):
            loss_cnn = torch.tensor(float(loss_cnn), device=device, requires_grad=True)
            
        loss = loss_fusion + lambda_cnn * loss_cnn
        
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss), device=device, requires_grad=True)

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
            
            print(f"Iteration {i+1}/{len(loader)}:")
            print(f"  Total Loss: {loss_value:.4f} (avg: {sum(running_losses.get('total', [loss_value])[-50:]) / min(50, len(running_losses.get('total', [loss_value]))):.4f})")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            for k, v in avg_losses.items():
                clean_key = k.replace("_", " ").title()
                print(f"  {clean_key}: {v:.4f}")
            print("-" * 50)

        loss.backward()
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
            training_logs[f"training/loss/{clean_key}"] = v_value
            
        wandb.log({
            "training/loss": loss_value,
            "training/lr": scheduler.get_last_lr()[-1],
            **training_logs
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def run_test(loader: DataLoader,
             model: torch.nn.Module,
             dry_run_steps: int = -1,
             dataset="gen1"):

    model.eval()
    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)
        detections, targets = model(data)
        if i % 10 == 0:
            torch.cuda.empty_cache()
        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()
    return mapcalc

if __name__ == '__main__':
    from torch_geometric.seed import seed_everything
    seed = 42
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()
    setattr(args, "lambda_cnn_loss", 1.0)

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

    lr_func = LRSchedule(warmup_epochs=0.3, num_iters_per_epoch=num_iters_per_epoch, tot_num_epochs=args.tot_num_epochs)
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
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        mapcalc.compute()

    print("starting to train")
    for epoch in range(start_epoch, args.tot_num_epochs):
        avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        checkpointer.checkpoint(epoch, name="last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            print(f"mAP: {metrics.get('mAP', 'N/A')}")
            checkpointer.process(metrics, epoch)
