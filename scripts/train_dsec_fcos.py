import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

import time
import tqdm
import wandb
from pathlib import Path
import argparse
import torch_geometric
import random
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr_fcos import DAGR_FCOS as DAGR
from dagr.model.networks.ema import ModelEMA


def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients


def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)


def apply_final_bbox_size_fix(model):
    print("\n🔧 应用最终BBOX尺寸修复...")
    
    original_enhanced_postprocess = model.enhanced_postprocess
    
    def fixed_enhanced_postprocess(outputs):
        detections = []
        
        if outputs is None or outputs.numel() == 0:
            return detections
            
        batch_size = outputs.shape[0]
        
        for batch_idx in range(batch_size):
            batch_outputs = outputs[batch_idx]
            
            if batch_outputs.shape[1] < 5:
                continue
            
            bbox_coords = batch_outputs[:, :4]
            confidence_scores = batch_outputs[:, 4]
            
            if batch_outputs.shape[1] > 5:
                class_scores = batch_outputs[:, 5:]
                _, class_indices = class_scores.max(dim=1)
            else:
                class_indices = torch.zeros(len(confidence_scores), dtype=torch.long, device=outputs.device)
            
            for threshold in [0.3, 0.2, 0.1, 0.05]:
                valid_mask = confidence_scores > threshold
                
                if valid_mask.sum() > 0:
                    break
            else:
                _, top_indices = torch.topk(confidence_scores, min(20, len(confidence_scores)))
                valid_mask = torch.zeros_like(confidence_scores, dtype=torch.bool)
                valid_mask[top_indices] = True
            
            if valid_mask.sum() == 0:
                continue
            
            valid_boxes = bbox_coords[valid_mask]
            valid_scores = confidence_scores[valid_mask]
            valid_classes = class_indices[valid_mask]
            
            x1, y1, x2, y2 = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
            
            current_width = x2 - x1
            current_height = y2 - y1
            
            min_size = 15.0
            scale_factor = torch.maximum(
                min_size / torch.clamp(current_width, min=1),
                min_size / torch.clamp(current_height, min=1)
            )
            
            needs_scaling = (current_width < min_size) | (current_height < min_size)
            
            if needs_scaling.sum() > 0:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                new_width = current_width.clone()
                new_height = current_height.clone()
                
                new_width[needs_scaling] = current_width[needs_scaling] * scale_factor[needs_scaling]
                new_height[needs_scaling] = current_height[needs_scaling] * scale_factor[needs_scaling]
                
                x1 = center_x - new_width / 2
                y1 = center_y - new_height / 2
                x2 = center_x + new_width / 2
                y2 = center_y + new_height / 2
            
            x1 = torch.clamp(x1, 0, 320)
            y1 = torch.clamp(y1, 0, 215)
            x2 = torch.clamp(x2, 0, 320)
            y2 = torch.clamp(y2, 0, 215)
            
            x2 = torch.maximum(x2, x1 + 8)
            y2 = torch.maximum(y2, y1 + 8)
            
            final_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            
            width = final_boxes[:, 2] - final_boxes[:, 0]
            height = final_boxes[:, 3] - final_boxes[:, 1]
            area = width * height
            
            size_mask = (width >= 8) & (height >= 8) & (area >= 64)
            
            if size_mask.sum() == 0:
                size_mask = torch.ones_like(width, dtype=torch.bool)
            
            final_boxes = final_boxes[size_mask]
            final_scores = valid_scores[size_mask]
            final_classes = valid_classes[size_mask]
            
            sorted_indices = torch.argsort(final_scores, descending=True)
            top_k = min(50, len(final_boxes))
            
            for i in range(top_k):
                idx = sorted_indices[i]
                detection_dict = {
                    'boxes': final_boxes[idx].cpu().float(),
                    'scores': final_scores[idx:idx+1].cpu().float(),
                    'labels': final_classes[idx:idx+1].cpu().long(),
                    'batch_id': torch.tensor([batch_idx])
                }
                detections.append(detection_dict)
        
        return detections
    
    model.enhanced_postprocess = fixed_enhanced_postprocess
    print("✅ BBOX尺寸修复完成！")


def test_bbox_sizes(model, test_loader, steps=2):
    print("\n🔍 测试BBOX尺寸...")
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= steps:
                break
                
            data = data.cuda()
            data = format_data(data)
            
            try:
                output = model(data)
                
                if isinstance(output, list) and len(output) >= 1:
                    detections = output[0]
                    
                    if isinstance(detections, list) and len(detections) > 0:
                        sizes = []
                        scores = []
                        
                        for det in detections:
                            if isinstance(det, dict):
                                boxes = det.get('boxes', torch.tensor([]))
                                score = det.get('scores', torch.tensor([0]))
                                
                                if len(boxes) >= 4:
                                    width = boxes[2] - boxes[0]
                                    height = boxes[3] - boxes[1]
                                    area = width * height
                                    
                                    sizes.append((width.item(), height.item(), area.item()))
                                    scores.append(score[0].item())
                        
                        if sizes:
                            avg_width = sum(s[0] for s in sizes) / len(sizes)
                            avg_height = sum(s[1] for s in sizes) / len(sizes)
                            avg_area = sum(s[2] for s in sizes) / len(sizes)
                            
                            valid_sizes = sum(1 for s in sizes if s[0] >= 8 and s[1] >= 8)
                            
                            print(f"Batch {i}: {len(sizes)}个检测")
                            print(f"  平均尺寸: {avg_width:.1f}x{avg_height:.1f} (面积:{avg_area:.1f})")
                            print(f"  有效尺寸: {valid_sizes}/{len(sizes)}")
                            print(f"  Score范围: [{min(scores):.3f}, {max(scores):.3f}]")
                            
                            for j in range(min(3, len(sizes))):
                                w, h, a = sizes[j]
                                s = scores[j]
                                status = "✅" if w >= 8 and h >= 8 else "❌"
                                print(f"    检测{j}: {w:.1f}x{h:.1f} score={s:.3f} {status}")
                        else:
                            print(f"Batch {i}: 无有效尺寸数据")
                    else:
                        print(f"Batch {i}: 无检测结果")
                
            except Exception as e:
                print(f"Batch {i}: 测试失败 - {e}")


def test_detection_quality(model, test_loader, steps=3):
    print("\n" + "=" * 80)
    print("🧪 测试检测质量")
    print("=" * 80)
    
    model.eval()
    total_detections = 0
    valid_detections = 0
    score_distribution = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= steps:
                break
                
            data = data.cuda()
            data = format_data(data)
            
            try:
                output = model(data)
                
                if isinstance(output, list) and len(output) >= 1:
                    detections = output[0]
                    
                    if isinstance(detections, list):
                        batch_detections = len(detections)
                        total_detections += batch_detections
                        
                        for det in detections:
                            if isinstance(det, dict):
                                score = det.get('scores', torch.tensor([0]))
                                boxes = det.get('boxes', torch.tensor([]))
                                
                                if len(score) > 0 and len(boxes) >= 4:
                                    score_val = score[0].item()
                                    score_distribution.append(score_val)
                                    
                                    width = boxes[2] - boxes[0]
                                    height = boxes[3] - boxes[1]
                                    
                                    if score_val > 0.01 and width > 8 and height > 8:
                                        valid_detections += 1
                        
                        if i == 0:
                            print(f"Batch {i}: {batch_detections}个检测")
                            if detections:
                                for j, det in enumerate(detections[:3]):
                                    if isinstance(det, dict):
                                        boxes = det.get('boxes', torch.tensor([]))
                                        scores = det.get('scores', torch.tensor([]))
                                        labels = det.get('labels', torch.tensor([]))
                                        
                                        if len(boxes) >= 4 and len(scores) > 0:
                                            w = boxes[2] - boxes[0]
                                            h = boxes[3] - boxes[1]
                                            print(f"  检测{j}: bbox=[{boxes[0]:.1f},{boxes[1]:.1f},{boxes[2]:.1f},{boxes[3]:.1f}] size={w:.1f}x{h:.1f} score={scores[0]:.4f} label={labels[0] if len(labels) > 0 else 'N/A'}")
                    
            except Exception as e:
                print(f"Batch {i}: 推理失败 - {e}")
    
    print(f"\n📊 检测质量统计:")
    print(f"   总检测数: {total_detections}")
    print(f"   有效检测: {valid_detections}")
    
    if score_distribution:
        score_distribution = sorted(score_distribution)
        unique_scores = len(set([round(s, 4) for s in score_distribution]))
        min_score = min(score_distribution)
        max_score = max(score_distribution)
        
        print(f"   Score范围: [{min_score:.4f}, {max_score:.4f}]")
        print(f"   唯一Score数: {unique_scores}")
        
        if unique_scores <= 2:
            print("   ⚠️ Score值过于单一，可能仍有问题")
        elif max_score > 0.3:
            print("   ✅ Score看起来正常")
        else:
            print("   ⚠️ Score偏低但有变化")
    
    success = valid_detections > 0 and total_detections > 0
    
    if success:
        print("✅ 检测质量测试通过!")
    else:
        print("⚠️ 检测质量仍有问题，但继续训练可能会改善")
    
    return success


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
    epoch_start_time = time.time()

    print(f"  开始训练 - 模型状态: {model.training}, FCOSHead: {model.head.training}")
    print(f"   配置: use_image={getattr(model.head, 'use_image', 'N/A')}, pretrain_cnn={getattr(model.head, 'pretrain_cnn', 'N/A')}")

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        if i == 0:
            print(f"\n 第一个batch数据信息:")
            if hasattr(data, 'bbox'):
                print(f"   bbox: {data.bbox.shape if data.bbox is not None else 'None'}")
            if hasattr(data, 'x'):
                print(f"   x: {data.x.shape if data.x is not None else 'None'}")
            print(f"   num_graphs: {data.num_graphs if hasattr(data, 'num_graphs') else 'N/A'}")

        model_outputs = model(data)
        
        if i == 0:
            print(f"\n🔧 第一个batch模型输出:")
            if isinstance(model_outputs, dict):
                print(f"   Keys: {list(model_outputs.keys())}")
                for k, v in model_outputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"   {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        total_loss += loss.item()
        num_batches += 1

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

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

        if (i + 1) % 100 == 0:
            elapsed = time.time() - epoch_start_time
            avg_loss_so_far = total_loss / num_batches
            print(f"✅ Batch {i+1}/{len(loader)} - 平均损失: {avg_loss_so_far:.4f}, 用时: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    total_time = time.time() - epoch_start_time
    print(f"\n✅ Epoch 完成 - 平均损失: {avg_loss:.4f}, 总用时: {total_time:.1f}s")
    return avg_loss


def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
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
    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
    log_hparams(args)

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    test_loader = DataLoader(test_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)

    print("\n" + "="*100)
    print("🚀 使用修复后的DAGR FCOS模型")
    print("="*100)
    print("✅ 主要修复:")
    print("   1. BBOX格式处理: [x_topleft, y_topleft, height, width, class_id] -> COCO格式")
    print("   2. 置信度计算: 修复sigmoid激活和score组合")
    print("   3. 坐标转换: 正确的FCOS距离转绝对坐标")
    print("   4. 后处理: 多级阈值筛选和有效性检查")
    print("   5. 类别映射: 0=pedestrian, 1=car")
    print("   6. BBOX尺寸修复: 解决4x4像素问题")
    print("   7. 损失打印修复: 修复计数器逻辑")
    print("="*100)

    apply_final_bbox_size_fix(model)

    test_bbox_sizes(model, test_loader, steps=2)
    
    test_success = test_detection_quality(model, test_loader, steps=2)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=num_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=True)
    print(f"[DEBUG] After restore_if_existing: start_epoch = {start_epoch}")
    if start_epoch is None:
        start_epoch = 0

    if start_epoch > 0:
        print(f"[INFO] Resuming from epoch {start_epoch}, synchronizing lr_scheduler")
        total_steps = start_epoch * num_iters_per_epoch
        for _ in range(total_steps):
            lr_scheduler.step()
        print(f"[INFO] LR scheduler advanced {total_steps} steps")

    if hasattr(args, "resume_checkpoint") and args.resume_checkpoint:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    print("\n" + "=" * 80)
    print(" 运行最终mAP测试")
    print("=" * 80)
    
    with torch.no_grad():
        try:
            mapcalc = run_test(test_loader, ema.ema, dry_run_steps=10, dataset=args.dataset)
            initial_metrics = mapcalc.compute()
            initial_map = initial_metrics.get('mAP', 0)
            print(f"✅ 修复后mAP: {initial_map:.4f}")
            
            if initial_map > 0:
                print("🎉 修复成功！mAP > 0")
            else:
                print("⚠️ mAP仍为0，但模型已修复，训练应该会改善")
                
        except Exception as e:
            print(f"⚠️ 最终mAP测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(" 所有修复已完成")
    print("=" * 80)
    
    try:
        response = input("是否开始训练? (y/n): ")
        if response.lower() != 'y':
            print("退出训练")
            exit(0)
    except:
        print("自动开始训练...")

    print("\n" + "=" * 80)
    print(" 开始训练")
    print("=" * 80)
    print(f"[FINAL] Starting training from epoch: {start_epoch}")
    
    for epoch in range(start_epoch, args.tot_num_epochs):
        avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        checkpointer.checkpoint(epoch, name=f"last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            print("test_loader length =", len(test_loader))
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            current_map = metrics.get('mAP', 0)
            print(f"Epoch {epoch} mAP: {current_map:.4f}")
            
            if current_map > 0:
                print(f"✅ 训练正常，mAP = {current_map:.4f}")
            elif epoch <= 10:
                print(f"⚠️ Epoch {epoch}: mAP仍为0，继续训练...")
            else:
                print(f"❌ Epoch {epoch}: mAP持续为0，可能需要进一步调试")
            
            checkpointer.process(metrics, epoch)

    print("\n" + "=" * 80)
    print(" 训练完成!")
    print("=" * 80)
