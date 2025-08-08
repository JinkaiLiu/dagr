# avoid matlab error on server
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

from torch_geometric.data import DataLoader

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

#DAGR with FCOS Head
from dagr.model.networks.dagr_fcos import DAGR_FCOS as DAGR

#DAGR with YOLOX Head
#from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA


def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

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

    for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        optimizer.zero_grad(set_to_none=True)

        model_outputs = model(data)

        loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
        loss = loss_dict.pop("total_loss")

        total_loss += loss.item()
        num_batches += 1

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

        fix_gradients(model)

        # 确保所有梯度和参数在同一设备上
        device = next(model.parameters()).device
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = p.grad.to(device)

        # 确保优化器状态在正确设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        optimizer.step()
        scheduler.step()

        ema.update(model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def calculate_iou(box1, box2):
    """计算两个边界框的IoU
    
    Args:
        box1: [x1, y1, x2, y2] 格式的边界框
        box2: [x1, y1, x2, y2] 格式的边界框
        
    Returns:
        iou: 两个边界框的IoU
    """
    # 计算交集面积
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算IoU
    iou = intersection / (area1 + area2 - intersection + 1e-6)
    return iou


def match_detections_to_targets(detections, targets, iou_threshold=0.5):
    """匹配检测结果和目标框，计算每个类的TP、FP、FN
    
    Args:
        detections: 检测结果，格式为[x1, y1, x2, y2, score, class_id]
        targets: 目标框，格式为[x1, y1, x2, y2, class_id]
        iou_threshold: IoU阈值，大于该阈值视为匹配成功
        
    Returns:
        matches: 匹配成功的检测结果和目标框的索引
        ious: 每对匹配的IoU值
    """
    matches = []
    ious = []
    
    # 如果没有检测结果或目标框，直接返回空列表
    if len(detections) == 0 or len(targets) == 0:
        return matches, ious
    
    # 计算所有可能的检测框和目标框的IoU
    for i, det in enumerate(detections):
        for j, tgt in enumerate(targets):
            # 检查类别是否匹配
            if det[5] == tgt[4]:
                iou = calculate_iou(det[:4], tgt[:4])
                if iou >= iou_threshold:
                    matches.append((i, j))
                    ious.append(iou)
    
    return matches, ious


def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)
    
    # 新增：详细的调试统计
    total_detections = 0
    total_targets = 0
    class_distribution_det = {i: 0 for i in range(len(loader.dataset.classes))}
    class_distribution_gt = {i: 0 for i in range(len(loader.dataset.classes))}
    score_stats = {i: {'scores': [], 'matched': []} for i in range(len(loader.dataset.classes))}
    box_sizes_det = []
    box_sizes_gt = []
    
    # 统计每个类别的TP、FP、FN
    tp_fp_fn = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(loader.dataset.classes))}
    
    # 记录每个批次的IoU分布
    all_ious = []
    
    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        detections, targets = model(data)
        
        # 重要修复：在传递给mAP计算器之前，确保检测框格式正确
        # 查看检测框格式并打印
        fixed_detections = []
        for det in detections:
            fixed_det = det.copy()  # 创建一个副本
            if 'boxes' in fixed_det and fixed_det['boxes'].size(0) > 0:
                boxes = fixed_det['boxes']
                
                # 检查框格式：如果是[x1, y1, w, h]格式，转换为[x1, y1, x2, y2]格式
                # [x1, y1, w, h]是FCOS的预测输出格式，不是GT框原始格式
                # 通常DetectionBuffer期望的是[x1, y1, x2, y2]格式
                if boxes.size(1) == 4:
                    print(f"[FORMAT-CHECK] Original boxes format: {boxes[0]}")
                    
                    # 检测第一个框的格式
                    first_box = boxes[0].cpu().numpy()
                    if first_box[2] < first_box[0] or first_box[3] < first_box[1]:
                        # 看起来已经是[x1, y1, w, h]格式，转换为[x1, y1, x2, y2]
                        print("[FORMAT-CONVERT] Converting from [x1, y1, w, h] to [x1, y1, x2, y2]")
                        x1 = boxes[:, 0]
                        y1 = boxes[:, 1]
                        w = boxes[:, 2]
                        h = boxes[:, 3]
                        
                        # 检查宽度和高度是否为负值
                        if (w < 0).any() or (h < 0).any():
                            print(f"[WARNING] Found negative width/height: w_min={w.min().item()}, h_min={h.min().item()}")
                            # 纠正负值
                            w = torch.abs(w)
                            h = torch.abs(h)
                        
                        # 转换为x2, y2
                        x2 = x1 + w
                        y2 = y1 + h
                        
                        # 创建新的boxes
                        fixed_det['boxes'] = torch.stack([x1, y1, x2, y2], dim=1)
                    else:
                        # 已经是[x1, y1, x2, y2]格式
                        print("[FORMAT-CHECK] Boxes already in [x1, y1, x2, y2] format")
                    
                    print(f"[FORMAT-CHECK] Converted boxes format: {fixed_det['boxes'][0]}")
            
            fixed_detections.append(fixed_det)
        
        # 使用修复后的检测结果
        detections = fixed_detections
        
        if i % 10 == 0:
            torch.cuda.empty_cache()

        # 添加到mAP计算器
        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])
        
        # 新增：详细分析每个批次的检测结果
        batch_dets = 0
        batch_targets = 0
        
        print(f"\n[BATCH-{i}] 分析检测结果:")
        
        # 将结果转移到CPU以便分析
        cpu_detections = [{k: v.cpu() for k, v in det.items()} for det in detections]
        cpu_targets = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
        
        for j, (det, tgt) in enumerate(zip(cpu_detections, cpu_targets)):
            # 检测结果分析
            num_dets = det['boxes'].shape[0]
            batch_dets += num_dets
            total_detections += num_dets
            
            # 目标框分析
            num_targets = tgt['boxes'].shape[0]
            batch_targets += num_targets
            total_targets += num_targets
            
            # 打印前三个检测框
            if num_dets > 0:
                print(f"  检测框样例 (batch {j}):")
                for k in range(min(3, num_dets)):
                    box = det['boxes'][k].tolist()
                    print(f"    Box {k}: {box}")
            
            # 打印前三个目标框
            if num_targets > 0:
                print(f"  目标框样例 (batch {j}):")
                for k in range(min(3, num_targets)):
                    box = tgt['boxes'][k].tolist()
                    print(f"    Box {k}: {box}")
            
            # 转换为标准格式进行匹配
            det_std = []
            if num_dets > 0:
                for k in range(num_dets):
                    box = det['boxes'][k].tolist()
                    score = det['scores'][k].item()
                    label = det['labels'][k].item()
                    
                    # 检查是否为[x1, y1, w, h]格式
                    if len(box) == 4 and (box[2] < box[0] or box[3] < box[1]):
                        # 转换为[x1, y1, x2, y2]格式
                        box = [box[0], box[1], box[0] + abs(box[2]), box[1] + abs(box[3])]
                    
                    det_std.append(box + [score, label])
                    
                    # 统计类别分布
                    class_distribution_det[label] += 1
                    score_stats[label]['scores'].append(score)
                    
                    # 统计框大小
                    w, h = box[2] - box[0], box[3] - box[1]
                    box_sizes_det.append((w, h))
            
            tgt_std = []
            if num_targets > 0:
                for k in range(num_targets):
                    box = tgt['boxes'][k].tolist()
                    label = tgt['labels'][k].item()
                    
                    # 检查是否为[x1, y1, w, h]格式
                    if len(box) == 4 and (box[2] < box[0] or box[3] < box[1]):
                        # 转换为[x1, y1, x2, y2]格式
                        box = [box[0], box[1], box[0] + abs(box[2]), box[1] + abs(box[3])]
                    
                    tgt_std.append(box + [label])
                    
                    # 统计类别分布
                    class_distribution_gt[label] += 1
                    
                    # 统计框大小
                    w, h = box[2] - box[0], box[3] - box[1]
                    box_sizes_gt.append((w, h))
            
            # 匹配检测结果和目标框
            matches, ious = match_detections_to_targets(det_std, tgt_std, iou_threshold=0.5)
            all_ious.extend(ious)
            
            # 统计每个类别的TP、FP、FN
            if num_dets > 0 and num_targets > 0:
                # 记录哪些检测框被匹配
                matched_dets = set()
                matched_tgts = set()
                
                for (d_idx, t_idx), iou in zip(matches, ious):
                    det_class = det_std[d_idx][5]
                    tgt_class = tgt_std[t_idx][4]
                    
                    # 记录被匹配的检测框的分数
                    score_stats[det_class]['matched'].append(det_std[d_idx][4])
                    
                    tp_fp_fn[det_class]['tp'] += 1
                    matched_dets.add(d_idx)
                    matched_tgts.add(t_idx)
                
                # 统计FP (未匹配的检测框)
                for d_idx in range(len(det_std)):
                    if d_idx not in matched_dets:
                        det_class = det_std[d_idx][5]
                        tp_fp_fn[det_class]['fp'] += 1
                
                # 统计FN (未匹配的目标框)
                for t_idx in range(len(tgt_std)):
                    if t_idx not in matched_tgts:
                        tgt_class = tgt_std[t_idx][4]
                        tp_fp_fn[tgt_class]['fn'] += 1
            
            # 如果没有匹配，所有检测都是FP，所有目标都是FN
            elif num_dets > 0 and num_targets == 0:
                for d_idx in range(len(det_std)):
                    det_class = det_std[d_idx][5]
                    tp_fp_fn[det_class]['fp'] += 1
            
            elif num_dets == 0 and num_targets > 0:
                for t_idx in range(len(tgt_std)):
                    tgt_class = tgt_std[t_idx][4]
                    tp_fp_fn[tgt_class]['fn'] += 1
                
        # 打印批次统计信息
        print(f"  批次 {i}: 检测框数量={batch_dets}, 目标框数量={batch_targets}")
        
        if batch_dets > 0 and batch_targets > 0:
            print(f"  成功匹配数量: {len(matches)}")
            if len(ious) > 0:
                print(f"  平均IoU: {sum(ious)/len(ious):.4f}, 最小IoU: {min(ious):.4f}, 最大IoU: {max(ious):.4f}")
        
        # 如果是干运行，提前退出
        if dry_run_steps > 0 and i == dry_run_steps:
            break
    
    # 打印总体统计信息
    print("\n==== 检测结果总体统计 ====")
    print(f"总检测框数量: {total_detections}, 总目标框数量: {total_targets}")
    
    print("\n类别分布:")
    for cls_id in range(len(loader.dataset.classes)):
        gt_count = class_distribution_gt.get(cls_id, 0)
        det_count = class_distribution_det.get(cls_id, 0)
        print(f"  类别 {cls_id} ({loader.dataset.classes[cls_id]}): GT={gt_count}, 检测={det_count}")
    
    print("\n检测分数统计:")
    for cls_id in range(len(loader.dataset.classes)):
        scores = score_stats[cls_id]['scores']
        matched_scores = score_stats[cls_id]['matched']
        
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  类别 {cls_id}: 平均分数={avg_score:.4f}, 最小={min_score:.4f}, 最大={max_score:.4f}, 总数={len(scores)}")
            
            if len(matched_scores) > 0:
                match_rate = len(matched_scores) / len(scores) * 100
                avg_matched_score = sum(matched_scores) / len(matched_scores)
                print(f"    匹配成功: {len(matched_scores)}/{len(scores)} ({match_rate:.2f}%), 平均分数={avg_matched_score:.4f}")
            else:
                print(f"    匹配成功: 0/{len(scores)} (0.00%), 没有匹配成功的检测")
    
    print("\nTP/FP/FN统计:")
    for cls_id in range(len(loader.dataset.classes)):
        tp = tp_fp_fn[cls_id]['tp']
        fp = tp_fp_fn[cls_id]['fp']
        fn = tp_fp_fn[cls_id]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  类别 {cls_id}: TP={tp}, FP={fp}, FN={fn}")
        print(f"    精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")
    
    print("\nIoU统计:")
    if len(all_ious) > 0:
        avg_iou = sum(all_ious) / len(all_ious)
        min_iou = min(all_ious)
        max_iou = max(all_ious)
        print(f"  平均IoU={avg_iou:.4f}, 最小IoU={min_iou:.4f}, 最大IoU={max_iou:.4f}, 匹配数量={len(all_ious)}")
        
        # 统计IoU分布
        iou_bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        iou_dist = {f"{bin:.2f}-{bin+0.05:.2f}": 0 for bin in iou_bins[:-1]}
        iou_dist["0.95-1.00"] = 0
        
        for iou in all_ious:
            for i, bin in enumerate(iou_bins[:-1]):
                if bin <= iou < iou_bins[i+1]:
                    iou_dist[f"{bin:.2f}-{bin+0.05:.2f}"] += 1
                    break
            if iou >= iou_bins[-1]:
                iou_dist["0.95-1.00"] += 1
        
        print("  IoU分布:")
        for bin, count in iou_dist.items():
            percentage = count / len(all_ious) * 100
            print(f"    {bin}: {count} ({percentage:.2f}%)")
    else:
        print("  没有匹配成功的检测框和目标框")
    
    print("\n框大小统计:")
    if len(box_sizes_det) > 0:
        avg_w_det = sum(w for w, _ in box_sizes_det) / len(box_sizes_det)
        avg_h_det = sum(h for _, h in box_sizes_det) / len(box_sizes_det)
        print(f"  检测框: 平均宽度={avg_w_det:.2f}, 平均高度={avg_h_det:.2f}")
    
    if len(box_sizes_gt) > 0:
        avg_w_gt = sum(w for w, _ in box_sizes_gt) / len(box_sizes_gt)
        avg_h_gt = sum(h for _, h in box_sizes_gt) / len(box_sizes_gt)
        print(f"  目标框: 平均宽度={avg_w_gt:.2f}, 平均高度={avg_h_gt:.2f}")
    
    # 比较检测框和目标框的大小差异
    if len(box_sizes_det) > 0 and len(box_sizes_gt) > 0:
        w_ratio = avg_w_det / avg_w_gt if avg_w_gt > 0 else float('inf')
        h_ratio = avg_h_det / avg_h_gt if avg_h_gt > 0 else float('inf')
        print(f"  检测框/目标框比例: 宽度比={w_ratio:.2f}, 高度比={h_ratio:.2f}")
    
    # 检查是否存在框格式问题
    print("\n框格式检查:")
    invalid_gt_boxes = sum(1 for w, h in box_sizes_gt if w <= 0 or h <= 0)
    invalid_det_boxes = sum(1 for w, h in box_sizes_det if w <= 0 or h <= 0)
    
    if invalid_gt_boxes > 0:
        print(f"  警告: 发现 {invalid_gt_boxes} 个无效的目标框 (宽度或高度<=0)")
    else:
        print("  目标框格式正常")
    
    if invalid_det_boxes > 0:
        print(f"  警告: 发现 {invalid_det_boxes} 个无效的检测框 (宽度或高度<=0)")
    else:
        print("  检测框格式正常")
    
    # 详细检查mAP计算流程中的关键点
    print("\nmAP计算分析:")
    print("  用于评估的DetectionBuffer包含:")
    print(f"    检测结果数量: {len(mapcalc.detections.buffer)}")
    print(f"    目标标签数量: {len(mapcalc.ground_truth.buffer)}")
    
    # 分析常见的mAP为0的原因
    print("\n常见的mAP为0的可能原因:")
    
    # 1. 检查是否有检测结果
    if total_detections == 0:
        print("  ✗ 模型没有产生任何检测结果，需要降低检测阈值或检查模型输出")
    else:
        print("  ✓ 模型产生了检测结果")
    
    # 2. 检查类别预测是否正确
    if sum(tp_fp_fn[i]['tp'] for i in range(len(loader.dataset.classes))) == 0:
        print("  ✗ 没有一个检测结果的类别和位置同时匹配GT，需要检查类别预测的准确性")
    else:
        print("  ✓ 有一些检测结果的类别和位置同时匹配GT")
    
    # 3. 检查IoU阈值
    if len(all_ious) > 0:
        above_threshold = sum(1 for iou in all_ious if iou >= 0.5)
        if above_threshold == 0:
            print("  ✗ 没有检测结果的IoU超过0.5，需要提高定位精度")
        else:
            print(f"  ✓ {above_threshold}/{len(all_ious)} 的匹配IoU超过0.5")
    
    # 4. 检查分数阈值
    lowest_score_threshold = 0.05  # 假设模型使用的最低阈值
    low_score_matches = sum(1 for cls_id in range(len(loader.dataset.classes)) 
                          for score in score_stats[cls_id]['matched'] 
                          if score < lowest_score_threshold)
    if low_score_matches > 0:
        print(f"  ✗ 有 {low_score_matches} 个匹配的检测分数低于阈值 {lowest_score_threshold}，这些检测可能被过滤掉")
    
    # 5. 检查是否存在类别不平衡
    max_gt = max(class_distribution_gt.values()) if class_distribution_gt else 0
    min_gt = min([v for v in class_distribution_gt.values() if v > 0], default=0)
    if max_gt > 0 and min_gt > 0 and max_gt / min_gt > 10:
        print(f"  ✗ 类别严重不平衡，最多类别/最少类别 = {max_gt}/{min_gt} = {max_gt/min_gt:.2f}")
    
    # 6. 检查类别1的特殊问题
    if 1 in class_distribution_gt and class_distribution_gt[1] > 0:
        class1_recall = tp_fp_fn[1]['tp'] / (tp_fp_fn[1]['tp'] + tp_fp_fn[1]['fn']) if (tp_fp_fn[1]['tp'] + tp_fp_fn[1]['fn']) > 0 else 0
        if class1_recall < 0.1:
            print(f"  ✗ 类别1的召回率极低: {class1_recall:.4f}，几乎检测不到类别1")
            
            if len(score_stats[1]['scores']) > 0:
                above_threshold = sum(1 for score in score_stats[1]['scores'] if score >= 0.3)
                if above_threshold == 0:
                    print(f"    所有类别1的检测分数都低于0.3，需要调整类别1的分数提升或阈值")

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

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
    #dataset_path = args.dataset_directory / args.dataset
    dataset_path = args.dataset_directory

    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10)

    train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    num_iters_per_epoch = len(train_loader)

    #sampler = np.random.permutation(np.arange(len(test_dataset)))
    test_loader = DataLoader(test_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    model = model.cuda()
    ema = ModelEMA(model)

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
    #start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)
    print(f"[DEBUG] After restore_if_existing: start_epoch = {start_epoch}")
    if start_epoch is None:
        start_epoch = 0

    # 如果从断点恢复，同步学习率调度器
    if start_epoch > 0:
        print(f"[INFO] Resuming from epoch {start_epoch}, synchronizing lr_scheduler")
        # 快进学习率调度器到正确的步数
        total_steps = start_epoch * num_iters_per_epoch
        for _ in range(total_steps):
            lr_scheduler.step()
        print(f"[INFO] LR scheduler advanced {total_steps} steps")

    #start_epoch = 0
    #print(f"[DEBUG] After reset: start_epoch = {start_epoch}")
    if "resume_checkpoint" in args:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    with torch.no_grad():
        mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
        mapcalc.compute()

    print("starting to train")
    print(f"[FINAL] Starting training from epoch: {start_epoch}")
    for epoch in range(start_epoch, args.tot_num_epochs):
        #train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        checkpointer.checkpoint(epoch, name=f"last_model")

        if epoch % 3 > 0:
            continue

        with torch.no_grad():
            print("test_loader length =", len(test_loader))
            mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
            metrics = mapcalc.compute()
            print(f"mAP: {metrics.get('mAP', 'N/A')}")
            checkpointer.process(metrics, epoch)




# # avoid matlab error on server
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# import torch

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(False)

# import tqdm
# import wandb
# from pathlib import Path
# import argparse

# from torch_geometric.data import DataLoader

# from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
# from dagr.utils.buffers import DetectionBuffer
# from dagr.utils.args import FLAGS
# from dagr.utils.learning_rate_scheduler import LRSchedule

# from dagr.data.augment import Augmentations
# from dagr.utils.buffers import format_data
# from dagr.data.dsec_data import DSEC

# #DAGR with FCOS Head
# from dagr.model.networks.dagr_fcos import DAGR_FCOS as DAGR

# #DAGR with YOLOX Head
# #from dagr.model.networks.dagr import DAGR
# from dagr.model.networks.ema import ModelEMA


# def gradients_broken(model):
#     valid_gradients = True
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
#             valid_gradients = not (torch.isnan(param.grad).any())
#             if not valid_gradients:
#                 break
#     return not valid_gradients

# def fix_gradients(model):
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0)


# def train(loader: DataLoader,
#           model: torch.nn.Module,
#           ema: ModelEMA,
#           scheduler: torch.optim.lr_scheduler.LambdaLR,
#           optimizer: torch.optim.Optimizer,
#           args: argparse.ArgumentParser,
#           run_name=""):

#     model.train()
#     total_loss = 0.0
#     num_batches = 0

#     for i, data in enumerate(tqdm.tqdm(loader, desc=f"Training {run_name}")):
#         data = data.cuda(non_blocking=True)
#         data = format_data(data)

#         optimizer.zero_grad(set_to_none=True)

#         model_outputs = model(data)

#         loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
#         loss = loss_dict.pop("total_loss")

#         total_loss += loss.item()
#         num_batches += 1

#         loss.backward()

#         torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)

#         fix_gradients(model)

#         # 确保所有梯度和参数在同一设备上
#         device = next(model.parameters()).device
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 if p.grad is not None:
#                     p.grad = p.grad.to(device)

#         # 确保优化器状态在正确设备上
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if isinstance(v, torch.Tensor):
#                     state[k] = v.to(device)

#         optimizer.step()
#         scheduler.step()

#         ema.update(model)

#         training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
#         wandb.log({"training/loss": loss.item(), "training/lr": scheduler.get_last_lr()[-1], **training_logs})

#     avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
#     return avg_loss


# def run_test(loader: DataLoader,
#          model: torch.nn.Module,
#          dry_run_steps: int=-1,
#          dataset="gen1"):

#     model.eval()

#     mapcalc = DetectionBuffer(height=loader.dataset.height, width=loader.dataset.width, classes=loader.dataset.classes)

#     for i, data in enumerate(tqdm.tqdm(loader)):
#         data = data.cuda()
#         data = format_data(data)

#         detections, targets = model(data)
#         if i % 10 == 0:
#             torch.cuda.empty_cache()

#         mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

#         if dry_run_steps > 0 and i == dry_run_steps:
#             break

#     torch.cuda.empty_cache()

#     return mapcalc

# if __name__ == '__main__':
#     import torch_geometric
#     import random
#     import numpy as np

#     seed = 42
#     torch_geometric.seed.seed_everything(seed)
#     torch.random.manual_seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

#     args = FLAGS()

#     output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
#     log_hparams(args)

#     augmentations = Augmentations(args)

#     print("init datasets")
#     #dataset_path = args.dataset_directory / args.dataset
#     dataset_path = args.dataset_directory

#     train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
#                          min_bbox_diag=15, min_bbox_height=10)
#     test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
#                         min_bbox_diag=15, min_bbox_height=10)

#     train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
#     num_iters_per_epoch = len(train_loader)

#     #sampler = np.random.permutation(np.arange(len(test_dataset)))
#     test_loader = DataLoader(test_dataset, follow_batch=['bbox', 'bbox0'], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

#     print("init net")
#     # load a dummy sample to get height, width
#     model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

#     num_params = sum([np.prod(p.size()) for p in model.parameters()])
#     print(f"Training with {num_params} number of parameters.")

#     model = model.cuda()
#     ema = ModelEMA(model)

#     nominal_batch_size = 64
#     lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
#     optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

#     if torch.cuda.is_available():
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if isinstance(v, torch.Tensor):
#                     state[k] = v.cuda()

#     lr_func = LRSchedule(warmup_epochs=.3,
#                          num_iters_per_epoch=num_iters_per_epoch,
#                          tot_num_epochs=args.tot_num_epochs)

#     lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

#     checkpointer = Checkpointer(output_directory=output_directory,
#                                 model=model, optimizer=optimizer,
#                                 scheduler=lr_scheduler, ema=ema,
#                                 args=args)

#     start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=True)
#     #start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)
#     print(f"[DEBUG] After restore_if_existing: start_epoch = {start_epoch}")
#     if start_epoch is None:
#         start_epoch = 0

#     # 如果从断点恢复，同步学习率调度器
#     if start_epoch > 0:
#         print(f"[INFO] Resuming from epoch {start_epoch}, synchronizing lr_scheduler")
#         # 快进学习率调度器到正确的步数
#         total_steps = start_epoch * num_iters_per_epoch
#         for _ in range(total_steps):
#             lr_scheduler.step()
#         print(f"[INFO] LR scheduler advanced {total_steps} steps")

#     #start_epoch = 0
#     #print(f"[DEBUG] After reset: start_epoch = {start_epoch}")
#     if "resume_checkpoint" in args:
#         start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
#         print(f"Resume from checkpoint at epoch {start_epoch}")

#     with torch.no_grad():
#         mapcalc = run_test(test_loader, ema.ema, dry_run_steps=2, dataset=args.dataset)
#         mapcalc.compute()

#     print("starting to train")
#     print(f"[FINAL] Starting training from epoch: {start_epoch}")
#     for epoch in range(start_epoch, args.tot_num_epochs):
#         #train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
#         avg_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, run_name=wandb.run.name)
#         print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
#         checkpointer.checkpoint(epoch, name=f"last_model")

#         if epoch % 3 > 0:
#             continue

#         with torch.no_grad():
#             print("test_loader length =", len(test_loader))
#             mapcalc = run_test(test_loader, ema.ema, dataset=args.dataset)
#             metrics = mapcalc.compute()
#             print(f"mAP: {metrics.get('mAP', 'N/A')}")
#             checkpointer.process(metrics, epoch)