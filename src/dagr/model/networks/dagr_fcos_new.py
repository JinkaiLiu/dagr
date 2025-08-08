import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch_geometric.data import Data
from dagr.model.networks.net import Net
from dagr.model.networks.fcos_head import FCOSHead
from dagr.model.utils import (
    convert_to_training_format,
    convert_to_evaluation_format,
    init_subnetwork
)


class DebugLogger:
    def __init__(self, print_interval=200):  # 修改为200，减少打印频率
        self.print_interval = print_interval
        self.iteration = 0
        self.feature_stats_printed = False
        self.stats = {
            'feature_shapes': [],
            'reg_pred_raw': {'min': [], 'max': [], 'mean': []},
            'reg_pred_scaled': {'min': [], 'max': [], 'mean': []},
            'pred_boxes': {'min': [], 'max': []},
            'target_boxes': {'min': [], 'max': []}
        }
        self.pos_samples_count = []
        self.cls_distribution = {}
    
    def log_feature_shapes(self, features_unpacked):
        if not self.feature_stats_printed:
            self.stats['feature_shapes'] = [f.shape for f in features_unpacked]
            self.feature_stats_printed = True
    
    def log_reg_pred(self, raw_pred, scaled_pred):
        self.stats['reg_pred_raw']['min'].append(raw_pred.min().item())
        self.stats['reg_pred_raw']['max'].append(raw_pred.max().item())
        self.stats['reg_pred_raw']['mean'].append(raw_pred.mean().item())
        
        self.stats['reg_pred_scaled']['min'].append(scaled_pred.min().item())
        self.stats['reg_pred_scaled']['max'].append(scaled_pred.max().item())
        self.stats['reg_pred_scaled']['mean'].append(scaled_pred.mean().item())
    
    def log_boxes(self, pred_boxes, target_boxes):
        if pred_boxes.numel() > 0:
            self.stats['pred_boxes']['min'].append(pred_boxes.min().item())
            self.stats['pred_boxes']['max'].append(pred_boxes.max().item())
        
        if target_boxes.numel() > 0:
            self.stats['target_boxes']['min'].append(target_boxes.min().item())
            self.stats['target_boxes']['max'].append(target_boxes.max().item())
    
    def log_pos_samples(self, num_pos):
        self.pos_samples_count.append(num_pos)
    
    def log_cls_distribution(self, gt_cls):
        for cls_id in gt_cls.unique():
            cls_id = cls_id.item()
            count = (gt_cls == cls_id).sum().item()
            if cls_id not in self.cls_distribution:
                self.cls_distribution[cls_id] = []
            self.cls_distribution[cls_id].append(count)
    
    def log_detailed_box_info(self, gt_boxes, pred_boxes=None, batch_idx=0, max_boxes=5):
        # 仅在特定迭代次数打印
        if self.iteration % self.print_interval != 0:
            return
            
        print(f"[BOX-INFO] Batch {batch_idx} GT boxes details:")
        num_boxes = min(gt_boxes.shape[0], max_boxes)
        for i in range(num_boxes):
            box = gt_boxes[i]
            if box.shape[0] >= 5:
                cls_id = box[0].item() if box.shape[0] > 4 else -1
                cx, cy, w, h = box[1:5].tolist()
                print(f"  GT[{i}]: cls={cls_id}, cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, area={w*h:.1f}")
        
        if pred_boxes is not None and pred_boxes.numel() > 0:
            print(f"[BOX-INFO] Batch {batch_idx} Predicted boxes details:")
            num_pred = min(pred_boxes.shape[0], max_boxes)
            for i in range(num_pred):
                box = pred_boxes[i]
                if len(box) >= 5:
                    cls_id = box[0].item()
                    score = box[1].item() if len(box) > 5 else 1.0
                    cx, cy, w, h = box[2:6].tolist() if len(box) > 5 else box[1:5].tolist()
                    print(f"  PRED[{i}]: cls={cls_id}, score={score:.2f}, cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}, area={w*h:.1f}")
    
    def log_iteration(self):
        self.iteration += 1
        
        if self.iteration % self.print_interval == 0:
            self._print_stats()
            self._reset_stats()
    
    def _print_stats(self):
        if self.stats['feature_shapes']:
            print(f"[DEBUG-{self.iteration}] Feature shapes: {self.stats['feature_shapes']}")
        
        reg_raw_min = sum(self.stats['reg_pred_raw']['min']) / len(self.stats['reg_pred_raw']['min']) if self.stats['reg_pred_raw']['min'] else float('nan')
        reg_raw_max = sum(self.stats['reg_pred_raw']['max']) / len(self.stats['reg_pred_raw']['max']) if self.stats['reg_pred_raw']['max'] else float('nan')
        reg_raw_mean = sum(self.stats['reg_pred_raw']['mean']) / len(self.stats['reg_pred_raw']['mean']) if self.stats['reg_pred_raw']['mean'] else float('nan')
        
        reg_scaled_min = sum(self.stats['reg_pred_scaled']['min']) / len(self.stats['reg_pred_scaled']['min']) if self.stats['reg_pred_scaled']['min'] else float('nan')
        reg_scaled_max = sum(self.stats['reg_pred_scaled']['max']) / len(self.stats['reg_pred_scaled']['max']) if self.stats['reg_pred_scaled']['max'] else float('nan')
        reg_scaled_mean = sum(self.stats['reg_pred_scaled']['mean']) / len(self.stats['reg_pred_scaled']['mean']) if self.stats['reg_pred_scaled']['mean'] else float('nan')
        
        if not math.isnan(reg_raw_min):
            print(f"[DEBUG-{self.iteration}] Reg pred raw stats: min={reg_raw_min:.4f}, max={reg_raw_max:.4f}, mean={reg_raw_mean:.4f}")
        
        if not math.isnan(reg_scaled_min):
            print(f"[DEBUG-{self.iteration}] Reg pred after scale exp: min={reg_scaled_min:.4f}, max={reg_scaled_max:.4f}, mean={reg_scaled_mean:.4f}")
        
        pred_boxes_min = sum(self.stats['pred_boxes']['min']) / len(self.stats['pred_boxes']['min']) if self.stats['pred_boxes']['min'] else float('nan')
        pred_boxes_max = sum(self.stats['pred_boxes']['max']) / len(self.stats['pred_boxes']['max']) if self.stats['pred_boxes']['max'] else float('nan')
        
        target_boxes_min = sum(self.stats['target_boxes']['min']) / len(self.stats['target_boxes']['min']) if self.stats['target_boxes']['min'] else float('nan')
        target_boxes_max = sum(self.stats['target_boxes']['max']) / len(self.stats['target_boxes']['max']) if self.stats['target_boxes']['max'] else float('nan')
        
        if not math.isnan(pred_boxes_min):
            print(f"[DEBUG-{self.iteration}] Pred boxes stats: min={pred_boxes_min:.4f}, max={pred_boxes_max:.4f}")
        
        if not math.isnan(target_boxes_min):
            print(f"[DEBUG-{self.iteration}] Target boxes stats: min={target_boxes_min:.4f}, max={target_boxes_max:.4f}")
        
        avg_pos_samples = sum(self.pos_samples_count) / len(self.pos_samples_count) if self.pos_samples_count else 0
        print(f"[DEBUG-{self.iteration}] Average positive samples per batch: {avg_pos_samples:.2f}")
        
        print(f"[DEBUG-{self.iteration}] Class distribution:")
        for cls_id, counts in self.cls_distribution.items():
            avg_count = sum(counts) / len(counts)
            print(f"  Class {cls_id}: avg {avg_count:.2f} instances per batch")
    
    def _reset_stats(self):
        for stat_type in ['reg_pred_raw', 'reg_pred_scaled', 'pred_boxes', 'target_boxes']:
            for metric in self.stats[stat_type]:
                self.stats[stat_type][metric] = []
        self.pos_samples_count = []
        self.cls_distribution = {}


def unpack_fused_features(fused_feat, debug_logger=None):
    features_tensor = []
    # 只在第一次迭代或200次间隔时打印详细信息
    verbose = debug_logger is None or (debug_logger.iteration <= 1 or debug_logger.iteration % 200 == 0)
    
    if verbose:
        print(f"[DEBUG] unpack_fused_features: received {len(fused_feat)} features")
    
    for i, f in enumerate(fused_feat):
        try:
            if isinstance(f, list):
                if verbose:
                    print(f"[DEBUG] Feature {i} is a list with {len(f)} elements")
                    print(f"[DEBUG] List contents types: {[type(item) for item in f]}")
                
                if len(f) > 0 and all(isinstance(item, torch.Tensor) for item in f):
                    if verbose:
                        print(f"[DEBUG] All list elements are tensors, shapes: {[item.shape for item in f]}")
                    
                    if f[0].dim() == 4:
                        features_tensor.append(f[0])
                        if len(f) > 1 and f[1].dim() == 4:
                            features_tensor.append(f[1])
                        if verbose:
                            print(f"[DEBUG] Added 4D tensor from list with shape: {f[0].shape}")
                    elif f[0].dim() == 3:
                        features_tensor.append(f[0].unsqueeze(0))
                        if len(f) > 1 and f[1].dim() == 3:
                            features_tensor.append(f[1].unsqueeze(0))
                        if verbose:
                            print(f"[DEBUG] Added 3D tensor from list with shape: {f[0].unsqueeze(0).shape}")
                    elif f[0].dim() == 2:
                        if len(f) > 1 and f[1].numel() == 2:
                            try:
                                height, width = f[1].tolist()
                                if verbose:
                                    print(f"[DEBUG] Found height and width in list: {height}x{width}")
                                
                                nodes = f[0]
                                if nodes.shape[0] * nodes.shape[1] == height * width:
                                    reshaped = nodes.reshape(nodes.shape[0], height, width)
                                else:
                                    if nodes.shape[0] > height * width:
                                        nodes = nodes[:height * width]
                                    elif nodes.shape[0] < height * width:
                                        padded = torch.zeros(height * width, nodes.shape[1], device=nodes.device)
                                        padded[:nodes.shape[0]] = nodes
                                        nodes = padded
                                    
                                    reshaped = nodes.reshape(height, width, nodes.shape[1]).permute(2, 0, 1)
                                
                                features_tensor.append(reshaped.unsqueeze(0))
                                if verbose:
                                    print(f"[DEBUG] Added reshaped tensor with shape: {reshaped.unsqueeze(0).shape}")
                            except Exception as e:
                                if verbose:
                                    print(f"[ERROR] Failed to reshape tensor using list dimensions: {e}")
                                nodes = f[0]
                                sqrt_nodes = int(math.sqrt(nodes.shape[0]))
                                if sqrt_nodes * sqrt_nodes == nodes.shape[0]:
                                    reshaped = nodes.reshape(sqrt_nodes, sqrt_nodes, nodes.shape[1]).permute(2, 0, 1)
                                    features_tensor.append(reshaped.unsqueeze(0))
                                    if verbose:
                                        print(f"[DEBUG] Added square reshaped tensor with shape: {reshaped.unsqueeze(0).shape}")
                                else:
                                    if verbose:
                                        print(f"[WARNING] Could not determine proper reshaping for tensor")
                        else:
                            pass
                else:
                    pass
            elif hasattr(f, "x") and isinstance(f.x, torch.Tensor):
                pass
            elif isinstance(f, torch.Tensor):
                pass
            else:
                if verbose:
                    print(f"[WARNING] Feature {i} is neither Data nor Tensor or List: {type(f)}")
        except Exception as e:
            print(f"[ERROR] Failed to process feature {i}: {e}")
            import traceback
            traceback.print_exc()
    
    if verbose:
        print(f"[DEBUG] Returning {len(features_tensor)} processed features")
    return features_tensor


class DAGR_FCOS(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        self.conf_threshold = args.score_threshold if hasattr(args, 'score_threshold') else 0.05
        self.nms_threshold = args.nms_iou_threshold if hasattr(args, 'nms_iou_threshold') else 0.6
        self.height = height
        self.width = width
        self.args = args
        
        self.debug_logger = DebugLogger(print_interval=200)  # 修改为200，减少打印频率

        self.backbone = Net(args, height=height, width=width)
        self.use_image = getattr(args, "use_image", False)
        self.no_events = getattr(args, "no_events", False)
        self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

        strides = [8, 16]
        in_channels = [256, 256]
        
        print(f"[INFO] Using in_channels: {in_channels} for FCOS head")
        
        # 修改初始化参数
        self.head = FCOSHead(
            num_classes=self.backbone.num_classes,
            in_channels=in_channels,
            strides=strides,
            use_gn=True,
            init_prior=0.01,  # 降低分类器先验概率以减少早期阶段的大量假阳性
            scale_exp_init=1.0  # 重要：影响回归值的尺度
        )

        if self.use_image:
            cnn_channels = getattr(self.backbone, "out_channels_cnn", [256, 256])
            if not isinstance(cnn_channels, list):
                cnn_channels = [cnn_channels, cnn_channels]
            elif isinstance(cnn_channels[0], list):
                cnn_channels = [cnn_channels[0][0], cnn_channels[1][0]]
            
            self.cnn_head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=cnn_channels[:2],
                strides=strides,
                use_gn=True
            )

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            
        self.loss_recorder = {
            'loss_cls': [],
            'loss_reg': [],
            'loss_ctr': [],
            'total_loss': [],
            'iteration': 0
        }
    
    def _record_loss(self, losses):
        self.loss_recorder['iteration'] += 1
        
        for k, v in losses.items():
            if k in self.loss_recorder:
                self.loss_recorder[k].append(v.item())
        
        if self.loss_recorder['iteration'] % 200 == 0:  # 修改为200，减少打印频率
            avg_losses = {}
            for k in ['loss_cls', 'loss_reg', 'loss_ctr', 'total_loss']:
                if len(self.loss_recorder[k]) > 0:
                    avg_losses[k] = sum(self.loss_recorder[k]) / len(self.loss_recorder[k])
                else:
                    avg_losses[k] = 0.0
            
            print(f"[LOSS] Iter: {self.loss_recorder['iteration']}, "
                  f"Cls: {avg_losses['loss_cls']:.4f}, Reg: {avg_losses['loss_reg']:.4f}, "
                  f"Ctr: {avg_losses['loss_ctr']:.4f}, Total: {avg_losses['total_loss']:.4f}")
            
            for k in ['loss_cls', 'loss_reg', 'loss_ctr', 'total_loss']:
                self.loss_recorder[k] = []
    
    def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
        """将边界框转换为FCOS格式"""
        targets = []
        
        # 检查输入有效性
        if bbox is None or bbox.numel() == 0 or bbox_batch is None or bbox_batch.numel() == 0:
            device = self.head.cls_preds[0].weight.device if hasattr(self, 'head') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for _ in range(num_graphs):
                targets.append(torch.zeros(0, 5, device=device if bbox is None else bbox.device))
            return targets
        
        # 仅在迭代点整除200时打印
        should_print = self.debug_logger.iteration % self.debug_logger.print_interval == 0
        
        # 新增: 打印初始边界框信息
        if should_print:
            print(f"[DEBUG] 原始边界框: shape={bbox.shape}, 范围=[{bbox.min().item():.2f}, {bbox.max().item():.2f}]")
            if bbox.shape[0] > 0 and bbox.shape[1] >= 5:
                print(f"[DEBUG] 类别ID范围: [{bbox[:, 4].min().item():.2f}, {bbox[:, 4].max().item():.2f}]")
                print(f"[DEBUG] 高度范围: [{bbox[:, 2].min().item():.2f}, {bbox[:, 2].max().item():.2f}]")
                print(f"[DEBUG] 宽度范围: [{bbox[:, 3].min().item():.2f}, {bbox[:, 3].max().item():.2f}]")
        
        # 新增: 修复高度为0的问题
        if bbox.shape[1] >= 5:
            zero_height_mask = bbox[:, 2] == 0
            if zero_height_mask.any():
                if should_print:
                    print(f"[WARN] 发现{zero_height_mask.sum().item()}个高度为0的框，使用宽度值替代")
                # 创建副本以避免修改原始张量
                bbox = bbox.clone()
                # 使用宽度作为高度的估计值
                bbox[zero_height_mask, 2] = bbox[zero_height_mask, 3]
        
        # 新增: 修复类别ID异常值问题
        if bbox.shape[1] >= 5:
            abnormal_cls_mask = bbox[:, 4] > 10  # 假设类别ID不会超过10
            if abnormal_cls_mask.any():
                if should_print:
                    print(f"[WARN] 发现{abnormal_cls_mask.sum().item()}个异常类别ID (>10)，设置为类别0")
                # 创建副本以避免修改原始张量
                bbox = bbox.clone()
                # 将异常类别ID设为0
                bbox[abnormal_cls_mask, 4] = 0
        
        # 为每个批次处理边界框
        for batch_idx in range(num_graphs):
            # 获取当前批次的边界框
            batch_mask = bbox_batch == batch_idx
            batch_bboxes = bbox[batch_mask]
            
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
                
                # 新增: 确保类别ID是整数
                cls = cls.long()
                
                # 检查是否为归一化坐标
                max_x = x_tl.max().item()
                max_y = y_tl.max().item()
                min_x = x_tl.min().item()
                min_y = y_tl.min().item()
                
                # 是否为归一化坐标
                is_normalized = max_x <= 1.0 and max_y <= 1.0 and min_x >= 0.0 and min_y >= 0.0
                
                # 如果坐标明显小于图像尺寸，可能是归一化坐标，放大到绝对尺寸
                if is_normalized:
                    x_tl = x_tl * self.width
                    y_tl = y_tl * self.height
                    width = width * self.width
                    height = height * self.height
                
                # 新增: 确保高度和宽度为正值
                height = torch.clamp(height, min=1.0)
                width = torch.clamp(width, min=1.0)
                
                # 过滤异常大的边界框
                max_reasonable_size = max(self.width, self.height) * 2  # 合理的最大尺寸
                valid_size_mask = (width < max_reasonable_size) & (height < max_reasonable_size)
                
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
                
                # 新增: 打印转换后的统计信息，仅在特定迭代次数
                if should_print and (batch_idx == 0 or batch_bboxes.shape[0] > 0):
                    print(f"[DEBUG] Batch {batch_idx}: 有效框 {x_center.shape[0]}/{batch_bboxes.shape[0]}")
                    if x_center.shape[0] > 0:
                        print(f"  - 类别分布: {torch.bincount(cls.long(), minlength=2)}")
                        print(f"  - 中心点 x: [{x_center.min().item():.1f}, {x_center.max().item():.1f}], y: [{y_center.min().item():.1f}, {y_center.max().item():.1f}]")
                        print(f"  - 尺寸 w: [{width.min().item():.1f}, {width.max().item():.1f}], h: [{height.min().item():.1f}, {height.max().item():.1f}]")
                
                # 创建FCOS格式的目标: [class_id, center_x, center_y, width, height]
                fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
                targets.append(fcos_target)
            else:
                # 边界框格式不正确，添加空目标
                targets.append(torch.zeros(0, 5, device=bbox.device))
                
        return targets
    
    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if self.training:
            self.debug_logger.log_iteration()
            
            # 判断是否应该在当前迭代打印信息
            should_print = self.debug_logger.iteration % self.debug_logger.print_interval == 0
            is_first_iter = self.debug_logger.iteration <= 1
            
            # 获取并打印GT框的统计信息，仅在特定迭代次数
            if should_print and hasattr(x, 'bbox') and x.bbox is not None:
                print(f"[GT-STATS] Input GT boxes: {x.bbox.shape}, batch indices: {x.bbox_batch.shape}")
                print(f"[GT-STATS] Unique batch indices: {x.bbox_batch.unique().tolist()}")
                print(f"[GT-STATS] Num graphs: {x.num_graphs}")
            
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            # 打印转换后的目标信息，仅在特定迭代次数
            if should_print:
                print(f"[TARGET-INFO] Converted targets: {len(targets)} batches")
                for i, target in enumerate(targets):
                    print(f"  Batch {i}: {target.shape[0]} targets")
                    if target.shape[0] > 0:
                        print(f"    Classes: {target[:, 0].unique().tolist()}")
                        print(f"    Box stats - cx: [{target[:, 1].min():.1f}, {target[:, 1].max():.1f}], "
                              f"cy: [{target[:, 2].min():.1f}, {target[:, 2].max():.1f}], "
                              f"w: [{target[:, 3].min():.1f}, {target[:, 3].max():.1f}], "
                              f"h: [{target[:, 4].min():.1f}, {target[:, 4].max():.1f}]")
            
            features = self.backbone(x)
            
            if is_first_iter:
                print(f"[DEBUG] Backbone features type: {type(features)}")
                if isinstance(features, list) or isinstance(features, tuple):
                    print(f"[DEBUG] Backbone features list length: {len(features)}")
                    for i, feat in enumerate(features):
                        print(f"[DEBUG] Feature {i} type: {type(feat)}")
                        if isinstance(feat, list):
                            print(f"[DEBUG] Feature {i} list length: {len(feat)}")
                            for j, subfeat in enumerate(feat):
                                print(f"[DEBUG] Subfeat {i}.{j} type: {type(subfeat)}")
                                if isinstance(subfeat, torch.Tensor):
                                    print(f"[DEBUG] Subfeat {i}.{j} shape: {subfeat.shape}")
                        elif isinstance(feat, torch.Tensor):
                            print(f"[DEBUG] Feature {i} shape: {feat.shape}")
            
            if isinstance(features, list) or isinstance(features, tuple):
                if len(features) == 1:
                    if isinstance(features[0], list) and len(features[0]) > 1 and isinstance(features[0][1], torch.Tensor):
                        second_feature = features[0][1] if len(features[0]) > 1 else None
                        features = [features[0], [features[0][0], second_feature]]
                    else:
                        features = [features[0], features[0]]
                elif len(features) > 2:
                    features = features[:2]
            
            if is_first_iter:
                print(f"[DEBUG] Processed features:")
                for i, feat in enumerate(features):
                    if isinstance(feat, list):
                        for j, subfeat in enumerate(feat):
                            if isinstance(subfeat, torch.Tensor):
                                print(f"[DEBUG] Feature {i}.{j} shape: {subfeat.shape}")
            
            try:
                if is_first_iter:
                    print(f"[DEBUG] Features before unpack:")
                    for i, feat in enumerate(features):
                        if isinstance(feat, list):
                            for j, subfeat in enumerate(feat):
                                if isinstance(subfeat, torch.Tensor):
                                    print(f"[DEBUG] Feature {i}.{j} shape: {subfeat.shape}")
                
                features_unpacked = unpack_fused_features(features, self.debug_logger)
                
                if is_first_iter:
                    print(f"[DEBUG] Features after unpack: {[f.shape for f in features_unpacked]}")
                
                self.debug_logger.log_feature_shapes(features_unpacked)
                
                if len(features_unpacked) == 1 and features_unpacked[0].dim() == 4:
                    feat1 = features_unpacked[0]
                    feat2 = F.avg_pool2d(feat1, kernel_size=2, stride=2)
                    features_unpacked = [feat1, feat2]
                    if is_first_iter:
                        print(f"[DEBUG] Generated second feature by downsampling: {feat2.shape}")
                
            except Exception as e:
                print(f"[ERROR] Error unpacking features: {e}")
                import traceback
                traceback.print_exc()
                
                dummy_param = next(self.parameters())
                zero_loss = dummy_param.mean() * 0
                
                return {
                    'loss_cls': zero_loss,
                    'loss_reg': zero_loss,
                    'loss_ctr': zero_loss,
                    'total_loss': zero_loss
                }
            
            if not features_unpacked or len(features_unpacked) == 0:
                print("[WARNING] No valid features extracted from backbone, returning zero loss")
                dummy_param = next(self.parameters())
                zero_loss = dummy_param.mean() * 0
                
                return {
                    'loss_cls': zero_loss,
                    'loss_reg': zero_loss,
                    'loss_ctr': zero_loss,
                    'total_loss': zero_loss
                }
            
            try:
                self.head.debug_logger = self.debug_logger
                
                cls_scores, reg_preds, centernesses = self.head(features_unpacked)
                
                if is_first_iter:
                    print(f"[DEBUG] Feature shapes: {[f.shape for f in features_unpacked]}")
                    print(f"[DEBUG] cls_scores length: {len(cls_scores)}, shapes: {[cs.shape for cs in cls_scores if isinstance(cs, torch.Tensor)]}")
                    print(f"[DEBUG] reg_preds length: {len(reg_preds)}, shapes: {[rp.shape for rp in reg_preds if isinstance(rp, torch.Tensor)]}")
                    print(f"[DEBUG] centernesses length: {len(centernesses)}, shapes: {[cn.shape for cn in centernesses if isinstance(cn, torch.Tensor)]}")
                
                # 在损失计算前记录分类预测分布，仅在特定迭代次数
                if should_print:
                    for level_idx, cls_score in enumerate(cls_scores):
                        # 计算每个类别的预测分布
                        cls_prob = torch.sigmoid(cls_score)
                        print(f"[CLS-STATS] Level {level_idx} classification probabilities:")
                        print(f"  Mean: {cls_prob.mean(dim=[0,2,3])}")
                        print(f"  Max: {cls_prob.max(dim=2)[0].max(dim=2)[0].mean(dim=0)}")
                        print(f"  # Preds > 0.5: {(cls_prob > 0.5).sum(dim=[0,2,3])}")
                
                losses = self.head.loss(cls_scores, reg_preds, centernesses, targets)
                
                for k, v in losses.items():
                    if not v.requires_grad:
                        dummy_param = next(self.parameters())
                        losses[k] = v + dummy_param.sum() * 0
                
                self._record_loss(losses)
                
                return losses
            except Exception as e:
                print(f"[ERROR] Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                
                dummy_param = next(self.parameters())
                zero_loss = dummy_param.mean() * 0
                
                return {
                    'loss_cls': zero_loss,
                    'loss_reg': zero_loss,
                    'loss_ctr': zero_loss,
                    'total_loss': zero_loss
                }
        
        else:
            x.reset = reset
            
            features = self.backbone(x)
            
            if isinstance(features, list) or isinstance(features, tuple):
                if len(features) == 1:
                    features = [features[0], features[0]]
                elif len(features) > 2:
                    features = features[:2]
            
            features = unpack_fused_features(features)
            
            if not features or len(features) == 0:
                device = next(self.parameters()).device
                empty_result = []
                for _ in range(x.num_graphs):
                    empty_result.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device),
                        'height': torch.tensor(self.height, device=device),
                        'width': torch.tensor(self.width, device=device)
                    })
                
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    return empty_result, targets
                return empty_result
            
            if len(features) == 1 and features[0].dim() == 4:
                feat1 = features[0]
                feat2 = F.avg_pool2d(feat1, kernel_size=2, stride=2)
                features = [feat1, feat2]
            
            cls_scores, reg_preds, centernesses = self.head(features)
            
            if not cls_scores or len(cls_scores) == 0:
                device = next(self.parameters()).device
                empty_result = []
                for _ in range(x.num_graphs):
                    empty_result.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device),
                        'height': torch.tensor(self.height, device=device),
                        'width': torch.tensor(self.width, device=device)
                    })
                
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    return empty_result, targets
                return empty_result
            
            try:
                # 打印评估阶段的预测分布，在评估阶段保留频繁打印以便于调试
                for level_idx, cls_score in enumerate(cls_scores):
                    cls_prob = torch.sigmoid(cls_score)
                    print(f"[EVAL-CLS] Level {level_idx} classification probabilities:")
                    print(f"  Mean: {cls_prob.mean(dim=[0,2,3])}")
                    print(f"  Max: {cls_prob.max(dim=2)[0].max(dim=2)[0].mean(dim=0)}")
                    print(f"  # Preds > 0.5: {(cls_prob > 0.5).sum(dim=[0,2,3])}")
                
                detections = self.head.get_bboxes(
                    cls_scores, 
                    reg_preds, 
                    centernesses, 
                    score_thr=self.conf_threshold,
                    nms_thr=self.nms_threshold
                )
                
                # 打印检测结果统计，在评估阶段保留频繁打印
                print(f"[DETECTION-STATS] Detection results:")
                for batch_idx, det in enumerate(detections):
                    print(f"  Batch {batch_idx}: {det['boxes'].shape[0]} detections")
                    if det['boxes'].shape[0] > 0:
                        print(f"    Score range: [{det['scores'].min():.3f}, {det['scores'].max():.3f}]")
                        print(f"    Class distribution: {torch.bincount(det['labels'])}")
                        
                        # 详细打印一些检测框
                        max_boxes_to_print = min(3, det['boxes'].shape[0])
                        print(f"    Detection examples (top {max_boxes_to_print}):")
                        scores, indices = det['scores'].sort(descending=True)
                        for i in range(max_boxes_to_print):
                            idx = indices[i]
                            box = det['boxes'][idx]
                            x1, y1, x2, y2 = box.tolist()
                            w, h = x2 - x1, y2 - y1
                            cls = det['labels'][idx].item()
                            score = det['scores'][idx].item()
                            print(f"      Det #{i}: class={cls}, score={score:.3f}, box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], w={w:.1f}, h={h:.1f}")
                
            except Exception as e:
                print(f"Error in get_bboxes: {e}")
                device = next(self.parameters()).device
                empty_result = []
                for _ in range(x.num_graphs):
                    empty_result.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.long, device=device),
                        'height': torch.tensor(self.height, device=device),
                        'width': torch.tensor(self.width, device=device)
                    })
                
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    return empty_result, targets
                return empty_result
            
            formatted_detections = []
            for det in detections:
                boxes = det['boxes']
                scores = det['scores']
                labels = det['labels']
                
                if boxes.size(0) > 0:
                    x1, y1, x2, y2 = boxes.unbind(1)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    boxes = torch.stack([cx, cy, w, h], dim=1)
                
                height_tensor = torch.tensor(self.height, device=boxes.device)
                width_tensor = torch.tensor(self.width, device=boxes.device)
                
                formatted_detections.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                    'height': height_tensor,
                    'width': width_tensor
                })
            
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                return formatted_detections, targets
            return formatted_detections



            
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# from torch_geometric.data import Data
# from dagr.model.networks.net import Net
# from dagr.model.networks.fcos_head import FCOSHead
# from dagr.model.utils import (
#     convert_to_training_format,
#     convert_to_evaluation_format,
#     init_subnetwork
# )


# class DebugLogger:
#     def __init__(self, print_interval=100):
#         self.print_interval = print_interval
#         self.iteration = 0
#         self.feature_stats_printed = False
#         self.stats = {
#             'feature_shapes': [],
#             'reg_pred_raw': {'min': [], 'max': [], 'mean': []},
#             'reg_pred_scaled': {'min': [], 'max': [], 'mean': []},
#             'pred_boxes': {'min': [], 'max': []},
#             'target_boxes': {'min': [], 'max': []}
#         }
    
#     def log_feature_shapes(self, features_unpacked):
#         """记录特征形状信息"""
#         # 只在第一次迭代时记录特征形状
#         if not self.feature_stats_printed:
#             self.stats['feature_shapes'] = [f.shape for f in features_unpacked]
#             self.feature_stats_printed = True
    
#     def log_reg_pred(self, raw_pred, scaled_pred):
#         """记录回归预测的统计信息"""
#         self.stats['reg_pred_raw']['min'].append(raw_pred.min().item())
#         self.stats['reg_pred_raw']['max'].append(raw_pred.max().item())
#         self.stats['reg_pred_raw']['mean'].append(raw_pred.mean().item())
        
#         self.stats['reg_pred_scaled']['min'].append(scaled_pred.min().item())
#         self.stats['reg_pred_scaled']['max'].append(scaled_pred.max().item())
#         self.stats['reg_pred_scaled']['mean'].append(scaled_pred.mean().item())
    
#     def log_boxes(self, pred_boxes, target_boxes):
#         """记录边界框的统计信息"""
#         if pred_boxes.numel() > 0:
#             self.stats['pred_boxes']['min'].append(pred_boxes.min().item())
#             self.stats['pred_boxes']['max'].append(pred_boxes.max().item())
        
#         if target_boxes.numel() > 0:
#             self.stats['target_boxes']['min'].append(target_boxes.min().item())
#             self.stats['target_boxes']['max'].append(target_boxes.max().item())
    
#     def log_iteration(self):
#         """递增迭代计数"""
#         self.iteration += 1
        
#         # 每print_interval次迭代打印一次统计信息
#         if self.iteration % self.print_interval == 0:
#             self._print_stats()
#             self._reset_stats()
    
#     def _print_stats(self):
#         """打印统计信息"""
#         # 打印特征形状信息（如果有）
#         if self.stats['feature_shapes']:
#             print(f"[DEBUG-{self.iteration}] Feature shapes: {self.stats['feature_shapes']}")
        
#         # 计算回归预测统计值的平均值
#         reg_raw_min = sum(self.stats['reg_pred_raw']['min']) / len(self.stats['reg_pred_raw']['min']) if self.stats['reg_pred_raw']['min'] else float('nan')
#         reg_raw_max = sum(self.stats['reg_pred_raw']['max']) / len(self.stats['reg_pred_raw']['max']) if self.stats['reg_pred_raw']['max'] else float('nan')
#         reg_raw_mean = sum(self.stats['reg_pred_raw']['mean']) / len(self.stats['reg_pred_raw']['mean']) if self.stats['reg_pred_raw']['mean'] else float('nan')
        
#         reg_scaled_min = sum(self.stats['reg_pred_scaled']['min']) / len(self.stats['reg_pred_scaled']['min']) if self.stats['reg_pred_scaled']['min'] else float('nan')
#         reg_scaled_max = sum(self.stats['reg_pred_scaled']['max']) / len(self.stats['reg_pred_scaled']['max']) if self.stats['reg_pred_scaled']['max'] else float('nan')
#         reg_scaled_mean = sum(self.stats['reg_pred_scaled']['mean']) / len(self.stats['reg_pred_scaled']['mean']) if self.stats['reg_pred_scaled']['mean'] else float('nan')
        
#         # 打印回归预测统计
#         if not math.isnan(reg_raw_min):
#             print(f"[DEBUG-{self.iteration}] Reg pred raw stats: min={reg_raw_min:.4f}, max={reg_raw_max:.4f}, mean={reg_raw_mean:.4f}")
        
#         if not math.isnan(reg_scaled_min):
#             print(f"[DEBUG-{self.iteration}] Reg pred after scale exp: min={reg_scaled_min:.4f}, max={reg_scaled_max:.4f}, mean={reg_scaled_mean:.4f}")
        
#         # 计算边界框统计值的平均值
#         pred_boxes_min = sum(self.stats['pred_boxes']['min']) / len(self.stats['pred_boxes']['min']) if self.stats['pred_boxes']['min'] else float('nan')
#         pred_boxes_max = sum(self.stats['pred_boxes']['max']) / len(self.stats['pred_boxes']['max']) if self.stats['pred_boxes']['max'] else float('nan')
        
#         target_boxes_min = sum(self.stats['target_boxes']['min']) / len(self.stats['target_boxes']['min']) if self.stats['target_boxes']['min'] else float('nan')
#         target_boxes_max = sum(self.stats['target_boxes']['max']) / len(self.stats['target_boxes']['max']) if self.stats['target_boxes']['max'] else float('nan')
        
#         # 打印边界框统计
#         if not math.isnan(pred_boxes_min):
#             print(f"[DEBUG-{self.iteration}] Pred boxes stats: min={pred_boxes_min:.4f}, max={pred_boxes_max:.4f}")
        
#         if not math.isnan(target_boxes_min):
#             print(f"[DEBUG-{self.iteration}] Target boxes stats: min={target_boxes_min:.4f}, max={target_boxes_max:.4f}")
    
#     def _reset_stats(self):
#         """重置统计信息"""
#         for stat_type in ['reg_pred_raw', 'reg_pred_scaled', 'pred_boxes', 'target_boxes']:
#             for metric in self.stats[stat_type]:
#                 self.stats[stat_type][metric] = []


# def unpack_fused_features(fused_feat, debug_logger=None):
#     """
#     将融合特征解包成标准张量
    
#     Args:
#         fused_feat: 融合特征
#         debug_logger: 调试记录器，用于控制输出
    
#     Returns:
#         features_tensor: 解包后的特征张量列表
#     """
#     features_tensor = []
    
#     # 只在第一次调用时输出详细信息
#     verbose = debug_logger is None or debug_logger.iteration <= 1
    
#     if verbose:
#         print(f"[DEBUG] unpack_fused_features: received {len(fused_feat)} features")
    
#     for i, f in enumerate(fused_feat):
#         try:
#             # 如果特征是列表，递归处理
#             if isinstance(f, list):
#                 if verbose:
#                     print(f"[DEBUG] Feature {i} is a list with {len(f)} elements")
#                     print(f"[DEBUG] List contents types: {[type(item) for item in f]}")
                
#                 # 如果列表元素是张量，直接处理
#                 if len(f) > 0 and all(isinstance(item, torch.Tensor) for item in f):
#                     if verbose:
#                         print(f"[DEBUG] All list elements are tensors, shapes: {[item.shape for item in f]}")
                    
#                     # 根据张量的维度处理
#                     if f[0].dim() == 4:  # [B, C, H, W]
#                         # 如果已经是4D张量，直接使用
#                         features_tensor.append(f[0])
#                         if len(f) > 1 and f[1].dim() == 4:
#                             features_tensor.append(f[1])
#                         if verbose:
#                             print(f"[DEBUG] Added 4D tensor from list with shape: {f[0].shape}")
#                     elif f[0].dim() == 3:  # [C, H, W]
#                         # 如果是3D张量，增加批次维度
#                         features_tensor.append(f[0].unsqueeze(0))
#                         if len(f) > 1 and f[1].dim() == 3:
#                             features_tensor.append(f[1].unsqueeze(0))
#                         if verbose:
#                             print(f"[DEBUG] Added 3D tensor from list with shape: {f[0].unsqueeze(0).shape}")
#                     elif f[0].dim() == 2:  # [N, C] or [C, N]
#                         # 如果是2D张量，需要猜测正确的形状
#                         if len(f) > 1 and f[1].numel() == 2:  # 可能包含高度和宽度信息
#                             try:
#                                 height, width = f[1].tolist()
#                                 if verbose:
#                                     print(f"[DEBUG] Found height and width in list: {height}x{width}")
                                
#                                 # 根据高度和宽度重塑张量
#                                 nodes = f[0]
#                                 if nodes.shape[0] * nodes.shape[1] == height * width:
#                                     # 如果张量大小和高宽匹配，重塑为[C, H, W]
#                                     reshaped = nodes.reshape(nodes.shape[0], height, width)
#                                 else:
#                                     # 否则假设是[N, C]，需要重塑为[C, H, W]
#                                     if nodes.shape[0] > height * width:
#                                         # 裁剪
#                                         nodes = nodes[:height * width]
#                                     elif nodes.shape[0] < height * width:
#                                         # 填充
#                                         padded = torch.zeros(height * width, nodes.shape[1], device=nodes.device)
#                                         padded[:nodes.shape[0]] = nodes
#                                         nodes = padded
                                    
#                                     reshaped = nodes.reshape(height, width, nodes.shape[1]).permute(2, 0, 1)
                                
#                                 # 添加批次维度
#                                 features_tensor.append(reshaped.unsqueeze(0))
#                                 if verbose:
#                                     print(f"[DEBUG] Added reshaped tensor with shape: {reshaped.unsqueeze(0).shape}")
#                             except Exception as e:
#                                 if verbose:
#                                     print(f"[ERROR] Failed to reshape tensor using list dimensions: {e}")
#                                 # 回退到基本处理
#                                 nodes = f[0]
#                                 sqrt_nodes = int(math.sqrt(nodes.shape[0]))
#                                 if sqrt_nodes * sqrt_nodes == nodes.shape[0]:
#                                     # 完美平方数
#                                     reshaped = nodes.reshape(sqrt_nodes, sqrt_nodes, nodes.shape[1]).permute(2, 0, 1)
#                                     features_tensor.append(reshaped.unsqueeze(0))
#                                     if verbose:
#                                         print(f"[DEBUG] Added square reshaped tensor with shape: {reshaped.unsqueeze(0).shape}")
#                                 else:
#                                     if verbose:
#                                         print(f"[WARNING] Could not determine proper reshaping for tensor")
#                         else:
#                             # 其他处理逻辑...（省略详细代码以节省空间）
#                             # 仅在必要时输出调试信息
#                             pass
#                 # 否则，尝试递归处理子列表
#                 else:
#                     # 递归处理子列表...（省略详细代码以节省空间）
#                     pass
#             elif hasattr(f, "x") and isinstance(f.x, torch.Tensor):
#                 # PyG数据处理...（省略详细代码以节省空间）
#                 # 仅在必要时输出调试信息
#                 pass
#             elif isinstance(f, torch.Tensor):
#                 # 张量处理...（省略详细代码以节省空间）
#                 # 仅在必要时输出调试信息
#                 pass
#             else:
#                 if verbose:
#                     print(f"[WARNING] Feature {i} is neither Data nor Tensor or List: {type(f)}")
#         except Exception as e:
#             # 只在出错时输出详细信息
#             print(f"[ERROR] Failed to process feature {i}: {e}")
#             import traceback
#             traceback.print_exc()
    
#     if verbose:
#         print(f"[DEBUG] Returning {len(features_tensor)} processed features")
#     return features_tensor


# class DAGR_FCOS(nn.Module):
#     def __init__(self, args, height, width):
#         super().__init__()
#         self.conf_threshold = args.score_threshold if hasattr(args, 'score_threshold') else 0.05
#         self.nms_threshold = args.nms_iou_threshold if hasattr(args, 'nms_iou_threshold') else 0.6
#         self.height = height
#         self.width = width
#         self.args = args
        
#         # 创建调试记录器
#         self.debug_logger = DebugLogger(print_interval=100)

#         # 原DAGR的backbone
#         self.backbone = Net(args, height=height, width=width)
#         self.use_image = getattr(args, "use_image", False)
#         self.no_events = getattr(args, "no_events", False)
#         self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

#         # 使用FCOS头部
#         strides = [8, 16]  # 固定使用两个尺度的stride
        
#         # 直接使用固定的输入通道数，不要依赖backbone的out_channels属性
#         in_channels = [256, 256]  # 明确设置为256通道
        
#         print(f"[INFO] Using in_channels: {in_channels} for FCOS head")
        
#         # 确保通道数正确，创建新的头部
#         self.head = FCOSHead(
#             num_classes=self.backbone.num_classes,
#             in_channels=in_channels,
#             strides=strides,
#             use_gn=True
#         )

#         # 添加CNN头部处理图像特征（如果需要）
#         if self.use_image:
#             cnn_channels = getattr(self.backbone, "out_channels_cnn", [256, 256])
#             # 确保是单个整数值而不是列表
#             if not isinstance(cnn_channels, list):
#                 cnn_channels = [cnn_channels, cnn_channels]
#             elif isinstance(cnn_channels[0], list):
#                 cnn_channels = [cnn_channels[0][0], cnn_channels[1][0]]
            
#             self.cnn_head = FCOSHead(
#                 num_classes=self.backbone.num_classes,
#                 in_channels=cnn_channels[:2],  # 只使用前两个通道
#                 strides=strides,
#                 use_gn=True
#             )

#         # 加载预训练的图像网络（如果提供）
#         if "img_net_checkpoint" in args:
#             state_dict = torch.load(args.img_net_checkpoint)
#             init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            
#         # 添加损失记录器，用于记录并周期性输出损失
#         self.loss_recorder = {
#             'loss_cls': [],
#             'loss_reg': [],
#             'loss_ctr': [],
#             'total_loss': [],
#             'iteration': 0
#         }
    
#     def _record_loss(self, losses):
#         """记录损失，并在每100次迭代后打印一次"""
#         # 更新迭代计数器
#         self.loss_recorder['iteration'] += 1
        
#         # 记录各项损失
#         for k, v in losses.items():
#             if k in self.loss_recorder:
#                 self.loss_recorder[k].append(v.item())
        
#         # 每100次迭代打印一次损失
#         if self.loss_recorder['iteration'] % 100 == 0:
#             # 计算平均损失
#             avg_losses = {}
#             for k in ['loss_cls', 'loss_reg', 'loss_ctr', 'total_loss']:
#                 if len(self.loss_recorder[k]) > 0:
#                     avg_losses[k] = sum(self.loss_recorder[k]) / len(self.loss_recorder[k])
#                 else:
#                     avg_losses[k] = 0.0
            
#             # 打印损失信息
#             print(f"[LOSS] Iter: {self.loss_recorder['iteration']}, "
#                   f"Cls: {avg_losses['loss_cls']:.4f}, Reg: {avg_losses['loss_reg']:.4f}, "
#                   f"Ctr: {avg_losses['loss_ctr']:.4f}, Total: {avg_losses['total_loss']:.4f}")
            
#             # 清空记录器
#             for k in ['loss_cls', 'loss_reg', 'loss_ctr', 'total_loss']:
#                 self.loss_recorder[k] = []
    
#     def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
#         """将边界框转换为FCOS格式"""
#         targets = []
        
#         # 检查输入有效性
#         if bbox is None or bbox.numel() == 0 or bbox_batch is None or bbox_batch.numel() == 0:
#             # 创建空目标
#             device = self.head.cls_preds[0].weight.device if hasattr(self, 'head') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             for _ in range(num_graphs):
#                 targets.append(torch.zeros(0, 5, device=device if bbox is None else bbox.device))
#             return targets
        
#         # 为每个批次处理边界框
#         for batch_idx in range(num_graphs):
#             # 获取当前批次的边界框
#             batch_mask = bbox_batch == batch_idx
#             batch_bboxes = bbox[batch_mask]
            
#             # 如果当前批次没有边界框，添加空目标
#             if batch_bboxes.numel() == 0:
#                 targets.append(torch.zeros(0, 5, device=bbox.device))
#                 continue
            
#             # 解析边界框信息
#             if batch_bboxes.shape[1] >= 5:  # 确保有足够的列
#                 x_tl = batch_bboxes[:, 0]    # 左上角x坐标
#                 y_tl = batch_bboxes[:, 1]    # 左上角y坐标
#                 height = batch_bboxes[:, 2]  # 高度
#                 width = batch_bboxes[:, 3]   # 宽度
#                 cls = batch_bboxes[:, 4]     # 类别
                
#                 # 检查是否为归一化坐标
#                 max_x = x_tl.max().item()
#                 max_y = y_tl.max().item()
#                 min_x = x_tl.min().item()
#                 min_y = y_tl.min().item()
                
#                 # 如果坐标明显小于图像尺寸，可能是归一化坐标，放大到绝对尺寸
#                 if max_x <= 1.0 and max_y <= 1.0 and min_x >= 0.0 and min_y >= 0.0:
#                     x_tl = x_tl * self.width
#                     y_tl = y_tl * self.height
#                     width = width * self.width
#                     height = height * self.height
                
#                 # 过滤异常大的边界框
#                 max_reasonable_size = max(self.width, self.height) * 2  # 合理的最大尺寸
#                 valid_size_mask = (width < max_reasonable_size) & (height < max_reasonable_size)
                
#                 # 应用有效性过滤
#                 x_tl = x_tl[valid_size_mask]
#                 y_tl = y_tl[valid_size_mask]
#                 height = height[valid_size_mask]
#                 width = width[valid_size_mask]
#                 cls = cls[valid_size_mask]
                
#                 # 计算面积以检查有效性
#                 areas = width * height
#                 valid_mask = areas > 0
                
#                 if valid_mask.sum() == 0:
#                     targets.append(torch.zeros(0, 5, device=bbox.device))
#                     continue
                    
#                 # 应用面积有效性过滤
#                 x_tl = x_tl[valid_mask]
#                 y_tl = y_tl[valid_mask]
#                 height = height[valid_mask]
#                 width = width[valid_mask]
#                 cls = cls[valid_mask]
                
#                 # 从左上角转换为中心点（FCOS需要中心点格式）
#                 x_center = x_tl + width / 2
#                 y_center = y_tl + height / 2
                
#                 # 检查坐标是否在合理范围内
#                 x_center = torch.clamp(x_center, 0, self.width)
#                 y_center = torch.clamp(y_center, 0, self.height)
                
#                 # 创建FCOS格式的目标: [class_id, center_x, center_y, width, height]
#                 fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
#                 targets.append(fcos_target)
#             else:
#                 # 边界框格式不正确，添加空目标
#                 targets.append(torch.zeros(0, 5, device=bbox.device))
                
#         return targets
    
#     def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
#         """前向传播方法"""
#         # 训练模式
#         if self.training:
#             # 递增迭代计数
#             self.debug_logger.log_iteration()
            
#             # 转换标签为FCOS格式
#             targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
#             # 获取backbone特征
#             features = self.backbone(x)
            
#             # 只在第一次迭代时输出backbone特征信息
#             if self.debug_logger.iteration <= 1:
#                 print(f"[DEBUG] Backbone features type: {type(features)}")
#                 if isinstance(features, list) or isinstance(features, tuple):
#                     print(f"[DEBUG] Backbone features list length: {len(features)}")
#                     for i, feat in enumerate(features):
#                         print(f"[DEBUG] Feature {i} type: {type(feat)}")
#                         if isinstance(feat, list):
#                             print(f"[DEBUG] Feature {i} list length: {len(feat)}")
#                             for j, subfeat in enumerate(feat):
#                                 print(f"[DEBUG] Subfeat {i}.{j} type: {type(subfeat)}")
#                                 if isinstance(subfeat, torch.Tensor):
#                                     print(f"[DEBUG] Subfeat {i}.{j} shape: {subfeat.shape}")
#                         elif isinstance(feat, torch.Tensor):
#                             print(f"[DEBUG] Feature {i} shape: {feat.shape}")
            
#             # 处理特征
#             if isinstance(features, list) or isinstance(features, tuple):
#                 if len(features) == 1:
#                     # 检查第一个元素是否是列表
#                     if isinstance(features[0], list) and len(features[0]) > 1 and isinstance(features[0][1], torch.Tensor):
#                         # 提取出第二个特征，假设第一个元素是列表
#                         second_feature = features[0][1] if len(features[0]) > 1 else None
#                         features = [features[0], [features[0][0], second_feature]]
#                     else:
#                         features = [features[0], features[0]]  # 复制第一个特征
#                 elif len(features) > 2:
#                     features = features[:2]  # 只使用前两个特征
            
#             # 只在第一次迭代时分析特征
#             if self.debug_logger.iteration <= 1:
#                 print(f"[DEBUG] Processed features:")
#                 for i, feat in enumerate(features):
#                     if isinstance(feat, list):
#                         for j, subfeat in enumerate(feat):
#                             if isinstance(subfeat, torch.Tensor):
#                                 print(f"[DEBUG] Feature {i}.{j} shape: {subfeat.shape}")
            
#             # 只在第一次迭代时打印GT框信息
#             if self.debug_logger.iteration <= 1 and hasattr(x, 'bbox') and x.bbox is not None:
#                 batch_size = x.num_graphs
#                 print(f"[INFO] GT Boxes Info - Batch Size: {batch_size}")
#                 for batch_idx in range(min(batch_size, 2)):  # 只打印前两个batch的信息
#                     batch_mask = x.bbox_batch == batch_idx
#                     batch_boxes = x.bbox[batch_mask]
#                     if batch_boxes.numel() > 0:
#                         num_boxes = batch_boxes.shape[0]
#                         print(f"[INFO] Batch {batch_idx}: {num_boxes} boxes")
#                         if num_boxes > 0:
#                             # 打印前3个框的信息
#                             for box_idx in range(min(num_boxes, 3)):
#                                 box = batch_boxes[box_idx]
#                                 if box.shape[0] >= 5:
#                                     x_tl, y_tl, h, w, cls_id = box
#                                     print(f"[INFO]   Box {box_idx}: [x_tl={x_tl:.2f}, y_tl={y_tl:.2f}, h={h:.2f}, w={w:.2f}, cls={cls_id}]")
            
#             # 将PyG数据转换为标准张量
#             try:
#                 # 只在第一次迭代时打印特征形状信息
#                 if self.debug_logger.iteration <= 1:
#                     print(f"[DEBUG] Features before unpack:")
#                     for i, feat in enumerate(features):
#                         if isinstance(feat, list):
#                             for j, subfeat in enumerate(feat):
#                                 if isinstance(subfeat, torch.Tensor):
#                                     print(f"[DEBUG] Feature {i}.{j} shape: {subfeat.shape}")
                
#                 # 使用调试记录器来控制输出
#                 features_unpacked = unpack_fused_features(features, self.debug_logger)
                
#                 # 只在第一次迭代时打印解包后的特征形状
#                 if self.debug_logger.iteration <= 1:
#                     print(f"[DEBUG] Features after unpack: {[f.shape for f in features_unpacked]}")
                
#                 # 记录特征形状信息
#                 self.debug_logger.log_feature_shapes(features_unpacked)
                
#                 # 确保有两个特征
#                 if len(features_unpacked) == 1 and features_unpacked[0].dim() == 4:
#                     # 如果只有一个特征，从第一个特征中复制并下采样生成第二个特征
#                     feat1 = features_unpacked[0]
#                     feat2 = F.avg_pool2d(feat1, kernel_size=2, stride=2)
#                     features_unpacked = [feat1, feat2]
#                     if self.debug_logger.iteration <= 1:
#                         print(f"[DEBUG] Generated second feature by downsampling: {feat2.shape}")
                
#             except Exception as e:
#                 print(f"[ERROR] Error unpacking features: {e}")
#                 import traceback
#                 traceback.print_exc()
                
#                 # 创建一个需要梯度的零损失
#                 dummy_param = next(self.parameters())
#                 zero_loss = dummy_param.mean() * 0
                
#                 return {
#                     'loss_cls': zero_loss,
#                     'loss_reg': zero_loss,
#                     'loss_ctr': zero_loss,
#                     'total_loss': zero_loss
#                 }
            
#             # 检查特征是否有效
#             if not features_unpacked or len(features_unpacked) == 0:
#                 print("[WARNING] No valid features extracted from backbone, returning zero loss")
#                 # 创建一个需要梯度的零损失
#                 dummy_param = next(self.parameters())
#                 zero_loss = dummy_param.mean() * 0
                
#                 return {
#                     'loss_cls': zero_loss,
#                     'loss_reg': zero_loss,
#                     'loss_ctr': zero_loss,
#                     'total_loss': zero_loss
#                 }
            
#             # 使用FCOS头部进行预测
#             try:
#                 # 传递调试记录器给头部
#                 self.head.debug_logger = self.debug_logger
                
#                 cls_scores, reg_preds, centernesses = self.head(features_unpacked)
                
#                 # 只在第一次迭代时输出调试信息
#                 if self.debug_logger.iteration <= 1:
#                     print(f"[DEBUG] Feature shapes: {[f.shape for f in features_unpacked]}")
#                     print(f"[DEBUG] cls_scores length: {len(cls_scores)}, shapes: {[cs.shape for cs in cls_scores if isinstance(cs, torch.Tensor)]}")
#                     print(f"[DEBUG] reg_preds length: {len(reg_preds)}, shapes: {[rp.shape for rp in reg_preds if isinstance(rp, torch.Tensor)]}")
#                     print(f"[DEBUG] centernesses length: {len(centernesses)}, shapes: {[cn.shape for cn in centernesses if isinstance(cn, torch.Tensor)]}")
                
#                 # 计算损失
#                 losses = self.head.loss(cls_scores, reg_preds, centernesses, targets)
                
#                 # 确保损失有梯度
#                 for k, v in losses.items():
#                     if not v.requires_grad:
#                         dummy_param = next(self.parameters())
#                         losses[k] = v + dummy_param.sum() * 0
                
#                 # 记录损失值用于后续打印
#                 self._record_loss(losses)
                
#                 return losses
#             except Exception as e:
#                 print(f"[ERROR] Error during forward pass: {e}")
#                 import traceback
#                 traceback.print_exc()
                
#                 # 创建一个需要梯度的零损失
#                 dummy_param = next(self.parameters())
#                 zero_loss = dummy_param.mean() * 0
                
#                 return {
#                     'loss_cls': zero_loss,
#                     'loss_reg': zero_loss,
#                     'loss_ctr': zero_loss,
#                     'total_loss': zero_loss
#                 }
        
#         # 评估模式
#         else:
#             x.reset = reset
            
#             # 获取backbone特征
#             features = self.backbone(x)
            
#             # 处理特征
#             if isinstance(features, list) or isinstance(features, tuple):
#                 if len(features) == 1:
#                     features = [features[0], features[0]]  # 复制第一个特征
#                 elif len(features) > 2:
#                     features = features[:2]  # 只使用前两个特征
            
#             # 将PyG数据转换为标准张量
#             features = unpack_fused_features(features)
            
#             # 检查特征是否有效
#             if not features or len(features) == 0:
#                 device = next(self.parameters()).device
#                 # 返回空结果
#                 empty_result = []
#                 for _ in range(x.num_graphs):
#                     empty_result.append({
#                         'boxes': torch.zeros((0, 4), device=device),
#                         'scores': torch.zeros(0, device=device),
#                         'labels': torch.zeros(0, dtype=torch.long, device=device),
#                         'height': torch.tensor(self.height, device=device),
#                         'width': torch.tensor(self.width, device=device)
#                     })
                
#                 if return_targets and hasattr(x, 'bbox'):
#                     targets = convert_to_evaluation_format(x)
#                     return empty_result, targets
#                 return empty_result
            
#             # 确保有两个特征
#             if len(features) == 1 and features[0].dim() == 4:
#                 # 如果只有一个特征，从第一个特征中复制并下采样生成第二个特征
#                 feat1 = features[0]
#                 feat2 = F.avg_pool2d(feat1, kernel_size=2, stride=2)
#                 features = [feat1, feat2]
            
#             # 使用FCOS头部进行预测
#             cls_scores, reg_preds, centernesses = self.head(features)
            
#             # 检查预测结果是否有效
#             if not cls_scores or len(cls_scores) == 0:
#                 device = next(self.parameters()).device
#                 # 返回空结果
#                 empty_result = []
#                 for _ in range(x.num_graphs):
#                     empty_result.append({
#                         'boxes': torch.zeros((0, 4), device=device),
#                         'scores': torch.zeros(0, device=device),
#                         'labels': torch.zeros(0, dtype=torch.long, device=device),
#                         'height': torch.tensor(self.height, device=device),
#                         'width': torch.tensor(self.width, device=device)
#                     })
                
#                 if return_targets and hasattr(x, 'bbox'):
#                     targets = convert_to_evaluation_format(x)
#                     return empty_result, targets
#                 return empty_result
            
#             # 解码预测结果
#             try:
#                 detections = self.head.get_bboxes(
#                     cls_scores, 
#                     reg_preds, 
#                     centernesses, 
#                     score_thr=self.conf_threshold,
#                     nms_thr=self.nms_threshold
#                 )
#             except Exception as e:
#                 print(f"Error in get_bboxes: {e}")
#                 device = next(self.parameters()).device
#                 # 返回空结果
#                 empty_result = []
#                 for _ in range(x.num_graphs):
#                     empty_result.append({
#                         'boxes': torch.zeros((0, 4), device=device),
#                         'scores': torch.zeros(0, device=device),
#                         'labels': torch.zeros(0, dtype=torch.long, device=device),
#                         'height': torch.tensor(self.height, device=device),
#                         'width': torch.tensor(self.width, device=device)
#                     })
                
#                 if return_targets and hasattr(x, 'bbox'):
#                     targets = convert_to_evaluation_format(x)
#                     return empty_result, targets
#                 return empty_result
            
#             # 适配DAGR的输出格式
#             formatted_detections = []
#             for det in detections:
#                 boxes = det['boxes']  # 已经是张量
#                 scores = det['scores']  # 已经是张量
#                 labels = det['labels']  # 已经是张量
                
#                 # 将边界框从XYXY格式转换为XYWH格式
#                 if boxes.size(0) > 0:
#                     # 提取坐标
#                     x1, y1, x2, y2 = boxes.unbind(1)
#                     # 计算中心点和宽高
#                     cx = (x1 + x2) / 2
#                     cy = (y1 + y2) / 2
#                     w = x2 - x1
#                     h = y2 - y1
#                     boxes = torch.stack([cx, cy, w, h], dim=1)
                
#                 # 确保所有值都是张量，不是标量或整数
#                 height_tensor = torch.tensor(self.height, device=boxes.device)
#                 width_tensor = torch.tensor(self.width, device=boxes.device)
                
#                 formatted_detections.append({
#                     'boxes': boxes,
#                     'scores': scores,
#                     'labels': labels,
#                     'height': height_tensor,
#                     'width': width_tensor
#                 })
            
#             if return_targets and hasattr(x, 'bbox'):
#                 targets = convert_to_evaluation_format(x)
#                 return formatted_detections, targets
#             return formatted_detections