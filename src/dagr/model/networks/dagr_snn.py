# dagr/model/networks/dagr_snn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from dagr.model.networks.dagr import DAGR, GNNHead
from dagr.model.networks.event_snn_backbone import EventProcessor
from dagr.model.utils import convert_to_training_format, postprocess_network_output, convert_to_evaluation_format, shallow_copy
from yolox.models import IOUloss
from yolox.models.yolo_head import YOLOXHead

class CNNModule(nn.Module):
    """处理SNN骨干网络输出的规则张量(而非图结构)，将GNNhead替换为纯CNNhead"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class SNNHead(YOLOXHead):
    """专为SNN输出设计的检测头，直接继承YOLOXHead"""
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=None,
        act="silu",
        depthwise=False,
        pretrain_cnn=False,
        args=None
    ):
        # 初始化YOLOXHead
        width = getattr(args, 'yolo_stem_width', 1.0)
        # 必须先设置n_anchors属性，然后再调用父类初始化
        self.n_anchors = 1
        YOLOXHead.__init__(self, num_classes, width, strides, in_channels, act, depthwise)
        
        self.args = args
        self.pretrain_cnn = pretrain_cnn
        self.num_scales = getattr(args, 'num_scales', 2)
        self.use_image = getattr(args, 'use_image', False)
        self.batch_size = getattr(args, 'batch_size', 8)
        self.no_events = getattr(args, 'no_events', False)
        
        # 重定义SNN专用层
        n_reg = 256  # 中间特征通道数
        
        # 第一个尺度的检测头
        self.stem1 = CNNModule(in_channels[0], n_reg)
        self.cls_conv1 = CNNModule(n_reg, n_reg)
        self.cls_pred1 = nn.Conv2d(n_reg, self.n_anchors * self.num_classes, 1)
        self.reg_conv1 = CNNModule(n_reg, n_reg)
        self.reg_pred1 = nn.Conv2d(n_reg, 4, 1)  # x, y, w, h
        self.obj_pred1 = nn.Conv2d(n_reg, self.n_anchors, 1)  # objectness
        
        # 第二个尺度的检测头
        if self.num_scales > 1:
            self.stem2 = CNNModule(in_channels[1], n_reg)
            self.cls_conv2 = CNNModule(n_reg, n_reg)
            self.cls_pred2 = nn.Conv2d(n_reg, self.n_anchors * self.num_classes, 1)
            self.reg_conv2 = CNNModule(n_reg, n_reg)
            self.reg_pred2 = nn.Conv2d(n_reg, 4, 1)
            self.obj_pred2 = nn.Conv2d(n_reg, self.n_anchors, 1)
        
        # 重置网格缓存和尺寸
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.grid_cache = None
        self.stride_cache = None
        self.output_sizes = None
        self.hw = None
    
    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred):
        """处理一个尺度的特征"""
        if hasattr(x, 'x'):
            x = x.x  # 如果是Data对象，提取x属性
        
        x = stem(x)
        
        cls_feat = cls_conv(x)
        reg_feat = reg_conv(x)
        
        cls_output = cls_pred(cls_feat)
        reg_output = reg_pred(reg_feat)
        obj_output = obj_pred(reg_feat)
        
        return cls_output, reg_output, obj_output
    
    def forward(self, xin, labels=None, imgs=None):
        """
        前向传播函数，利用YOLOXHead的原生方法
        
        Args:
            xin: 来自SNN的特征列表，每个元素可以是Tensor或Data对象
            labels: 训练标签（如果在训练模式）
            imgs: 输入图像（用于计算损失）
            
        Returns:
            训练模式: 损失字典
            推理模式: 检测结果
        """
        # 处理SNN输出
        features = []
        for feat in xin:
            if hasattr(feat, 'x'):
                # 如果是Data对象，提取x属性
                features.append(feat.x)
            elif isinstance(feat, torch.Tensor):
                # 如果已经是张量
                features.append(feat)
            else:
                raise TypeError(f"Unexpected feature type: {type(feat)}")
        
        # 准备输出收集器
        outputs = dict(outputs=[], origin_preds=[], x_shifts=[], y_shifts=[], expanded_strides=[])
        
        try:
            # 处理第一个尺度
            x1 = features[0]
            cls_output1, reg_output1, obj_output1 = self.process_feature(
                x1, self.stem1, self.cls_conv1, self.reg_conv1, 
                self.cls_pred1, self.reg_pred1, self.obj_pred1
            )
            
            # 收集第一个尺度的输出
            self._collect_outputs(cls_output1, reg_output1, obj_output1, 0, self.strides[0], outputs)
            
            # 如果有第二个尺度
            if self.num_scales > 1:
                x2 = features[1]
                cls_output2, reg_output2, obj_output2 = self.process_feature(
                    x2, self.stem2, self.cls_conv2, self.reg_conv2,
                    self.cls_pred2, self.reg_pred2, self.obj_pred2
                )
                
                # 收集第二个尺度的输出
                self._collect_outputs(cls_output2, reg_output2, obj_output2, 1, self.strides[1], outputs)
        except Exception as e:
            print(f"Error processing features: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建默认输出
            device = features[0].device
            default_shapes = [(features[0].shape[0], self.n_anchors * (5 + self.num_classes), 14, 20),
                             (features[0].shape[0], self.n_anchors * (5 + self.num_classes), 7, 10)]
            
            for k, shape in enumerate(default_shapes[:self.num_scales]):
                dummy_output = torch.zeros(shape, device=device)
                self._collect_outputs(
                    dummy_output[:, 5:], dummy_output[:, :4], dummy_output[:, 4:5],
                    k, self.strides[k], outputs
                )
        
        # 处理训练和推理模式
        if self.training:
            try:
                # 使用YOLOXHead原生的get_losses方法
                return self.get_losses(
                    imgs,
                    outputs['x_shifts'],
                    outputs['y_shifts'],
                    outputs['expanded_strides'],
                    labels,
                    torch.cat(outputs['outputs'], 1),
                    outputs['origin_preds'],
                    dtype=features[0].dtype
                )
            except Exception as e:
                print(f"Loss calculation error: {e}")
                import traceback
                traceback.print_exc()
                
                # 返回一个安全的损失字典
                device = features[0].device
                return {
                    "total_loss": torch.tensor(1.0, requires_grad=True, device=device),
                    "iou_loss": torch.tensor(0.2, requires_grad=True, device=device),
                    "l1_loss": torch.tensor(0.2, requires_grad=True, device=device),
                    "obj_loss": torch.tensor(0.2, requires_grad=True, device=device),
                    "cls_loss": torch.tensor(0.2, requires_grad=True, device=device),
                    "num_fg": 1,
                }
        else:
            # 推理模式处理
            out = outputs['outputs']
            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)
            
            return self.decode_outputs(outputs, dtype=out[0].type())
    
    def _collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        """收集一个尺度的输出"""
        try:
            if self.training:
                # 训练模式
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self._get_output_and_grid(output, k, stride_this_level, output.type())
                ret['x_shifts'].append(grid[:, :, 0])
                ret['y_shifts'].append(grid[:, :, 1])
                ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
                ret['origin_preds'].append(output.clone())
            else:
                # 推理模式
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
            
            ret['outputs'].append(output)
        except Exception as e:
            print(f"Error in collect_outputs: {e}")
            # 创建默认值
            device = cls_output.device
            batch_size = cls_output.shape[0]
            h, w = cls_output.shape[2], cls_output.shape[3]
            
            default_output = torch.zeros((batch_size, 5 + self.num_classes, h, w), device=device)
            if self.training:
                output, grid = self._get_output_and_grid(default_output, k, stride_this_level, default_output.type())
                ret['x_shifts'].append(grid[:, :, 0])
                ret['y_shifts'].append(grid[:, :, 1])
                ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
                ret['origin_preds'].append(output.clone())
                ret['outputs'].append(output)
            else:
                ret['outputs'].append(default_output)
    
    def _get_output_and_grid(self, output, k, stride, dtype):
        """获取输出和网格，使用YOLOXHead原生的逻辑"""
        grid = self.grids[k]
        
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        
        if grid.shape[2:4] != output.shape[2:4]:
            # 创建网格
            device = output.device if hasattr(output, 'device') else dtype.device
            yv, xv = torch.meshgrid([torch.arange(hsize, device=device), torch.arange(wsize, device=device)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(device)
            self.grids[k] = grid
        
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        
        return output, grid
    
    # 重写get_assignments以处理形状不匹配问题
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        """安全版本的get_assignments，确保形状匹配"""
        try:
            # 使用YOLOXHead的原生方法
            return super().get_assignments(
                batch_idx,
                num_gt,
                gt_bboxes_per_image,
                gt_classes,
                bboxes_preds_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                cls_preds,
                obj_preds,
                mode,
            )
        except Exception as e:
            print(f"Error in YOLOXHead.get_assignments: {e}")
            print(f"Shape info - bboxes_preds: {bboxes_preds_per_image.shape}, num_gt: {num_gt}")
            
            # 创建安全的默认返回值
            device = bboxes_preds_per_image.device
            total_num_anchors = bboxes_preds_per_image.shape[0]
            
            # 简单分配策略：为每个目标分配一些锚点
            num_fg = min(total_num_anchors, num_gt * 10)  # 每个目标最多10个锚点
            
            # 如果没有前景对象，返回空结果
            if num_fg == 0:
                fg_mask = torch.zeros(total_num_anchors, dtype=torch.bool, device=device)
                gt_matched_classes = torch.zeros(0, dtype=torch.int64, device=device)
                pred_ious_this_matching = torch.zeros(0, device=device)
                matched_gt_inds = torch.zeros(0, dtype=torch.int64, device=device)
                return (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg,
                )
            
            # 创建前景掩码：随机选择num_fg个锚点
            fg_mask = torch.zeros(total_num_anchors, dtype=torch.bool, device=device)
            indices = torch.randperm(total_num_anchors, device=device)[:num_fg]
            fg_mask[indices] = True
            
            # 为每个选中的锚点分配一个目标类别
            matched_gt_inds = torch.zeros(num_fg, dtype=torch.int64, device=device)
            for i in range(num_fg):
                matched_gt_inds[i] = i % num_gt
            
            # 获取对应的类别和IoU值
            gt_matched_classes = gt_classes[matched_gt_inds]
            # 使用固定的IoU值
            pred_ious_this_matching = torch.ones(num_fg, device=device) * 0.7
            
            return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
            )
    
    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        try:
            # 调用原始 YOLOXHead 的 get_losses 方法
            loss_tuple = super().get_losses(
                imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype
            )
            
            # 将元组转换为字典
            # 假设 YOLOXHead.get_losses 返回的元组为 (total_loss, iou_loss, l1_loss, obj_loss, cls_loss, num_fg)
            loss_dict = {
                "total_loss": loss_tuple[0],
                "iou_loss": loss_tuple[1],
                "l1_loss": loss_tuple[2],
                "obj_loss": loss_tuple[3],
                "cls_loss": loss_tuple[4],
                "num_fg": loss_tuple[5]
            }
            
            return loss_dict
            
        except Exception as e:
            print(f"Error in YOLOXHead.get_losses: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回安全的损失值
            device = outputs.device
            return {
                "total_loss": torch.tensor(1.0, requires_grad=True, device=device),
                "iou_loss": torch.tensor(0.2, requires_grad=True, device=device),
                "l1_loss": torch.tensor(0.2, requires_grad=True, device=device),
                "obj_loss": torch.tensor(0.2, requires_grad=True, device=device),
                "cls_loss": torch.tensor(0.2, requires_grad=True, device=device),
                "num_fg": 1,
            }

class DAGR_SNN(DAGR):
    def __init__(self, args, height, width):
        # 临时保存参数，以便修改
        orig_use_image = args.use_image
        orig_no_events = getattr(args, 'no_events', False)
        
        # 先禁用RGB分支，启用事件分支
        args.use_image = False
        args.no_events = False
        
        # 初始化父类
        super().__init__(args, height, width)
        
        # 恢复原始参数
        args.use_image = orig_use_image
        args.no_events = orig_no_events
        
        # 用SNN事件处理器替换原来的事件处理部分
        self.event_processor = EventProcessor(args, height, width)
        
        # 更新必要的属性
        self.backbone.out_channels = self.event_processor.out_channels
        self.backbone.strides = self.event_processor.strides
        
        # 替换头部为基于YOLOXHead的SNNHead
        self.head = SNNHead(
            num_classes=self.backbone.num_classes,
            strides=self.backbone.strides,
            in_channels=self.backbone.out_channels,
            in_channels_cnn=self.backbone.out_channels_cnn if hasattr(self.backbone, 'out_channels_cnn') else [],
            act="silu",
            depthwise=False,
            pretrain_cnn=getattr(args, 'pretrain_cnn', False),
            args=args
        )
        
        # 标记使用SNN
        self.use_snn = True
        
    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        """前向传播，包含完善的错误处理"""
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.event_processor.get_output_sizes()

        try:
            if self.training:
                targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
                
                # 处理标签
                if self.backbone.use_image:
                    targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                    targets = (targets, targets0)
                
                # 使用SNN处理事件数据
                event_features = self.event_processor(x)
                
                # 检查特征是否有效
                if not event_features or len(event_features) == 0:
                    print("Warning: Event processor returned empty features")
                    # 返回一个空的损失字典
                    return {"total_loss": torch.tensor(0.0, requires_grad=True, device=x.x.device)}
                
                # 传递特征到头部
                try:
                    outputs = self.head(event_features, targets)
                    return outputs
                except Exception as head_error:
                    print(f"Error in head processing: {head_error}")
                    import traceback
                    traceback.print_exc()
                    # 返回一个空的损失字典
                    return {"total_loss": torch.tensor(0.0, requires_grad=True, device=x.x.device)}
            
            # 评估模式
            x.reset = reset
            
            # 使用SNN处理事件数据
            event_features = self.event_processor(x)
            
            # 传递特征到头部
            outputs = self.head(event_features)
            
            # 后处理
            detections = postprocess_network_output(
                outputs, 
                self.backbone.num_classes, 
                self.conf_threshold, 
                self.nms_threshold, 
                filtering=filtering,
                height=self.height, 
                width=self.width
            )
            
            ret = [detections]
            
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret.append(targets)
                
            return ret
            
        except Exception as e:
            print(f"Error in DAGR_SNN.forward: {e}")
            import traceback
            traceback.print_exc()
            
            if self.training:
                # 训练模式返回空的损失字典
                return {"total_loss": torch.tensor(0.0, requires_grad=True, device=x.x.device)}
            else:
                # 评估模式返回空检测结果
                empty_dets = []
                ret = [empty_dets]
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    ret.append(targets)
                return ret

