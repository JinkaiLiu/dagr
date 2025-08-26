# dagr/model/networks/dagr_snn.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
from dagr.model.networks.dagr import DAGR, GNNHead
from dagr.model.networks.event_snn_backbone import EventProcessor
from dagr.model.utils import convert_to_training_format, postprocess_network_output, convert_to_evaluation_format, shallow_copy
from yolox.models import IOUloss

class CNNModule(nn.Module):
    """标准CNN模块，替代ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class SNNHead(nn.Module):
    """专为SNN输出设计的检测头"""
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
        super().__init__()
        self.args = args
        
        self.num_classes = num_classes
        self.strides = strides
        self.in_channels = in_channels
        self.n_anchors = 1
        self.use_image = getattr(args, 'use_image', False)
        self.batch_size = getattr(args, 'batch_size', 8)
        self.num_scales = getattr(args, 'num_scales', 2)
        self.no_events = getattr(args, 'no_events', False)
        
        # 使用标准CNN操作
        n_reg = 256  # 中间特征通道数
        
        # 第一个尺度的检测头
        self.stem1 = CNNModule(in_channels[0], n_reg)
        self.cls_conv1 = CNNModule(n_reg, n_reg)
        self.cls_pred1 = nn.Conv2d(n_reg, self.n_anchors * self.num_classes, 1)
        self.reg_conv1 = CNNModule(n_reg, n_reg)
        self.reg_pred1 = nn.Conv2d(n_reg, 4, 1)  # x, y, w, h
        self.obj_pred1 = nn.Conv2d(n_reg, self.n_anchors, 1)  # objectness
        
        # 第二个尺度的检测头（如果需要）
        if self.num_scales > 1:
            self.stem2 = CNNModule(in_channels[1], n_reg)
            self.cls_conv2 = CNNModule(n_reg, n_reg)
            self.cls_pred2 = nn.Conv2d(n_reg, self.n_anchors * self.num_classes, 1)
            self.reg_conv2 = CNNModule(n_reg, n_reg)
            self.reg_pred2 = nn.Conv2d(n_reg, 4, 1)
            self.obj_pred2 = nn.Conv2d(n_reg, self.n_anchors, 1)
        
        # 损失函数
        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        
        # 网格和步长缓存
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.grid_cache = None
        self.stride_cache = None
        self.output_sizes = None  # 将在forward中设置
        self.hw = None
    
    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred):
        """处理一个尺度的特征"""
        x = stem(x)
        
        cls_feat = cls_conv(x)
        reg_feat = reg_conv(x)
        
        cls_output = cls_pred(cls_feat)
        reg_output = reg_pred(reg_feat)
        obj_output = obj_pred(reg_feat)
        
        return cls_output, reg_output, obj_output
    
    def forward(self, xin, labels=None, imgs=None):
        """
        前向传播函数
        
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
        
        # 处理训练和推理模式
        if self.training:
            # 计算损失
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
        else:
            # 推理模式处理
            out = outputs['outputs']
            self.hw = [x.shape[-2:] for x in out]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in out], dim=2).permute(0, 2, 1)
            
            return self.decode_outputs(outputs, dtype=out[0].type())
    
    def _collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        """收集一个尺度的输出"""
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
    
    def _get_output_and_grid(self, output, k, stride, dtype):
        """获取输出和网格"""
        grid = self.grids[k]
        
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        
        if grid.shape[2:4] != output.shape[2:4]:
            # 将字符串类型转换为设备对象
            if isinstance(dtype, str):
                device = output.device
            else:
                device = dtype.device
                
            yv, xv = yv.to(device), xv.to(device)
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(output.device)
            self.grids[k] = grid
        
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        
        return output, grid
    
    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
        """计算损失"""
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # 计算损失
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
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
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory" in str(e):
                        print("OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                               CPU mode is applied in this batch. If you want to avoid this issue, \
                               try to reduce the batch size or image size.")
                        torch.cuda.empty_cache()
                        (
                            gt_matched_classes,
                            fg_mask,
                            pred_ious_this_matching,
                            matched_gt_inds,
                            num_fg_img,
                        ) = self.get_assignments(  # noqa
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
                            "cpu",
                        )
                    else:
                        raise e

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return {
            "total_loss": loss,
            "iou_loss": loss_iou,
            "l1_loss": loss_l1,
            "obj_loss": loss_obj,
            "cls_loss": loss_cls,
            "num_fg": num_fg,
        }
    
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        """计算L1损失目标"""
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target
    
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
        """分配目标给预测框"""
        # 为了保持兼容性，从GNNHead借用这个方法
        try:
            from dagr.model.networks.dagr import GNNHead
            dummy_gnn_head = GNNHead(num_classes=self.num_classes, in_channels=self.in_channels, args=self.args)
            return dummy_gnn_head.get_assignments(
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
        except IndexError as e:
            print(f"IndexError in get_assignments: {e}")
            print(f"Shape info - bboxes_preds_per_image: {bboxes_preds_per_image.shape}, num_gt: {num_gt}")
            
            # 创建安全的默认返回值
            num_fg = 0
            device = bboxes_preds_per_image.device
            
            # 创建空的掩码和匹配
            fg_mask = torch.zeros(bboxes_preds_per_image.shape[0], dtype=torch.bool, device=device)
            matched_gt_inds = torch.tensor([], dtype=torch.int64, device=device)
            gt_matched_classes = torch.tensor([], dtype=torch.int64, device=device)
            pred_ious_this_matching = torch.tensor([], device=device)
            
            return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
            )
    
    def decode_outputs(self, outputs, dtype):
        """解码输出为检测框"""
        from dagr.model.utils import init_grid_and_stride
        if self.grid_cache is None and self.hw is not None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs

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
        
        # 替换头部为完全基于CNN的SNNHead
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
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.event_processor.get_output_sizes()

        if self.training:
            try:
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
                outputs = self.head(event_features, targets)
                
                return outputs
            except Exception as e:
                print(f"Error in DAGR_SNN.forward (training mode): {e}")
                import traceback
                traceback.print_exc()
                # 返回一个空的损失字典，允许训练继续
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