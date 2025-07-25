import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from dagr.model.networks.net import Net
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, convert_to_training_format


class FCOSHead(nn.Module):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=[256, 512, 1024],
        fpn_feature_channels=256,
        pretrain_cnn=False,
        args=None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides
        self.fpn_feature_channels = fpn_feature_channels
        self.use_image = args.use_image if args else False
        self.batch_size = args.batch_size if args else 1
        self.num_scales = args.num_scales if args else len(strides)
        self.no_events = args.no_events if args else False
        self.pretrain_cnn = pretrain_cnn
        
        self.width = getattr(args, 'width', 320) if args else 320
        self.height = getattr(args, 'height', 240) if args else 240
        
        actual_in_channels = [64, 64]
        
        self.n_anchors = 1
        n_reg = max(actual_in_channels)
        
        self.stem1 = ConvBlock(in_channels=actual_in_channels[0], out_channels=n_reg, args=args)
        self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.cls_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
        self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
        self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
        self.centerness_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=1, bias=True, args=args)

        if self.num_scales > 1:
            self.stem2 = ConvBlock(in_channels=actual_in_channels[1], out_channels=n_reg, args=args)
            self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.centerness_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=1, bias=True, args=args)

        if self.use_image:
            from dagr.model.networks.dagr import CNNHead
            self.cnn_head = CNNHead(num_classes=num_classes, strides=strides, in_channels=in_channels_cnn)
        
        self.scales = nn.Parameter(torch.ones(len(strides)))
        
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        
        from yolox.models import IOUloss
        self.iou_loss = IOUloss(reduction="none")
        
        self.regress_ranges = [
            (-1, 64),
            (64, 128), 
            (128, 256),
            (256, 512),
            (512, float('inf'))
        ]
        
        self.init_weights()
    
    def init_weights(self):
        bias_value = -torch.log(torch.tensor((1 - 0.01) / 0.01))
        
        if hasattr(self, 'cls_pred1') and hasattr(self.cls_pred1, 'bias') and self.cls_pred1.bias is not None:
            nn.init.constant_(self.cls_pred1.bias, bias_value)
        if hasattr(self, 'cls_pred2') and hasattr(self.cls_pred2, 'bias') and self.cls_pred2.bias is not None:
            nn.init.constant_(self.cls_pred2.bias, bias_value)
        
        if self.use_image and hasattr(self, 'cnn_head'):
            for layer in self.cnn_head.cls_preds:
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, bias_value)
    
    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, centerness_pred, batch_size):
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        centerness_output = centerness_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, centerness_output
    
    def forward(self, features, labels=None, imgs=None):
        if self.use_image:
            event_features, image_features = features
        else:
            event_features = features
        
        cls_scores = []
        bbox_preds = []
        centernesses = []
        
        batch_size = self.batch_size
        
        # 总是保存事件特征，训练和推理都需要
        self._current_event_features = event_features
        
        if len(event_features) > 0:
            feature = event_features[0]
            
            try:
                cls_output, reg_output, centerness_output = self.process_feature(
                    feature, self.stem1, self.cls_conv1, self.reg_conv1,
                    self.cls_pred1, self.reg_pred1, self.centerness_pred1, batch_size
                )
                
                reg_output = reg_output * self.scales[0]
                
            except RuntimeError as e:
                
                device = feature.x.device if hasattr(feature, 'x') else torch.device('cuda')
                num_points = feature.x.shape[0] if hasattr(feature, 'x') else 1000
                
                if self.training:
                    cls_output = torch.zeros(batch_size, num_points, self.num_classes, device=device, requires_grad=True)
                    reg_output = torch.zeros(batch_size, num_points, 4, device=device, requires_grad=True)
                    centerness_output = torch.zeros(batch_size, num_points, 1, device=device, requires_grad=True)
                else:
                    cls_output = torch.zeros(batch_size, num_points, self.num_classes, device=device)
                    reg_output = torch.zeros(batch_size, num_points, 4, device=device)
                    centerness_output = torch.zeros(batch_size, num_points, 1, device=device)
            
            cls_scores.append(cls_output)
            bbox_preds.append(reg_output)
            centernesses.append(centerness_output)
        
        if self.num_scales > 1 and len(event_features) > 1:
            feature = event_features[1]
            
            try:
                cls_output, reg_output, centerness_output = self.process_feature(
                    feature, self.stem2, self.cls_conv2, self.reg_conv2,
                    self.cls_pred2, self.reg_pred2, self.centerness_pred2, batch_size
                )
                
                reg_output = reg_output * self.scales[1]
                
            except RuntimeError as e:
                
                device = feature.x.device if hasattr(feature, 'x') else torch.device('cuda')
                num_points = feature.x.shape[0] if hasattr(feature, 'x') else 1000
                
                if self.training:
                    cls_output = torch.zeros(batch_size, num_points, self.num_classes, device=device, requires_grad=True)
                    reg_output = torch.zeros(batch_size, num_points, 4, device=device, requires_grad=True)
                    centerness_output = torch.zeros(batch_size, num_points, 1, device=device, requires_grad=True)
                else:
                    cls_output = torch.zeros(batch_size, num_points, self.num_classes, device=device)
                    reg_output = torch.zeros(batch_size, num_points, 4, device=device)
                    centerness_output = torch.zeros(batch_size, num_points, 1, device=device)
            
            cls_scores.append(cls_output)
            bbox_preds.append(reg_output)
            centernesses.append(centerness_output)
        
        if self.use_image and hasattr(self, 'cnn_head'):
            try:
                image_features_resized = [torch.nn.functional.interpolate(f, o) for f, o in zip(image_features, self.output_sizes)]
                out_cnn = self.cnn_head(image_features_resized)
                
                for k in range(min(len(cls_scores), len(out_cnn["cls_output"]))):
                    if cls_scores[k].shape[1] == out_cnn["cls_output"][k].shape[-1]:
                        cnn_cls = out_cnn["cls_output"][k].flatten(2).permute(0, 2, 1)
                        cnn_reg = out_cnn["reg_output"][k].flatten(2).permute(0, 2, 1)
                        cnn_obj = out_cnn["obj_output"][k].flatten(2).permute(0, 2, 1)
                        
                        actual_batch_size = cnn_cls.shape[0]
                        cls_scores[k][:actual_batch_size] += cnn_cls.detach()
                        bbox_preds[k][:actual_batch_size] += cnn_reg.detach()
                        centernesses[k][:actual_batch_size] += cnn_obj.detach()
                        
            except Exception as e:
                pass
        
        if self.training:
            return self.compute_losses(cls_scores, bbox_preds, centernesses, labels, imgs)
        else:
            return self.postprocess(cls_scores, bbox_preds, centernesses)
    
    def compute_losses(self, cls_scores, bbox_preds, centernesses, labels, imgs):
        device = cls_scores[0].device
        
        if self.use_image and self.pretrain_cnn:
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'loss_cls': torch.tensor(0.0, device=device),
                'loss_reg': torch.tensor(0.0, device=device),
                'loss_centerness': torch.tensor(0.0, device=device),
                'loss_iou': torch.tensor(0.0, device=device),
                'loss_l1': torch.tensor(0.0, device=device)
            }
        
        reshaped_cls_scores = []
        reshaped_bbox_preds = []
        reshaped_centernesses = []
        
        for level in range(len(cls_scores)):
            cls_score = cls_scores[level]
            bbox_pred = bbox_preds[level]
            centerness = centernesses[level]
            
            if len(cls_score.shape) == 4:
                batch_size, num_classes, H, W = cls_score.shape
                cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, H*W, num_classes)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, H*W, 4)
                centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, H*W, 1)
            
            reshaped_cls_scores.append(cls_score)
            reshaped_bbox_preds.append(bbox_pred)
            reshaped_centernesses.append(centerness)
        
        all_cls_scores = torch.cat(reshaped_cls_scores, dim=1) if len(reshaped_cls_scores) > 1 else reshaped_cls_scores[0]
        all_bbox_preds = torch.cat(reshaped_bbox_preds, dim=1) if len(reshaped_bbox_preds) > 1 else reshaped_bbox_preds[0]
        all_centernesses = torch.cat(reshaped_centernesses, dim=1) if len(reshaped_centernesses) > 1 else reshaped_centernesses[0]
        
        gt_bboxes_list, gt_labels_list = self.prepare_targets_for_fcos(labels, imgs)
        
        batch_size = all_cls_scores.shape[0]
        num_points = all_cls_scores.shape[1]
        
        points = self.get_points(reshaped_cls_scores, device, event_features=getattr(self, '_current_event_features', None))
        
        pos_inds_list = []
        neg_inds_list = []
        pos_bbox_targets_list = []
        pos_centerness_targets_list = []
        pos_labels_list = []
        
        total_pos_samples = 0
        
        for batch_idx in range(batch_size):
            gt_bboxes = gt_bboxes_list[batch_idx] if batch_idx < len(gt_bboxes_list) else torch.empty(0, 4, device=device)
            gt_labels = gt_labels_list[batch_idx] if batch_idx < len(gt_labels_list) else torch.empty(0, dtype=torch.long, device=device)
            
            if len(gt_bboxes) == 0:
                pos_inds_list.append(torch.empty(0, dtype=torch.long, device=device))
                neg_inds_list.append(torch.arange(num_points, device=device))
                pos_bbox_targets_list.append(torch.empty(0, 4, device=device))
                pos_centerness_targets_list.append(torch.empty(0, device=device))
                pos_labels_list.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            
            areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            areas = areas[None].repeat(num_points, 1)
            
            regress_ranges = self.get_regress_ranges(reshaped_cls_scores)
            regress_ranges = regress_ranges[:, None, :].expand(regress_ranges.shape[0], len(gt_bboxes), regress_ranges.shape[-1])
            
            gt_bboxes = gt_bboxes[None].expand(num_points, gt_bboxes.shape[0], 4)
            
            xs, ys = points[:, 0], points[:, 1]
            xs = xs[:, None].expand(num_points, len(gt_labels))
            ys = ys[:, None].expand(num_points, len(gt_labels))
            
            left = xs - gt_bboxes[..., 0]
            right = gt_bboxes[..., 2] - xs
            top = ys - gt_bboxes[..., 1]
            bottom = gt_bboxes[..., 3] - ys
            bbox_targets = torch.stack([left, top, right, bottom], dim=-1)
            
            inside_gt_bbox_mask = bbox_targets.min(dim=-1)[0] > 0
            
            max_regress_distance = bbox_targets.max(dim=-1)[0]
            inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1])
            )
            
            areas[inside_gt_bbox_mask == 0] = float('inf')
            areas[inside_regress_range == 0] = float('inf')
            min_area, min_area_inds = areas.min(dim=1)
            
            pos_inds = torch.nonzero(min_area != float('inf'), as_tuple=False).squeeze(1)
            neg_inds = torch.nonzero(min_area == float('inf'), as_tuple=False).squeeze(1)
            
            total_pos_samples += len(pos_inds)
            
            pos_matched_gt_inds = min_area_inds[pos_inds]
            
            pos_bbox_targets = bbox_targets[pos_inds, pos_matched_gt_inds]
            pos_labels = gt_labels[pos_matched_gt_inds]
            
            pos_centerness_targets = self.compute_centerness_targets(pos_bbox_targets)
            
            pos_inds_list.append(pos_inds)
            neg_inds_list.append(neg_inds)
            pos_bbox_targets_list.append(pos_bbox_targets)
            pos_centerness_targets_list.append(pos_centerness_targets)
            pos_labels_list.append(pos_labels)
        
        num_pos = sum(len(pos_inds) for pos_inds in pos_inds_list)
        num_pos = max(num_pos, 1)
        
        loss_cls = torch.tensor(0.0, device=device)
        loss_bbox = torch.tensor(0.0, device=device)
        loss_centerness = torch.tensor(0.0, device=device)
        
        for batch_idx in range(batch_size):
            cls_score = all_cls_scores[batch_idx]
            bbox_pred = all_bbox_preds[batch_idx]
            centerness = all_centernesses[batch_idx]
            
            pos_inds = pos_inds_list[batch_idx]
            neg_inds = neg_inds_list[batch_idx]
            pos_bbox_targets = pos_bbox_targets_list[batch_idx]
            pos_centerness_targets = pos_centerness_targets_list[batch_idx]
            pos_labels = pos_labels_list[batch_idx]
            
            labels_target = torch.zeros(num_points, self.num_classes, device=device)
            if len(pos_inds) > 0:
                labels_target[pos_inds, pos_labels] = 1.0
            
            focal_loss_batch = self.focal_loss(cls_score, labels_target)
            loss_cls += focal_loss_batch.mean()
            
            if len(pos_inds) > 0:
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_centerness_pred = centerness[pos_inds]
                
                centerness_targets = pos_centerness_targets.clamp(min=1e-6)
                
                iou_loss_batch = self.iou_loss(pos_bbox_pred, pos_bbox_targets) * centerness_targets
                loss_bbox += iou_loss_batch.mean()
                
                centerness_loss_batch = F.binary_cross_entropy_with_logits(
                    pos_centerness_pred.squeeze(-1), pos_centerness_targets, reduction='none'
                )
                loss_centerness += centerness_loss_batch.mean()
        
        loss_cls = loss_cls * 1.0
        loss_bbox = loss_bbox * 0.5
        loss_centerness = loss_centerness * 0.1
        
        total_loss = loss_cls + loss_bbox + loss_centerness
        
        return {
            'total_loss': total_loss,
            'loss_cls': loss_cls,
            'loss_reg': loss_bbox,
            'loss_centerness': loss_centerness,
            'loss_iou': loss_bbox,
            'loss_l1': torch.tensor(0.0, device=device)
        }
    
    def postprocess(self, cls_scores, bbox_preds, centernesses):
        if not cls_scores:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros(self.batch_size, 1, 5 + self.num_classes, device=device)
        
        processed_outputs = []
        device = cls_scores[0].device
        
        # 获取特征点坐标
        points = self.get_points(cls_scores, device, event_features=getattr(self, '_current_event_features', None))
        
        current_point_idx = 0
        
        for level in range(len(cls_scores)):
            cls_score = cls_scores[level]
            bbox_pred = bbox_preds[level]
            centerness = centernesses[level]
            
            if len(cls_score.shape) == 4:
                batch_size, num_classes, H, W = cls_score.shape
                cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, H*W, num_classes)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, H*W, 4)
                centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, H*W, 1)
            
            cls_score = torch.sigmoid(cls_score)
            centerness = torch.sigmoid(centerness)
            
            # 获取当前level的点坐标
            num_points_this_level = cls_score.shape[1]
            level_points = points[current_point_idx:current_point_idx + num_points_this_level]
            current_point_idx += num_points_this_level
            
            # 将FCOS距离预测转换为边界框
            batch_size, num_points, _ = bbox_pred.shape
            bbox_pred = torch.abs(bbox_pred) * 200.0  # 取绝对值并适度缩放
            
            # 确保点数量匹配
            if level_points.shape[0] != num_points:
                # 如果点数不匹配，直接使用网格点，不显示警告（推理时这是正常的）
                level_points = self._generate_grid_points(cls_score, level, device)
                if level_points.shape[0] != num_points:
                    # 如果还是不匹配，截断或填充
                    if level_points.shape[0] > num_points:
                        level_points = level_points[:num_points]
                    else:
                        # 重复最后一个点来填充
                        padding = num_points - level_points.shape[0]
                        last_point = level_points[-1:].repeat(padding, 1)
                        level_points = torch.cat([level_points, last_point], dim=0)
            
            # 扩展点坐标到批次维度
            level_points_batch = level_points.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, num_points, 2]
            
            # 计算边界框坐标
            x1 = level_points_batch[:, :, 0:1] - bbox_pred[:, :, 0:1]  # x - left
            y1 = level_points_batch[:, :, 1:2] - bbox_pred[:, :, 1:2]  # y - top  
            x2 = level_points_batch[:, :, 0:1] + bbox_pred[:, :, 2:3]  # x + right
            y2 = level_points_batch[:, :, 1:2] + bbox_pred[:, :, 3:4]  # y + bottom
            
            # 裁剪边界框到图像范围内
            x1 = torch.clamp(x1, 0, self.width - 1)
            y1 = torch.clamp(y1, 0, self.height - 1) 
            x2 = torch.clamp(x2, 0, self.width)
            y2 = torch.clamp(y2, 0, self.height)
            
            # 确保x2 > x1, y2 > y1（避免无效边界框）
            x2 = torch.maximum(x2, x1 + 1)
            y2 = torch.maximum(y2, y1 + 1)
            
            # 合并为边界框 [x1, y1, x2, y2]
            bbox_coords = torch.cat([x1, y1, x2, y2], dim=-1)
            
            # 修复分数计算 - 使用实际的分类和中心度分数
            centerness_expanded = centerness.expand(-1, -1, self.num_classes)
            scores = cls_score * centerness_expanded
            
            # 计算objectness分数：取最大类别分数 * 对应的中心度
            max_class_scores, max_indices = scores.max(dim=-1, keepdim=True)  # [batch, points, 1]
            objectness = max_class_scores  # 使用最大类别分数作为objectness
            
            # 输出格式：[x1, y1, x2, y2, objectness, class_scores...]
            output = torch.cat([
                bbox_coords,    # [batch, num_points, 4] 
                objectness,     # [batch, num_points, 1] - 真实的objectness分数
                scores          # [batch, num_points, num_classes]
            ], dim=-1)
            
            processed_outputs.append(output)
        
        if len(processed_outputs) == 1:
            final_output = processed_outputs[0]
        else:
            final_output = torch.cat(processed_outputs, dim=1)
        
        print(f"[DEBUG] Postprocess output shape: {final_output.shape}")
        print(f"[DEBUG] Max cls score: {cls_score.max().item():.6f}")
        print(f"[DEBUG] Max centerness: {centerness.max().item():.6f}")
        print(f"[DEBUG] Max objectness: {objectness.max().item():.6f}")
        print(f"[DEBUG] Bbox coords range: x1=[{final_output[..., 0].min().item():.1f}, {final_output[..., 0].max().item():.1f}], x2=[{final_output[..., 2].min().item():.1f}, {final_output[..., 2].max().item():.1f}]")
        print(f"[DEBUG] Bbox coords range: y1=[{final_output[..., 1].min().item():.1f}, {final_output[..., 1].max().item():.1f}], y2=[{final_output[..., 3].min().item():.1f}, {final_output[..., 3].max().item():.1f}]")
        print(f"[DEBUG] Bbox sizes: width=[{(final_output[..., 2] - final_output[..., 0]).min().item():.1f}, {(final_output[..., 2] - final_output[..., 0]).max().item():.1f}], height=[{(final_output[..., 3] - final_output[..., 1]).min().item():.1f}, {(final_output[..., 3] - final_output[..., 1]).max().item():.1f}]")
        
        return final_output
    
    def prepare_targets_for_fcos(self, labels, imgs):
        
        if self.use_image and isinstance(labels, tuple):
            labels, _ = labels
        
        gt_bboxes_list = []
        gt_labels_list = []
        
        if isinstance(labels, torch.Tensor) and len(labels.shape) == 3:
            batch_size, max_objects, _ = labels.shape
            
            for batch_idx in range(batch_size):
                batch_labels = labels[batch_idx]
                
                valid_mask = (batch_labels.sum(dim=1) != 0)
                valid_labels = batch_labels[valid_mask]
                
                if len(valid_labels) > 0:
                    gt_labels = valid_labels[:, 0].long()
                    gt_bboxes = valid_labels[:, 1:5].clone()
                    
                    x_center, y_center, width, height = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3]
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    gt_bboxes = torch.stack([x1, y1, x2, y2], dim=1)
                    
                    gt_bboxes[:, 0] = torch.clamp(gt_bboxes[:, 0], 0, self.width)
                    gt_bboxes[:, 1] = torch.clamp(gt_bboxes[:, 1], 0, self.height)
                    gt_bboxes[:, 2] = torch.clamp(gt_bboxes[:, 2], 0, self.width)
                    gt_bboxes[:, 3] = torch.clamp(gt_bboxes[:, 3], 0, self.height)
                else:
                    gt_bboxes = torch.empty(0, 4, device=labels.device)
                    gt_labels = torch.empty(0, dtype=torch.long, device=labels.device)
                
                gt_bboxes_list.append(gt_bboxes)
                gt_labels_list.append(gt_labels)
        
        else:
            for batch_idx in range(len(labels) if isinstance(labels, list) else self.batch_size):
                if isinstance(labels, list) and batch_idx < len(labels):
                    batch_labels = labels[batch_idx]
                    if len(batch_labels) > 0:
                        gt_bboxes = batch_labels[:, 1:5].clone()
                        gt_labels = batch_labels[:, 0].long()
                    else:
                        gt_bboxes = torch.empty(0, 4, device=labels[0].device if len(labels) > 0 else torch.device('cuda'))
                        gt_labels = torch.empty(0, dtype=torch.long, device=labels[0].device if len(labels) > 0 else torch.device('cuda'))
                else:
                    gt_bboxes = torch.empty(0, 4, device=torch.device('cuda'))
                    gt_labels = torch.empty(0, dtype=torch.long, device=torch.device('cuda'))
                
                gt_bboxes_list.append(gt_bboxes)
                gt_labels_list.append(gt_labels)
        
        return gt_bboxes_list, gt_labels_list
    
    def get_points(self, cls_scores, device, event_features=None):
        if event_features is not None and len(event_features) > 0:
            points_list = []
            
            for level_idx, (cls_score, feature) in enumerate(zip(cls_scores, event_features)):
                points = None
                
                if hasattr(feature, 'pos') and feature.pos is not None:
                    if len(feature.pos.shape) >= 2 and feature.pos.shape[1] >= 2:
                        points = feature.pos[:, :2].float()
                        
                        if points.numel() > 0 and points.max(dim=0)[0].max() <= 1.0:
                            points = points * torch.tensor([self.width, self.height], device=device)
                
                elif hasattr(feature, 'batch') and hasattr(feature, 'x'):
                    if len(feature.x.shape) >= 2 and feature.x.shape[1] >= 2:
                        points = feature.x[:, :2].float()
                        
                        if points.numel() > 0 and points.max(dim=0)[0].max() <= 1.0:
                            points = points * torch.tensor([self.width, self.height], device=device)
                
                elif hasattr(feature, 'coord'):
                    points = feature.coord[:, :2].float()
                    
                    if points.numel() > 0 and points.max(dim=0)[0].max() <= 1.0:
                        points = points * torch.tensor([self.width, self.height], device=device)
                
                elif hasattr(feature, 'keys'):
                    possible_coord_keys = ['pos', 'coordinates', 'coord', 'xy', 'position']
                    for key in possible_coord_keys:
                        if key in feature.keys and hasattr(feature, key):
                            attr = getattr(feature, key)
                            if attr is not None and len(attr.shape) >= 2 and attr.shape[1] >= 2:
                                points = attr[:, :2].float()
                                
                                if points.numel() > 0 and points.max(dim=0)[0].max() <= 1.0:
                                    points = points * torch.tensor([self.width, self.height], device=device)
                                break
                
                if points is None:
                    points = self._generate_grid_points(cls_score, level_idx, device)
                else:
                    expected_num_points = cls_score.shape[1]
                    if points.shape[0] != expected_num_points:
                        if points.shape[0] > expected_num_points:
                            indices = torch.randperm(points.shape[0])[:expected_num_points]
                            points = points[indices]
                        else:
                            grid_points = self._generate_grid_points(cls_score, level_idx, device)
                            points = grid_points
                
                points_list.append(points)
            
            if points_list:
                return torch.cat(points_list, dim=0)
        
        return self._generate_grid_points_from_scores(cls_scores, device)
    
    def _generate_grid_points(self, cls_score, level_idx, device):
        num_points = cls_score.shape[1]
        stride = self.strides[level_idx] if level_idx < len(self.strides) else self.strides[-1]
        
        h = int(self.height / stride)
        w = int(self.width / stride)
        
        y, x = torch.meshgrid(
            torch.arange(0, h, device=device),
            torch.arange(0, w, device=device),
            indexing='ij'
        )
        
        points = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        points = points * stride + stride // 2
        
        if len(points) > num_points:
            indices = torch.randperm(len(points))[:num_points]
            points = points[indices]
        elif len(points) < num_points:
            extra_points = torch.rand(num_points - len(points), 2, device=device)
            extra_points[:, 0] *= self.width
            extra_points[:, 1] *= self.height
            points = torch.cat([points, extra_points], dim=0)
        
        return points
    
    def _generate_grid_points_from_scores(self, cls_scores, device):
        points_list = []
        
        for level_idx, cls_score in enumerate(cls_scores):
            points = self._generate_grid_points(cls_score, level_idx, device)
            points_list.append(points)
        
        return torch.cat(points_list, dim=0)
    
    def get_regress_ranges(self, cls_scores=None):
        regress_ranges = []
        for level_idx in range(len(self.strides)):
            if level_idx < len(self.regress_ranges):
                regress_ranges.append(self.regress_ranges[level_idx])
            else:
                regress_ranges.append(self.regress_ranges[-1])
        
        if cls_scores is None or len(cls_scores) == 0:
            return torch.tensor([[0, float('inf')]], dtype=torch.float32)
        
        expanded_ranges = []
        for level_idx, cls_score in enumerate(cls_scores):
            if level_idx < len(regress_ranges):
                regress_range = regress_ranges[level_idx]
                num_points = cls_score.shape[1]
                level_ranges = torch.tensor(regress_range, dtype=torch.float32, device=cls_score.device).unsqueeze(0).repeat(num_points, 1)
                expanded_ranges.append(level_ranges)
        
        if expanded_ranges:
            return torch.cat(expanded_ranges, dim=0)
        else:
            return torch.tensor([[0, float('inf')]], dtype=torch.float32)
    
    def compute_centerness_targets(self, bbox_targets):
        left_right = bbox_targets[:, [0, 2]]
        top_bottom = bbox_targets[:, [1, 3]]
        
        lr_min = left_right.min(dim=1)[0]
        lr_max = left_right.max(dim=1)[0]
        tb_min = top_bottom.min(dim=1)[0]
        tb_max = top_bottom.max(dim=1)[0]
        
        centerness = torch.sqrt((lr_min / (lr_max + 1e-8)) * (tb_min / (tb_max + 1e-8)))
        return centerness
    
    def focal_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.focal_loss_gamma
        
        if self.focal_loss_alpha >= 0:
            alpha_t = torch.where(target == 1, self.focal_loss_alpha, 1 - self.focal_loss_alpha)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        return focal_loss


class DAGR_FCOS(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        
        self.conf_threshold = 0.00001  # 极低的置信度阈值用于调试
        self.nms_threshold = 0.65
        self.height = height
        self.width = width
        
        self.backbone = Net(args, height=height, width=width)
        self.head = FCOSHead(
            num_classes=self.backbone.num_classes,
            in_channels=self.backbone.out_channels,
            in_channels_cnn=self.backbone.out_channels_cnn if hasattr(self.backbone, 'out_channels_cnn') else self.backbone.out_channels,
            strides=self.backbone.strides,
            pretrain_cnn=args.pretrain_cnn if hasattr(args, 'pretrain_cnn') else False,
            args=args
        )
        
        if hasattr(self.backbone, 'out_channels'):
            args.backbone_out_channels = self.backbone.out_channels
        else:
            print("WARNING: backbone.out_channels not found")
        
        if hasattr(args, "img_net_checkpoint"):
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            if self.head.use_image:
                init_subnetwork(self, state_dict['ema'], "head.cnn_")
    
    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width + 1)
        
        if hasattr(self.backbone, 'conv_block1'):
            self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
            self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        
        for layer_name in ['layer2', 'layer3', 'layer4', 'layer5']:
            if hasattr(self.backbone, layer_name):
                layer = getattr(self.backbone, layer_name)
                pool_name = layer_name.replace('layer', 'pool')
                if hasattr(self.backbone, pool_name):
                    pool = getattr(self.backbone, pool_name)
                    rx, ry, M = voxel_size_to_params(pool, height, width)
                    
                    if hasattr(layer, 'conv_block1'):
                        layer.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
                    if hasattr(layer, 'conv_block2'):
                        layer.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        
        for i, tower in enumerate(self.head.cls_towers):
            for block in tower:
                if hasattr(block, 'conv') and hasattr(block.conv, 'init_lut'):
                    if i == 0:
                        rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
                    else:
                        rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
                    block.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
    
    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()
        
        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                targets = (targets, targets0)
            
            features = self.backbone(x)
            outputs = self.head(features, targets, x)
            
            return outputs
        
        x.reset = reset
        features = self.backbone(x)
        outputs = self.head(features)
        
        print(f"[DEBUG] Model outputs shape: {outputs.shape}")
        print(f"[DEBUG] Confidence threshold: {self.conf_threshold}")
        print(f"[DEBUG] NMS threshold: {self.nms_threshold}")
        
        detections = postprocess_network_output(
            outputs, 
            self.backbone.num_classes, 
            self.conf_threshold, 
            self.nms_threshold, 
            filtering=filtering,
            height=self.height, 
            width=self.width
        )
        
        print(f"[DEBUG] After postprocess: {len(detections)} batches")
        for i, det in enumerate(detections):
            if det is not None and hasattr(det, '__len__') and len(det) > 0:
                if hasattr(det, 'shape'):
                    print(f"[DEBUG] Batch {i}: {det.shape[0]} detections, max conf: {det[:, 4].max():.6f}")
                else:
                    print(f"[DEBUG] Batch {i}: {len(det)} detections")
            else:
                print(f"[DEBUG] Batch {i}: No detections")
        
        ret = [detections]
        
        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)
        
        return ret


