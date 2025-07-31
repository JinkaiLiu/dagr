import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from dagr.model.networks.net import sampling_skip
from dagr.model.networks.fcos_head import FCOSHead
from dagr.model.networks.net import Net
from dagr.model.utils import (
    convert_to_training_format,
    convert_to_evaluation_format,
    init_subnetwork
)

def postprocess_network_output(prediction, conf_thres, nms_thres):
    batch_size = prediction.shape[0]
    
    if prediction.dim() == 3:
        num_classes = prediction.shape[2] - 5
    else:
        num_classes = prediction.shape[-1] - 5
    
    if torch.is_tensor(num_classes):
        num_classes = num_classes.item()
    num_classes = int(num_classes)
    
    output = []
    
    for batch_idx in range(batch_size):
        image_pred = prediction[batch_idx]
        
        if image_pred.dim() == 1:
            image_pred = image_pred.unsqueeze(0)
            
        if image_pred.shape[0] == 0:
            output.append([])
            continue
            
        conf_scores = image_pred[:, 4]
        conf_mask = conf_scores >= conf_thres
        image_pred = image_pred[conf_mask]
        
        if image_pred.shape[0] == 0:
            output.append([])
            continue
        
        if num_classes > 0:
            class_scores = image_pred[:, 5:5 + num_classes]
            class_conf, class_pred = torch.max(class_scores, 1, keepdim=True)
            
            detections = torch.cat((
                image_pred[:, :4],
                conf_scores.unsqueeze(1),
                class_conf,
                class_pred.float()
            ), 1)
        else:
            detections = torch.cat((
                image_pred[:, :4],
                conf_scores.unsqueeze(1),
                torch.zeros(image_pred.shape[0], 1, device=image_pred.device),
                torch.zeros(image_pred.shape[0], 1, device=image_pred.device)
            ), 1)
        
        batch_detections = []
        if detections.shape[0] > 0:
            boxes = detections[:, :4].cpu()
            scores = detections[:, 4].cpu()
            labels = detections[:, 6].cpu().long() if num_classes > 0 else torch.zeros(detections.shape[0]).long()
            
            det_dict = {
                'boxes': boxes,
                'scores': scores,  
                'labels': labels
            }
            batch_detections.append(det_dict)
        
        output.append(batch_detections)
    
    return output

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = target * p + (1 - target) * (1 - p)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        loss *= alpha_t
    return loss.sum()

def unpack_fused_features(fused_feat):
    features_tensor = []
    features_hw = []
    
    print(f"[DEBUG] Unpacking {len(fused_feat)} fused features")
    
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x  # [num_nodes, channels]
            height = f.height.item() if hasattr(f, "height") and torch.is_tensor(f.height) else None
            width = f.width.item() if hasattr(f, "width") and torch.is_tensor(f.width) else None
            batch_info = f.batch if hasattr(f, "batch") else None
            
            print(f"[DEBUG] Feature {i}: x.shape = {x.shape}, height = {height}, width = {width}")
            if batch_info is not None:
                print(f"[DEBUG] Feature {i}: batch info available, unique batches = {batch_info.unique()}")
            
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None and batch_info is not None:
                    # 有batch信息的情况，需要正确重建batch维度
                    batch_size = len(batch_info.unique())
                    expected_nodes_per_batch = height * width
                    
                    print(f"[DEBUG] Reconstructing batch dimension: batch_size={batch_size}, nodes_per_batch={expected_nodes_per_batch}")
                    
                    # 创建[batch_size, channels, height, width]的tensor
                    full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
                    # 为每个batch分别处理
                    for batch_idx in range(batch_size):
                        batch_mask = batch_info == batch_idx
                        batch_nodes = x[batch_mask]  # [nodes_in_batch, channels]
                        
                        if batch_nodes.shape[0] > 0:
                            # 限制到特征图大小
                            max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                            
                            # 重塑为特征图
                            if max_nodes == expected_nodes_per_batch:
                                # 完整填充
                                reshaped = batch_nodes[:max_nodes].transpose(0, 1).view(channels, height, width)
                            else:
                                # 部分填充
                                temp_features = torch.zeros(expected_nodes_per_batch, channels, device=x.device, dtype=x.dtype)
                                temp_features[:max_nodes] = batch_nodes[:max_nodes]
                                reshaped = temp_features.transpose(0, 1).view(channels, height, width)
                            
                            full_tensor[batch_idx] = reshaped
                    
                    x = full_tensor
                    print(f"[DEBUG] Reconstructed to batch tensor: {x.shape}")
                    
                elif height is not None and width is not None:
                    # 没有batch信息，假设是单个batch
                    expected_nodes = height * width
                    
                    if num_nodes <= expected_nodes:
                        full_features = torch.zeros(expected_nodes, channels, device=x.device, dtype=x.dtype)
                        full_features[:num_nodes] = x
                        x = full_features.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    else:
                        x_truncated = x[:expected_nodes]
                        x = x_truncated.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    
                    print(f"[DEBUG] Single batch reshaped to: {x.shape}")
                else:
                    # 推断空间维度
                    sqrt_nodes = int(num_nodes ** 0.5)
                    if sqrt_nodes * sqrt_nodes == num_nodes:
                        x = x.transpose(0, 1).contiguous().view(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                        height, width = sqrt_nodes, sqrt_nodes
                    else:
                        x = x.transpose(0, 1).contiguous().view(channels, num_nodes, 1).unsqueeze(0)
                        height, width = num_nodes, 1
                    print(f"[DEBUG] Inferred reshape to: {x.shape}")
                        
            elif x.dim() == 3:
                x = x.unsqueeze(0)
            elif x.dim() == 4:
                pass
            else:
                raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            
            features_tensor.append(x)
            features_hw.append((height, width))
            
        else:
            if isinstance(f, torch.Tensor):
                if f.dim() == 2:
                    print(f"[WARNING] Processing bare 2D tensor: {f.shape}")
                    f = f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                elif f.dim() == 3:
                    f = f.unsqueeze(0)
                features_tensor.append(f)
                features_hw.append((None, None))
            else:
                raise ValueError(f"Unknown feature type: {type(f)}")
    
    print(f"[DEBUG] Final unpacked shapes: {[f.shape for f in features_tensor]}")
    return features_tensor, features_hw


class DAGR(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        self.conf_threshold = 0.05
        self.nms_threshold = 0.6
        self.height = height
        self.width = width
        self.args = args

        self.backbone = Net(args, height=height, width=width)
        self.use_image = getattr(args, "use_image", True)
        self.no_events = getattr(args, "no_events", False)
        self.pretrain_cnn = getattr(args, "pretrain_cnn", False)

        self.head = None
        self.cnn_head = None
        self._head_initialized = False

        if self.use_image:
            self.cnn_head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=self.backbone.out_channels_cnn[:2],
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            )

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)

    def _convert_bbox_to_fcos_format(self, bbox, bbox_batch, num_graphs):
        """
        将原始bbox转换为FCOS训练格式
        原始格式: [x, y, h, w, class] (按DSEC官方文档)
        FCOS格式: [class, x_center, y_center, width, height]
        """
        targets = []
        
        for batch_idx in range(num_graphs):
            # 获取当前batch的bbox
            batch_mask = bbox_batch == batch_idx
            batch_bboxes = bbox[batch_mask]
            
            if batch_bboxes.numel() == 0:
                # 空的target
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
            
            # 按照DSEC格式解析: [x, y, h, w, class]
            x_left = batch_bboxes[:, 0]      # x-coordinate of top-left corner
            y_top = batch_bboxes[:, 1]       # y-coordinate of top-left corner  
            height = batch_bboxes[:, 2]      # height
            width = batch_bboxes[:, 3]       # width
            cls = batch_bboxes[:, 4]         # class_id
            
            # 过滤有效的bbox (面积大于0)
            areas = width * height
            valid_mask = areas > 0
            
            if valid_mask.sum() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
                
            # 只保留有效的boxes
            x_left = x_left[valid_mask]
            y_top = y_top[valid_mask]
            height = height[valid_mask]
            width = width[valid_mask]
            cls = cls[valid_mask]
            
            # 转换为中心点坐标
            x_center = x_left + width / 2
            y_center = y_top + height / 2
            
            # 组装FCOS格式: [class, x_center, y_center, width, height]
            fcos_target = torch.stack([cls, x_center, y_center, width, height], dim=1)
            targets.append(fcos_target)
            
        return targets

    def _initialize_head_if_needed(self, fused_feat):
        if not self._head_initialized:
            actual_channels = []
            for f in fused_feat[:2]:
                if hasattr(f, "x"):
                    channels = f.x.shape[1]
                    actual_channels.append(channels)
                else:
                    actual_channels.append(f.shape[1])
            
            print(f"[DEBUG] Initializing FCOS head with actual channels: {actual_channels}")
            
            self.head = FCOSHead(
                num_classes=self.backbone.num_classes,
                in_channels=actual_channels,
                strides=self.backbone.strides[:2],
                use_focal_loss=True,
                use_iou_loss=True
            ).to(next(self.parameters()).device)
            
            self._head_initialized = True

    def forward(self, x, reset=True, return_targets=True, filtering=True):
        if self.use_image:
            event_feat, image_feat = self.backbone(x)
        else:
            event_feat = self.backbone(x)
            image_feat = None

        if self.training and self.pretrain_cnn:
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[0] // (h * w)
                        tensor = tensor.view(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            return self.cnn_head(image_feat_tensors, targets=targets, training=True)

        if not self.training and self.no_events:
            image_feat_tensors = []
            for img_f in image_feat:
                if hasattr(img_f, 'x'):
                    tensor = img_f.x
                    if tensor.dim() == 2:
                        h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                        w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                        c = tensor.shape[0] // (h * w)
                        tensor = tensor.view(c, h, w).unsqueeze(0)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    image_feat_tensors.append(tensor)
                else:
                    image_feat_tensors.append(img_f)
            outputs = self.cnn_head(image_feat_tensors, training=False)
            
            if filtering:
                processed_outputs = torch.cat([
                    outputs[..., :4],
                    outputs[..., 4:5],
                    outputs[..., 5:]
                ], dim=-1)
                batch_detections = postprocess_network_output(processed_outputs, self.conf_threshold, self.nms_threshold)
                outputs = []
                for batch_det in batch_detections:
                    outputs.extend(batch_det)
            else:
                outputs = torch.cat([
                    outputs[..., :4],
                    outputs[..., 4:5],
                    outputs[..., 5:]
                ], dim=-1)
            
            ret = outputs
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [outputs, targets]
            
            return ret

        fused_feat = event_feat
        if self.use_image and image_feat is not None:
            print(f"[DEBUG] Fusing image and event features")
            for i in range(len(event_feat)):
                event_feat[i].width = torch.tensor([image_feat[i].shape[-1]])
                event_feat[i].height = torch.tensor([image_feat[i].shape[-2]])
                event_feat[i].x = sampling_skip(event_feat[i], image_feat[i].detach())
                
        fused_feat = event_feat

        if self.training:
            # 直接使用原始bbox数据，不经过convert_to_training_format
            targets = self._convert_bbox_to_fcos_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            print(f"[DEBUG] Training mode - processing {len(targets)} targets")
            for i, t in enumerate(targets):
                if t.numel() > 0:
                    print(f"[DEBUG] Target {i}: {t.shape[0]} objects")
                    valid_boxes = t[t[:, 1:].sum(dim=1) > 0]
                    print(f"[DEBUG] Target {i}: {valid_boxes.shape[0]} valid objects")
                    if valid_boxes.shape[0] > 0:
                        print(f"[DEBUG] Target {i} bbox range: {valid_boxes[:, 1:].min(0)[0]} to {valid_boxes[:, 1:].max(0)[0]}")
            
            self._initialize_head_if_needed(fused_feat)
            
            fused_feat_tensors, _ = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Calling fusion head with {len(fused_feat_tensors)} features")
            loss_fused = self.head(fused_feat_tensors, targets=targets, training=True)

            result = {}
            
            if isinstance(loss_fused, dict):
                for k, v in loss_fused.items():
                    if torch.is_tensor(v):
                        result[f"fusion_{k}"] = v
                    else:
                        device = next(self.parameters()).device
                        result[f"fusion_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
            else:
                device = next(self.parameters()).device
                result["fusion_total_loss"] = torch.tensor(float(loss_fused), device=device, requires_grad=True) if not torch.is_tensor(loss_fused) else loss_fused

            if self.use_image:
                image_feat_tensors = []
                for img_f in image_feat:
                    if hasattr(img_f, 'x'):
                        tensor = img_f.x
                        if tensor.dim() == 2:
                            h = img_f.height.item() if hasattr(img_f, 'height') else int(tensor.shape[0]**0.5)
                            w = img_f.width.item() if hasattr(img_f, 'width') else int(tensor.shape[0]**0.5)
                            c = tensor.shape[0] // (h * w)
                            tensor = tensor.view(c, h, w).unsqueeze(0)
                        elif tensor.dim() == 3:
                            tensor = tensor.unsqueeze(0)
                        image_feat_tensors.append(tensor)
                    else:
                        image_feat_tensors.append(img_f)
                        
                print(f"[DEBUG] Calling CNN head with {len(image_feat_tensors)} features")
                # 使用相同的targets格式
                loss_image = self.cnn_head(image_feat_tensors, targets=targets, training=True)
                
                if isinstance(loss_image, dict):
                    for k, v in loss_image.items():
                        if torch.is_tensor(v):
                            result[f"cnn_{k}"] = v
                        else:
                            device = next(self.parameters()).device
                            result[f"cnn_{k}"] = torch.tensor(float(v), device=device, requires_grad=True)
                else:
                    device = next(self.parameters()).device
                    result["cnn_total_loss"] = torch.tensor(float(loss_image), device=device, requires_grad=True) if not torch.is_tensor(loss_image) else loss_image

            if "fusion_total_loss" not in result:
                if "fusion_loss" in result:
                    result["fusion_total_loss"] = result["fusion_loss"]
                else:
                    device = next(self.parameters()).device
                    result["fusion_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)
                    
            if self.use_image and "cnn_total_loss" not in result:
                if "cnn_loss" in result:
                    result["cnn_total_loss"] = result["cnn_loss"] 
                else:
                    device = next(self.parameters()).device
                    result["cnn_total_loss"] = torch.tensor(0.0, device=device, requires_grad=True)

            return result

        for i, f in enumerate(fused_feat):
            print(f"[DEBUG] Feature {i}: x.shape = {f.x.shape}, height = {getattr(f, 'height', 'N/A')}, width = {getattr(f, 'width', 'N/A')}")

        x.reset = reset
        
        self._initialize_head_if_needed(fused_feat)
        
        try:
            fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Unpacked features shapes: {[f.shape for f in fused_feat_x]}")
            
            # 检查unpacked features是否有效
            valid_features = []
            for i, feat in enumerate(fused_feat_x):
                if feat.numel() > 0 and all(d > 0 for d in feat.shape):
                    valid_features.append(feat)
                    print(f"[DEBUG] Feature {i} is valid: {feat.shape}")
                else:
                    print(f"[WARNING] Feature {i} is invalid: {feat.shape}, numel={feat.numel()}")
            
            if len(valid_features) == 0:
                print(f"[ERROR] No valid features after unpacking!")
                # 返回空的检测结果字典格式
                ret = []
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    ret = [ret, targets]
                return ret
                
            outputs = self.head(valid_features, training=False)
            
        except Exception as e:
            print(f"[ERROR] Failed to unpack features: {e}")
            print(f"[DEBUG] fused_feat types: {[type(f) for f in fused_feat]}")
            for i, f in enumerate(fused_feat):
                if hasattr(f, 'x'):
                    print(f"[DEBUG] fused_feat[{i}].x.shape: {f.x.shape}")
                else:
                    print(f"[DEBUG] fused_feat[{i}] is tensor with shape: {f.shape if hasattr(f, 'shape') else 'no shape'}")
            # 返回空的检测结果字典格式
            ret = []
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [ret, targets]
            return ret
            
        if filtering:
            processed_outputs = torch.cat([
                outputs[..., :4],
                outputs[..., 4:5],
                outputs[..., 5:]
            ], dim=-1)
            batch_detections = postprocess_network_output(processed_outputs, self.conf_threshold, self.nms_threshold)
            outputs = []
            for batch_det in batch_detections:
                outputs.extend(batch_det)
        else:
            outputs = torch.cat([
                outputs[..., :4],
                outputs[..., 4:5], 
                outputs[..., 5:]
            ], dim=-1)
        
        ret = outputs

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret = [outputs, targets]

        return ret
