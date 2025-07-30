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
        
        # 转换为COCO评估期望的格式
        batch_detections = []
        if detections.shape[0] > 0:
            boxes = detections[:, :4].cpu()
            scores = detections[:, 4].cpu()
            labels = detections[:, 6].cpu().long() if num_classes > 0 else torch.zeros(detections.shape[0]).long()
            
            # 创建符合COCO格式的检测结果
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
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x
            height = f.height.item() if hasattr(f, "height") and torch.is_tensor(f.height) else None
            width = f.width.item() if hasattr(f, "width") and torch.is_tensor(f.width) else None
            
            print(f"[DEBUG] Feature {i}: x.shape = {x.shape}, height = {height}, width = {width}")
            
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None:
                    expected_nodes = height * width
                    
                    if num_nodes <= expected_nodes:
                        # 节点数少于预期，可能是稀疏表示，需要填充
                        print(f"[DEBUG] Sparse nodes {num_nodes} < expected {expected_nodes}")
                        # 创建一个完整的特征图，用零填充缺失的位置
                        full_features = torch.zeros(expected_nodes, channels, device=x.device, dtype=x.dtype)
                        full_features[:num_nodes] = x  # 假设节点是按顺序排列的
                        x = full_features.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                        print(f"[DEBUG] Padded and reshaped to: [1, {channels}, {height}, {width}]")
                        
                    elif num_nodes > expected_nodes:
                        # 节点数多于预期，可能需要下采样或裁剪
                        print(f"[DEBUG] Extra nodes {num_nodes} > expected {expected_nodes}")
                        # 简单裁剪到预期大小
                        x_truncated = x[:expected_nodes]
                        x = x_truncated.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                        print(f"[DEBUG] Truncated and reshaped to: [1, {channels}, {height}, {width}]")
                        
                    else:
                        # 节点数正好匹配
                        x = x.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                        print(f"[DEBUG] Exact match reshaped to: [1, {channels}, {height}, {width}]")
                else:
                    # 没有空间信息，尝试推断合理的空间维度
                    sqrt_nodes = int(num_nodes ** 0.5)
                    if sqrt_nodes * sqrt_nodes == num_nodes:
                        x = x.transpose(0, 1).contiguous().view(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                        height, width = sqrt_nodes, sqrt_nodes
                        print(f"[DEBUG] Inferred square reshape to: [1, {channels}, {sqrt_nodes}, {sqrt_nodes}]")
                    else:
                        # 作为1D特征处理
                        x = x.transpose(0, 1).contiguous().view(channels, num_nodes, 1).unsqueeze(0)
                        height, width = num_nodes, 1
                        print(f"[DEBUG] 1D reshape to: [1, {channels}, {num_nodes}, 1]")
                        
            elif x.dim() == 3:
                x = x.unsqueeze(0)
            elif x.dim() == 4:
                pass  # Already correct format
            else:
                raise ValueError(f"Unexpected tensor dimension: {x.dim()}, shape: {x.shape}")
            
            features_tensor.append(x)
            features_hw.append((height, width))
        else:
            if isinstance(f, torch.Tensor):
                if f.dim() == 2:
                    print(f"[WARNING] Processing bare 2D tensor: {f.shape}")
                    f = f.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [H, W] -> [1, W, H, 1]
                elif f.dim() == 3:
                    f = f.unsqueeze(0)
                features_tensor.append(f)
                features_hw.append((None, None))
            else:
                raise ValueError(f"Unknown feature type: {type(f)}")
    
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

        # 先创建一个临时的头，用于确定实际的通道数
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

    def _initialize_head_if_needed(self, fused_feat):
        if not self._head_initialized:
            # 从实际的融合特征中获取通道数
            actual_channels = []
            for f in fused_feat[:2]:  # 只取前两个特征层
                if hasattr(f, "x"):
                    channels = f.x.shape[1]  # [nodes, channels]
                    actual_channels.append(channels)
                else:
                    actual_channels.append(f.shape[1])  # 如果是普通tensor
            
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
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
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
            
            # 后处理输出格式
            if filtering:
                processed_outputs = torch.cat([
                    outputs[..., :4],    # bbox coordinates
                    outputs[..., 4:5],   # confidence scores  
                    outputs[..., 5:]     # class scores
                ], dim=-1)
                batch_detections = postprocess_network_output(processed_outputs, self.conf_threshold, self.nms_threshold)
                # 展平批次维度
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
            for i in range(len(event_feat)):
                event_feat[i].width = torch.tensor([image_feat[i].shape[-1]])
                event_feat[i].height = torch.tensor([image_feat[i].shape[-2]])
                print(f"[DEBUG] event_feat[{i}].x.shape = {event_feat[i].x.shape}")
                print(f"[DEBUG] image_feat[{i}].shape = {image_feat[i].shape}")
                event_feat[i].x = sampling_skip(event_feat[i], image_feat[i].detach())
        fused_feat = event_feat

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
            
            # 确保头部已经初始化
            self._initialize_head_if_needed(fused_feat)
            
            fused_feat_tensors, _ = unpack_fused_features(fused_feat)
            loss_fused = self.head(fused_feat_tensors, targets=targets, training=True)

            result = {}
            
            # 处理融合分支的损失
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

            # 处理CNN分支的损失
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

            # 确保至少有基本的损失项
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
        
        # 确保头部已经初始化
        self._initialize_head_if_needed(fused_feat)
        
        try:
            fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Unpacked features shapes: {[f.shape for f in fused_feat_x]}")
            outputs = self.head(fused_feat_x, training=False)
        except Exception as e:
            print(f"[ERROR] Failed to unpack features: {e}")
            print(f"[DEBUG] fused_feat types: {[type(f) for f in fused_feat]}")
            for i, f in enumerate(fused_feat):
                if hasattr(f, 'x'):
                    print(f"[DEBUG] fused_feat[{i}].x.shape: {f.x.shape}")
                else:
                    print(f"[DEBUG] fused_feat[{i}] is tensor with shape: {f.shape if hasattr(f, 'shape') else 'no shape'}")
            raise
            
        # 后处理输出格式，转换为检测结果
        if filtering:
            # 将 tensor 输出转换为正确格式供 postprocess_network_output 使用
            # 确保 outputs 是正确的格式: [batch_size, num_detections, 5 + num_classes]
            processed_outputs = torch.cat([
                outputs[..., :4],    # bbox coordinates 
                outputs[..., 4:5],   # confidence scores
                outputs[..., 5:]     # class scores
            ], dim=-1)
            batch_detections = postprocess_network_output(processed_outputs, self.conf_threshold, self.nms_threshold)
            # 展平批次维度，将所有检测结果合并到一个列表中
            outputs = []
            for batch_det in batch_detections:
                outputs.extend(batch_det)
        else:
            # 如果不需要过滤，直接格式化输出
            outputs = torch.cat([
                outputs[..., :4],
                outputs[..., 4:5], 
                outputs[..., 5:]
            ], dim=-1)
        
        ret = outputs  # 直接返回处理后的输出

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret = [outputs, targets]  # 返回 [detections, targets] 格式

        return ret
