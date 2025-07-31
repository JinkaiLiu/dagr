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
    if not isinstance(prediction, torch.Tensor):
        print(f"[WARNING] postprocess_network_output received {type(prediction)}, returning empty")
        return []
        
    if prediction.numel() == 0:
        print(f"[WARNING] Empty prediction tensor")
        return []
    
    print(f"[DEBUG] Postprocessing prediction shape: {prediction.shape}")
    print(f"[DEBUG] Confidence threshold: {conf_thres}")
        
    batch_size = prediction.shape[0]
    
    if prediction.dim() == 3:
        num_classes = prediction.shape[2] - 5
    else:
        num_classes = prediction.shape[-1] - 5
    
    if torch.is_tensor(num_classes):
        num_classes = num_classes.item()
    num_classes = int(num_classes)
    
    print(f"[DEBUG] Detected num_classes: {num_classes}")
    
    output = []
    total_detections = 0
    
    for batch_idx in range(batch_size):
        image_pred = prediction[batch_idx]
        
        if image_pred.dim() == 1:
            image_pred = image_pred.unsqueeze(0)
            
        if image_pred.shape[0] == 0:
            output.append([])
            continue
        
        print(f"[DEBUG] Batch {batch_idx}: Processing {image_pred.shape[0]} predictions")
            
        conf_scores = image_pred[:, 4]
        print(f"[DEBUG] Batch {batch_idx}: Confidence range: {conf_scores.min().item():.4f} to {conf_scores.max().item():.4f}")
        
        conf_mask = conf_scores >= conf_thres
        above_threshold = conf_mask.sum().item()
        print(f"[DEBUG] Batch {batch_idx}: {above_threshold} detections above threshold {conf_thres}")
        
        if above_threshold == 0:
            k = min(5, len(conf_scores))
            if k > 0:
                _, top_indices = torch.topk(conf_scores, k)
                conf_mask = torch.zeros_like(conf_scores, dtype=torch.bool)
                conf_mask[top_indices] = True
                print(f"[DEBUG] Batch {batch_idx}: Using top-{k} detections as fallback")
        
        image_pred = image_pred[conf_mask]
        
        if image_pred.shape[0] == 0:
            output.append([])
            continue
        
        if num_classes > 0:
            class_scores = image_pred[:, 5:5 + num_classes]
            class_conf, class_pred = torch.max(class_scores, 1, keepdim=True)
            
            detections = torch.cat((
                image_pred[:, :4],
                image_pred[:, 4:5],
                class_conf,
                class_pred.float()
            ), 1)
        else:
            detections = torch.cat((
                image_pred[:, :4],
                image_pred[:, 4:5],
                torch.ones(image_pred.shape[0], 1, device=image_pred.device),
                torch.zeros(image_pred.shape[0], 1, device=image_pred.device)
            ), 1)
        
        batch_detections = []
        if detections.shape[0] > 0:
            boxes_xyxy = detections[:, :4]
            valid_boxes = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
            
            if valid_boxes.sum() > 0:
                detections = detections[valid_boxes]
                
                boxes = detections[:, :4].cpu()
                scores = detections[:, 4].cpu()
                labels = detections[:, 6].cpu().long() if num_classes > 0 else torch.zeros(detections.shape[0]).long()
                
                det_dict = {
                    'boxes': boxes,
                    'scores': scores,  
                    'labels': labels
                }
                batch_detections.append(det_dict)
                total_detections += len(boxes)
                
                print(f"[DEBUG] Batch {batch_idx}: Generated {len(boxes)} valid detections")
            else:
                print(f"[DEBUG] Batch {batch_idx}: No valid boxes after filtering")
        
        output.append(batch_detections)
    
    print(f"[DEBUG] Total detections across all batches: {total_detections}")
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
    
    target_batch_size = None
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            batch_info = f.batch if hasattr(f, "batch") else None
            if batch_info is not None:
                current_batch_size = len(batch_info.unique())
                if target_batch_size is None:
                    target_batch_size = current_batch_size
                else:
                    target_batch_size = min(target_batch_size, current_batch_size)
    
    print(f"[DEBUG] Target batch size: {target_batch_size}")
    
    for i, f in enumerate(fused_feat):
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            x = f.x
            height = f.height.item() if hasattr(f, "height") and torch.is_tensor(f.height) else None
            width = f.width.item() if hasattr(f, "width") and torch.is_tensor(f.width) else None
            batch_info = f.batch if hasattr(f, "batch") else None
            
            print(f"[DEBUG] Feature {i}: x.shape = {x.shape}, height = {height}, width = {width}")
            if batch_info is not None:
                unique_batches = batch_info.unique()
                print(f"[DEBUG] Feature {i}: batch info available, unique batches = {unique_batches}")
            
            if x.dim() == 2:
                num_nodes, channels = x.shape
                
                if height is not None and width is not None and batch_info is not None and target_batch_size is not None:
                    batch_size = target_batch_size
                    expected_nodes_per_batch = height * width
                    
                    print(f"[DEBUG] Reconstructing batch dimension: batch_size={batch_size}, nodes_per_batch={expected_nodes_per_batch}")
                    
                    full_tensor = torch.zeros(batch_size, channels, height, width, device=x.device, dtype=x.dtype)
                    
                    for batch_idx in range(batch_size):
                        if batch_idx < len(batch_info.unique()):
                            batch_mask = batch_info == batch_idx
                            batch_nodes = x[batch_mask]
                            
                            if batch_nodes.shape[0] > 0:
                                max_nodes = min(batch_nodes.shape[0], expected_nodes_per_batch)
                                
                                if max_nodes == expected_nodes_per_batch:
                                    reshaped = batch_nodes[:max_nodes].transpose(0, 1).view(channels, height, width)
                                else:
                                    temp_features = torch.zeros(expected_nodes_per_batch, channels, device=x.device, dtype=x.dtype)
                                    temp_features[:max_nodes] = batch_nodes[:max_nodes]
                                    reshaped = temp_features.transpose(0, 1).view(channels, height, width)
                                
                                full_tensor[batch_idx] = reshaped
                    
                    x = full_tensor
                    print(f"[DEBUG] Reconstructed to batch tensor: {x.shape}")
                    
                elif height is not None and width is not None:
                    expected_nodes = height * width
                    
                    if num_nodes <= expected_nodes:
                        full_features = torch.zeros(expected_nodes, channels, device=x.device, dtype=x.dtype)
                        full_features[:num_nodes] = x
                        x = full_features.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    else:
                        x_truncated = x[:expected_nodes]
                        x = x_truncated.transpose(0, 1).contiguous().view(channels, height, width).unsqueeze(0)
                    
                    if target_batch_size is not None and target_batch_size > 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    
                    print(f"[DEBUG] Single batch reshaped to: {x.shape}")
                else:
                    sqrt_nodes = int(num_nodes ** 0.5)
                    if sqrt_nodes * sqrt_nodes == num_nodes:
                        x = x.transpose(0, 1).contiguous().view(channels, sqrt_nodes, sqrt_nodes).unsqueeze(0)
                        height, width = sqrt_nodes, sqrt_nodes
                    else:
                        x = x.transpose(0, 1).contiguous().view(channels, num_nodes, 1).unsqueeze(0)
                        height, width = num_nodes, 1
                    
                    if target_batch_size is not None and target_batch_size > 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                        
                    print(f"[DEBUG] Inferred reshape to: {x.shape}")
                        
            elif x.dim() == 3:
                x = x.unsqueeze(0)
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] == 1:
                        x = x.repeat(target_batch_size, 1, 1, 1)
                    else:
                        x = x[:target_batch_size]
            elif x.dim() == 4:
                if target_batch_size is not None and x.shape[0] != target_batch_size:
                    if x.shape[0] > target_batch_size:
                        x = x[:target_batch_size]
                    else:
                        repeats = target_batch_size // x.shape[0]
                        remainder = target_batch_size % x.shape[0]
                        if repeats > 0:
                            x_repeated = x.repeat(repeats, 1, 1, 1)
                            if remainder > 0:
                                x = torch.cat([x_repeated, x[:remainder]], dim=0)
                            else:
                                x = x_repeated
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
                
                if target_batch_size is not None and f.shape[0] != target_batch_size:
                    if f.shape[0] == 1:
                        f = f.repeat(target_batch_size, 1, 1, 1)
                    else:
                        f = f[:target_batch_size]
                        
                features_tensor.append(f)
                features_hw.append((None, None))
            else:
                raise ValueError(f"Unknown feature type: {type(f)}")
    
    print(f"[DEBUG] Final unpacked shapes: {[f.shape for f in features_tensor]}")
    return features_tensor, features_hw


class DAGR(nn.Module):
    def __init__(self, args, height, width):
        super().__init__()
        self.conf_threshold = 0.01  # 降低推理阈值
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
        targets = []
        
        for batch_idx in range(num_graphs):
            batch_mask = bbox_batch == batch_idx
            batch_bboxes = bbox[batch_mask]
            
            if batch_bboxes.numel() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
            
            x_left = batch_bboxes[:, 0]
            y_top = batch_bboxes[:, 1]
            height = batch_bboxes[:, 2]
            width = batch_bboxes[:, 3]
            cls = batch_bboxes[:, 4]
            
            areas = width * height
            valid_mask = areas > 0
            
            if valid_mask.sum() == 0:
                targets.append(torch.zeros(0, 5, device=bbox.device))
                continue
                
            x_left = x_left[valid_mask]
            y_top = y_top[valid_mask]
            height = height[valid_mask]
            width = width[valid_mask]
            cls = cls[valid_mask]
            
            x_center = x_left + width / 2
            y_center = y_top + height / 2
            
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
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    batch_detections = postprocess_network_output(outputs, self.conf_threshold, self.nms_threshold)
                    outputs = []
                    for batch_det in batch_detections:
                        outputs.extend(batch_det)
                else:
                    outputs = []
            
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

        # 推理模式
        for i, f in enumerate(fused_feat):
            print(f"[DEBUG] Inference Feature {i}: x.shape = {f.x.shape}, height = {getattr(f, 'height', 'N/A')}, width = {getattr(f, 'width', 'N/A')}")

        x.reset = reset
        
        self._initialize_head_if_needed(fused_feat)
        
        try:
            fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
            print(f"[DEBUG] Inference unpacked features shapes: {[f.shape for f in fused_feat_x]}")
            
            valid_features = []
            for i, feat in enumerate(fused_feat_x):
                if feat.numel() > 0 and all(d > 0 for d in feat.shape):
                    valid_features.append(feat)
                    print(f"[DEBUG] Inference feature {i} is valid: {feat.shape}")
                else:
                    print(f"[WARNING] Inference feature {i} is invalid: {feat.shape}, numel={feat.numel()}")
            
            if len(valid_features) == 0:
                print(f"[ERROR] No valid features after unpacking in inference!")
                ret = []
                if return_targets and hasattr(x, 'bbox'):
                    targets = convert_to_evaluation_format(x)
                    ret = [ret, targets]
                return ret
            
            print(f"[DEBUG] Calling head in inference mode with {len(valid_features)} features")
            outputs = self.head(valid_features, training=False)
            print(f"[DEBUG] Raw inference outputs type: {type(outputs)}")
            
            if isinstance(outputs, torch.Tensor):
                print(f"[DEBUG] Raw inference outputs shape: {outputs.shape}")
                print(f"[DEBUG] Raw inference outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                
                if outputs.dim() >= 3:
                    print(f"[DEBUG] Confidence scores range: {outputs[..., 4].min().item():.4f} to {outputs[..., 4].max().item():.4f}")
                    if outputs.shape[-1] > 5:
                        print(f"[DEBUG] Class scores range: {outputs[..., 5:].min().item():.4f} to {outputs[..., 5:].max().item():.4f}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process features in inference: {e}")
            import traceback
            traceback.print_exc()
            
            ret = []
            if return_targets and hasattr(x, 'bbox'):
                targets = convert_to_evaluation_format(x)
                ret = [ret, targets]
            return ret
            
        if filtering:
            try:
                if isinstance(outputs, torch.Tensor) and outputs.numel() > 0:
                    print(f"[DEBUG] Filtering detections with conf_threshold={self.conf_threshold}")
                    
                    if outputs.dim() >= 3:
                        conf_scores = outputs[..., 4]
                        above_threshold = (conf_scores > self.conf_threshold).sum().item()
                        print(f"[DEBUG] Detections above threshold {self.conf_threshold}: {above_threshold}")
                        
                        if above_threshold == 0:
                            temp_threshold = 0.001
                            above_temp = (conf_scores > temp_threshold).sum().item()
                            print(f"[DEBUG] Detections above temp threshold {temp_threshold}: {above_temp}")
                            
                            original_threshold = self.conf_threshold
                            self.conf_threshold = temp_threshold
                            batch_detections = postprocess_network_output(outputs, self.conf_threshold, self.nms_threshold)
                            self.conf_threshold = original_threshold
                            
                            print(f"[DEBUG] Generated detections with lowered threshold: {len(batch_detections)}")
                        else:
                            batch_detections = postprocess_network_output(outputs, self.conf_threshold, self.nms_threshold)
                    else:
                        batch_detections = postprocess_network_output(outputs, self.conf_threshold, self.nms_threshold)
                    
                    outputs = []
                    for batch_det in batch_detections:
                        outputs.extend(batch_det)
                    
                    print(f"[DEBUG] Final detection count: {len(outputs)}")
                    
                else:
                    print(f"[DEBUG] Invalid outputs for filtering: type={type(outputs)}, numel={outputs.numel() if hasattr(outputs, 'numel') else 'N/A'}")
                    outputs = []
                    
            except Exception as e:
                print(f"[ERROR] Failed to postprocess outputs: {e}")
                print(f"[DEBUG] outputs info: type={type(outputs)}, shape={outputs.shape if hasattr(outputs, 'shape') else 'no shape'}")
                outputs = []
        else:
            if not isinstance(outputs, torch.Tensor):
                print(f"[WARNING] Expected tensor output, got {type(outputs)}")
                outputs = torch.zeros(1, 1, 5 + self.backbone.num_classes, device=next(self.parameters()).device)
        
        ret = outputs

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret = [outputs, targets]

        return ret
