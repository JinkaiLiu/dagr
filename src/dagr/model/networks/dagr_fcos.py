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
    postprocess_network_output,
    init_subnetwork
)

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
    for f in fused_feat:
        if hasattr(f, "x") and isinstance(f.x, torch.Tensor):
            features_tensor.append(f.x)
            # 记录 height 和 width, int 而不是 tensor
            height = f.height.item() if hasattr(f, "height") else None
            width = f.width.item() if hasattr(f, "width") else None
            features_hw.append((height, width))
        else:
            # fallback 情况，例如纯 tensor，不是 PyG Data
            features_tensor.append(f)
            features_hw.append((None, None))
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

        # 主干 FCOSHead，用于融合特征
        self.head = FCOSHead(
            num_classes=self.backbone.num_classes,
            in_channels=self.backbone.out_channels[:2],
            strides=self.backbone.strides[:2],
            use_focal_loss=True,
            use_iou_loss=True
        )

        # CNN-only image 分支也使用 FCOSHead
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

    def forward(self, x, reset=True, return_targets=True, filtering=True):
        if self.use_image:
            event_feat, image_feat = self.backbone(x)
        else:
            event_feat = self.backbone(x)
            image_feat = None
        # CNN-only pretraining 模式
        if self.training and self.pretrain_cnn:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
            return self.cnn_head(image_feat, targets=targets, training=True)

        # 不使用 event 分支，只用 image 推理
        if not self.training and self.no_events:
            outputs = self.cnn_head(image_feat, training=False)
            outputs = torch.cat([
                outputs[..., :4],
                outputs[..., 4:5],
                outputs[..., 5:]
            ], dim=-1)
            return [outputs]

        # 正常训练或推理，使用融合特征送入主干 FCOSHead
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
            loss_fused = self.head(fused_feat, targets=targets, training=True)

            if self.use_image:
                loss_image = self.cnn_head(image_feat, targets=targets, training=True)
                return {
                    "loss": loss_fused["loss"] + loss_image["loss"],
                    "fused": loss_fused,
                    "image": loss_image
                }

            return loss_fused
        for i, f in enumerate(fused_feat):
            print(f"[DEBUG] Feature {i}: x.shape = {f.x.shape}, height = {getattr(f, 'height', 'N/A')}, width = {getattr(f, 'width', 'N/A')}")

        # 推理阶段
        x.reset = reset
        fused_feat_x, fused_hw = unpack_fused_features(fused_feat)
        outputs = self.head(fused_feat_x, hw=fused_hw, training=False)
        outputs = torch.cat([
            outputs[..., :4],
            outputs[..., 4:5],
            outputs[..., 5:]
        ], dim=-1)

        ret = [outputs]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret
