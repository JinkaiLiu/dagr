Wimport torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

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
        features = self.backbone(x)

        # CNN-only pretraining 模式
        if self.training and self.pretrain_cnn:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)
            return self.cnn_head(features["image"][:2], targets=targets, training=True)

        # 不使用 event 分支，只用 image 推理
        if not self.training and self.no_events:
            image_feat = features[2:4]
            outputs = self.cnn_head(image_feat, training=False)
            outputs = torch.cat([
                outputs[..., :4],
                outputs[..., 4:5],
                outputs[..., 5:]
            ], dim=-1)
            return [outputs]

        # 正常训练或推理，使用融合特征送入主干 FCOSHead
        event_feat = features[:2]
        image_feat = features[2:4]
        fused_feat = event_feat
        if self.use_image:
            image_feat = features[2:4]
            fused_feat = [a + b.detach() for a, b in zip(fused_feat, image_feat)]

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

        # 推理阶段
        x.reset = reset
        outputs = self.head(fused_feat, training=False)
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
