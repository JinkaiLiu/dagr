import torch
import torch.nn as nn

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


class SNNBackboneYAMLWrapper(nn.Module):
    def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        temporal_bins = getattr(args, 'snn_temporal_bins', 4)
        self.backbone = YAMLBackbone(yaml_path=yaml_path, scale=scale, in_ch=2, height=self.height, width=self.width, temporal_bins=temporal_bins)

        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.use_image = False
        self.is_snn = True
        self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return [[h, w] for h, w in sizes]


    def forward(self, data, reset: bool = True):
        # pass Data to backbone; MS_GetT_Voxel will voxelize to [T,B,2,H,W]
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)
        p3, p4, p5 = self.backbone(data)
        # aggregate time: mean over T -> BCHW
        p4_bchw = p4.mean(dim=0)
        p5_bchw = p5.mean(dim=0)
        return [p4_bchw, p5_bchw]


