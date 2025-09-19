import torch
import torch.nn as nn

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


class TemporalAggHybrid(nn.Module):
    """
    Parallel fusion of temporal statistics:
      - mean   (temporal average, stable features)
      - std    (temporal variation/energy)
      - channel-wise temporal attention (adaptive weighting across T)

    Input:  p [T, B, C, H, W]  or [B, C, H, W] (auto-expanded to T=1)
    Output: [B, C, H, W]
    """
    def __init__(self, c_in: int, use_gn: bool = True):
        super().__init__()
        proj = [nn.Conv2d(3 * c_in, c_in, kernel_size=1, bias=False)]
        if use_gn:
            proj += [nn.GroupNorm(32, c_in)]
        proj += [nn.SiLU()]
        self.proj = nn.Sequential(*proj)

    @staticmethod
    def _attn_pool(p: torch.Tensor) -> torch.Tensor:
        """
        Channel-wise temporal attention pooling:
          1) Global average pool over spatial dims -> g [T,B,C]
          2) Softmax along T -> attention weights [T,B,C]
          3) Weighted sum across T -> [B,C,H,W]
        """
        g = p.mean(dim=(3, 4))          # [T, B, C]
        w = torch.softmax(g, dim=0)     # [T, B, C]
        w = w.unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
        return (p * w).sum(dim=0)       # [B, C, H, W]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # Compatibility: if input already has no temporal dim, treat as T=1
        if p.dim() == 4:
            p = p.unsqueeze(0)  # [1,B,C,H,W]

        # mean branch (stable temporal context)
        mu  = p.mean(dim=0)                            # [B,C,H,W]
        # std branch (temporal variation, motion strength)
        std = (p.var(dim=0, unbiased=False) + 1e-6).sqrt()  # [B,C,H,W]
        # attention branch (adaptive emphasis across T)
        att = self._attn_pool(p)                       # [B,C,H,W]

        # concatenate and project back to C channels
        x = torch.cat([mu, std, att], dim=1)           # [B,3C,H,W]
        return self.proj(x)                            # [B,C,H,W]


class SNNBackboneYAMLWrapper(nn.Module):
    def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        temporal_bins = getattr(args, 'snn_temporal_bins', 4)

        # YAMLBackbone will voxelize events to [T,B,2,H,W] and extract multi-scale features
        self.backbone = YAMLBackbone(
            yaml_path=yaml_path, scale=scale,
            in_ch=2, height=self.height, width=self.width,
            temporal_bins=temporal_bins
        )

        # Metadata for downstream detection head
        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.use_image = False
        self.is_snn = True
        self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

        # temporal aggregation (parallel fusion) modules
        self.agg_p4 = TemporalAggHybrid(c_in=256, use_gn=True)
        self.agg_p5 = TemporalAggHybrid(c_in=512, use_gn=True)

    def get_output_sizes(self):
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return [[h, w] for h, w in sizes]

    def forward(self, data, reset: bool = True):
        # Pass spatial metadata for voxelization inside YAMLBackbone
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width', self.width)

        # Backbone outputs multi-scale features: [T,B,C,H/s,W/s]
        p3, p4, p5 = self.backbone(data)

        # Replace simple mean with hybrid fusion: retains stable + dynamic + attention info
        p4_bchw = self.agg_p4(p4)   # [B,256,H/16,W/16]
        p5_bchw = self.agg_p5(p5)   # [B,512,H/32,W/32]

        return [p4_bchw, p5_bchw]


# import torch
# import torch.nn as nn

# from dagr.model.snn.snn_yaml_builder import YAMLBackbone


# class SNNBackboneYAMLWrapper(nn.Module):
#     def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
#         super().__init__()
#         self.height = int(height)
#         self.width = int(width)
#         temporal_bins = getattr(args, 'snn_temporal_bins', 4)
#         self.backbone = YAMLBackbone(yaml_path=yaml_path, scale=scale, in_ch=2, height=self.height, width=self.width, temporal_bins=temporal_bins)

#         self.out_channels = [256, 512]
#         self.strides = [16, 32]
#         self.num_scales = 2
#         self.use_image = False
#         self.is_snn = True
#         self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

#     def get_output_sizes(self):
#         sizes = []
#         for s in self.strides:
#             sizes.append([max(1, self.height // s), max(1, self.width // s)])
#         return [[h, w] for h, w in sizes]


#     def forward(self, data, reset: bool = True):
#         # pass Data to backbone; MS_GetT_Voxel will voxelize to [T,B,2,H,W]
#         setattr(data, 'meta_height', self.height)
#         setattr(data, 'meta_width', self.width)
#         p3, p4, p5 = self.backbone(data)
#         # aggregate time: mean over T -> BCHW
#         p4_bchw = p4.mean(dim=0)
#         p5_bchw = p5.mean(dim=0)
#         return [p4_bchw, p5_bchw]


