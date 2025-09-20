import torch
import torch.nn as nn

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


def _make_group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """
    Uses min(max_groups, num_channels) to avoid 'num_channels % num_groups != 0' errors
    when channels are small or not divisible by 32.
    """
    num_groups = max(1, min(max_groups, num_channels))
    return nn.GroupNorm(num_groups, num_channels)


class TemporalAggHybrid(nn.Module):
    """
    Parallel fusion over the temporal axis (T) with stability tweaks.

    What it does (for p shaped [T, B, C, H, W]):
      1) mean  : temporal average (stable/steady context)
      2) std   : temporal standard deviation (variation / motion energy)
      3) attn  : channel-wise temporal attention (softmax across T per channel)

    The three branches are concatenated along channel dim -> [B, 3C, H, W],
    then projected back to [B, C, H, W] by a lightweight 1×1 Conv (+ optional GN + SiLU).
    Two stability features are included:
      - sqrt(T) scaling: prevents magnitude drop when slicing events into more bins.
      - residual-to-mean: start close to your old "mean over time" baseline, then learn gains.

    Inputs
    ------
    p : Tensor
        Either [T, B, C, H, W] or [B, C, H, W] (the latter auto-expanded to T=1).

    Output
    ------
    Tensor
        [B, C, H, W] — same spatial resolution and channels as a single temporal slice.
    """
    def __init__(
        self,
        c_in: int,
        use_gn: bool = True,
        residual: bool = True,
        gamma_init: float = 1.0,
        zero_init_proj: bool = True,
        scale_by_sqrt_t: bool = True,
    ):
        super().__init__()

        # 1×1 projection: [B, 3C, H, W] -> [B, C, H, W]
        layers = [nn.Conv2d(3 * c_in, c_in, kernel_size=1, bias=False)]
        if use_gn:
            layers += [_make_group_norm(c_in)]
        layers += [nn.SiLU()]
        self.proj = nn.Sequential(*layers)

        # (Optional) start exactly at the "mean" behavior:
        # zero init makes proj ≈ 0 at the beginning (so output ≈ residual branch).
        if zero_init_proj:
            nn.init.zeros_(self.proj[0].weight)

        # Residual to mean (learnable gate). Keeps behavior close to baseline at init,
        # then allows learning temporal gains on top.
        self.residual = residual
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # If True, multiply input feature sequence by sqrt(T) before statistics.
        # This keeps magnitude comparable when you move from T=1 to T>1.
        self.scale_by_sqrt_t = scale_by_sqrt_t

    @staticmethod
    def _attn_pool(p: torch.Tensor) -> torch.Tensor:
        """
        Channel-wise temporal attention pooling:
          g = GAP_spatial(p) -> [T, B, C]
          w = softmax_T(g)   -> [T, B, C]
          sum_t(p * w)       -> [B, C, H, W]
        """
        g = p.mean(dim=(3, 4))                               # [T, B, C]
        w = torch.softmax(g, dim=0).unsqueeze(-1).unsqueeze(-1)  # [T, B, C, 1, 1]
        return (p * w).sum(dim=0)                            # [B, C, H, W]

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # Accept both [B, C, H, W] and [T, B, C, H, W]
        if p.dim() == 4:
            p = p.unsqueeze(0)  # -> [1, B, C, H, W]
        T = int(p.shape[0])

        # Keep magnitude healthy when T>1 (mitigates "temporal mean dilution").
        if self.scale_by_sqrt_t and T > 1:
            p = p * (T ** 0.5)

        # Temporal statistics
        # mean: steady context; std: variation strength; attn: adaptive emphasis over T
        mu  = p.mean(dim=0)                                   # [B, C, H, W]
        std = (p.var(dim=0, unbiased=False) + 1e-6).sqrt()    # [B, C, H, W]
        att = self._attn_pool(p)                               # [B, C, H, W]

        # Fuse then project back to C channels
        x = torch.cat([mu, std, att], dim=1)                  # [B, 3C, H, W]
        out = self.proj(x)                                    # [B, C, H, W]

        # Residual to mean (optional). At init, output ≈ mean if proj is zero-initialized.
        return out + (self.gamma * mu if self.residual else 0.0)


class SNNBackboneYAMLWrapper(nn.Module):
    """
      - passes spatial metadata to YAMLBackbone (so it can voxelize events),
      - receives multi-scale temporal features shaped roughly [T, B, C, H/s, W/s],
      - aggregates the temporal axis with TemporalAggHybrid,
      - returns standard FPN-like feature maps as [B, C_l, H/s_l, W/s_l] for the head.
    Notes:
    * YAMLBackbone is expected to voxelize events into [T, B, 2, H, W], then extract features.
    * We keep output channels/strides consistent with your downstream head (e.g., FCOS/YOLOX).
    * TemporalAggHybrid keeps time information (steady + variation + adaptive emphasis)
      while remaining lightweight and stable for small batch training.
    """
    def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
        super().__init__()
        self.height = int(height)
        self.width  = int(width)

        temporal_bins = int(getattr(args, 'snn_temporal_bins', 4))

        # Backbone: will voxelize to [T, B, 2, H, W] internally and produce multi-scale features
        self.backbone = YAMLBackbone(
            yaml_path=yaml_path, scale=scale,
            in_ch=2, height=self.height, width=self.width,
            temporal_bins=temporal_bins
        )

        # Metadata for the detection head
        self.out_channels = [256, 512]  # p4, p5 channels
        self.strides      = [16, 32]
        self.num_scales   = 2
        self.use_image    = False
        self.is_snn       = True
        self.num_classes  = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

        # Temporal aggregation per scale (keeps interface unchanged: 256/512 channels out)
        self.agg_p4 = TemporalAggHybrid(
            c_in=256, use_gn=True,
            residual=True, gamma_init=1.0,
            zero_init_proj=True, scale_by_sqrt_t=True
        )
        self.agg_p5 = TemporalAggHybrid(
            c_in=512, use_gn=True,
            residual=True, gamma_init=1.0,
            zero_init_proj=True, scale_by_sqrt_t=True
        )

    def get_output_sizes(self):
        """
        returns [[H/stride, W/stride], ...] for each scale.
        """
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        return [[h, w] for h, w in sizes]

    def forward(self, data, reset: bool = True):
        # Provide spatial metadata so voxelizer can size the grids
        setattr(data, 'meta_height', self.height)
        setattr(data, 'meta_width',  self.width)

        # Backbone is expected to output: p3, p4, p5 each shaped ~ [T, B, C, H/s, W/s]
        p3, p4, p5 = self.backbone(data)  # p3 is unused here

        # Aggregate over T with hybrid fusion (steady + variation + attention) + stability tricks
        p4_bchw = self.agg_p4(p4)  # [B, 256, H/16, W/16]
        p5_bchw = self.agg_p5(p5)  # [B, 512, H/32, W/32]

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


