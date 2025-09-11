import torch
import torch.nn as nn

from dagr.model.snn.snn_yaml_builder import YAMLBackbone


class SNNBackboneYAMLWrapper(nn.Module):
    def __init__(self, args, height: int, width: int, yaml_path: str, scale: str = 's'):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.backbone = YAMLBackbone(yaml_path=yaml_path, scale=scale, in_ch=2)

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

    @staticmethod
    def _events_to_frames(data: 'torch_geometric.data.Data', height: int, width: int) -> torch.Tensor:
        device = data.x.device
        batch_size = int(getattr(data, 'num_graphs', 1))
        frames = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=device)

        if hasattr(data, 'batch') and data.batch is not None:
            b = data.batch.long()
        else:
            b = torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device)

        x_norm = data.pos[:, 0]
        y_norm = data.pos[:, 1]
        x_pix = torch.clamp((x_norm * (width - 1)).round().long(), 0, width - 1)
        y_pix = torch.clamp((y_norm * (height - 1)).round().long(), 0, height - 1)

        p = (data.x[:, 0] > 0).long()
        frames.index_put_((b, p, y_pix, x_pix), torch.ones_like(p, dtype=frames.dtype), accumulate=True)
        frames.clamp_(max=1.0)
        return frames  # [B,2,H,W]

    def forward(self, data, reset: bool = True):
        frames = self._events_to_frames(data, self.height, self.width)
        p3, p4, p5 = self.backbone(frames)
        # aggregate time: mean over T -> BCHW
        p4_bchw = p4.mean(dim=0)
        p5_bchw = p5.mean(dim=0)
        return [p4_bchw, p5_bchw]


