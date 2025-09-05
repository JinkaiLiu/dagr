import torch
import torch.nn as nn


class MemUpdate(nn.Module):
    """
    Minimal spiking placeholder that simply forwards inputs.
    Keeps the T x B x C x H x W contract without modification.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SpikeConv(nn.Module):
    """
    2D convolution applied to each time-step independently.
    Input:  x in shape [T, B, C, H, W]
    Output: y in shape [T, B, C_out, H_out, W_out]
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mem = MemUpdate()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x = self.mem(x)
        x = self.conv(x.flatten(0, 1))
        x = self.bn(x)
        C_out, H_out, W_out = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(T, B, C_out, H_out, W_out)
        return x


class DownSampling(nn.Module):
    """
    Downsampling block implemented as a stride>1 SpikeConv.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = SpikeConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AllConvBlock(nn.Module):
    """
    Lightweight residual block composed of two SpikeConv layers.
    Keeps channels constant.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = SpikeConv(channels, channels, kernel_size=kernel_size, stride=1)
        self.conv2 = SpikeConv(channels, channels, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class ConvBlock(nn.Module):
    """
    Another lightweight residual block to mimic MS_ConvBlock behavior.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = SpikeConv(channels, channels, kernel_size=kernel_size, stride=1)
        self.conv2 = SpikeConv(channels, channels, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class SpikeSPPF(nn.Module):
    """
    SPPF-like module adapted for [T, B, C, H, W] tensors.
    Reduces then aggregates multi-scale pooled features.
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = SpikeConv(in_channels, hidden, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = SpikeConv(hidden * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        x = self.cv1(x)
        T, B, C, H, W = x.shape
        y1 = self.pool(x.flatten(0, 1)).view(T, B, C, H, W)
        y2 = self.pool(y1.flatten(0, 1)).view(T, B, C, H, W)
        y3 = self.pool(y2.flatten(0, 1)).view(T, B, C, H, W)
        x = torch.cat((x, y1, y2, y3), dim=2)  # concat on channels
        x = self.cv2(x)
        return x


class SNNBackbone(nn.Module):
    """
    Minimal SNN backbone approximating snn_yolov8.yaml (scale 's') backbone.
    Outputs three feature maps (P3, P4, P5) at strides 8, 16, 32 respectively.
    All tensors use [T, B, C, H, W].
    """
    def __init__(self, in_channels: int = 2):
        super().__init__()
        # Stage P2 -> P3 (overall stride 8)
        self.down1 = DownSampling(in_channels, 128, kernel_size=7, stride=4)
        self.b1 = nn.Sequential(
            AllConvBlock(128), AllConvBlock(128), AllConvBlock(128)
        )

        # Stage P3 -> P4 (stride 16)
        self.down2 = DownSampling(128, 256, kernel_size=3, stride=2)
        self.b2 = nn.Sequential(
            AllConvBlock(256), AllConvBlock(256), AllConvBlock(256),
            AllConvBlock(256), AllConvBlock(256), AllConvBlock(256)
        )

        # Stage P4 -> P5 (stride 32)
        self.down3 = DownSampling(256, 512, kernel_size=3, stride=2)
        self.b3 = nn.Sequential(
            ConvBlock(512), ConvBlock(512), ConvBlock(512)
        )

        self.sppf = SpikeSPPF(512, 512, k=5)

    def forward(self, x: torch.Tensor):
        # x: [T, B, 2, H, W]
        x = self.down1(x)
        p3 = self.b1(x)

        x = self.down2(p3)
        p4 = self.b2(x)

        x = self.down3(p4)
        x = self.b3(x)
        p5 = self.sppf(x)

        return [p3, p4, p5]


class SNNBackboneWrapper(nn.Module):
    """
    Adapter around SNNBackbone to:
    - Convert event Data (torch_geometric.data.Data) to dense frames [T=1, B, C=2, H, W]
    - Run SNN backbone and aggregate over time (mean over T)
    - Return two-scale BCHW features (P4, P5) and expose channels/strides for YOLOXHead
    """
    def __init__(self, args, height: int, width: int):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.backbone = SNNBackbone(in_channels=2)
        # choose P4 and P5
        self.out_channels = [256, 512]
        self.strides = [16, 32]
        self.num_scales = 2
        self.use_image = False
        self.is_snn = True
        self.num_classes = dict(dsec=2, ncaltech101=100).get(getattr(args, 'dataset', 'dsec'), 2)

    def get_output_sizes(self):
        # sizes for P4, P5 given current H, W
        sizes = []
        for s in self.strides:
            sizes.append([max(1, self.height // s), max(1, self.width // s)])
        # return as [H, W] per original convention
        return [[h, w] for h, w in sizes]

    @staticmethod
    def _events_to_frames(data: 'torch_geometric.data.Data', height: int, width: int) -> torch.Tensor:
        device = data.x.device
        if hasattr(data, 'num_graphs'):
            batch_size = int(data.num_graphs)
        else:
            batch_size = 1

        frames = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=device)

        if hasattr(data, 'batch') and data.batch is not None:
            b = data.batch.long()
        else:
            b = torch.zeros((data.pos.shape[0],), dtype=torch.long, device=device)

        # pos was normalized in format_data: pos[:,0]=x/W, pos[:,1]=y/H
        x_norm = data.pos[:, 0]
        y_norm = data.pos[:, 1]
        x_pix = torch.clamp((x_norm * (width - 1)).round().long(), 0, width - 1)
        y_pix = torch.clamp((y_norm * (height - 1)).round().long(), 0, height - 1)

        # polarity in data.x[:,0] set to {-1, 1}; map to {0,1}
        p = (data.x[:, 0] > 0).long()

        frames.index_put_((b, p, y_pix, x_pix), torch.ones_like(p, dtype=frames.dtype), accumulate=True)
        frames.clamp_(max=1.0)
        # add time dimension T=1
        return frames.unsqueeze(0)

    def forward(self, data, reset: bool = True):
        # Convert events to spike frames
        frames = self._events_to_frames(data, self.height, self.width)
        # Run backbone -> [P3, P4, P5] in [T,B,C,H,W]
        feats_t = self.backbone(frames)
        # Aggregate time (mean over T); pick P4, P5
        p4 = feats_t[1].mean(dim=0)
        p5 = feats_t[2].mean(dim=0)
        return [p4, p5]


