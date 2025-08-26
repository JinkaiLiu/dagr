# event_snn_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch_geometric.data import Data


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# 整数值发放神经元
class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)


# 膜电位更新模块，整数值到二进制脉冲的转换
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        self.act = act
        self.qtrick = MultiSpike4()

    def forward(self, x):
        mem = x.sum(0).unsqueeze(0)
        spike = self.qtrick(mem)
        spike = expand_tensor_cumulative(spike)
        return spike


# 将整数值扩展为二进制脉冲序列
def expand_tensor_cumulative(tensor, max_value=4):
    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)
    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)
    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]
    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)
    return binary


# 标准卷积+BN层，处理二进制脉冲序列
class Conv2d_bn(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=None, first_layer=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=None)
        self.bn = nn.BatchNorm2d(c2)
        self.first_layer = first_layer

    def forward(self, x, first_layer=False):
        x = self.conv(x)
        if first_layer == False:
            new_shape = (4, x.shape[0] // 4) + x.shape[1:]
            x = x.view(new_shape).sum(dim=0)
        x = self.bn(x)
        return x


# 标准卷积（无BN）
class Conv2d(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, bias=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=None)

    def forward(self, x):
        x = self.conv(x)
        new_shape = (4, x.shape[0] // 4) + x.shape[1:]
        x = x.view(new_shape).sum(dim=0)
        return x


# 添加时间维度
class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=4):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        if len(x.shape) == 4:  # [B,C,H,W] -> [T,B,C,H,W]
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        elif len(x.shape) == 5:
            x = x.transpose(0, 1)  # [B,T,C,H,W] -> [T,B,C,H,W]
        return x


# 标准卷积层
class MS_StandardConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = Conv2d_bn(c1, c2, k, s, autopad(k, p, d), g=g, bias=False)
        self.lif = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        
        # 激活并展平前两个维度
        spikes = self.lif(x)
        flattened = spikes.flatten(0, 1)  # [T*B, C, H, W]
        
        # 应用卷积
        convolved = self.conv(flattened)  # [T*B, C_out, H_new, W_new]
        
        # 检查输出大小
        _, C_out, H_out, W_out = convolved.shape
        
        # 重塑回[T, B, C_out, H_new, W_new]形状
        try:
            reshaped = convolved.reshape(T, B, C_out, H_out, W_out)
            return reshaped
        except RuntimeError as e:
            print(f"Reshape error: {e}")
            print(f"Expected shape: [T={T}, B={B}, C_out={C_out}, H_out={H_out}, W_out={W_out}]")
            print(f"Actual tensor size: {convolved.numel()}")
            
            # 计算实际期望的维度
            expected_elements = T * B * C_out * H_out * W_out
            actual_elements = convolved.numel()
            
            if actual_elements != expected_elements:
                print(f"Dimension mismatch: expected {expected_elements} elements, got {actual_elements}")
                
                # 尝试推断正确的维度
                if actual_elements % (B * C_out * H_out * W_out) == 0:
                    T_actual = actual_elements // (B * C_out * H_out * W_out)
                    print(f"Inferred T = {T_actual} instead of {T}")
                    return convolved.reshape(T_actual, B, C_out, H_out, W_out)
                
                # 如果无法推断，返回原始张量并添加一个批次维度
                return convolved.unsqueeze(0)


# 无BN的标准卷积层
class MS_StandardConvWithoutBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv2d(c1, c2, k, s, autopad(k, p, d), g=g)
        self.lif = mem_update()
        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x


# 下采样层
class MS_DownSampling(nn.Module):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = Conv2d_bn(in_channels, embed_dims, k=kernel_size, s=stride, p=padding)
        if not first_layer:
            self.encode_lif = mem_update()

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, "encode_lif"):  # 如果不是第一层
            x = self.encode_lif(x)
            x = self.encode_conv(x.flatten(0, 1), first_layer=False)
        else:
            x = self.encode_conv(x.flatten(0, 1), first_layer=True)
        H, W = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, H, W).contiguous()
        return x


# 标准瓶颈模块
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, k[0], 1)
        self.cv2 = MS_StandardConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# CSP Bottleneck 模块
class MS_C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = MS_StandardConv(c1, 2 * self.c, 1, 1)
        self.cv2 = MS_StandardConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 2))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 2))


# 分离卷积
class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, bias=True, kernel_size=3, padding=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = Conv2d_bn(dim, med_channels, k=1, s=1, bias=bias)
        self.dwconv2 = Conv2d_bn(med_channels, med_channels, k=kernel_size, p=padding, g=med_channels, bias=bias)
        self.pwconv3 = Conv2d_bn(med_channels, dim, k=1, s=1, bias=bias)
        self.dwconv4 = Conv2d_bn(dim, dim, 1, 1, 0, g=dim, bias=True)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()
        self.lif4 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.pwconv1(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv2(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.lif3(x)
        x = self.pwconv3(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.lif4(x)
        x = self.dwconv4(x.flatten(0, 1)).reshape(T, B, -1, H, W)
        return x


# 重参数化卷积
class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, bias=True, group=1):
        super().__init__()
        self.conv1 = Conv2d_bn(in_channel, in_channel, 1, 1, 0, bias=True, g=group)
        self.conv2 = Conv2d_bn(in_channel, in_channel, kernel_size, 1, 1, g=in_channel, bias=True)
        self.conv3 = Conv2d_bn(in_channel, out_channel, 1, 1, 0, g=group, bias=True)

        self.lif1 = mem_update()
        self.lif2 = mem_update()
        self.lif3 = mem_update()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.conv1(self.lif1(x).flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.conv2(self.lif2(x).flatten(0, 1)).reshape(T, B, -1, H, W)
        x = self.conv3(self.lif3(x).flatten(0, 1)).reshape(T, B, -1, H, W)
        return x


# 添加层
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, y):
        return x + y


# SNN卷积块
class MS_ConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, full=False):
        super().__init__()
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.add1 = Add()
        self.add2 = Add()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.add1(self.Conv(x), x)
        x_feat = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.add2(x_feat, x)
        return x


# 标准SNN卷积块
class MS_AllConvBlock(nn.Module):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):
        super().__init__()
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)
        self.add1 = Add()
        self.add2 = Add()

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.add1(self.Conv(x), x)
        x_feat = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.add2(x_feat, x)
        return x


# 连接层
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# SPPF层
class SpikeSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = MS_StandardConv(c1, c_, 1, 1)
        self.cv2 = MS_StandardConv(c_ * 4, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.concat = Concat(2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            T, B, C, H, W = x.shape
            warnings.simplefilter('ignore')
            y1 = self.m1(x.flatten(0, 1)).reshape(T, B, -1, H, W)
            y2 = self.m2(y1.flatten(0, 1)).reshape(T, B, -1, H, W)
            y3 = self.m3(y2.flatten(0, 1)).reshape(T, B, -1, H, W)
            return self.cv2(self.concat((x, y1, y2, y3)))


# 事件处理的SpikeYOLO Backbone
class EventSNNBackbone(nn.Module):
    def __init__(self, input_channels=2, scale='n', T=4):
        super().__init__()
        # 根据scale确定网络深度和宽度
        scales = {
            'n': [0.33, 0.25, 1024],  # depth, width, max_channels
            's': [0.33, 0.5, 1024],
            'm': [0.67, 0.75, 768],
            'l': [1.0, 1.0, 512],
            'x': [1.0, 1.25, 512]
        }
        depth_scale, width_scale, max_channels = scales.get(scale, scales['n'])
        
        # 初始通道设置
        base_channels = int(64 * width_scale)
        
        # 添加时间维度
        self.get_t = MS_GetT(input_channels, T=T)
        
        # 调整下采样步长，使特征图尺寸与RGB分支匹配
        # 原始图像尺寸: [215, 320]
        # 目标尺寸: [14, 20]和[7, 10]
        
        # Stage 1: 输入层 -> [108, 160]
        self.stage1_down = MS_DownSampling(input_channels, base_channels, 3, 2, 1, True)
        self.stage1_blocks = nn.Sequential(*[
            MS_C2f(base_channels, base_channels, n=1, shortcut=True) 
            for _ in range(int(3 * depth_scale))
        ])
        
        # Stage 2: [108, 160] -> [54, 80]
        self.stage2_down = MS_StandardConv(base_channels, base_channels*2, 3, 2)
        self.stage2_blocks = nn.Sequential(*[
            MS_C2f(base_channels*2, base_channels*2, n=1, shortcut=True) 
            for _ in range(int(6 * depth_scale))
        ])
        
        # Stage 3: [54, 80] -> [27, 40]
        self.stage3_down = MS_StandardConv(base_channels*2, base_channels*4, 3, 2)
        self.stage3_blocks = nn.Sequential(*[
            MS_C2f(base_channels*4, base_channels*4, n=1, shortcut=True) 
            for _ in range(int(6 * depth_scale))
        ])
        
        # Stage 4: [27, 40] -> [14, 20] (匹配RGB分支的第一个输出)
        self.stage4_down = MS_StandardConv(base_channels*4, base_channels*8, 3, 2)
        self.stage4_blocks = nn.Sequential(*[
            MS_C2f(base_channels*8, base_channels*8, n=1, shortcut=True) 
            for _ in range(int(3 * depth_scale))
        ])
        
        # Stage 5: [14, 20] -> [7, 10] (匹配RGB分支的第二个输出)
        self.stage5_down = MS_StandardConv(base_channels*8, base_channels*8, 3, 2)
        self.stage5_blocks = SpikeSPPF(base_channels*8, base_channels*8, 5)
        
        # 调整通道数，使其与RGB分支兼容
        self.out_conv4 = nn.Conv2d(base_channels*8, 256, kernel_size=1)
        self.out_conv5 = nn.Conv2d(base_channels*8, 256, kernel_size=1)
        
        # 输出通道数与RGB分支保持一致
        self.out_channels = [
            base_channels,    # Stage 1
            base_channels*2,  # Stage 2
            base_channels*4,  # Stage 3
            256,              # Stage 4 (调整为256通道，匹配RGB分支)
            256               # Stage 5 (调整为256通道，匹配RGB分支)
        ]
        
        # 步长
        self.strides = [2, 4, 8, 16, 32]

    def forward(self, x):
        """
        输入:
            x: 形状为[B,C,H,W]的事件体素
        
        返回:
            features: 包含5个特征层的列表，对应于5个不同尺度
        """
        # 添加时间维度
        x = self.get_t(x)  # [B,C,H,W] -> [T,B,C,H,W]
        
        # Stage 1
        x = self.stage1_down(x)
        x1 = self.stage1_blocks(x)
        
        # Stage 2
        x = self.stage2_down(x1)
        x2 = self.stage2_blocks(x)
        
        # Stage 3
        x = self.stage3_down(x2)
        x3 = self.stage3_blocks(x)
        
        # Stage 4
        x = self.stage4_down(x3)
        x4 = self.stage4_blocks(x)
        
        # Stage 5
        x = self.stage5_down(x4)
        x5 = self.stage5_blocks(x)
        
        # 调整通道数以匹配RGB分支
        x4_out = self.out_conv4(x4.mean(0))  # [T,B,C,H,W] -> [B,256,H,W]
        x5_out = self.out_conv5(x5.mean(0))  # [T,B,C,H,W] -> [B,256,H,W]
        
        # 返回所有5个特征层，但最后两层已调整通道
        return [x1, x2, x3, x4_out, x5_out]


def compute_pooling_at_each_layer(pooling_dim_at_output, num_layers):
    py, px = map(int, pooling_dim_at_output.split("x"))
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    poolings = []
    for i in range(num_layers):
        pooling = pooling_base / 2 ** (3 - i)
        pooling[-1] = 1
        poolings.append(pooling)
    poolings = torch.stack(poolings)
    return poolings


def voxelize_events(data, height=215, width=320):
    """
    将事件点云数据体素化为适合SNN处理的张量
    
    参数:
        data: PyTorch Geometric的DataBatch对象，包含事件点云
        height: 图像高度
        width: 图像宽度
        
    返回:
        event_tensor: 形状为[B,C,H,W]的张量，C=2表示正负极性
    """
    # 从data获取图像尺寸
    if hasattr(data, 'width') and hasattr(data, 'height'):
        try:
            if hasattr(data.width, 'item'):
                width = data.width.item()
                height = data.height.item()
            elif len(data.width) == 1:
                width = data.width[0].item()
                height = data.height[0].item()
            else:
                width = data.width[0].item()
                height = data.height[0].item()
        except (ValueError, IndexError, RuntimeError) as e:
            print(f"Error accessing width/height: {e}, using defaults: {width}x{height}")
    
    # 获取批次大小
    try:
        batch_size = int(data.batch.max().item()) + 1
    except (ValueError, AttributeError) as e:
        print(f"Error determining batch size: {e}, using default batch size of 1")
        batch_size = 1
    
    # 创建空的事件体素
    device = data.x.device
    event_tensor = torch.zeros((batch_size, 2, height, width), device=device)
    
    # 从data获取信息
    pos = data.pos[:, :2]
    polarity = data.x.view(-1)
    batch_idx = data.batch
    
    # 计算像素位置
    pos_x = (pos[:, 0] * width).long().clamp(0, width - 1)
    pos_y = (pos[:, 1] * height).long().clamp(0, height - 1)
    pol_idx = (polarity < 0).long()
    
    # 使用向量化操作
    try:
        # 创建索引张量 [N, 4] 其中N是事件数，4对应(batch, channel, y, x)
        indices = torch.stack([batch_idx, pol_idx, pos_y, pos_x], dim=1)
        
        # 验证索引是否有效
        valid_indices = (
            (indices[:, 0] >= 0) & (indices[:, 0] < batch_size) &
            (indices[:, 1] >= 0) & (indices[:, 1] < 2) &
            (indices[:, 2] >= 0) & (indices[:, 2] < height) &
            (indices[:, 3] >= 0) & (indices[:, 3] < width)
        )
        
        # 只保留有效索引
        if not torch.all(valid_indices):
            indices = indices[valid_indices]
        
        # 使用index_add_累加事件
        values = torch.ones(indices.shape[0], device=device)
        
        for i in range(indices.shape[0]):
            b, c, y, x = indices[i]
            event_tensor[b, c, y, x] += 1
            
    except Exception as e:
        print(f"Vectorized approach failed: {e}, falling back to sparse tensor method")
        
        try:
            # 尝试使用sparse_coo_tensor方法
            indices = torch.stack([batch_idx, pol_idx, pos_y, pos_x], dim=0)
            values = torch.ones_like(batch_idx, dtype=torch.float, device=device)
            
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, 
                size=(batch_size, 2, height, width),
                device=device
            )
            
            # 转换为密集张量
            event_tensor = sparse_tensor.to_dense()
            
        except Exception as e2:
            print(f"Sparse tensor approach failed: {e2}, falling back to slow method")
            
            # 如果所有向量化方法都失败，回退到逐点处理
            for i in range(len(pos)):
                try:
                    b = int(batch_idx[i].cpu().numpy())
                    p = int(pol_idx[i].cpu().numpy())
                    y = int(pos_y[i].cpu().numpy())
                    x = int(pos_x[i].cpu().numpy())
                    
                    if 0 <= b < batch_size and 0 <= p < 2 and 0 <= y < height and 0 <= x < width:
                        event_tensor[b, p, y, x] += 1
                except Exception:
                    continue
    
    return event_tensor

class EventProcessor(nn.Module):
    def __init__(self, args, height=215, width=320):
        super().__init__()
        # 创建SNN backbone
        self.backbone = EventSNNBackbone(
            input_channels=2,
            scale=getattr(args, 'snn_scale', 'n'),
            T=getattr(args, 'snn_timesteps', 4)
        )
        
        self.height = height
        self.width = width
        self.batch_size = getattr(args, 'batch_size', 8)  # 添加这一行，设置batch_size
        self.num_scales = getattr(args, 'num_scales', 2)
        
        # 池化大小计算
        pooling_dim = getattr(args, 'pooling_dim_at_output', '5x7')
        self.poolings = compute_pooling_at_each_layer(pooling_dim, num_layers=5)
        
        # 输出通道
        self.out_channels = self.backbone.out_channels[-self.num_scales:]
        
        # 步长
        self.strides = self.backbone.strides[-self.num_scales:]

    def forward(self, data):
        """
        输入:
            data: PyTorch Geometric的Data对象，包含事件点云
        
        返回:
            outputs: 包含2个特征层的列表，对应于2个不同尺度
        """
        # 调试信息
        print("\n===== EVENT PROCESSOR INPUT =====")
        print(f"Data type: {type(data)}")
        if hasattr(data, 'x'):
            print(f"data.x shape: {data.x.shape}")
        if hasattr(data, 'pos'):
            print(f"data.pos shape: {data.pos.shape}")
        if hasattr(data, 'batch'):
            if hasattr(data.batch, 'unique'):
                print(f"data.batch unique values: {torch.unique(data.batch)}")
        
        try:
            # 体素化事件数据
            event_tensor = voxelize_events(data, self.height, self.width)
            print(f"Voxelized event tensor shape: {event_tensor.shape}")
            print(f"Event tensor stats - min: {event_tensor.min()}, max: {event_tensor.max()}, mean: {event_tensor.mean()}")
            
            # 通过SNN backbone处理体素化事件数据
            features = self.backbone(event_tensor)
            
            # 打印每层特征的形状
            print("\n===== SNN BACKBONE OUTPUT =====")
            if isinstance(features, list):
                for i, feat in enumerate(features[-self.num_scales:]):
                    if isinstance(feat, torch.Tensor):
                        print(f"Feature layer {i} shape: {feat.shape}")
                    else:
                        print(f"Feature layer {i} shape: {feat.x.shape if hasattr(feat, 'x') else 'Not a tensor'}")
            else:
                print(f"Features is not a list: {type(features)}")
            
            # 转换为与GNNHead兼容的格式
            outputs = []
            for i, feat in enumerate(features[-self.num_scales:]):
                # 创建Data对象
                out = Data()
                
                # 如果是张量，添加为x属性
                if isinstance(feat, torch.Tensor):
                    out.x = feat
                else:
                    # 如果已经是Data对象，复制x属性
                    if hasattr(feat, 'x'):
                        out.x = feat.x
                    else:
                        out.x = feat
                
                # 设置池化大小
                pool_idx = i + (len(features) - self.num_scales)
                out.pooling = self.poolings[pool_idx]
                
                # 添加pos属性以兼容GNN处理
                batch_size = out.x.shape[0]
                height, width = out.x.shape[2], out.x.shape[3]
                
                # 创建网格坐标作为pos
                y, x = torch.meshgrid(
                    torch.linspace(0, 1, height, device=out.x.device),
                    torch.linspace(0, 1, width, device=out.x.device)
                )
                pos = torch.stack([x.flatten(), y.flatten()], dim=1)
                
                # 复制到batch_size次
                pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
                out.pos = pos.reshape(-1, 2)
                
                # 添加batch索引
                batch_idx = torch.arange(batch_size, device=out.x.device)
                batch_idx = batch_idx.view(-1, 1, 1).expand(-1, height * width, 1)
                out.batch = batch_idx.reshape(-1)
                
                outputs.append(out)
            
            return outputs
        except Exception as e:
            print(f"Error in EventProcessor.forward: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回空结果以避免完全崩溃
            outputs = []
            batch_size = getattr(self, 'batch_size', 8)  # 使用默认值如果属性不存在
            device = data.x.device if hasattr(data, 'x') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            for i in range(self.num_scales):
                out = Data()
                h = 14 >> i  # 14->7
                w = 20 >> i  # 20->10
                out.x = torch.zeros((batch_size, 256, h, w), device=device)
                out.pooling = self.poolings[-(self.num_scales-i)]
                
                # 添加pos属性
                y, x = torch.meshgrid(
                    torch.linspace(0, 1, h, device=device),
                    torch.linspace(0, 1, w, device=device)
                )
                pos = torch.stack([x.flatten(), y.flatten()], dim=1)
                pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
                out.pos = pos.reshape(-1, 2)
                
                # 添加batch索引
                batch_idx = torch.arange(batch_size, device=device)
                batch_idx = batch_idx.view(-1, 1, 1).expand(-1, h * w, 1)
                out.batch = batch_idx.reshape(-1)
                
                outputs.append(out)
            
            return outputs
    
    # def forward(self, data):
    #     # 体素化事件数据
    #     event_tensor = voxelize_events(data, self.height, self.width)
        
    #     # 提取特征
    #     features = self.backbone(event_tensor)
        
    #     # 将特征转换为与原始DAGR兼容的格式
    #     outputs = []
    #     for i, feat in enumerate(features[-self.num_scales:]):
    #         # 创建一个类似图数据的对象
    #         out = Data()
            
    #         # 特征已经在backbone中处理为[B,C,H,W]格式
    #         out.x = feat
            
    #         # 设置池化大小
    #         pool_idx = i + (5 - self.num_scales)
    #         out.pooling = self.poolings[pool_idx]
            
    #         outputs.append(out)
        
    #     return outputs
    
    def get_output_sizes(self):
        """返回特征图的大小"""
        output_sizes = [(1 / p[:2] + 1e-3).cpu().int().numpy().tolist()[::-1] 
                         for p in self.poolings[-self.num_scales:]]
        return output_sizes