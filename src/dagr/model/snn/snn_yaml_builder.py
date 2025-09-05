import yaml
from typing import List, Tuple

import torch
import torch.nn as nn

from dagr.model.snn.snn_modules import (
    MS_GetT,
    MS_CancelT,
    MS_DownSampling,
    MS_AllConvBlock,
    MS_ConvBlock,
    MS_StandardConv,
    SpikeSPPF,
)


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor - 1) // divisor) * divisor


def parse_model(d: dict, ch: int, scale: str = 's', verbose: bool = False) -> Tuple[nn.Sequential, List[int], List[int]]:
    scales = d.get('scales') or {}
    depth, width, max_channels = scales.get(scale, (1.0, 1.0, float('inf')))

    layers: list = []
    save: list = []
    ch_list: list = [ch]
    tap_indices: List[int] = []

    backbone = d['backbone']
    for i, (f, n, m, args) in enumerate(backbone):
        if isinstance(m, str) and m.startswith('nn.'):
            raise NotImplementedError('nn.* modules are not required in backbone for this integration')

        module = globals()[m]
        n_ = int(round(n * depth)) if n > 1 else n

        # resolve any string args like '(1,2,2)'
        resolved_args = []
        for a in args:
            if isinstance(a, str) and a.startswith('(') and a.endswith(')'):
                resolved_args.append(eval(a))
            else:
                resolved_args.append(a)
        args = resolved_args

        if module is MS_GetT:
            c1 = ch_list[f]
            c2 = args[0]
            # Guard: MS_GetT should only appear at the first layer
            if i != 0:
                raise ValueError('MS_GetT is only allowed at the first layer (i == 0).')
            mod_args = [c1, c2, *args[1:]]
            c_out = c1
        elif module is MS_CancelT:
            c1 = ch_list[f]
            c2 = args[0]
            mod_args = [c1, c2, *args[1:]]
            c_out = c1
        elif module is MS_DownSampling:
            c1 = ch_list[f]
            c2 = _make_divisible(min(int(args[0] * width), int(max_channels)))
            mod_args = [c1, c2, *args[1:]]
            c_out = c2
        elif module in (MS_ConvBlock, MS_AllConvBlock):
            c1 = ch_list[f]
            c2 = c1
            mod_args = [c1, *args]
            c_out = c1
        elif module is MS_StandardConv:
            c1 = ch_list[f]
            c2 = _make_divisible(min(int(args[0] * width), int(max_channels)))
            mod_args = [c1, c2, *args[1:]]
            c_out = c2
        elif module is SpikeSPPF:
            c1 = ch_list[f]
            c2 = args[0]
            mod_args = [c1, c2, *args[1:]]
            c_out = c2
        else:
            raise NotImplementedError(f'Module {m} not supported in backbone parser')

        m_ = nn.Sequential(*(module(*mod_args) for _ in range(n_))) if n_ > 1 else module(*mod_args)
        m_.i, m_.f, m_.type = i, f, m,
        layers.append(m_)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

        # track output channels correctly per module
        ch_list.append(c_out)

        # tap indices for P3/P4/P5 based on YAML order (indices 4,6,9 after parse)
        if i in (4, 6, 9):
            tap_indices.append(i)

    return nn.Sequential(*layers), sorted(set(save)), tap_indices


class YAMLBackbone(nn.Module):
    def __init__(self, yaml_path: str, scale: str = 's', in_ch: int = 2):
        super().__init__()
        with open(yaml_path, 'r') as f:
            d = yaml.safe_load(f)
        self.model, self.save, self.tap_indices = parse_model(d, ch=in_ch, scale=scale, verbose=False)

    def forward(self, x4d: torch.Tensor):
        # x4d: [B, C=2, H, W]
        y = []
        x = x4d
        taps = {}
        for m in self.model:
            if m.f != -1:
                x = y[m.f]
            x = m(x)
            y.append(x)
            if m.i in self.tap_indices:
                taps[m.i] = x
        # Order taps by indices: expect [4,6,9]
        p3 = taps.get(self.tap_indices[0])
        p4 = taps.get(self.tap_indices[1])
        p5 = taps.get(self.tap_indices[2])
        return [p3, p4, p5]


