"""Core modules used by SD-MSCF-DET."""

import torch
import torch.nn as nn

from ultralytics.utils.ops import make_divisible

from .block import C2PSA, C3k
from .conv import Conv, autopad

__all__ = ("MSCF_C3", "SD_PSA")


class MSCFContextAttention(nn.Module):
    """Aggregate horizontal and vertical context for MSCF features."""

    def __init__(self, channels, h_kernel_size=11, v_kernel_size=11):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(channels, channels)
        self.h_conv = nn.Conv2d(
            channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels
        )
        self.v_conv = nn.Conv2d(
            channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels
        )
        self.conv2 = Conv(channels, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))


class MSCFBlock(nn.Module):
    """Multi-scale convolutional feature block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(3, 5, 7, 9, 11),
        expansion=1.0,
        with_context=True,
        context_kernel_size=11,
        add_identity=True,
    ):
        super().__init__()
        hidden_channels = make_divisible(int(out_channels * expansion), 8)
        self.pre_conv = Conv(in_channels, hidden_channels)
        self.depthwise_convs = nn.ModuleList(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=autopad(kernel_size),
                groups=hidden_channels,
            )
            for kernel_size in kernel_sizes
        )
        self.pointwise_conv = Conv(hidden_channels, hidden_channels)
        self.post_conv = Conv(hidden_channels, out_channels)
        self.context = (
            MSCFContextAttention(hidden_channels, context_kernel_size, context_kernel_size) if with_context else None
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        x = self.pre_conv(x)
        context = x
        x = self.depthwise_convs[0](x)
        x = torch.stack([x] + [layer(x) for layer in self.depthwise_convs[1:]], dim=0).sum(dim=0)
        x = self.pointwise_conv(x)
        if self.context is not None:
            context = self.context(context)
        gated = x * context
        x = x + gated if self.add_identity else gated
        return self.post_conv(x)


class C3kMSCF(C3k):
    """C3k block whose bottlenecks are replaced by MSCF blocks."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        kernel_sizes=(3, 5, 7, 9, 11),
        expansion=1.0,
        with_context=True,
        context_kernel_size=11,
        add_identity=True,
        g=1,
        e=0.5,
        k=3,
    ):
        super().__init__(c1, c2, n, True, g, e, k)
        hidden_channels = int(c2 * e)
        self.m = nn.Sequential(
            *(
                MSCFBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_sizes,
                    expansion,
                    with_context,
                    context_kernel_size,
                    add_identity,
                )
                for _ in range(n)
            )
        )


class MSCF_C3(nn.Module):
    """CSP-style MSCF-C3 feature extraction module."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        kernel_sizes=(3, 5, 7, 9, 11),
        expansion=1.0,
        with_context=True,
        context_kernel_size=11,
        add_identity=True,
        c3k=False,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        block = C3kMSCF if c3k else MSCFBlock
        self.m = nn.ModuleList(
            block(
                self.c,
                self.c,
                2,
                kernel_sizes,
                expansion,
                with_context,
                context_kernel_size,
                add_identity,
            )
            if c3k
            else block(
                self.c,
                self.c,
                kernel_sizes,
                expansion,
                with_context,
                context_kernel_size,
                add_identity,
            )
            for _ in range(n)
        )

    def forward(self, x):
        features = list(self.cv1(x).chunk(2, 1))
        features.extend(block(features[-1]) for block in self.m)
        return self.cv2(torch.cat(features, 1))


class StatisticalAttention(nn.Module):
    """Token-statistics self-attention used by SD-PSA."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1))
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x):
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        values = self.qkv(x).reshape(batch, tokens, self.num_heads, head_dim).permute(0, 2, 1, 3)
        normalized = torch.nn.functional.normalize(values, dim=-2)
        weights = self.attend(torch.sum(normalized.square(), dim=-1) * self.temperature)
        energy = torch.matmul(
            (weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), values.square()
        )
        attention = self.attn_drop(1.0 / (1.0 + energy))
        output = -(values * weights.unsqueeze(-1) * attention)
        output = output.permute(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.proj(output)


class DynamicTanh(nn.Module):
    """Learnable dynamic Tanh normalization."""

    def __init__(self, channels, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SDPSABlock(nn.Module):
    """Statistical attention and dynamic normalization block."""

    def __init__(self, channels, num_heads=4, shortcut=True):
        super().__init__()
        self.norm1 = DynamicTanh(channels)
        self.norm2 = DynamicTanh(channels)
        self.attention = StatisticalAttention(channels, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(channels, channels * 2, 1), Conv(channels * 2, channels, 1, act=False))
        self.shortcut = shortcut

    def forward(self, x):
        batch, channels, height, width = x.shape
        attention = self.attention(self.norm1(x).flatten(2).permute(0, 2, 1))
        attention = attention.permute(0, 2, 1).reshape(batch, channels, height, width).contiguous()
        x = x + attention if self.shortcut else attention
        ffn = self.ffn(self.norm2(x))
        return x + ffn if self.shortcut else ffn


class SD_PSA(C2PSA):
    """C2PSA variant built from SD-PSA blocks."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(SDPSABlock(self.c, num_heads=self.c // 64) for _ in range(n)))
