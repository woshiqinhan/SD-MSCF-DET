"""Feature-pyramid modules used by SD-MSCF-DET.

SNI follows Rethinking Feature Pyramid Network, while GSConvE is based on the
GSConv family introduced by Slim-Neck by GSConv.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv

__all__ = ("SNI", "GSConvE")


class SNI(nn.Module):
    def __init__(self, up_f=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=up_f, mode="nearest")
        self.scale = 1 / (up_f**2)

    def forward(self, x):
        return self.scale * self.upsample(x)


class GSConvE(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = Conv(c1, hidden, k, s, None, g, d, act)
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, bias=False),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y = torch.cat((x1, self.cv2(x1)), dim=1)
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
