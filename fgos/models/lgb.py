"""LightGate Bottleneck."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _valid_groups(channels: int, preferred: int) -> int:
    for groups in range(min(channels, preferred), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LightGateBottleneck(nn.Module):
    """C -> C/4 -> C bottleneck modulated by ECA/GSE gates."""

    def __init__(
        self,
        channels: int,
        bottleneck_ratio: float = 0.25,
        gate_type: str = "gse",
        gate_groups: int = 4,
        gate_ratio: float = 0.16,
    ):
        super().__init__()
        mid = max(8, int(channels * bottleneck_ratio))
        self.path = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GroupNorm(_valid_groups(mid, 8), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.GroupNorm(_valid_groups(mid, 8), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.gate_type = gate_type.lower()
        if self.gate_type == "eca":
            kernel = int(abs(math.log2(max(1, channels)) / 2.0 + 1.0))
            kernel = kernel + 1 if kernel % 2 == 0 else kernel
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.eca = nn.Conv1d(1, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
            self.sigmoid = nn.Sigmoid()
        elif self.gate_type in {"se", "gse"}:
            groups = 1 if self.gate_type == "se" else _valid_groups(channels, gate_groups)
            per_group = channels // groups
            hidden = max(4, int(per_group * gate_ratio)) * groups
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1, groups=groups),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 1, groups=groups),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "eca":
            pooled = self.pool(x).squeeze(-1).transpose(1, 2)
            return self.sigmoid(self.eca(pooled).transpose(1, 2).unsqueeze(-1))
        return self.gate(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._gate(x) * self.path(x)

