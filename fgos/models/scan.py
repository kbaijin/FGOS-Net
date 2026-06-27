"""Frequency-aligned serialization and lightweight sequence scanning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import HaarDWT, HaarIDWT


def _snake_order(h: int, w: int, device: torch.device) -> torch.Tensor:
    rows: list[int] = []
    for i in range(h):
        row = range(w) if i % 2 == 0 else range(w - 1, -1, -1)
        rows.extend(i * w + j for j in row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def _vertical_order(h: int, w: int, device: torch.device) -> torch.Tensor:
    order = [i * w + j for j in range(w) for i in range(h)]
    return torch.tensor(order, dtype=torch.long, device=device)


def _hilbert_order(h: int, w: int, device: torch.device) -> torch.Tensor:
    def is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    if h != w or not is_power_of_two(h):
        return _snake_order(h, w, device)

    def d2xy(n: int, d: int) -> tuple[int, int]:
        x = 0
        y = 0
        s = 1
        while s < n:
            rx = 1 & (d // 2)
            ry = 1 & (d ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d //= 4
            s *= 2
        return x, y

    return torch.tensor([d2xy(h, d)[1] * w + d2xy(h, d)[0] for d in range(h * w)], dtype=torch.long, device=device)


class SharedSequenceSSM(nn.Module):
    """A small sequence mixer used as the public fallback SSM operator."""

    def __init__(self, dim: int, d_conv: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.scan_conv = nn.Conv1d(dim, dim, kernel_size=d_conv, padding=d_conv - 1, groups=dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.scan_conv.weight, 1.0 / float(d_conv))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = self.scan_conv(value.transpose(1, 2))[:, :, : value.shape[1]].transpose(1, 2)
        return self.out_proj(F.silu(mixed) * torch.sigmoid(gate))


class FrequencyAlignedScan(nn.Module):
    """FA-Scan: sub-band aligned traversal for topology modeling."""

    branch_modes = {
        "LL": "hilbert",
        "LH": "horizontal",
        "HL": "vertical",
        "HH": "hilbert",
    }

    def __init__(self, dim: int):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.ssm = SharedSequenceSSM(dim)
        self.norm = nn.GroupNorm(1, dim)

    def _order(self, mode: str, h: int, w: int, device: torch.device) -> torch.Tensor:
        if mode == "horizontal":
            return torch.arange(h * w, device=device)
        if mode == "vertical":
            return _vertical_order(h, w, device)
        if mode == "hilbert":
            return _hilbert_order(h, w, device)
        raise ValueError(f"Unknown FA-Scan mode: {mode}")

    def _scan(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        order = self._order(mode, h, w, x.device)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=x.device)
        ordered = seq.index_select(1, order)
        mixed = self.ssm(ordered)
        restored = mixed.index_select(1, inv)
        return restored.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, (lh, hl, hh), shape = self.dwt(x)
        y_ll = self._scan(ll, self.branch_modes["LL"])
        y_lh = self._scan(lh, self.branch_modes["LH"])
        y_hl = self._scan(hl, self.branch_modes["HL"])
        y_hh = self._scan(hh, self.branch_modes["HH"])
        y = self.idwt(y_ll, [y_lh, y_hl, y_hh], shape)
        return self.norm(y)

