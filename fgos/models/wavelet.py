"""Pure PyTorch Haar DWT/IDWT utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WaveletShape:
    height: int
    width: int
    pad_h: int
    pad_w: int


class HaarDWT(nn.Module):
    """One-level Haar transform for feature maps."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], WaveletShape]:
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        tl = x[:, :, 0::2, 0::2]
        tr = x[:, :, 0::2, 1::2]
        bl = x[:, :, 1::2, 0::2]
        br = x[:, :, 1::2, 1::2]

        ll = (tl + tr + bl + br) * 0.5
        lh = (tl + tr - bl - br) * 0.5
        hl = (tl - tr + bl - br) * 0.5
        hh = (tl - tr - bl + br) * 0.5
        return ll, [lh, hl, hh], WaveletShape(h, w, pad_h, pad_w)


class HaarIDWT(nn.Module):
    """Inverse one-level Haar transform for feature maps."""

    def forward(
        self,
        ll: torch.Tensor,
        bands: list[torch.Tensor],
        shape: WaveletShape | None = None,
    ) -> torch.Tensor:
        lh, hl, hh = bands
        b, c, h2, w2 = ll.shape
        y = torch.empty(b, c, h2 * 2, w2 * 2, device=ll.device, dtype=ll.dtype)

        y[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        y[:, :, 0::2, 1::2] = (ll + lh - hl - hh) * 0.5
        y[:, :, 1::2, 0::2] = (ll - lh + hl - hh) * 0.5
        y[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5

        if shape is not None:
            y = y[:, :, : shape.height, : shape.width]
        return y

