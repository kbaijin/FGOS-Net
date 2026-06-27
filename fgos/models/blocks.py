"""FGOS encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .asgp import ActiveSpectralGeometricProbing
from .lgb import LightGateBottleneck
from .scan import FrequencyAlignedScan
from .wavelet import HaarDWT, HaarIDWT


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class FGOSBlock(nn.Module):
    """Split, align, scan, gate, and reconstruct."""

    def __init__(
        self,
        dim: int,
        stage_index: int,
        use_asgp: bool = True,
        asgp_mode: str = "paper",
        asgp_num_probes: int = 64,
        asgp_num_steps: int = 3,
        offset_scale: float = 0.1,
    ):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.offset_scale = offset_scale
        self.norm_lf = nn.GroupNorm(1, dim)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, max(8, dim // 4), 3, padding=1, bias=False),
            nn.GroupNorm(1, max(8, dim // 4)),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(8, dim // 4), 2, 3, padding=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.offset_predictor[-2].weight)
        self.fa_scan = FrequencyAlignedScan(dim)
        gate_type = "eca" if stage_index < 2 else "gse"
        gate_groups = 1 if gate_type == "eca" else 16
        self.lgb = LightGateBottleneck(dim, gate_type=gate_type, gate_groups=gate_groups)
        self.parallel_gate = nn.Parameter(torch.tensor(0.0))
        self.use_asgp = use_asgp
        self.asgp = ActiveSpectralGeometricProbing(dim, asgp_num_probes, asgp_num_steps, asgp_mode) if use_asgp else None
        self.hf_gate = nn.Sequential(nn.Conv2d(dim * 3, dim, 1, bias=False), nn.GroupNorm(1, dim), nn.Sigmoid())
        self.norm_out = nn.GroupNorm(1, dim)

    @staticmethod
    def _grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1, 1, height, device=device)
        x = torch.linspace(-1, 1, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)

    def set_asgp_mode(self, mode: str) -> None:
        if self.asgp is not None:
            self.asgp.set_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        ll, hf_bands, shape = self.dwt(x)
        hf = torch.cat(hf_bands, dim=1)
        ll = self.norm_lf(ll)
        offsets = self.offset_predictor(hf) * self.offset_scale
        grid = self._grid(ll.shape[0], ll.shape[2], ll.shape[3], ll.device)
        aligned = F.grid_sample(ll, (grid + offsets.permute(0, 2, 3, 1)).clamp(-1, 1), mode="bilinear", padding_mode="border", align_corners=False)
        scanned = self.fa_scan(aligned)
        local = self.lgb(aligned)
        gate = torch.sigmoid(self.parallel_gate)
        topology = gate * scanned + (1.0 - gate) * local
        if self.asgp is not None:
            topology, _ = self.asgp(topology, hf)
        else:
            topology = topology * (1.0 + self.hf_gate(hf))
        y = self.idwt(topology, hf_bands, shape)
        return self.norm_out(residual + y)


class FGOSStage(nn.Module):
    def __init__(self, dim: int, depth: int, stage_index: int, asgp_mode: str, use_asgp: bool = True):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                FGOSBlock(
                    dim=dim,
                    stage_index=stage_index,
                    use_asgp=use_asgp,
                    asgp_mode=asgp_mode,
                    asgp_num_probes=64,
                    asgp_num_steps=3,
                )
                for _ in range(depth)
            ]
        )

    def set_asgp_mode(self, mode: str) -> None:
        for block in self.blocks:
            block.set_asgp_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
