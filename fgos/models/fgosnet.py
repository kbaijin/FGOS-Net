"""FGOS-Net model and builder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, FGOSStage


@dataclass(frozen=True)
class FGOSConfig:
    dims: tuple[int, int, int, int] = (32, 64, 128, 192)
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    decoder_dim: int = 96
    head_dim: int = 32
    use_asgp: bool = True
    decoder: str = "hybrid_gfa"


class ScaleGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class FGOSEncoder(nn.Module):
    def __init__(self, in_channels: int, cfg: FGOSConfig, asgp_mode: str):
        super().__init__()
        dims = cfg.dims
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, dims[0], 7, stride=2),
            ConvBNAct(dims[0], dims[0], 3, stride=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        prev = dims[0]
        for index, (dim, depth) in enumerate(zip(cfg.dims, cfg.depths)):
            if index == 0:
                self.downsamples.append(nn.Identity() if prev == dim else nn.Conv2d(prev, dim, 1))
            else:
                self.downsamples.append(nn.Sequential(nn.Conv2d(prev, dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(dim)))
            self.stages.append(FGOSStage(dim, depth, index, asgp_mode, use_asgp=cfg.use_asgp))
            prev = dim

    def set_asgp_mode(self, mode: str) -> None:
        for stage in self.stages:
            stage.set_asgp_mode(mode)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for downsample, stage in zip(self.downsamples, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return features


class GFADecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    ScaleGate(out_channels),
                )
                for channels in in_channels
            ]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        aligned = []
        for feature, branch in zip(features, self.branches):
            y = branch(feature)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(y)
        return self.refine(sum(aligned))


class HybridGFADecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(channels, out_channels, 1, bias=False) for channels in in_channels])
        self.fpn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )
        self.gates = nn.ModuleList([ScaleGate(out_channels) for _ in in_channels])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [layer(feature) for layer, feature in zip(self.lateral, features)]
        outputs: list[torch.Tensor] = []
        top_down: torch.Tensor | None = None
        for index in range(len(laterals) - 1, -1, -1):
            current = laterals[index]
            if top_down is not None:
                top_down = F.interpolate(top_down, size=current.shape[-2:], mode="bilinear", align_corners=False)
                current = current + top_down
            current = self.gates[index](self.fpn[index](current))
            outputs.insert(0, current)
            top_down = current
        return outputs


class BoundaryRefinementModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.edge = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(torch.cat([self.context(x), self.edge(x)], dim=1))


class MultiBranchBRMHead(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int, int], hidden_dim: int, num_classes: int):
        super().__init__()
        self.lateral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for channels in in_channels
            ]
        )
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )
        self.brm = BoundaryRefinementModule(hidden_dim)
        self.pred = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim // 2, num_classes, 1),
        )

    def forward(self, features: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        fused = None
        for feature, lateral, refine in zip(features, self.lateral, self.refine):
            y = lateral(feature)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            y = refine(y)
            fused = y if fused is None else fused + y
        logits = self.pred(self.brm(fused))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class SimpleBRMHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.brm = BoundaryRefinementModule(hidden_dim)
        self.pred = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.pred(self.brm(self.project(x))), size=output_size, mode="bilinear", align_corners=False)


class FGOSNet(nn.Module):
    """FGOS-Net segmentation model."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, cfg: FGOSConfig | None = None, asgp_mode: str = "paper"):
        super().__init__()
        self.cfg = cfg or FGOSConfig()
        self.variant = "custom"
        self.asgp_mode = asgp_mode
        self.encoder = FGOSEncoder(in_channels, self.cfg, asgp_mode)
        if self.cfg.decoder == "hybrid_gfa":
            self.decoder = HybridGFADecoder(self.cfg.dims, self.cfg.decoder_dim)
            self.head = MultiBranchBRMHead((self.cfg.decoder_dim,) * 4, self.cfg.head_dim, num_classes)
        elif self.cfg.decoder == "gfa":
            self.decoder = GFADecoder(self.cfg.dims, self.cfg.decoder_dim)
            self.head = SimpleBRMHead(self.cfg.decoder_dim, self.cfg.head_dim, num_classes)
        else:
            raise ValueError("decoder must be 'hybrid_gfa' or 'gfa'.")
        self.paper_profile = {
            "params_m": 6.26,
            "flops_g": 7.87,
            "model_size_mb": 23.92,
            "fps": 80.2,
            "latency_ms": 12.47,
            "device": "NVIDIA RTX 3090",
            "input_size": "1x3x256x256",
        }

    def set_asgp_mode(self, mode: str) -> None:
        if mode not in {"paper", "fast"}:
            raise ValueError("ASGP mode must be 'paper' or 'fast'.")
        self.asgp_mode = mode
        self.encoder.set_asgp_mode(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        decoded = self.decoder(features)
        if isinstance(decoded, list):
            return self.head(decoded, input_size)
        return self.head(decoded, input_size)


def _config_for_variant(variant: str) -> FGOSConfig:
    if variant == "eccv2026_paper":
        return FGOSConfig(depths=(2, 2, 2, 2), use_asgp=True, decoder="hybrid_gfa")
    if variant == "current_best":
        return FGOSConfig(depths=(2, 2, 2, 2), use_asgp=True, decoder="hybrid_gfa")
    raise ValueError("variant must be 'eccv2026_paper' or 'current_best'.")


def build_fgosnet(
    variant: str = "eccv2026_paper",
    num_classes: int = 1,
    in_channels: int = 3,
    asgp_mode: str = "paper",
) -> FGOSNet:
    """Build the public FGOS-Net model."""

    model = FGOSNet(in_channels=in_channels, num_classes=num_classes, cfg=_config_for_variant(variant), asgp_mode=asgp_mode)
    model.variant = variant
    return model
