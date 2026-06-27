"""Active Spectral-Geometric Probing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActiveSpectralGeometricProbing(nn.Module):
    """Topology-conditioned high-frequency gating with paper/fast modes."""

    def __init__(
        self,
        dim: int,
        num_probes: int = 64,
        num_steps: int = 3,
        mode: str = "paper",
    ):
        super().__init__()
        if mode not in {"paper", "fast"}:
            raise ValueError("ASGP mode must be 'paper' or 'fast'.")
        self.dim = dim
        self.num_probes = num_probes
        self.num_steps = num_steps
        self.mode = mode

        self.probes = nn.Parameter(torch.randn(1, num_probes, dim) * 0.02)
        self.register_buffer("init_coords", self._init_coords(num_probes), persistent=False)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.static_value = nn.Sequential(nn.Linear(dim, max(8, dim // 2)), nn.ReLU(inplace=True), nn.Linear(max(8, dim // 2), 1), nn.Sigmoid())
        self.dynamic_value = nn.Sequential(nn.Linear(dim, max(8, dim // 2)), nn.ReLU(inplace=True), nn.Linear(max(8, dim // 2), 1), nn.Sigmoid())
        self.gru = nn.GRUCell(dim, dim)
        self.offset = nn.Sequential(nn.Linear(dim, max(8, dim // 2)), nn.ReLU(inplace=True), nn.Linear(max(8, dim // 2), 2), nn.Tanh())
        self.offset_scale = nn.Parameter(torch.tensor(0.20))
        self.attraction_scale = nn.Parameter(torch.tensor(0.30))
        self.stage_weight = nn.Parameter(torch.tensor(0.0))
        self.output_gate = nn.Parameter(torch.tensor(0.0))

        self.last_coords_log: list[torch.Tensor] = []
        self.last_coarse_mask: torch.Tensor | None = None
        self.last_final_mask: torch.Tensor | None = None

    def set_mode(self, mode: str) -> None:
        if mode not in {"paper", "fast"}:
            raise ValueError("ASGP mode must be 'paper' or 'fast'.")
        self.mode = mode

    @staticmethod
    def _init_coords(num_probes: int) -> torch.Tensor:
        side = int(math.ceil(math.sqrt(num_probes)))
        ys = torch.linspace(-0.8, 0.8, side)
        xs = torch.linspace(-0.8, 0.8, side)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:num_probes]
        return coords.unsqueeze(0)

    def _sample(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        sampled = F.grid_sample(x, coords.unsqueeze(1), mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.squeeze(2).permute(0, 2, 1)

    def _sparse_to_dense(self, x: torch.Tensor, state: torch.Tensor, value_net: nn.Module) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query_proj(x.flatten(2).transpose(1, 2))
        k = self.key_proj(state)
        values = value_net(state)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        mask = torch.bmm(attn, values).transpose(1, 2).reshape(b, 1, h, w)
        return mask.clamp(1e-5, 1.0 - 1e-5)

    def _attraction(self, mask: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, _, h, w = mask.shape
        y = torch.linspace(-1, 1, h, device=mask.device)
        x = torch.linspace(-1, 1, w, device=mask.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        px = coords[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        py = coords[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        dist = torch.sqrt((xx - px) ** 2 + (yy - py) ** 2 + 1e-6)
        weight = mask / (dist + 0.2)
        denom = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        cx = (weight * xx).sum(dim=(2, 3), keepdim=True) / denom
        cy = (weight * yy).sum(dim=(2, 3), keepdim=True) / denom
        vec = torch.stack([(cx.squeeze(-1).squeeze(-1) - coords[:, :, 0]), (cy.squeeze(-1).squeeze(-1) - coords[:, :, 1])], dim=-1)
        norm = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return vec / norm * torch.clamp(norm, max=0.15)

    @staticmethod
    def _repulsion(coords: torch.Tensor, min_dist: float = 0.15) -> torch.Tensor:
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eye = torch.eye(coords.shape[1], device=coords.device).view(1, coords.shape[1], coords.shape[1], 1)
        strength = torch.clamp(min_dist - dist, min=0.0) / min_dist
        force = (diff / dist) * strength * (1.0 - eye)
        force = force.sum(dim=2)
        norm = force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return force / norm * torch.clamp(norm, max=0.10)

    def _should_evolve(self) -> bool:
        return self.training or self.mode == "paper"

    def forward(self, x_lf: torch.Tensor, x_hf: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, c, _, _ = x_lf.shape
        coords = self.init_coords.to(x_lf.device).expand(b, -1, -1).clone()
        static_state = self.probes.expand(b, -1, -1) + 0.1 * self._sample(x_lf, coords)
        coarse = self._sparse_to_dense(x_lf, static_state, self.static_value)
        coords_log = [coords.clone()]

        if self._should_evolve() and self.num_steps > 0:
            state = self.probes.expand(b, -1, -1).clone()
            for _ in range(self.num_steps):
                sampled = self._sample(x_lf, coords)
                state = self.gru(sampled.reshape(b * self.num_probes, c), state.reshape(b * self.num_probes, c)).reshape(b, self.num_probes, c)
                coords = coords + self.offset(state) * self.offset_scale.abs()
                coords = coords + self._attraction(coarse, coords) * self.attraction_scale.abs()
                coords = coords + self._repulsion(coords)
                coords = coords.clamp(-0.95, 0.95)
                coords_log.append(coords.clone())
            dynamic_state = state
        else:
            dynamic_state = static_state

        refined = self._sparse_to_dense(x_lf, dynamic_state, self.dynamic_value)
        weight = torch.sigmoid(self.stage_weight)
        mask = weight * refined + (1.0 - weight) * coarse
        hf = x_hf.reshape(b, 3, c, x_hf.shape[-2], x_hf.shape[-1]).mean(dim=1)
        out = x_lf + torch.sigmoid(self.output_gate) * hf * mask

        self.last_coords_log = coords_log
        self.last_coarse_mask = coarse.detach()
        self.last_final_mask = mask.detach()
        self._coarse_mask = self.last_coarse_mask
        self._final_mask = self.last_final_mask
        return out, coords_log

