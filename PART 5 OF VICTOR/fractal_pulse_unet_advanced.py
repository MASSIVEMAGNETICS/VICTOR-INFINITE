"""
FractalPulseUNetAdvanced
========================

This module defines a forward‑thinking U‑Net backbone for latent diffusion
models.  It integrates **Pulse Blocks** with learnable sinusoidal gating,
**Fractal Skip Connections** with attention‑based fusion and supports
timestep‑aware gating.  The design is inspired by SDXL‑Lightning but
introduces novel architectural elements aimed at reducing the number of
sampling steps without sacrificing quality.

The implementation is self‑contained and does not depend on the diffusers
UNet2DConditionModel.  It can serve as a starting point for training
a new diffusion model or be adapted into existing pipelines.

Key features:

* **PulseBlockEnhanced** – Caches coordinate grids, modulates the
  sinusoidal gating with the diffusion timestep and supports dynamic
  frequency scaling.
* **FractalSkipConnectionAdvanced** – Fuses multi‑scale skip
  connections using learnable attention weights rather than simple
  averaging.
* **DownBlockAdvanced / UpBlockAdvanced** – Integrate pulse gating
  directly into the convolutional blocks and propagate fractal echoes
  across scales.
* **SinusoidalPositionEmbeddings** – Provides timestep embeddings for
  diffusion models.

Note: This code defines the architecture only; training on a large
dataset and distillation to a small number of steps are required to
achieve SDXL‑Lightning‑level performance.  Use this as a blueprint
for experimentation.
"""

from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Create sinusoidal embeddings for timesteps.

    This module produces a vector of size ``embedding_dim`` for each
    timestep, using sine and cosine functions of different frequencies.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
        self.register_buffer("inv_freq", torch.exp(-math.log(10000) * exponent))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) timesteps normalised in [0,1]
        pos = t[:, None] * self.inv_freq[None, :]
        embeddings = torch.cat([pos.sin(), pos.cos()], dim=-1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class PulseBlockEnhanced(nn.Module):
    """Learnable sinusoidal gating with timestep modulation and coordinate caching."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.learnable_freq = nn.Parameter(torch.randn(1, channels, 1, 1) * 0.1 + 1.0)
        self.learnable_phase = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.SiLU()
        # coordinate cache
        self.register_buffer("grid_cache", torch.empty(0))

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use cached grid if resolution matches
        if self.grid_cache.numel() == h * w * 2 and self.grid_cache.device == device:
            grid = self.grid_cache.view(2, h, w)
            return grid[0], grid[1]
        # Otherwise create new grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, steps=h, device=device, dtype=dtype),
            torch.linspace(-1, 1, steps=w, device=device, dtype=dtype),
            indexing="ij",
        )
        self.grid_cache = torch.stack([grid_y, grid_x], dim=0)
        return grid_y, grid_x

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), t_emb: (B, D)
        b, c, h, w = x.shape
        # Compute frequency scaling from timestep embedding (t_emb)
        # Use a simple linear projection; could be replaced with MLP
        freq_scale = torch.tanh(t_emb.mean(dim=-1, keepdim=True)).unsqueeze(-1).unsqueeze(-1)
        phase_shift = torch.tanh(t_emb.std(dim=-1, keepdim=True)).unsqueeze(-1).unsqueeze(-1)
        grid_y, grid_x = self._get_grid(h, w, x.device, x.dtype)
        # Broadcast learnable parameters
        freq = self.learnable_freq * (1 + freq_scale)
        phase = self.learnable_phase + phase_shift
        pulse_x = torch.cos(freq * grid_x + phase)
        pulse_y = torch.sin(freq * grid_y + phase)
        gate = pulse_x + pulse_y
        gated = x * gate
        out = self.conv(gated)
        out = self.norm(out)
        return self.act(out)


class FractalSkipConnectionAdvanced(nn.Module):
    """Attention‑based fusion of multi‑scale skip connections."""

    def __init__(self, channels: int, max_levels: int = 4):
        super().__init__()
        self.channels = channels
        self.max_levels = max_levels
        # Learnable attention weights for each level (including the primary skip)
        self.attn_weights = nn.Parameter(torch.ones(max_levels + 1))
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.SiLU()

    def forward(self, primary: torch.Tensor, echoes: List[torch.Tensor]) -> torch.Tensor:
        # primary: skip from current resolution
        # echoes: previous skip features from coarser resolutions (list of tensors)
        target_h, target_w = primary.shape[-2:]
        # Pad echoes list to max_levels
        padded = echoes[-self.max_levels:]
        while len(padded) < self.max_levels:
            padded.insert(0, torch.zeros_like(primary))
        # Resize echoes to target resolution
        resized = []
        for echo in padded:
            if echo.shape[-2:] != (target_h, target_w):
                resized_echo = F.interpolate(echo, size=(target_h, target_w), mode="bilinear", align_corners=False)
            else:
                resized_echo = echo
            resized.append(resized_echo)
        # Stack primary + echoes: shape (L+1, B, C, H, W)
        features = torch.stack([primary] + resized, dim=0)
        # Compute softmax attention over levels
        weights = F.softmax(self.attn_weights[: features.size(0)], dim=0)
        fused = torch.sum(weights[:, None, None, None, None] * features, dim=0)
        fused = self.fusion_conv(fused)
        fused = self.norm(fused)
        return self.act(fused)


class DownBlockAdvanced(nn.Module):
    """A downsampling block with pulse gating and optional residual layers."""

    def __init__(self, in_channels: int, out_channels: int, num_res_layers: int = 2, downsample: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pulse = PulseBlockEnhanced(out_channels)
        # First conv to change channel dim
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        for _ in range(num_res_layers - 1):
            self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if downsample else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x -> convs
        out = x
        for layer in self.layers:
            out = layer(out)
            out = self.norm(out)
            out = self.act(out)
        # Apply pulse gating
        out = self.pulse(out, t_emb)
        skip = out  # skip before downsampling
        # Downsample
        if self.downsample is not None:
            out = self.downsample(out)
        return out, skip


class UpBlockAdvanced(nn.Module):
    """An upsampling block with fractal skip fusion and pulse gating."""

    def __init__(self, in_channels: int, out_channels: int, num_res_layers: int = 2, upsample: bool = True, max_levels: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pulse = PulseBlockEnhanced(out_channels)
        self.fractal_fuse = FractalSkipConnectionAdvanced(out_channels, max_levels=max_levels)
        # First conv to change channel dim
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        for _ in range(num_res_layers - 1):
            self.layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act = nn.SiLU()
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if upsample else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, echoes: List[torch.Tensor], t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fractal fuse skip and echoes before upsampling
        skip_fused = self.fractal_fuse(skip, echoes)
        # Upsample input x if needed to match skip resolution
        if self.upsample is not None:
            x = self.upsample(x)
        # Combine upsampled x with fused skip
        # If channel dims differ, project x to out_channels
        if x.shape[1] != skip_fused.shape[1]:
            proj = nn.Conv2d(x.shape[1], skip_fused.shape[1], kernel_size=1).to(x.device)
            x = proj(x)
        out = x + skip_fused
        # Residual conv layers
        for layer in self.layers:
            out = layer(out)
            out = self.norm(out)
            out = self.act(out)
        # Pulse gating
        out = self.pulse(out, t_emb)
        return out, skip_fused


class FractalPulseUNetAdvanced(nn.Module):
    """Full U‑Net with fractal pulse architecture."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 320, channel_mults: Tuple[int, ...] = (1, 2, 4, 8), max_fractal_levels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        # Input projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # Down blocks
        in_ch = base_channels
        self.downs = nn.ModuleList()
        for mult in channel_mults:
            out_ch = base_channels * mult
            down_block = DownBlockAdvanced(in_ch, out_ch, num_res_layers=2, downsample=True)
            self.downs.append(down_block)
            in_ch = out_ch
        # Bottleneck
        self.mid_block1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.mid_pulse = PulseBlockEnhanced(in_ch)
        self.mid_block2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        # Up blocks
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            up_block = UpBlockAdvanced(in_ch, out_ch, num_res_layers=2, upsample=True, max_levels=max_fractal_levels)
            self.ups.append(up_block)
            in_ch = out_ch
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, in_ch), num_channels=in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fractal pulse UNet.

        Args:
            x (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            timestep (torch.Tensor): Normalised diffusion timesteps of shape (B,).
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        # Compute timestep embeddings
        t_emb = self.time_mlp(timestep)
        # Project input
        h = self.conv_in(x)
        # Downsample and collect skips
        skips: List[torch.Tensor] = []
        fractal_echoes: List[torch.Tensor] = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
            fractal_echoes.append(skip)
        # Bottleneck
        h = self.mid_block1(h)
        h = self.mid_pulse(h, t_emb)
        h = self.mid_block2(h)
        # Upsample
        for up in self.ups:
            skip = skips.pop()
            # Provide fractal echoes collected so far
            h, fused_skip = up(h, skip, fractal_echoes, t_emb)
            fractal_echoes.append(fused_skip)
        # Output
        return self.conv_out(h)


__all__ = [
    "SinusoidalPositionEmbeddings",
    "PulseBlockEnhanced",
    "FractalSkipConnectionAdvanced",
    "DownBlockAdvanced",
    "UpBlockAdvanced",
    "FractalPulseUNetAdvanced",
]