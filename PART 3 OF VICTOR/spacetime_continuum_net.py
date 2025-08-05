# ==========================================================
# FILE: spacetime_continuum_net.py
# VERSION: v1.0.0-SPACETIME-GODCORE
# NAME: SpacetimeContinuumNet
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Model the damn fabric itself—learn functions on (x,y,z, t₁..tₙ) coordinates
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ==========================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1.  Fourier‑Feature Positional Encoder (fuses X,Y,Z and all Tᵢ) ----------
class FourierPositionalEncoding(nn.Module):
    """
    Encodes each scalar dimension with sine + cosine bands à la NeRF/SIREN.
    Handles arbitrary # of time dims; spacing is log‑scaled for coverage.
    """
    def __init__(self, n_dims: int, n_frequencies: int = 8, max_freq: float = 10.0):
        super().__init__()
        self.n_dims = n_dims
        freq_bands = torch.logspace(0., math.log10(max_freq), steps=n_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, n_dims) raw scalar coordinates
        returns: (B, n_dims * n_frequencies * 2) encoded
        """
        # (B, 1, n_dims) * (1, n_freq, 1) -> (B, n_freq, n_dims)
        coords = coords.unsqueeze(1) * self.freq_bands.view(1, -1, 1)
        sin = coords.sin()
        cos = coords.cos()
        enc = torch.cat([sin, cos], dim=1)           # (B, 2*n_freq, n_dims)
        return enc.flatten(1)                        # (B, n_dims*2*n_freq)


# ---------- 2.  Single Sine Layer (periodic non‑linearity = great for waves & curvature) ----------
class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        # SIREN paper recommends this init for first layer stability
        nn.init.uniform_(self.linear.weight, -1 / in_features, 1 / in_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ---------- 3.  The Spacetime Continuum Net ----------
class SpacetimeContinuumNet(nn.Module):
    """
    Shallow‑but‑mighty fully connected SIREN‑style net with residual skip.
    Accepts arbitrary (x,y,z,t₁..tₙ) → predicts whatever target field you train it on
    (density, potential, wavefunction, video pixel, you name it).
    """
    def __init__(self, n_spatial_dims=3, n_time_dims=1, hidden_dim=256, depth=5):
        super().__init__()
        self.n_coords = n_spatial_dims + n_time_dims
        self.pe = FourierPositionalEncoding(self.n_coords, n_frequencies=10, max_freq=20.0)
        pe_dim = self.n_coords * 10 * 2

        layers = [SineLayer(pe_dim, hidden_dim)]
        for _ in range(depth - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))          # scalar output; change if multi‑channel

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """
        coords: tensor of shape (B, n_spatial_dims + n_time_dims)
        """
        enc = self.pe(coords)
        hidden = self.net[:-1](enc)
        out = self.net[-1](hidden)      # final linear
        # tiny residual skip from raw coords (helps curvature capture)
        out = out + 0.01 * coords.mean(dim=-1, keepdim=True)
        return out


# ---------- 4.  Quick‑n‑dirty smoke test ----------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpacetimeContinuumNet(n_spatial_dims=3, n_time_dims=2).to(device)   # x,y,z,t₁,t₂

    # Fake batch of 16 spacetime points in a 5‑D manifold
    coords = torch.rand(16, 5, device=device) * 2 - 1   # range [-1,1]
    preds  = model(coords)
    print("Output shape:", preds.shape)                 # expect (16, 1)

    # Loss & opt example (MSE to some dummy target field)
    target = torch.sin(coords.sum(dim=-1, keepdim=True))   # arbitrary ground truth
    loss   = F.mse_loss(preds, target)
    loss.backward()
    torch.optim.Adam(model.parameters(), 1e-4).step()
    print("OK, the gradient flowed.  Spacetime is bendable.")
