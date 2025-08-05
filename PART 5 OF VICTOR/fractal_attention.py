# fractal_attention.py
# Patched to ensure correct integer typing in torch.randn()

import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalAttention(nn.Module):
    def __init__(self, d_model, num_heads, recursion_depth, entropy_threshold=0.01):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.recursion_depth = int(recursion_depth)
        self.entropy_threshold = float(entropy_threshold)

        assert self.d_model % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = self.d_model // self.num_heads
        self.query_proj = nn.Linear(self.d_model, self.d_model)
        self.key_proj = nn.Linear(self.d_model, self.d_model)
        self.value_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # ✅ Ensure integer type
        self.fractal_coords = nn.Parameter(torch.randn(int(self.recursion_depth), 1, 1, int(self.d_model)))

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.query_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)

        # Apply entropy threshold logic (can be expanded)
        ent = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1).mean()
        if ent < self.entropy_threshold:
            weights = F.dropout(weights, p=0.1, training=self.training)

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attended)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
