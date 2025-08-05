import torch.nn as nn
from fractal_attention import FractalAttention
from fractal_feedforward import FractalFeedForward

class FractalTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, recursion_depth=2):
        super().__init__()
        self.attention = FractalAttention(d_model, num_heads, recursion_depth)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FractalFeedForward(d_model, ff_hidden_dim, recursion_depth)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + attn_output
        ffn_output = self.ffn(self.norm2(x))
        return x + ffn_output

# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
