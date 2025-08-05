import torch.nn as nn
import torch
from fractal_transformer_block import FractalTransformerBlock
from fractal_positional_encoding import mandelbrot_positional_encoding

class FractalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, ff_hidden_dim, recursion_depth, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = mandelbrot_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            FractalTransformerBlock(d_model, num_heads, ff_hidden_dim, recursion_depth) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x, mask=None):
        x = self.embedding(x).to(self.device) + self.pos_encoding[:x.shape[1], :].to(self.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
