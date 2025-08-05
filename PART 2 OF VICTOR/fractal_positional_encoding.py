import torch
import math
import numpy as np

def mandelbrot_positional_encoding(seq_len, d_model, max_iter=20, escape_radius=2):
    pos_enc = torch.zeros(seq_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            z = complex(0, 0)
            c = complex(pos / seq_len, i / d_model)
            iteration = 0
            
            while abs(z) < escape_radius and iteration < max_iter:
                z = z**2 + c
                iteration += 1
            
            pos_enc[pos, i] = math.sin(iteration / max_iter * math.pi)
            if i + 1 < d_model:
                pos_enc[pos, i+1] = math.cos(iteration / max_iter * math.pi)
    
    return pos_enc

# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
