#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: neural_ca_full.py
VERSION: v1.0.0-GODCORE-BANDO
NAME: NeuralCellularAutomata
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Standalone, differentiable, regenerative Neural Cellular Automata system.
         Grow, damage, and regrow arbitrary patterns from a single cell, trainable end-to-end.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

# ---------------------- CORE NCA MODEL ----------------------

class NeuralCA(nn.Module):
    def __init__(self, channels=16, hidden_size=128):
        super().__init__()
        self.channels = channels
        # Per-channel depthwise Sobel filter for gradients
        self.perc_conv = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1, bias=False, groups=channels)
        self._init_sobel()
        # Update rule: (state+dx+dy) --> hidden --> update vector
        self.update = nn.Sequential(
            nn.Conv2d(channels*3, hidden_size, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, channels, 1, bias=False)
        )
        nn.init.zeros_(self.update[-1].weight)

    def _init_sobel(self):
        # Initialize with Sobel x and y for each channel
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = sobel_x.T
        kernel = torch.zeros(self.channels*2,1,3,3)
        for ch in range(self.channels):
            kernel[2*ch,0] = sobel_x
            kernel[2*ch+1,0] = sobel_y
        self.perc_conv.weight.data = kernel

    def perceive(self, x):
        grads = self.perc_conv(x)
        x_cat = torch.cat([x, grads], 1)  # [B, 16+32, H, W]
        return x_cat

    def forward(self, x, steps=1, fire_rate=0.5, damage_mask=None):
        for step in range(steps):
            x = self.step(x, fire_rate, damage_mask if step==0 else None)
        return x

    def step(self, x, fire_rate=0.5, damage_mask=None):
        # Perceive + update
        x_in = self.perceive(x)
        dx = self.update(x_in)
        # Stochastic cell update
        stochastic_mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
        dx = dx * stochastic_mask
        x = x + dx
        # Living cell masking
        x = self.alive_mask(x)
        # Damage: zero out region if mask provided
        if damage_mask is not None:
            x = x * damage_mask
        return x

    def alive_mask(self, x):
        alpha = x[:, 3:4]
        alive = F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1
        return x * alive.float()

# ---------------------- DATA UTILITIES ----------------------

def seed_grid(batch_size, size=64, channels=16):
    x = torch.zeros(batch_size, channels, size, size)
    x[:, 3:, size//2, size//2] = 1.0  # Alpha+hidden channels alive in center
    return x

def load_target_pattern(size=64):
    # Simple: white square in the middle
    pattern = np.zeros((4, size, size), dtype=np.float32)
    pattern[:,16:48,16:48] = 1.0
    return torch.tensor(pattern).unsqueeze(0)

def random_damage_mask(x, kind='circle', strength=1.0):
    # x: [B, C, H, W]
    B,C,H,W = x.shape
    mask = torch.ones(B,1,H,W)
    for b in range(B):
        if kind == 'circle':
            r = random.randint(H//10, H//4)
            cx,cy = random.randint(r,W-r-1), random.randint(r,H-r-1)
            Y,X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            circle = ((X-cx)**2 + (Y-cy)**2) < r**2
            mask[b,0][circle] = 0.0
        elif kind == 'square':
            w = random.randint(H//8, H//4)
            x0, y0 = random.randint(0,W-w-1), random.randint(0,H-w-1)
            mask[b,0,y0:y0+w,x0:x0+w] = 0.0
    return mask

def visualize_grid(x, savefile=None):
    # x: [B, 16, H, W], shows RGB (0:3) with alpha as mask (3)
    img = x[0,:3].detach().cpu().permute(1,2,0).numpy()
    alpha = x[0,3].detach().cpu().numpy()
    img = np.clip(img,0,1)
    alpha = np.clip(alpha,0,1)
    img = img * alpha[...,None] + (1-alpha[...,None]) * 1.0  # on white
    plt.imshow(img)
    plt.axis('off')
    if savefile:
        plt.savefig(savefile, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

# ---------------------- TRAINING LOOP ----------------------

def train_nca(
    nca,
    target,
    pool_size=1024,
    batch_size=8,
    grid_size=64,
    channels=16,
    iters=2000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    nca.to(device)
    pool = [seed_grid(1, grid_size, channels).to(device) for _ in range(pool_size)]
    optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)
    target = target.to(device)

    for step in range(iters):
        batch_idx = np.random.choice(len(pool), batch_size, replace=False)
        batch = torch.cat([deepcopy(pool[i]) for i in batch_idx], 0).to(device)

        # Randomly damage half the batch
        for i in range(batch_size//2):
            mask = random_damage_mask(batch[i:i+1], kind=random.choice(['circle','square']))
            batch[i:i+1] = batch[i:i+1] * mask

        # Evolve CAs for random steps
        steps = random.randint(48, 96)
        out = nca(batch, steps=steps, fire_rate=0.5)

        # L2 loss on RGBA channels
        loss = ((out[:,:4] - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update pool with outputs
        for i, idx in enumerate(batch_idx):
            pool[idx] = out[i:i+1].detach().clone()

        if step % 100 == 0 or step == iters-1:
            print(f"Step {step:04d}: loss={loss.item():.5f}")
            visualize_grid(out)
    print("Training finished.")

    return nca, pool

# ---------------------- DEMO: TRAIN, DAMAGE, REGENERATE ----------------------

if __name__ == "__main__":
    SIZE = 64
    CHANNELS = 16
    BATCH = 8

    print("== Initializing NCA ==")
    nca = NeuralCA(channels=CHANNELS)
    target = load_target_pattern(SIZE)
    print("== Training NCA (watch the pattern grow and heal) ==")
    nca, pool = train_nca(nca, target, pool_size=256, batch_size=BATCH, grid_size=SIZE, channels=CHANNELS, iters=1200)

    # Test: Grow from scratch
    print("== Grow from single seed ==")
    x = seed_grid(1, SIZE, CHANNELS)
    out = nca(x, steps=80)
    visualize_grid(out)

    # Test: Damage, then regrow
    print("== Damage and Regenerate ==")
    mask = random_damage_mask(out, kind='square')
    visualize_grid(out * mask)  # show damaged
    out2 = nca(out * mask, steps=60)  # regrow
    visualize_grid(out2)

    # Save model
    torch.save(nca.state_dict(), "neural_ca_full.pt")
    print("== All done. Neural CA ready for fractal/ASI hacking ==")
