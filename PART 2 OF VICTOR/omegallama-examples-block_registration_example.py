# omegallama/examples/block_registration_example.py

from omegallama.registry import OmegaBlockRegistry

# Example: Register a custom attention block
@OmegaBlockRegistry.register('flash_attn')
class FlashAttention:
    def __init__(self, dim, n_heads, **kwargs):
        self.dim = dim
        self.n_heads = n_heads
        # ...initialize weights, etc.

    def __call__(self, x, *args, **kwargs):
        # ...implement FlashAttention logic
        return x # placeholder

# Example: Register a custom FFN block
@OmegaBlockRegistry.register('moe')
class MixtureOfExpertsFFN:
    def __init__(self, dim, hidden_dim, num_experts=4, **kwargs):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        # ...initialize experts, gating, etc.

    def __call__(self, x, *args, **kwargs):
        # ...implement MoE logic
        return x # placeholder

# Usage: Instantiate a block from config
from omegallama.blocks import BlockConfig, TransformerBlock

block_cfg = BlockConfig(attn_type='flash_attn', ffn_type='moe', residual_style='sandwich')
block = TransformerBlock(block_cfg, dim=512, n_heads=8, hidden_dim=2048)
print("Block instantiated with:", block.attn.__class__.__name__, block.ffn.__class__.__name__)