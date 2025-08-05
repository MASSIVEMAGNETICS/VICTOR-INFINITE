# omegallama/blocks.py

from omegallama.registry import OmegaBlockRegistry

class BlockConfig:
    """
    Configuration for a single transformer block.
    """
    def __init__(
        self,
        attn_type='attention',  # e.g., 'attention', 'flash_attn', 'local_attn'
        ffn_type='feedforward', # e.g., 'feedforward', 'moe', 'adapter'
        residual_style='prenorm', # 'prenorm', 'postnorm', 'sandwich', 'gated'
        norm_cls=None,           # e.g., RMSNorm, LayerNorm
        **kwargs
    ):
        self.attn_type = attn_type
        self.ffn_type = ffn_type
        self.residual_style = residual_style
        self.norm_cls = norm_cls
        self.kwargs = kwargs

def apply_residual(x, fn, norm, style, gate_proj=None):
    """
    Apply a residual connection with the specified normalization style.
    """
    if style == "prenorm":
        return x + fn(norm(x))
    elif style == "postnorm":
        return norm(x + fn(x))
    elif style == "sandwich":
        return norm(x + fn(norm(x)))
    elif style == "gated":
        if gate_proj is None:
            raise ValueError("Gated residual requires a gate_proj function.")
        gate = gate_proj(x).sigmoid()
        return x + gate * fn(norm(x))
    else:
        raise ValueError(f"Unknown residual style: {style}")

class TransformerBlock(OmegaLayer):
    """
    Configurable transformer block using OmegaBlockRegistry and flexible residuals.
    """
    def __init__(self, config: BlockConfig, **layer_args):
        super().__init__()
        # Instantiate attention and FFN via registry
        self.attn = OmegaBlockRegistry.create(config.attn_type, **layer_args)
        self.ffn = OmegaBlockRegistry.create(config.ffn_type, **layer_args)
        # Norm layers (default to RMSNorm if not specified)
        norm_cls = config.norm_cls or RMSNorm
        self.norm1 = norm_cls(layer_args['dim'], name="block_norm1")
        self.norm2 = norm_cls(layer_args['dim'], name="block_norm2")
        self.residual_style = config.residual_style
        # Optional: Gated residual projection
        self.gate_proj = None
        if self.residual_style == "gated":
            self.gate_proj = OmegaBlockRegistry.create('gate_proj', **layer_args)
        # Register sub-layers for parameter collection
        self._register_layer("attn", self.attn)
        self._register_layer("ffn", self.ffn)
        self._register_layer("norm1", self.norm1)
        self._register_layer("norm2", self.norm2)
        if self.gate_proj:
            self._register_layer("gate_proj", self.gate_proj)

    def __call__(self, x, *args, **kwargs):
        x = apply_residual(x, self.attn, self.norm1, self.residual_style, gate_proj=self.gate_proj)
        x = apply_residual(x, self.ffn, self.norm2, self.residual_style, gate_proj=self.gate_proj)
        return x