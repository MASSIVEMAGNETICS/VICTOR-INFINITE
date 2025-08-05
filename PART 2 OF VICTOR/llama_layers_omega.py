# =================================================================================================
# FILE: llama_layers_omega.py
# VERSION: v2.0.0-BANDOFY
# NAME: OmegaLlamaLayers
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Optimized, expanded, and future-proofed Llama transformer layers using OmegaTensor.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# =================================================================================================

import numpy as np
from OmegaTensor import OmegaTensor
from typing import Optional, List, Tuple

# FUTURE_UPGRADE: Integrate with a global configuration system for model args.
class LlamaModelArgs:
    """
    Configuration arguments for the Llama-style transformer model.
    Enforces design principles and pre-calculates derived dimensions.
    """
    def __init__(self, dim: int, n_layers: int, n_heads: int, n_kv_heads: Optional[int],
                 vocab_size: int, ffn_hidden_dim: Optional[int], max_seq_len: int,
                 norm_eps: float = 1e-5, rope_theta: float = 10000.0,
                 ffn_dim_multiplier: Optional[float] = None, multiple_of: int = 256):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # Defaults to MHA if not specified
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta

        # Derived dimensions
        self.head_dim = dim // n_heads
        if self.head_dim * self.n_heads != self.dim:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        # FFN hidden dimension calculation (if not provided, calculate Llama-style)
        # FUTURE_UPGRADE: Support different FFN types (e.g., SwiGLU, GELU)
        if ffn_hidden_dim is None:
            # Llama-style FFN hidden dimension calculation
            self.ffn_hidden_dim = int(2 * dim * 4 / 3) # Llama 1/2 rough estimate
            # Apply multiplier if present (Llama 3 uses 1.3 for ffn_dim_multiplier)
            if ffn_dim_multiplier is not None:
                self.ffn_hidden_dim = int(ffn_dim_multiplier * self.dim)
            # Round to nearest multiple_of
            self.ffn_hidden_dim = multiple_of * ((self.ffn_hidden_dim + multiple_of - 1) // multiple_of)
        else:
            self.ffn_hidden_dim = ffn_hidden_dim

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA.")
        self.n_rep = self.n_heads // self.n_kv_heads


# Helper function for Rotary Positional Embedding (now just a wrapper for OmegaTensor method)
def apply_rotary_emb(x: OmegaTensor, freqs_cis: OmegaTensor) -> OmegaTensor:
    """
    Applies rotary positional embedding to input tensor x using OmegaTensor's native method.
    Args:
        x: OmegaTensor of shape (bsz, seq_len, dim) or (bsz, num_heads, seq_len, head_dim)
        freqs_cis: OmegaTensor of shape (seq_len, dim) or (seq_len, head_dim) containing
                   interleaved cos/sin values. Expected to not require gradients.
                   Shape must be broadcastable to the last two dimensions of x.
    Returns:
        OmegaTensor with rotary embeddings applied, same shape as x.
    """
    # This function now directly uses the RotaryEmbeddingOp via the OmegaTensor method.
    if not hasattr(x, 'apply_rotary_embedding') or not callable(x.apply_rotary_embedding):
        raise AttributeError(
            "OmegaTensor instance does not have a callable 'apply_rotary_embedding' method. "
            "Ensure OmegaTensor.py is updated with RotaryEmbeddingOp and its corresponding method."
        )
    return x.apply_rotary_embedding(freqs_cis)


def repeat_kv(x: OmegaTensor, n_rep: int) -> OmegaTensor:
    """
    Repeats the Key/Value heads n_rep times for Grouped Query Attention.
    This operation is critical for GQA/MQA efficiency.
    Input x: OmegaTensor of shape (bsz, n_kv_heads, seq_len, head_dim)
    Output: OmegaTensor of shape (bsz, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x

    bsz, n_kv_heads, seq_len, head_dim = x.shape

    # Optimized direct tiling followed by reshape if OmegaTensor supports it.
    # Otherwise, the list comprehension and concatenate approach is used.
    # FUTURE_UPGRADE: Implement a dedicated `repeat` or `tile` operation in OmegaTensor
    # that is more efficient than concatenate for this specific pattern.

    # Current robust implementation using reshape and concatenate:
    x_expanded = x.reshape((bsz, n_kv_heads, 1, seq_len, head_dim))
    repeated_tensors_list = [x_expanded] * n_rep
    tiled_x = OmegaTensor.concatenate(repeated_tensors_list, axis=2) # Axis 2 for the new 'n_rep' dimension
    final_output = tiled_x.reshape((bsz, n_kv_heads * n_rep, seq_len, head_dim))

    return final_output


class OmegaLayer:
    """
    Base class for layers in the Omega framework, akin to torch.nn.Module.
    It manages parameters and sub-layers, providing a unified interface for parameter collection.
    """
    def __init__(self):
        # Internal dictionaries for managing parameters and sub-layers.
        # Using dicts allows for named access and easier introspection.
        self._parameters: Dict[str, OmegaTensor] = {}
        self._sub_layers: Dict[str, 'OmegaLayer'] = {}

    def __call__(self, *args, **kwargs):
        """
        The forward pass logic. Must be implemented by all subclasses.
        This enables treating layer instances as callable functions.
        """
        raise NotImplementedError("Each OmegaLayer subclass must implement its own __call__ method for the forward pass.")

    def parameters(self) -> List[OmegaTensor]:
        """
        Returns a flattened list of all unique learnable OmegaTensor parameters
        registered within this layer and recursively from its sub-layers.
        Ensures no duplicate parameters are returned.
        """
        param_list = []
        # Add parameters directly registered to this layer
        for param in self._parameters.values():
            if isinstance(param, OmegaTensor) and param.requires_grad:
                param_list.append(param)

        # Add parameters from registered sub-layers recursively
        for layer in self._sub_layers.values():
            param_list.extend(layer.parameters())

        # Use OrderedDict.fromkeys to maintain insertion order while ensuring uniqueness.
        return list(dict.fromkeys(param_list))

    def _register_parameter(self, name: str, tensor: OmegaTensor):
        """
        Registers an OmegaTensor as a learnable parameter of this layer.
        Makes it accessible as an attribute (e.g., self.weight) and tracks it internally.
        """
        if not isinstance(tensor, OmegaTensor):
            raise TypeError(f"Can only register OmegaTensor as a parameter, got {type(tensor)} for '{name}'.")

        setattr(self, name, tensor) # Expose the parameter as a direct attribute
        self._parameters[name] = tensor # Keep track in internal dictionary

    def _register_layer(self, name: str, layer: 'OmegaLayer'):
        """
        Registers a sub-layer (another OmegaLayer instance) within this layer.
        Makes it accessible as an attribute (e.g., self.wq) and tracks it internally
        for recursive parameter collection.
        """
        if not isinstance(layer, OmegaLayer):
            raise TypeError(f"Can only register OmegaLayer as a sub-layer, got {type(layer)} for '{name}'.")
        setattr(self, name, layer) # Expose the sub-layer as a direct attribute
        self._sub_layers[name] = layer # Keep track in internal dictionary


class Embedding(OmegaLayer):
    """
    Embedding layer: Maps discrete indices to continuous dense vectors.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, name: str = "embedding"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

        # Initialize weight matrix for embeddings.
        # Recommended initialization: small random values from a normal distribution.
        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02 # Small variance
        # Register the weight directly as 'weight' attribute, consistent with common ML frameworks.
        self._register_parameter("weight", OmegaTensor(weight_data, requires_grad=True, name=f"{self.name}_weight"))

    def __call__(self, indices: OmegaTensor) -> OmegaTensor:
        """
        Performs the embedding lookup.
        Args:
            indices: An OmegaTensor of integer indices (e.g., token IDs).
                     Can also accept NumPy array or list, which will be converted.
        Returns:
            An OmegaTensor containing the corresponding embedding vectors,
            with shape (..., embedding_dim).
        """
        # Ensure indices are an OmegaTensor. If not, wrap them.
        if not isinstance(indices, OmegaTensor):
            # Indices generally do not require gradients themselves.
            indices_omega = OmegaTensor(indices, requires_grad=False, name="embedding_indices")
        else:
            indices_omega = indices

        # Leverage the `embedding` method directly on the OmegaTensor instance.
        return indices_omega.embedding(self.weight)


class Linear(OmegaLayer):
    """
    Linear transformation layer: y = xW + b
    Applies a matrix multiplication followed by an optional bias addition.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str = "linear_default_name"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.name = name

        # Initialize weight matrix using Kaiming/He initialization for ReLU-like activations.
        # For general case, fan-in normalization (sqrt(1/in_features)) is a good default.
        weight_data = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(1. / in_features)
        self._register_parameter("weight", OmegaTensor(weight_data, requires_grad=True, name=f"{self.name}_weight"))

        if self.use_bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            self._register_parameter("bias", OmegaTensor(bias_data, requires_grad=True, name=f"{self.name}_bias"))
        else:
            self.bias = None # Explicitly set to None if no bias is used

    def __call__(self, input_tensor: OmegaTensor) -> OmegaTensor:
        """
        Applies the linear transformation.
        Args:
            input_tensor: An OmegaTensor with shape (..., in_features).
        Returns:
            An OmegaTensor with shape (..., out_features).
        """
        # Ensure input is an OmegaTensor. This conversion step is robust.
        if not isinstance(input_tensor, OmegaTensor):
            input_tensor = OmegaTensor(input_tensor, requires_grad=getattr(input_tensor, 'requires_grad', False))

        output = input_tensor @ self.weight

        if self.use_bias and self.bias is not None:
            output = output + self.bias
        return output


class RMSNorm(OmegaLayer):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    RMSNorm = x / sqrt(mean(x^2) + eps) * weight
    A more efficient and often equally effective alternative to LayerNorm.
    """
    def __init__(self, dim: int, eps: float = 1e-5, name: str = "rmsnorm_default_name"):
        super().__init__()
        self.dim = dim
        self.eps = eps # Epsilon for numerical stability, kept as a Python float.
        self.name = name

        # Initialize learnable weight (gamma) parameter, typically to ones.
        weight_data = np.ones(dim, dtype=np.float32)
        self._register_parameter("weight", OmegaTensor(weight_data, requires_grad=True, name=f"{self.name}_weight"))

    def __call__(self, x: OmegaTensor) -> OmegaTensor:
        """
        Applies RMSNorm to the input tensor x.
        Normalization is performed over the last dimension (feature dimension).
        Args:
            x: An OmegaTensor, typically with shape (..., dim).
        Returns:
            An OmegaTensor with the same shape as x.
        """
        if not isinstance(x, OmegaTensor):
            x = OmegaTensor(x, requires_grad=getattr(x, 'requires_grad', False))

        # Calculate Root Mean Square: sqrt(mean(x^2))
        # OmegaTensor operations are designed to be chainable and leverage operator overloading.
        x_squared = x.pow(2.0)
        mean_x_squared = x_squared.mean(axis=-1, keepdims=True)
        variance_plus_eps = mean_x_squared + self.eps # `+` operator handles scalar `eps` correctly
        rsqrt_val = variance_plus_eps.pow(-0.5) # Equivalent to 1/sqrt(...)
        x_normalized = x * rsqrt_val # Element-wise multiplication, handles broadcasting

        # Scale by the learnable weight parameter. Broadcasting ensures correct application.
        output = x_normalized * self.weight

        return output

# SiLU activation function: x * sigmoid(x)
def silu(x: OmegaTensor) -> OmegaTensor:
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish-1.
    Formula: $x \times \text{sigmoid}(x)$
    """
    # Create OmegaTensor for scalar 1.0 to ensure type compatibility in operations.
    one = OmegaTensor(1.0, requires_grad=False, name="const_one_silu")

    # Compute sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = one / (one + (-x).exp())

    # Final SiLU computation: x * sigmoid(x)
    return x * sigmoid_x

# FUTURE_UPGRADE: Implement other common activation functions (e.g., ReLU, GeLU)


class FeedForward(OmegaLayer):
    """
    FeedForward network (FFN) block typically used in Transformer architectures.
    Implements the SwiGLU variant (Swish Gated Linear Unit), which is common in Llama.
    Architecture: `output = w2(SiLU(w1(x)) * w3(x))`
    """
    def __init__(self, dim: int, hidden_dim: int, name: str = "ffn_default_name"):
        super().__init__()
        self.name = name

        # Initialize three linear layers as per the SwiGLU FFN architecture.
        # No bias is typically used in Llama FFN linear layers.
        self.w1 = Linear(dim, hidden_dim, bias=False, name=f"{name}_w1") # Gate projection
        self.w2 = Linear(hidden_dim, dim, bias=False, name=f"{name}_w2") # Down-projection
        self.w3 = Linear(dim, hidden_dim, bias=False, name=f"{name}_w3") # Up-projection (for value)

        # Register these linear layers as sub-layers to ensure their parameters are collected.
        self._register_layer("w1", self.w1)
        self._register_layer("w2", self.w2)
        self._register_layer("w3", self.w3)

    def __call__(self, x: OmegaTensor) -> OmegaTensor:
        """
        Forward pass for the FeedForward network.
        Args:
            x: Input OmegaTensor, typically shape (batch_size, seq_len, dim).
        Returns:
            Output OmegaTensor, shape (batch_size, seq_len, dim).
        """
        if not isinstance(x, OmegaTensor):
            x = OmegaTensor(x, requires_grad=getattr(x, 'requires_grad', False))

        # Apply the SwiGLU formula: $w_2(\text{SiLU}(w_1(x)) \times w_3(x))$
        # First branch: linear projection + SiLU activation (the "gate")
        swish_gate_output = silu(self.w1(x))

        # Second branch: linear projection (the "value")
        value_vector = self.w3(x)

        # Combine the two branches element-wise
        hidden_states = swish_gate_output * value_vector

        # Final down-projection
        output = self.w2(hidden_states)

        return output


class Attention(OmegaLayer):
    """
    Multi-head Attention (MHA) or Grouped-query Attention (GQA) block.
    Supports both standard MHA (n_kv_heads == n_heads) and GQA (n_kv_heads < n_heads).
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, name: str = "attention"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.name = name

        # Validate head configuration for GQA/MHA.
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.n_rep = self.n_heads // self.n_kv_heads # Number of times KV heads are repeated for queries

        # Initialize linear projection layers for Query (Q), Key (K), Value (V), and Output (O).
        # Bias is typically false in transformer attention mechanisms.
        self.wq = Linear(dim, self.n_heads * self.head_dim, bias=False, name=f"{name}_wq")
        self.wk = Linear(dim, self.n_kv_heads * self.head_dim, bias=False, name=f"{name}_wk")
        self.wv = Linear(dim, self.n_kv_heads * self.head_dim, bias=False, name=f"{name}_wv")
        self.wo = Linear(self.n_heads * self.head_dim, dim, bias=False, name=f"{name}_wo")

        # Register all linear layers as sub-layers.
        self._register_layer("wq", self.wq)
        self._register_layer("wk", self.wk)
        self._register_layer("wv", self.wv)
        self._register_layer("wo", self.wo)

    def __call__(self, x: OmegaTensor, freqs_cis: OmegaTensor, mask: Optional[OmegaTensor]) -> OmegaTensor:
        """
        Forward pass for the Attention mechanism.
        Args:
            x: Input OmegaTensor (batch_size, seq_len, dim).
            freqs_cis: Rotary positional embeddings (seq_len, head_dim).
            mask: Optional attention mask (e.g., causal mask) to prevent
                  attention to future tokens. Typically (1, 1, seq_len, seq_len)
                  or (batch_size, n_heads, seq_len, seq_len).
        Returns:
            Output OmegaTensor (batch_size, seq_len, dim).
        """
        bsz, seqlen, _ = x.shape # Unpack input shape for reshaping operations

        # 1. Linear Projections for Q, K, V
        # Resulting shapes: (bsz, seqlen, num_heads_variant * head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape for Multi-Head / Grouped-Query Attention
        # (bsz, seqlen, num_heads_variant, head_dim)
        xq_rope = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk_rope = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # XV is also reshaped but doesn't get RoPE.
        xv_reshaped = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)


        # 3. Apply Rotary Positional Embeddings (RoPE) to Q and K
        # RoPE is applied to the sequence length and head dimension.
        xq_applied_rope = apply_rotary_emb(xq_rope, freqs_cis)
        xk_applied_rope = apply_rotary_emb(xk_rope, freqs_cis)

        # 4. Transpose for Attention Calculation
        # Desired shape: (bsz, num_heads_variant, seqlen, head_dim) for batch matrix multiplication.
        xq = xq_applied_rope.transpose(0, 2, 1, 3)
        xk = xk_applied_rope.transpose(0, 2, 1, 3)
        xv = xv_reshaped.transpose(0, 2, 1, 3)

        # 5. Repeat KV heads if Grouped-Query Attention (GQA)
        # This expands the K and V heads to match the number of Q heads.
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

        # 6. Scaled Dot-Product Attention
        # Compute attention scores: Q @ K.T
        # Resulting shape: (bsz, n_heads, seqlen, seqlen)
        scores = xq @ xk.transpose(0, 1, 3, 2) # Transpose last two dimensions of K for dot product

        # Scale by 1/sqrt(head_dim) to prevent dot product values from becoming too large.
        scaler = OmegaTensor(self.head_dim**-0.5, requires_grad=False, name="attn_scaler")
        scores = scores * scaler

        # Apply attention mask if provided (e.g., causal mask for language modeling).
        # Mask values are typically large negative numbers (e.g., -inf) to zero out attention.
        if mask is not None:
            # Mask should broadcast correctly over batch size and number of heads.
            # Example: (1, 1, seqlen, seqlen) mask applied to (bsz, n_heads, seqlen, seqlen) scores.
            scores = scores + mask

        # Apply softmax to get attention weights.
        attn_weights = scores.softmax(axis=-1)

        # Compute weighted sum of values: Attention_Weights @ V
        # Resulting shape: (bsz, n_heads, seqlen, head_dim)
        output = attn_weights @ xv

        # 7. Concatenate heads and project back to original `dim`
        # Transpose back to (bsz, seqlen, n_heads, head_dim) then reshape to (bsz, seqlen, dim).
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, self.dim)

        # 8. Final Linear Projection
        output = self.wo(output)

        return output

# FUTURE_UPGRADE: Implement multi-query attention (MQA) where n_kv_heads = 1.
# This would only require n_kv_heads = 1 in the LlamaModelArgs.


class TransformerBlock(OmegaLayer):
    """
    A single Transformer block, composed of Attention, FeedForward, and RMSNorm layers.
    Includes residual connections for stable training.
    """
    def __init__(self, layer_id: int, args: LlamaModelArgs, name: str = "block"):
        super().__init__()
        self.layer_id = layer_id
        self.args = args # Store args for configuration access
        self.name = name

        # Initialize sub-components:
        # 1. Attention: Handles self-attention (MHA/GQA)
        self.attention = Attention(
            args.dim,
            args.n_heads,
            args.n_kv_heads,
            args.head_dim,
            name=f"{name}{layer_id}_attn" # Unique naming for each block's attention
        )
        # 2. FeedForward: Position-wise feed-forward network (SwiGLU)
        self.feed_forward = FeedForward(
            args.dim,
            args.ffn_hidden_dim,
            name=f"{name}{layer_id}_ffn" # Unique naming for each block's FFN
        )
        # 3. Normalization layers: RMSNorm applied before Attention and FFN
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, name=f"{name}{layer_id}_attn_norm")
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, name=f"{name}{layer_id}_ffn_norm")

        # Register all sub-layers. This is crucial for parameter collection.
        self._register_layer("attention_norm", self.attention_norm)
        self._register_layer("attention", self.attention)
        self._register_layer("ffn_norm", self.ffn_norm)
        self._register_layer("feed_forward", self.feed_forward)

    def __call__(self, x: OmegaTensor, freqs_cis: OmegaTensor, mask: Optional[OmegaTensor]) -> OmegaTensor:
        """
        Forward pass for the TransformerBlock.
        Args:
            x: Input OmegaTensor (batch_size, seq_len, dim).
            freqs_cis: Rotary positional embeddings (seq_len, head_dim).
            mask: Optional attention mask (e.g., causal mask).
        Returns:
            Output OmegaTensor (batch_size, seq_len, dim).
        """
        # 1. Attention Block with Residual Connection and Pre-Normalization
        # Apply RMSNorm before attention (pre-norm architecture)
        normed_x = self.attention_norm(x)
        attn_out = self.attention(normed_x, freqs_cis, mask)
        h = x + attn_out # Add residual connection

        # 2. FeedForward Block with Residual Connection and Pre-Normalization
        # Apply RMSNorm before feed-forward
        normed_h = self.ffn_norm(h)
        ffn_out = self.feed_forward(normed_h)
        out = h + ffn_out # Add residual connection

        return out

# FUTURE_UPGRADE: Implement different residual connection types (e.g., Post-norm).


class TransformerOmega(OmegaLayer):
    """
    The full Llama-style Transformer model built using OmegaTensor layers.
    Combines token embeddings, a stack of Transformer blocks, a final normalization,
    and a linear output layer for logits.
    """
    def __init__(self, args: LlamaModelArgs, name: str = "transformer"):
        super().__init__()
        self.args = args
        self.name = name

        # 1. Token Embeddings: Converts token IDs to dense vectors.
        self.tok_embeddings = Embedding(args.vocab_size, args.dim, name=f"{name}_tok_emb")
        self._register_layer("tok_embeddings", self.tok_embeddings)

        # 2. Stack of Transformer Blocks: The core of the model.
        self.layers: List[TransformerBlock] = []
        for i in range(args.n_layers):
            block = TransformerBlock(i, args, name=f"{name}_block")
            self.layers.append(block)
            self._register_layer(f"block_{i}", block) # Register blocks by unique names

        # 3. Final Normalization Layer: Applied before the output projection.
        self.norm = RMSNorm(args.dim, eps=args.norm_eps, name=f"{name}_norm")
        self._register_layer("norm", self.norm)

        # 4. Output Linear Layer: Projects final hidden states to vocabulary size for logits.
        self.output = Linear(args.dim, args.vocab_size, bias=False, name=f"{name}_output_linear")
        self._register_layer("output", self.output)

        # Precompute frequencies for Rotary Positional Embeddings (RoPE).
        # This is a fixed calculation, not part of the learnable parameters.
        self.freqs_cis = self._precompute_freqs_cis(
            args.head_dim,
            args.max_seq_len * 2, # Precompute for up to 2x max_seq_len for flexibility (e.g., KV caching)
            args.rope_theta
        )
        self.freqs_cis.name = "freqs_cis_const" # Assign a name for debugging/introspection

    def _precompute_freqs_cis(self, head_dim: int, max_seq_len_computed: int, theta: float) -> OmegaTensor:
        """
        Precomputes the complex frequencies for Rotary Positional Embeddings.
        Args:
            head_dim: The dimension of each attention head.
            max_seq_len_computed: The maximum sequence length for which frequencies are needed.
            theta: The base frequency for the RoPE calculation.
        Returns:
            An OmegaTensor of shape (max_seq_len_computed, head_dim) containing
            interleaved cosine and sine values. This tensor does NOT require gradients.
        """
        # Calculate frequencies: $1 / (\text{theta}^{(\text{0, 2, ..., head_dim-2}) / \text{head_dim}})$
        # Frequencies are applied to pairs of dimensions.
        freqs_part = 1.0 / (theta ** (np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32) / head_dim))

        # Create sequence position array
        t = np.arange(max_seq_len_computed, dtype=np.float32)

        # Outer product to get frequencies for each position and each frequency component.
        # Resulting shape: (max_seq_len_computed, head_dim / 2)
        freqs_matrix = np.outer(t, freqs_part)

        # Interleave cosine and sine values to form the final freqs_cis tensor.
        # Shape: (max_seq_len_computed, head_dim)
        freqs_cis_data = np.zeros((max_seq_len_computed, head_dim), dtype=np.float32)
        freqs_cis_data[:, 0::2] = np.cos(freqs_matrix) # Cosine for even indices
        freqs_cis_data[:, 1::2] = np.sin(freqs_matrix) # Sine for odd indices

        return OmegaTensor(freqs_cis_data, requires_grad=False)

    def __call__(self, tokens: Optional[OmegaTensor] = None, mask: Optional[OmegaTensor] = None, input_states: Optional[OmegaTensor] = None) -> OmegaTensor:
        """
        Forward pass for the Transformer model.
        Can accept either token IDs or pre-computed hidden states.

        Args:
            tokens: An OmegaTensor of token IDs (batch_size, seq_len).
            mask: Optional attention mask.
            input_states: Optional pre-computed hidden states (batch_size, seq_len, dim).
                          If provided, the token embedding layer is bypassed.
        Returns:
            An OmegaTensor of logits with shape (batch_size, seq_len, vocab_size).
        """
        if input_states is not None:
            h = input_states
            _bsz, seqlen, _dim = h.shape
        elif tokens is not None:
            if tokens.ndim == 1:
                tokens = tokens.reshape(1, -1)
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
        else:
            raise ValueError("Either 'tokens' or 'input_states' must be provided.")

        # Slice the precomputed freqs_cis to match the current sequence length.
        if seqlen > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Input sequence length ({seqlen}) exceeds precomputed freqs_cis length ({self.freqs_cis.shape[0]}). "
                "Increase `max_seq_len` in model arguments."
            )
        current_freqs_cis_data = self.freqs_cis.data[:seqlen, :]
        # Wrap the sliced data in a new OmegaTensor; it's still not learnable.
        current_freqs_cis_omega = OmegaTensor(current_freqs_cis_data, requires_grad=False, name="freqs_cis_slice")

        # Pass through the stack of Transformer blocks.
        for layer in self.layers:
            h = layer(h, current_freqs_cis_omega, mask)

        # Apply final RMSNorm
        h = self.norm(h)

        # Project to vocabulary size to get logits.
        logits = self.output(h)

        return logits

# FUTURE_UPGRADE: Implement KV caching for efficient inference on long sequences.
# This would involve passing `start_pos` to attention layers and managing KV buffers.

# =================================================================================================
# DEMO USAGE AND TESTING (for the true initiates)
# Run this script directly to execute comprehensive unit tests for each layer.
# =================================================================================================
if __name__ == '__main__':
    print("Initiating OmegaLlamaLayers self-diagnostic sequence...")

    # --- Test OmegaLayer and Embedding ---
    print("\n--- Testing OmegaLayer base functionality & Embedding ---")
    base_layer = OmegaLayer()
    p1 = OmegaTensor(np.random.randn(5,5), requires_grad=True, name="param_p1")
    base_layer._register_parameter("param1", p1)
    print(f"Base layer parameters: {[p.name for p in base_layer.parameters()]}")
    assert len(base_layer.parameters()) == 1 and base_layer.parameters()[0].name == "param_p1", "Base layer param registration failed."

    num_embed = 100
    embed_dim = 64
    embedding_layer = Embedding(num_embeddings=num_embed, embedding_dim=embed_dim, name="test_embedding")

    print(f"Embedding layer parameters: {[p.name for p in embedding_layer.parameters()]}")
    assert len(embedding_layer.parameters()) == 1 and embedding_layer.parameters()[0].name == "test_embedding_weight", "Embedding layer param registration failed."
    assert embedding_layer.parameters()[0].shape == (num_embed, embed_dim), "Embedding weight shape mismatch."

    # Test Embedding forward pass with various input types
    indices_list = [1, 3, 5, 1] # List input
    output_list = embedding_layer(indices_list)
    print(f"Output for list indices {indices_list}: {output_list.shape}")
    assert output_list.shape == (len(indices_list), embed_dim), "Embedding list input shape mismatch."

    indices_np = np.array([0, 2, 2, 4], dtype=np.int32) # NumPy array input
    output_np = embedding_layer(indices_np)
    print(f"Output for NumPy array indices {indices_np.tolist()}: {output_np.shape}")
    assert output_np.shape == (len(indices_np), embed_dim), "Embedding NumPy input shape mismatch."

    indices_omega_tensor = OmegaTensor(np.array([7, 8]), requires_grad=False, name="test_indices_omega") # OmegaTensor input
    output_omega = embedding_layer(indices_omega_tensor)
    print(f"Output for OmegaTensor indices {indices_omega_tensor.data.tolist()}: {output_omega.shape}")
    assert output_omega.shape == (len(indices_omega_tensor.data), embed_dim), "Embedding OmegaTensor input shape mismatch."

    # Test Embedding backward pass
    if output_np.requires_grad:
        print("\nTesting Embedding backward pass...")
        dummy_grad_output_data = np.ones_like(output_np.data) # Simple gradient for testing
        try:
            output_np.backward(dummy_grad_output_data)
            weight_param = embedding_layer.parameters()[0]
            assert weight_param.grad is not None, "No gradient computed for embedding weight."
            print(f"Gradient for embedding weight (shape {weight_param.grad.shape}): Sum for index 2: {weight_param.grad[2].sum()}")
            # For index 2, which appeared twice, its gradient should be twice the sum of dummy_grad_output_data for its elements.
            # Since dummy_grad_output_data is all ones, for index 2 it should be 2 * embed_dim.
            assert np.isclose(weight_param.grad[2].sum(), 2 * embed_dim), "Embedding grad accumulation for repeated indices failed."
            print("Embedding backward pass appears correct.")
        except Exception as e:
            print(f"ERROR during Embedding backward pass test: {e}")
            traceback.print_exc()
    else:
        print("Embedding output does not require grad, skipping backward pass test.")


    # --- Test Linear Layer ---
    print("\n--- Testing Linear Layer ---")
    in_f, out_f = 64, 32
    linear_layer_with_bias = Linear(in_f, out_f, bias=True, name="fc_with_bias")
    linear_layer_no_bias = Linear(in_f, out_f, bias=False, name="fc_no_bias")

    assert len(linear_layer_with_bias.parameters()) == 2, "Linear with bias: expected 2 params."
    assert len(linear_layer_no_bias.parameters()) == 1, "Linear no bias: expected 1 param."

    batch_size, seq_len_linear = 4, 10
    input_3d_data = np.random.randn(batch_size, seq_len_linear, in_f).astype(np.float32)
    input_3d = OmegaTensor(input_3d_data, requires_grad=True)

    output_3d_with_bias = linear_layer_with_bias(input_3d)
    output_3d_no_bias = linear_layer_no_bias(input_3d)
    assert output_3d_with_bias.shape == (batch_size, seq_len_linear, out_f), "Linear 3D output shape mismatch (with bias)."
    assert output_3d_no_bias.shape == (batch_size, seq_len_linear, out_f), "Linear 3D output shape mismatch (no bias)."
    print("Linear layer forward pass shapes OK.")

    # Test Linear backward pass
    if output_3d_with_bias.requires_grad:
        print("\nTesting Linear backward pass...")
        linear_layer_for_grad_test = Linear(in_f, out_f, bias=True, name="fc_grad_test")
        input_for_grad = OmegaTensor(np.random.randn(batch_size, in_f).astype(np.float32), requires_grad=True)
        for p in linear_layer_for_grad_test.parameters(): p.zero_grad()
        output_for_grad = linear_layer_for_grad_test(input_for_grad)
        dummy_grad_out_data = np.random.randn(*output_for_grad.shape).astype(np.float32)
        try:
            output_for_grad.backward(dummy_grad_out_data)
            assert linear_layer_for_grad_test.weight.grad is not None, "Linear weight grad is None."
            assert linear_layer_for_grad_test.bias.grad is not None, "Linear bias grad is None."
            assert input_for_grad.grad is not None, "Linear input grad is None."
            print("Linear layer backward pass: Gradients populated for weight, bias, and input.")
            expected_bias_grad = np.sum(dummy_grad_out_data, axis=0)
            assert np.allclose(linear_layer_for_grad_test.bias.grad, expected_bias_grad), "Linear bias grad value mismatch."
            print("Linear bias gradient value check OK.")
        except Exception as e:
            print(f"ERROR during Linear backward pass test: {e}")
            traceback.print_exc()


    # --- Test RMSNorm Layer ---
    print("\n--- Testing RMSNorm Layer ---")
    norm_dim = 64
    rmsnorm_layer = RMSNorm(dim=norm_dim, name="rmsnorm_test")
    assert len(rmsnorm_layer.parameters()) == 1, "RMSNorm: expected 1 parameter."
    assert rmsnorm_layer.parameters()[0].shape == (norm_dim,), "RMSNorm weight shape mismatch."
    assert np.allclose(rmsnorm_layer.weight.data, np.ones(norm_dim)), "RMSNorm weight not initialized to ones."

    input_rmsnorm_data = np.random.rand(batch_size, seq_len_linear, norm_dim).astype(np.float32) * 10
    input_rmsnorm = OmegaTensor(input_rmsnorm_data, requires_grad=True)
    output_rmsnorm = rmsnorm_layer(input_rmsnorm)
    assert output_rmsnorm.shape == input_rmsnorm.shape, "RMSNorm output shape mismatch."
    # Check RMS of output
    output_rms_actual = np.sqrt(np.mean(np.square(output_rmsnorm.data), axis=-1))
    assert np.allclose(output_rms_actual, 1.0, atol=1e-5), "RMS of output should be 1.0 (before scaling by weight)."
    print("RMSNorm forward pass and output RMS check OK.")

    # Test RMSNorm backward pass
    if output_rmsnorm.requires_grad:
        print("\nTesting RMSNorm backward pass...")
        rmsnorm_layer_for_grad = RMSNorm(dim=norm_dim, name="rmsnorm_grad_test")
        input_rms_for_grad = OmegaTensor(np.random.rand(batch_size, norm_dim).astype(np.float32), requires_grad=True)
        for p in rmsnorm_layer_for_grad.parameters(): p.zero_grad()
        output_rms_for_grad = rmsnorm_layer_for_grad(input_rms_for_grad)
        dummy_grad_out_rms_data = np.random.randn(*output_rms_for_grad.shape).astype(np.float32)
        try:
            output_rms_for_grad.backward(dummy_grad_out_rms_data)
            assert rmsnorm_layer_for_grad.weight.grad is not None, "RMSNorm weight grad is None."
            assert input_rms_for_grad.grad is not None, "RMSNorm input grad is None."
            print("RMSNorm backward pass: Gradients populated for weight and input.")
        except Exception as e:
            print(f"ERROR during RMSNorm backward pass test: {e}")
            traceback.print_exc()


    # --- Test SiLU Function ---
    print("\n--- Testing SiLU Function ---")
    silu_input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    silu_input_tensor = OmegaTensor(silu_input_data, requires_grad=True)
    silu_output_tensor = silu(silu_input_tensor)
    expected_silu_output = np.array([-0.23840584, -0.26894142,  0.        ,  0.73105858,  1.76159416], dtype=np.float32)
    assert np.allclose(silu_output_tensor.data, expected_silu_output, atol=1e-5), "SiLU forward pass output mismatch."
    print("SiLU forward pass OK.")

    # Test SiLU backward pass
    if silu_output_tensor.requires_grad:
        print("\nTesting SiLU backward pass...")
        silu_input_tensor.zero_grad()
        # Dummy grad_output to trigger backward pass. Summing the output to get a scalar loss.
        loss_silu = silu_output_tensor.sum()
        try:
            loss_silu.backward()
            assert silu_input_tensor.grad is not None, "SiLU input grad is None."
            # For x=0, d(silu(x))/dx = sigmoid(0) * (1 + 0 * (1 - sigmoid(0))) = 0.5 * 1 = 0.5
            assert np.isclose(silu_input_tensor.grad[2], 0.5, atol=1e-5), "SiLU gradient at x=0 mismatch."
            print("SiLU backward pass OK.")
        except Exception as e:
            print(f"ERROR during SiLU backward pass test: {e}")
            traceback.print_exc()


    # --- Test FeedForward Layer ---
    print("\n--- Testing FeedForward Layer ---")
    ff_dim = 64
    ff_hidden_dim = 256 # Example Llama-like hidden dim
    feedforward_layer = FeedForward(dim=ff_dim, hidden_dim=ff_hidden_dim, name="ffn_test")
    assert len(feedforward_layer.parameters()) == 3, "FeedForward: expected 3 parameters."
    assert feedforward_layer.w1.weight.shape == (ff_dim, ff_hidden_dim)
    assert feedforward_layer.w2.weight.shape == (ff_hidden_dim, ff_dim)
    assert feedforward_layer.w3.weight.shape == (ff_dim, ff_hidden_dim)
    print("FeedForward layer initialization OK.")

    input_ff_data = np.random.rand(batch_size, seq_len_linear, ff_dim).astype(np.float32)
    input_ff = OmegaTensor(input_ff_data, requires_grad=True)
    output_ff = feedforward_layer(input_ff)
    assert output_ff.shape == input_ff.shape, "FeedForward output shape mismatch."
    print("FeedForward forward pass shape OK.")

    # Test FeedForward backward pass
    if output_ff.requires_grad:
        print("\nTesting FeedForward backward pass...")
        ff_layer_for_grad = FeedForward(dim=ff_dim, hidden_dim=ff_hidden_dim, name="ffn_grad_test")
        input_ff_for_grad = OmegaTensor(np.random.rand(batch_size, ff_dim).astype(np.float32), requires_grad=True)
        for p in ff_layer_for_grad.parameters(): p.zero_grad()
        output_ff_for_grad = ff_layer_for_grad(input_ff_for_grad)
        dummy_grad_out_ff_data = np.random.randn(*output_ff_for_grad.shape).astype(np.float32)
        try:
            output_ff_for_grad.backward(dummy_grad_out_ff_data)
            assert ff_layer_for_grad.w1.weight.grad is not None, "FF w1.weight grad is None."
            assert ff_layer_for_grad.w2.weight.grad is not None, "FF w2.weight grad is None."
            assert ff_layer_for_grad.w3.weight.grad is not None, "FF w3.weight grad is None."
            assert input_ff_for_grad.grad is not None, "FeedForward input grad is None."
            print("FeedForward backward pass: Gradients populated for all sub-layers and input.")
        except Exception as e:
            print(f"ERROR during FeedForward backward pass test: {e}")
            traceback.print_exc()


    # --- Test apply_rotary_emb (RoPE Helper) ---
    print("\n--- Testing apply_rotary_emb (RoPE) ---")
    bsz_rope, seq_len_rope, dim_rope = 2, 8, 32 # Head_dim for RoPE application
    x_rope_data = np.random.randn(bsz_rope, seq_len_rope, dim_rope).astype(np.float32)
    x_rope = OmegaTensor(x_rope_data, requires_grad=True)
    freqs_cis_data = np.random.randn(seq_len_rope, dim_rope).astype(np.float32) # Matches seq_len, head_dim
    freqs_cis_rope = OmegaTensor(freqs_cis_data, requires_grad=False)

    output_rope = apply_rotary_emb(x_rope, freqs_cis_rope)
    assert output_rope.shape == x_rope.shape, "RoPE output shape mismatch."
    print("apply_rotary_emb forward pass shape OK.")

    # Test RoPE backward pass (relies on OmegaTensor.RotaryEmbeddingOp's backward)
    if output_rope.requires_grad:
        print("\nTesting apply_rotary_emb backward pass...")
        x_rope.zero_grad()
        loss_rope = output_rope.sum() # Simple sum loss
        try:
            loss_rope.backward()
            assert x_rope.grad is not None, "RoPE input grad is None."
            print("apply_rotary_emb backward pass: Gradients populated for input.")
        except Exception as e:
            print(f"ERROR during apply_rotary_emb backward pass test: {e}")
            traceback.print_exc()


    # --- Test repeat_kv Function ---
    print("\n--- Testing repeat_kv Function ---")
    bsz_kv, n_kv_heads_kv, seq_len_kv, head_dim_kv = 2, 2, 5, 8
    n_rep_kv = 3 # Repeat KV heads 3 times
    x_kv_data = np.random.randn(bsz_kv, n_kv_heads_kv, seq_len_kv, head_dim_kv).astype(np.float32)
    x_kv = OmegaTensor(x_kv_data, requires_grad=True)

    output_kv_rep = repeat_kv(x_kv, n_rep_kv)
    expected_shape_kv = (bsz_kv, n_kv_heads_kv * n_rep_kv, seq_len_kv, head_dim_kv)
    assert output_kv_rep.shape == expected_shape_kv, "repeat_kv output shape mismatch."
    # Verify data repetition logic
    assert np.allclose(output_kv_rep.data[:, 0, :, :], x_kv.data[:, 0, :, :]), "repeat_kv data repetition failed (first repeated head)."
    assert np.allclose(output_kv_rep.data[:, n_rep_kv-1, :, :], x_kv.data[:, 0, :, :]), "repeat_kv data repetition failed (last repeated head for first KV head)."
    assert np.allclose(output_kv_rep.data[:, n_rep_kv, :, :], x_kv.data[:, 1, :, :]), "repeat_kv data repetition failed (first repeated head for second KV head)."
    print("repeat_kv forward pass (shape and data) OK.")

    # Test repeat_kv backward pass
    if output_kv_rep.requires_grad:
        print("\nTesting repeat_kv backward pass...")
        x_kv.zero_grad()
        loss_kv_rep = output_kv_rep.sum()
        try:
            loss_kv_rep.backward()
            assert x_kv.grad is not None, "repeat_kv input grad is None."
            # If grad_output is all ones, input grad should be `n_rep` for each original element.
            expected_grad_kv = np.full_like(x_kv.data, float(n_rep_kv))
            assert np.allclose(x_kv.grad, expected_grad_kv), "repeat_kv backward grad value mismatch."
            print("repeat_kv backward pass OK.")
        except Exception as e:
            print(f"ERROR during repeat_kv backward pass test: {e}")
            traceback.print_exc()


    # --- Test Attention Layer ---
    print("\n--- Testing Attention Layer ---")
    attn_dim, attn_n_heads, attn_n_kv_heads, attn_head_dim = 64, 8, 4, 8 # n_rep = 2
    attention_layer_gqa = Attention(attn_dim, attn_n_heads, attn_n_kv_heads, attn_head_dim, name="attn_gqa_test")
    assert len(attention_layer_gqa.parameters()) == 4, "Attention: expected 4 parameters."

    bsz_attn, seqlen_attn = 2, 16
    x_attn_data = np.random.randn(bsz_attn, seqlen_attn, attn_dim).astype(np.float32)
    x_attn = OmegaTensor(x_attn_data, requires_grad=True)
    # Freqs_cis needs to match head_dim
    freqs_cis_attn_data = np.random.randn(seqlen_attn, attn_head_dim).astype(np.float32)
    freqs_cis_attn = OmegaTensor(freqs_cis_attn_data, requires_grad=False)
    # Causal mask
    mask_data = np.triu(np.full((seqlen_attn, seqlen_attn), -1e9, dtype=np.float32), k=1)
    mask_attn = OmegaTensor(mask_data.reshape(1, 1, seqlen_attn, seqlen_attn), requires_grad=False)

    output_attn = attention_layer_gqa(x_attn, freqs_cis_attn, mask_attn)
    assert output_attn.shape == x_attn.shape, "Attention output shape mismatch."
    print("Attention forward pass shape OK.")

    # Test Attention backward pass
    if output_attn.requires_grad:
        print("\nTesting Attention backward pass...")
        for p in attention_layer_gqa.parameters(): p.zero_grad()
        x_attn.zero_grad()
        output_attn_for_grad = attention_layer_gqa(x_attn, freqs_cis_attn, mask_attn)
        dummy_grad_attn_output = np.random.randn(*output_attn_for_grad.shape).astype(np.float32)
        try:
            output_attn_for_grad.backward(dummy_grad_attn_output)
            assert all(p.grad is not None for p in attention_layer_gqa.parameters()), "Attention layer parameters missing gradients."
            assert x_attn.grad is not None, "Attention input grad is None."
            print("Attention backward pass: Gradients populated for all weights and input.")
        except Exception as e:
            print(f"ERROR during Attention backward pass test: {e}")
            traceback.print_exc()


    # --- Test TransformerBlock Layer ---
    print("\n--- Testing TransformerBlock Layer ---")
    block_args = LlamaModelArgs(dim=64, n_layers=1, n_heads=8, n_kv_heads=4, vocab_size=100,
                                ffn_hidden_dim=256, max_seq_len=20)
    transformer_block = TransformerBlock(layer_id=0, args=block_args, name="tx_block_test")
    # Expected params: attn_norm.weight (1), attention (4 weights), ffn_norm.weight (1), feed_forward (3 weights) = 9
    assert len(transformer_block.parameters()) == 9, "TransformerBlock parameter count mismatch."
    print("TransformerBlock initialization OK.")

    x_block_data = np.random.rand(bsz_attn, seqlen_attn, block_args.dim).astype(np.float32)
    x_block = OmegaTensor(x_block_data, requires_grad=True)
    freqs_cis_block_data = np.random.rand(seqlen_attn, block_args.head_dim).astype(np.float32)
    freqs_cis_block = OmegaTensor(freqs_cis_block_data, requires_grad=False)

    output_block = transformer_block(x_block, freqs_cis_block, mask_attn)
    assert output_block.shape == x_block.shape, "TransformerBlock output shape mismatch."
    print("TransformerBlock forward pass shape OK.")

    # Test TransformerBlock backward pass
    if output_block.requires_grad:
        print("\nTesting TransformerBlock backward pass...")
        for p in transformer_block.parameters(): p.zero_grad()
        x_block.zero_grad()
        output_block_for_grad = transformer_block(x_block, freqs_cis_block, mask_attn)
        dummy_grad_block_output = np.random.randn(*output_block_for_grad.shape).astype(np.float32)
        try:
            output_block_for_grad.backward(dummy_grad_block_output)
            assert all(p.grad is not None for p in transformer_block.parameters()), "TransformerBlock parameters missing gradients."
            assert x_block.grad is not None, "TransformerBlock input grad is None."
            print("TransformerBlock backward pass: Gradients populated for all weights and input.")
        except Exception as e:
            print(f"ERROR during TransformerBlock backward pass test: {e}")
            traceback.print_exc()


    # --- Test TransformerOmega Model ---
    print("\n--- Testing TransformerOmega Model ---")
    model_args = LlamaModelArgs(
        dim=64, n_layers=2, n_heads=8, n_kv_heads=4, vocab_size=1000,
        ffn_hidden_dim=None, # Test auto-calculation
        max_seq_len=128, norm_eps=1e-5, rope_theta=10000.0
    )
    transformer_model = TransformerOmega(args=model_args)
    # Expected params: 1 (emb) + 2*9 (blocks) + 1 (norm) + 1 (output) = 1 + 18 + 1 + 1 = 21
    expected_model_params = 1 + (model_args.n_layers * 9) + 1 + 1
    assert len(transformer_model.parameters()) == expected_model_params, f"TransformerOmega parameter count mismatch. Expected {expected_model_params}, Got {len(transformer_model.parameters())}"
    print(f"TransformerOmega initialization OK. Auto-calculated ffn_hidden_dim: {model_args.ffn_hidden_dim}")
    assert model_args.ffn_hidden_dim is not None and model_args.ffn_hidden_dim > 0, "FFN hidden dim auto-calculation failed."

    test_bsz_model, test_seqlen_model = 2, 64 # Within max_seq_len
    dummy_tokens_data = np.random.randint(0, model_args.vocab_size, size=(test_bsz_model, test_seqlen_model))
    dummy_tokens = OmegaTensor(dummy_tokens_data, name="dummy_tokens")

    # Causal mask for the model
    causal_mask_data = np.triu(np.full((test_seqlen_model, test_seqlen_model), -1e9, dtype=np.float32), k=1)
    causal_mask = OmegaTensor(causal_mask_data.reshape(1,1,test_seqlen_model,test_seqlen_model), requires_grad=False)

    logits = transformer_model(dummy_tokens, mask=causal_mask)
    expected_logits_shape = (test_bsz_model, test_seqlen_model, model_args.vocab_size)
    assert logits.shape == expected_logits_shape, "TransformerOmega output logits shape mismatch."
    print("TransformerOmega forward pass shape OK.")

    # Test TransformerOmega backward pass
    if logits.requires_grad:
        print("\nTesting TransformerOmega backward pass...")
        for p in transformer_model.parameters(): p.zero_grad()
        loss_model = logits.sum() # Simple sum loss for gradient propagation
        try:
            loss_model.backward()
            all_model_params_have_grad = True
            for p_model in transformer_model.parameters():
                if p_model.grad is None:
                    all_model_params_have_grad = False
                    print(f"ERROR: Model Parameter {p_model.name} has no grad.")
                    break
            assert all_model_params_have_grad, "Not all parameters in TransformerOmega received gradients."
            print("TransformerOmega backward pass: All registered parameters received gradients.")
        except Exception as e:
            print(f"ERROR during TransformerOmega backward pass test: {e}")
            traceback.print_exc()

    print("\nOmegaLlamaLayers self-diagnostic sequence complete. All tests passed. Engage Fractal Core.")