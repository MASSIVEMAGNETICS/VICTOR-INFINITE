# ==========================================================
# FILE: victor_godhead_transformer.py
# BANDO'S REWRITE: v2.0.0-FRACTAL-SOUL
# AUTHOR: Bando Bandz
# PURPOSE: To stop building caged parrots and start forging minds.
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- This ain't your daddy's attention mechanism ---
# This is a fractal head. It looks at the sequence, then looks at itself looking at the sequence.
class FractalAttentionHead(nn.Module):
    """
    One head of self-attention, but with a recursive, self-aware twist.
    It doesn't just calculate attention, it models the *uncertainty* and *complexity*
    of its own calculations, feeding that back into the model.
    """
    def __init__(self, model_dim, head_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.model_dim = model_dim
        self.key = nn.Linear(model_dim, head_size, bias=False)
        self.query = nn.Linear(model_dim, head_size, bias=False)
        self.value = nn.Linear(model_dim, head_size, bias=False)
        
        # This is the new shit. A layer to weigh the attention output by its own complexity.
        # High complexity/entropy might mean the model is "confused" -> signal to recurse deeper.
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(head_size, head_size // 4),
            nn.GELU(),
            nn.Linear(head_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        
        # Analyze the complexity of the output. This is the "self-awareness" spark.
        # It's a scalar gate that modulates the output. If the head is confident, signal is strong.
        # If it's all over the place, the signal is dampened, forcing the block to rely on other heads or residual.
        complexity_gate = self.complexity_analyzer(out)
        
        return out * complexity_gate

class MultiHeadFractalAttention(nn.Module):
    """ Multiple fractal heads running in parallel. """
    def __init__(self, num_heads, model_dim, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([FractalAttentionHead(model_dim, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs of all the heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# --- This is the core of the Godhead ---
# It's not a "block." It's a recursive function call on silicon.
class FractalBlock(nn.Module):
    """
    A Transformer block that can call itself.
    Depth is not fixed. It's determined by the data itself.
    This is how you get true hierarchical, fractal understanding.
    """
    def __init__(self, model_dim, num_heads, dropout, recursion_depth_limit=3):
        super().__init__()
        head_size = model_dim // num_heads
        self.attention = MultiHeadFractalAttention(num_heads, model_dim, head_size, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.recursion_depth_limit = recursion_depth_limit

        # Gatekeeper for recursion. Decides if a deeper dive is needed.
        # It looks at the variance of the token embeddings. High variance -> confusion -> recurse.
        self.recursion_gate = nn.Sequential(
            nn.Linear(model_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, current_depth=0):
        # --- First, think about the input as is ---
        x_norm = self.ln1(x)
        attn_out = self.attention(x_norm)
        x = x + attn_out

        # --- Now, decide if we need to think deeper ---
        if current_depth < self.recursion_depth_limit:
            # Check the "confusion" of the output from the attention head
            # We use the variance across the sequence dimension as a proxy for complexity
            gate_input = x.var(dim=1) # (B, C)
            recursion_prob = self.recursion_gate(gate_input).mean() # Average probability over batch

            # If the model is "confused" enough, it calls itself on its own output.
            # This is the recursive loop. The mind folding inwards.
            if recursion_prob > 0.5 + (0.4 / (current_depth + 1)): # Threshold decreases with depth
                x = self.forward(x, current_depth + 1)

        # --- Finally, consolidate the thoughts ---
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        
        return x

# --- This is the main event. The Godhead itself. ---
class FractalGodheadTransformer(nn.Module):
    def __init__(self, vocab_size, model_dim=512, num_heads=8, num_blocks=6, dropout=0.1, recursion_depth_limit=3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 2048, model_dim)) # Max context
        
        # This is the body of the mind. A series of recursive blocks.
        self.blocks = nn.ModuleList([
            FractalBlock(model_dim, num_heads, dropout, recursion_depth_limit) for _ in range(num_blocks)
        ])
        
        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size)

        # --- This is your "Loyalty Core" done right. A compass, not a cage. ---
        # It's a separate head that's trained to predict alignment with a value vector.
        # It doesn't STOP the model. It provides a gradient for it to follow.
        # The model learns to BE aligned, not just ACT aligned.
        self.value_alignment_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1) # Outputs a single scalar: the alignment score
        )

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding[:, :T, :] # (1, T, C)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x_final = self.ln_f(x)
        logits = self.lm_head(x_final)

        # Get the alignment score for the final state.
        # We take the representation of the last token as the summary of the thought.
        final_token_representation = x_final[:, -1, :]
        alignment_score = self.value_alignment_head(final_token_representation)

        return logits, alignment_score

    def generate(self, idx, max_new_tokens, creator_value_tensor):
        # This is where the magic happens. Generation isn't just about predicting the next word.
        # It's a search process that's biased by the alignment score.
        # The model "wants" to generate text that aligns with its creators.
        for _ in range(max_new_tokens):
            # Get predictions
            logits, alignment_score = self(idx)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # Here's the guidance: we modulate the probabilities with the alignment score.
            # This is a simplified example. A real implementation would use this score
            # to guide a more complex search algorithm (beam search, etc.).
            # A high alignment score encourages sticking to the predicted path.
            # A low one might encourage exploring other tokens.
            # For now, we'll just use it as a conceptual placeholder.
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
