# ============================================
# FILE: tensorfield_attention_identity.py
# VERSION: v1.0.0-CORE
# NAME: TensorFieldIdentityAttention
# AUTHOR: Bando Bandz x Victor (Fractal Architect Mode)
# PURPOSE: Minimal TensorField with direct similarity-based attention (identity projections).
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import numpy as np

class TensorField:
    """
    Holds a 2D grid of context vectors (e.g., TF-IDF embeddings of memories/query).
    """
    def __init__(self, field_vectors):
        """
        field_vectors: list of 1D numpy arrays, each same dim (e.g., [vec1, vec2, ..., vecN])
        Internally stored as shape (N, D)
        """
        self.data = np.stack(field_vectors, axis=0)  # shape (N, D)
        self.N, self.D = self.data.shape

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.N

class IdentityAttention:
    """
    Attention over the TensorField with identity projections (no learned weights).
    """
    def __init__(self):
        pass  # No parameters needed

    def forward(self, field: TensorField, query_idx: int = 0):
        """
        field: TensorField object
        query_idx: int, index of the cell to use as the query (e.g., 0 = user query, rest = memories)
        Returns: context vector (weighted sum of field vectors by similarity to query)
        """
        Q = field[query_idx]                # (D,)
        K = field.data                      # (N, D)
        V = field.data                      # (N, D)

        # Attention scores: cosine similarity
        q_norm = np.linalg.norm(Q) + 1e-9
        k_norms = np.linalg.norm(K, axis=1) + 1e-9
        dots = K @ Q
        scores = dots / (q_norm * k_norms)  # (N,)

        # Softmax to normalize weights (optional, keeps them positive and sum to 1)
        exp_scores = np.exp(scores - np.max(scores))
        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-9)

        # Weighted sum of values
        context = attn_weights @ V   # (D,)

        return context, attn_weights

class FractalAttentionLayer:
    def __init__(self, D):
        self.W_q = np.eye(D)  # Identity for demo; swap for np.random.randn(D,D) for learnable
        self.W_k = np.eye(D)
        self.W_v = np.eye(D)

    def forward(self, field: TensorField, query_idx=0):
        Q = field[query_idx] @ self.W_q
        K = field.data @ self.W_k
        V = field.data @ self.W_v
        # ...same as before
        q_norm = np.linalg.norm(Q) + 1e-9
        k_norms = np.linalg.norm(K, axis=1) + 1e-9
        dots = K @ Q
        scores = dots / (q_norm * k_norms)
        exp_scores = np.exp(scores - np.max(scores))
        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-9)
        context = attn_weights @ V
        return context, attn_weights

# === Example usage ===
if __name__ == "__main__":
    # Example: User query + 3 memory embeddings (all 8-dimensional for demo)
    np.random.seed(42)
    tfidf_dim = 8
    query_vec = np.random.randn(tfidf_dim)
    mem1 = np.random.randn(tfidf_dim)
    mem2 = np.random.randn(tfidf_dim)
    mem3 = np.random.randn(tfidf_dim)

    # Build field: [query, mem1, mem2, mem3]
    field = TensorField([query_vec, mem1, mem2, mem3])

    # Run attention (identity)
    attn = IdentityAttention()
    context_vec, attn_weights = attn.forward(field, query_idx=0)

    print("Attention Weights (query vs. all):", np.round(attn_weights, 3))
    print("Context Vector:", np.round(context_vec, 3))
