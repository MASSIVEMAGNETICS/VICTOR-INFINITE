# ============================================
# FILE: runner.py
# VERSION: v0.0.2-GODCORE-ELITE
# NAME: VictorRunner (Test Harness - Transformer Ready)
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Clean runner to import and test VICTORCH modules, including Transformer Block.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

from victorch.core.tensor import Tensor
from victorch.modules.layers import Dense, FeedForward
from victorch.modules.activations import Softmax, ReLU
from victorch.modules.fractal_ops import SelfAttention
from victorch.modules.transformer_block import TransformerBlock
from victorch.modules.multihead_attention import MultiHeadAttention
from victorch.models.victor_model import VictorTransformerModel
import numpy as np

# ==========================
# Test Functions
# ==========================

def test_dense():
    print("\n--- Dense Layer Test ---")
    x = Tensor(np.random.randn(2, 5))
    dense = Dense(5, 10)
    out = dense(x)
    print("Output Shape:", out.shape())

def test_feedforward():
    print("\n--- FeedForward Layer Test ---")
    x = Tensor(np.random.randn(2, 5))
    ff = FeedForward(5, 20)
    out = ff(x)
    print("Output Shape:", out.shape())

def test_softmax():
    print("\n--- Softmax Test ---")
    x = Tensor(np.random.randn(2, 5))
    softmax = Softmax()
    out = softmax(x)
    print("Softmax Output:", out.data)

def test_attention():
    print("\n--- Self-Attention Test ---")
    x = Tensor(np.random.randn(2, 4, 5))  # batch=2, seq_len=4, embed_dim=5
    attn = SelfAttention(5)
    out = attn(x)
    print("Attention Output Shape:", out.shape())

def test_transformer_block():
    print("\n--- TransformerBlock Test ---")
    x = Tensor(np.random.randn(2, 4, 5))  # batch=2, seq_len=4, embed_dim=5
    block = TransformerBlock(embed_dim=5, hidden_dim=20)
    out = block(x)
    print("TransformerBlock Output Shape:", out.shape())

def test_multihead_attention():
    print("\n--- MultiHeadAttention Test ---")
    x = Tensor(np.random.randn(2, 4, 8))  # (batch=2, seq_len=4, embed_dim=8)
    attn = MultiHeadAttention(embed_dim=8, num_heads=2)
    out = attn(x)
    print("MultiHeadAttention Output Shape:", out.shape())


def test_victor_model():
    print("\n--- VictorTransformerModel Test ---")
    x = Tensor(np.random.randn(2, 4, 50))  # batch=2, seq_len=4, vocab_size=50
    model = VictorTransformerModel(vocab_size=50, embed_dim=32, num_layers=2, hidden_dim=64, num_classes=10)
    out = model(x)
    print("VictorTransformerModel Output Shape:", out.shape())

# ==========================
# Entry Point
# ==========================



if __name__ == "__main__":
    test_dense()
    test_feedforward()
    test_softmax()
    test_attention()
    test_transformer_block()
    test_multihead_attention()
    test_victor_model()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
