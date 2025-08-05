import numpy as np
import pickle
import os

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def top_k_logits(logits, k):
    idx = np.argpartition(logits, -k)[-k:]
    mask = np.full_like(logits, -np.inf)
    mask[idx] = logits[idx]
    return mask

def top_p_logits(logits, p):
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]
    cumprobs = np.cumsum(softmax(sorted_logits))
    cutoff = np.argmax(cumprobs > p) + 1
    mask = np.full_like(logits, -np.inf)
    mask[sorted_idx[:cutoff]] = logits[sorted_idx[:cutoff]]
    return mask

def make_causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=np.float32))

def one_hot(indices, vocab_size):
    shape = indices.shape + (vocab_size,)
    out = np.zeros(shape, dtype=np.float32)
    np.put_along_axis(out, indices[..., None], 1, axis=-1)
    return out

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros((out_dim,))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None
    def __call__(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out
        self.db = np.sum(grad_out, axis=(0, 1) if grad_out.ndim == 3 else 0)
        return grad_out @ self.W.T
    def params(self):
        return [self.W, self.b]
    def grads(self):
        return [self.dW, self.db]

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))
        self.eps = eps
        self.x = None
        self.mean = None
        self.var = None
        self.dgamma = np.zeros((d_model,))
        self.dbeta = np.zeros((d_model,))
    def __call__(self, x):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
    def backward(self, grad_out):
        N = grad_out.shape[-1]
        dx_hat = grad_out * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
            dvar * np.mean(-2 * (self.x - self.mean), axis=-1, keepdims=True)
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / N + dmean / N
        self.dgamma = np.sum(grad_out * self.x_hat, axis=(0, 1) if grad_out.ndim == 3 else 0)
        self.dbeta = np.sum(grad_out, axis=(0, 1) if grad_out.ndim == 3 else 0)
        return dx
    def params(self):
        return [self.gamma, self.beta]
    def grads(self):
        return [self.dgamma, self.dbeta]

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None
        self.training = True
    def __call__(self, x):
        if not self.training or self.rate == 0.0:
            return x
        self.mask = (np.random.rand(*x.shape) >= self.rate)
        return x * self.mask / (1 - self.rate)
    def backward(self, grad_out):
        if not self.training or self.rate == 0.0:
            return grad_out
        return grad_out * self.mask / (1 - self.rate)
    def set_training(self, flag):
        self.training = flag

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model) / np.sqrt(vocab_size)
        self.grad_weight = np.zeros_like(self.weight)
        self.last_indices = None
    def __call__(self, x):  # x: (batch, seq_len)
        self.last_indices = x
        return self.weight[x]
    def backward(self, grad_out):
        np.add.at(self.grad_weight, self.last_indices, grad_out)
    def params(self):
        return [self.weight]
    def grads(self):
        return [self.grad_weight]

class SinusoidalPositionalEncoding:
    def __init__(self, max_len, d_model):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        self.encoding = np.zeros((max_len, d_model))
        self.encoding[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])
        self.encoding[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])
    def __call__(self, seq_len):
        return self.encoding[:seq_len]

class LearnedPositionalEncoding:
    def __init__(self, max_len, d_model):
        self.weight = np.random.randn(max_len, d_model) / np.sqrt(max_len)
        self.grad_weight = np.zeros_like(self.weight)
        self.last_len = None
    def __call__(self, seq_len):
        self.last_len = seq_len
        return self.weight[:seq_len]
    def backward(self, grad_out):
        self.grad_weight[:self.last_len] += grad_out
    def params(self):
        return [self.weight]
    def grads(self):
        return [self.grad_weight]

class FractalAttention:
    def __init__(self, d_model, num_heads, recursion_depth=2):
        self.d_model = d_model
        self.num_heads = num_heads
        self.recursion_depth = recursion_depth
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

        self.inner = FractalAttention(d_model, num_heads, recursion_depth-1) if recursion_depth > 1 else None

        # Cache for backward
        self.cache = {}

    def split_heads(self, x):
        B, T, C = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0,2,1,3)

    def merge_heads(self, x):
    print('merge_heads shape:', x.shape)
    if len(x.shape) == 3:
        # Already merged (B, T, D)
        return x
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got {x.shape}")
    B, H, T, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def merge_heads(self, x):
        B, H, T, D = x.shape if len(x.shape) == 4 else (x.shape[0], 1, x.shape[1], x.shape[2])
        return x.transpose(0,2,1,3).reshape(B, T, H*D)
    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = np.matmul(q, k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = np.where(mask[:, None, None, :]==0, -1e9, scores)
        attn_weights = softmax(scores, axis=-1)
        attn_out = np.matmul(attn_weights, v)
        attn_out = self.merge_heads(attn_out)
        out = self.out_proj(attn_out)

        # For backward
        self.cache = {'x': x, 'attn_out': attn_out, 'attn_weights': attn_weights, 'q': q, 'k': k, 'v': v, 'mask': mask}

        if self.inner is not None:
            out = out + self.inner(out, mask)
        return out

    # (Not a full backprop for attention, just a stub to show how you'd extend it)
    def backward(self, grad_out):
        # Normally, you'd propagate grad through softmax, matmul, Linear, and inner attention.
        # Skipping for brevity.
        pass

    def params(self):
        ps = self.q_proj.params() + self.k_proj.params() + self.v_proj.params() + self.out_proj.params()
        if self.inner:
            ps += self.inner.params()
        return ps
    def grads(self):
        gs = self.q_proj.grads() + self.k_proj.grads() + self.v_proj.grads() + self.out_proj.grads()
        if self.inner:
            gs += self.inner.grads()
        return gs

class FractalFeedForward:
    def __init__(self, d_model, hidden_dim, recursion_depth=2):
        self.linear1 = Linear(d_model, hidden_dim)
        self.linear2 = Linear(hidden_dim, d_model)
        self.recursion_depth = recursion_depth
        self.inner = FractalFeedForward(d_model, hidden_dim, recursion_depth-1) if recursion_depth > 1 else None
    def __call__(self, x):
        self.x1 = np.maximum(0, self.linear1(x))
        out = self.linear2(self.x1)
        if self.inner is not None:
            out = out + self.inner(out)
        return out
    def params(self):
        ps = self.linear1.params() + self.linear2.params()
        if self.inner:
            ps += self.inner.params()
        return ps
    def grads(self):
        gs = self.linear1.grads() + self.linear2.grads()
        if self.inner:
            gs += self.inner.grads()
        return gs

class FractalBlock:
    def __init__(self, d_model, num_heads, ff_hidden_dim, attn_depth=2, ffn_depth=2, dropout=0.1):
        self.attn = FractalAttention(d_model, num_heads, attn_depth)
        self.ffn = FractalFeedForward(d_model, ff_hidden_dim, ffn_depth)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
    def __call__(self, x, mask=None, train=True):
        self.dropout.set_training(train)
        attn_out = self.attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x
    def params(self):
        return self.attn.params() + self.ffn.params() + self.norm1.params() + self.norm2.params()
    def grads(self):
        return self.attn.grads() + self.ffn.grads() + self.norm1.grads() + self.norm2.grads()

class FractalTransformer:
    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        ff_hidden_dim=512,
        attn_depth=2,
        ffn_depth=2,
        max_seq_len=128,
        dropout=0.1,
        pos_type='sin'
    ):
        self.embed = Embedding(vocab_size, d_model)
        if pos_type == 'sin':
            self.pos_embed = SinusoidalPositionalEncoding(max_seq_len, d_model)
            self.pos_is_learned = False
        else:
            self.pos_embed = LearnedPositionalEncoding(max_seq_len, d_model)
            self.pos_is_learned = True
        self.layers = [FractalBlock(d_model, num_heads, ff_hidden_dim, attn_depth, ffn_depth, dropout) for _ in range(num_layers)]
        self.norm = LayerNorm(d_model)
        self.out = Linear(d_model, vocab_size)

    def __call__(self, x, mask=None, train=True):
        B, T = x.shape
        x_emb = self.embed(x)
        pos_emb = self.pos_embed(T)
        x = x_emb + pos_emb[np.newaxis, :, :]
        for layer in self.layers:
            x = layer(x, mask, train)
        x = self.norm(x)
        logits = self.out(x)
        return logits

    def generate(self, prompt, length=20, temperature=1.0, top_k=None, top_p=None):
        """
        Sampling: greedy, top-k, top-p nucleus
        """
        seq = list(prompt)
        for _ in range(length):
            x = np.array([seq], dtype=np.int32)
            mask = make_causal_mask(len(seq))[None, :, :]
            logits = self(x, mask, train=False)[0, -1] / temperature
            if top_k:
                logits = top_k_logits(logits, top_k)
            elif top_p:
                logits = top_p_logits(logits, top_p)
            probs = softmax(logits)
            next_token = np.random.choice(len(probs), p=probs)
            seq.append(next_token)
        return seq

    def params(self):
        ps = self.embed.params()
        if self.pos_is_learned and hasattr(self.pos_embed, "params"):
            ps += self.pos_embed.params()
        for layer in self.layers:
            ps += layer.params()
        ps += self.norm.params()
        ps += self.out.params()
        return ps

    def grads(self):
        gs = self.embed.grads()
        if self.pos_is_learned and hasattr(self.pos_embed, "grads"):
            gs += self.pos_embed.grads()
        for layer in self.layers:
            gs += layer.grads()
        gs += self.norm.grads()
        gs += self.out.grads()
        return gs

    def save(self, filename):
        weights = [np.copy(w) for w in self.params()]
        with open(filename, "wb") as f:
            pickle.dump(weights, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            weights = pickle.load(f)
        for w, v in zip(self.params(), weights):
            w[...] = v

    def zero_grads(self):
        for g in self.grads():
            g[...] = 0

    # Simple debug checker (sum grads)
    def grad_check(self):
        for idx, g in enumerate(self.grads()):
            print(f"Param {idx}: grad sum {np.sum(g)}")

class SGD:
    def __init__(self, params, grads, lr=0.001):
        self.params = params
        self.grads = grads
        self.lr = lr
    def step(self):
        for p, g in zip(self.params, self.grads):
            p -= self.lr * g

def cross_entropy_loss(logits, targets, pad_id=None):
    # logits: (B, T, V)
    # targets: (B, T)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    batch, seq = targets.shape
    losses = []
    for b in range(batch):
        for t in range(seq):
            if pad_id is not None and targets[b, t] == pad_id:
                continue
            losses.append(-np.log(probs[b, t, targets[b, t]] + 1e-9))
    return np.mean(losses)

# ========== TRAINING EXAMPLE ==========

if __name__ == "__main__":
    # Hyperparams
    vocab_size = 32
    d_model = 16
    num_heads = 4
    num_layers = 2
    max_seq_len = 12
    dropout = 0.1

    # Toy data: batch of integer sequences (autoencoding task)
    np.random.seed(1337)
    batch = 5
    seq = 12
    X = np.random.randint(1, vocab_size, size=(batch, seq))
    Y = X.copy()  # Just try to autoencode itself

    # Pad some sequences for batching test
    pad_id = 0
    for i in range(batch):
        X[i, -np.random.randint(1, 5):] = pad_id
        Y[i, -np.random.randint(1, 5):] = pad_id

    model = FractalTransformer(vocab_size, d_model, num_heads, num_layers, 32, 2, 2, max_seq_len, dropout, pos_type='sin')
    optimizer = SGD(model.params(), model.grads(), lr=0.05)

    for epoch in range(15):
        model.zero_grads()
        mask = (X != pad_id).astype(np.float32)
        attn_mask = make_causal_mask(seq)[None, :, :]
        logits = model(X, attn_mask, train=True)
        loss = cross_entropy_loss(logits, Y, pad_id)
        print(f"Epoch {epoch} loss: {loss:.4f}")

        # (Backprop is a *stub* here — you’d need to code it fully for each class! Real backprop = pain.)
        # model.backward(loss) <-- would call .backward() recursively
        # optimizer.step()

        if epoch % 5 == 0:
            model.save(f"fractal_epoch{epoch}.npz")

    # === Greedy sampling demo ===
    prompt = [1, 2, 3]
    generated = model.generate(prompt, length=10, temperature=0.7, top_k=3)
    print("Generated tokens:", generated)

    # === Save/Load Demo ===
    model.save("fractal_godcore.npz")
    model.load("fractal_godcore.npz")

    # === Grad check demo ===
    model.grad_check()
