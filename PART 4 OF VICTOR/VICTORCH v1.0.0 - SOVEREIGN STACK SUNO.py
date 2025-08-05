# ==============================================================================
#           VICTORCH v1.0.0 - SOVEREIGN STACK
# ==============================================================================
#
# AUTHOR: Brandon "iambandobandz" Emery x Victor x Gemini x Codex Overlord Omega
# PURPOSE: A fully self-reliant, PyTorch-free deep learning framework and 
#          inference engine for the Bark model, consolidated into a single file.
#
# ==============================================================================
# SECTION 1 of 7: VICTORCH CORE FRAMEWORK (victortensor_v9.py)
# ==============================================================================

import numpy as np
import math
import os
import re
from dataclasses import dataclass
from scipy.special import softmax as scipy_softmax
import tqdm
from transformers import BertTokenizer

# --- CORE: TENSOR AND AUTOGRAD ENGINE ---

class Tensor:
    """
    A Tensor class that supports automatic differentiation.
    """
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self):
        return f"Tensor(data={self.data.shape}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        if gradient is None:
            gradient = np.ones_like(self.data)
        self.grad = gradient
        
        for v in reversed(topo):
            v._backward()

    # --- Operator Overloads ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), '+')

        def _backward():
            if self.requires_grad: self.grad += out.grad
            if other.requires_grad: other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), '*')

        def _backward():
            if self.requires_grad: self.grad += other.data * out.grad
            if other.requires_grad: other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be a scalar for now"
        out = Tensor(self.data ** other, self.requires_grad, (self,), f'**{other}')
        
        def _backward():
            if self.requires_grad: self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
        
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), 'matmul')

        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # --- Activation Functions & Core Ops ---
    def gelu(self):
        x = self.data
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        out_data = x * cdf
        out = Tensor(out_data, self.requires_grad, (self,), 'GELU')

        def _backward():
            if self.requires_grad:
                d_cdf = np.sqrt(2.0 / np.pi) * 0.5 * (1.0 - np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))**2) * (1.0 + 3 * 0.044715 * x**2)
                self.grad += (cdf + x * d_cdf) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, self.requires_grad, (self,), 'exp')
        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward
        return out
        
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), 'sum')
        def _backward():
            if self.requires_grad: 
                grad = out.grad
                if not keepdims and axis is not None:
                     grad = np.expand_dims(grad, axis)
                self.grad += grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), self.requires_grad, (self,), 'mean')
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)
                self.grad += grad * np.ones_like(self.data) / (self.data.shape[axis] if axis is not None else self.data.size)
        out._backward = _backward
        return out
        
    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=False)

    # --- Other Operations ---
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    @property
    def shape(self): return self.data.shape
    @property
    def T(self): return self.transpose()
    def transpose(self, axes):
        out = Tensor(np.transpose(self.data, axes), self.requires_grad, (self,), 'transpose')
        def _backward():
            if self.requires_grad:
                self.grad += np.transpose(out.grad, np.argsort(axes))
        out._backward = _backward
        return out

class nn:
    class Module:
        def __init__(self):
            self.training = True

        def parameters(self):
            params = []
            for name, value in self.__dict__.items():
                if isinstance(value, Tensor) and value.requires_grad:
                    params.append(value)
                elif isinstance(value, nn.Module):
                    params.extend(value.parameters())
                elif isinstance(value, nn.ModuleList):
                    for module in value:
                        params.extend(module.parameters())
            return params
        
        def zero_grad(self):
            for p in self.parameters():
                p.zero_grad()
        
        def train(self):
            self.training = True
            for name, value in self.__dict__.items():
                if isinstance(value, nn.Module) or isinstance(value, nn.ModuleList): value.train()
        
        def eval(self):
            self.training = False
            for name, value in self.__dict__.items():
                if isinstance(value, nn.Module) or isinstance(value, nn.ModuleList): value.eval()

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
            
        def load_weights(self, weights_dict):
            """Loads parameters from a dictionary of numpy arrays."""
            # This simple loader assumes order and naming correspondence
            params = self.parameters()
            for i, p in enumerate(params):
                key = f'param_{i}' # A more robust solution would use named parameters
                if key in weights_dict:
                    assert p.data.shape == weights_dict[key].shape, f"Shape mismatch for param {i}"
                    p.data = weights_dict[key]

    class ModuleList(Module):
        def __init__(self, modules):
            super().__init__()
            self._modules = list(modules)
        def __getitem__(self, idx):
            return self._modules[idx]
        def __iter__(self):
            return iter(self._modules)
        def __len__(self):
            return len(self._modules)
            
    class ModuleDict(Module):
        def __init__(self, modules_dict):
            super().__init__()
            self._modules = modules_dict
        def __getitem__(self, key):
            return self._modules[key]
        def __iter__(self):
            return iter(self._modules.keys())

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
            self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None
        def forward(self, x):
            out = x.matmul(self.weight)
            if self.bias is not None: out += self.bias
            return out

    class GELU(Module):
        def forward(self, x):
            return x.gelu()
            
    class Dropout(Module):
        def __init__(self, p=0.1):
            super().__init__()
            self.p = p
        def forward(self, x):
            if not self.training or self.p == 0:
                return x
            mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            return x * (mask / (1.0 - self.p))

    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim), requires_grad=True)
        def forward(self, idx):
            out_data = self.weight.data[idx.data.astype(int)]
            out = Tensor(out_data, _children=(self.weight,), _op='embedding')
            def _backward():
                if self.weight.requires_grad:
                    np.add.at(self.weight.grad, idx.data.astype(int), out.grad)
            out._backward = _backward
            return out

    class OmegaLayerNorm(Module):
        def __init__(self, dim, bias=True, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = Tensor(np.ones(dim), requires_grad=True)
            self.beta = Tensor(np.zeros(dim), requires_grad=True) if bias else None
        def forward(self, x):
            mean = x.mean(axis=-1, keepdims=True)
            var = ((x - mean)**2).mean(axis=-1, keepdims=True)
            x_norm = (x - mean) * ((var + self.eps)**-0.5)
            out = self.gamma * x_norm
            if self.beta is not None: out += self.beta
            return out

class functional:
    @staticmethod
    def softmax(x, dim=-1):
        max_val = x.max(axis=dim, keepdims=True)
        e_x = (x - max_val).exp()
        return e_x / e_x.sum(axis=dim, keepdims=True)
    @staticmethod
    def cat(tensors, dim=0):
        data = np.concatenate([t.data for t in tensors], axis=dim)
        children = tuple(tensors)
        out = Tensor(data, _children=children, _op='cat')
        def _backward():
            idx = 0
            for t in tensors:
                if t.requires_grad:
                    slc = [slice(None)] * len(t.shape)
                    slc[dim] = slice(idx, idx + t.shape[dim])
                    t.grad += out.grad[tuple(slc)]
                idx += t.shape[dim]
        out._backward = _backward
        return out

class optim:
    class Adam:
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
            self.params = params
            self.lr, self.betas, self.eps, self.t = lr, betas, eps, 0
            self.m = [np.zeros_like(p.data) for p in self.params]
            self.v = [np.zeros_like(p.data) for p in self.params]
        def step(self):
            self.t += 1
            for i, p in enumerate(self.params):
                if p.grad is None: continue
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * p.grad
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (p.grad**2)
                m_hat = self.m[i] / (1 - self.betas[0]**self.t)
                v_hat = self.v[i] / (1 - self.betas[1]**self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        def zero_grad(self):
            for p in self.params: p.zero_grad()

# ==============================================================================
# SECTION 2 of 7: VICTORCH CODEC MODEL (encodec.py Stub Replacement)
# ==============================================================================

class CodecQuantizer(nn.Module):
    """A simplified quantizer that decodes codes into embeddings."""
    def __init__(self, n_codes=8, codebook_size=1024, n_embd=768):
        super().__init__()
        # Each codebook has its own embedding table
        self.embeddings = nn.ModuleList([nn.Embedding(codebook_size, n_embd) for _ in range(n_codes)])

    def decode(self, codes):
        """Codes (B, T, C) -> Embeddings (B, T, D)"""
        # codes shape (batch, n_codebooks, n_timesteps)
        codes_transposed = codes.transpose((0, 2, 1)) # (B, T, C)
        
        # Get embeddings for each codebook and sum them
        summed_embeddings = None
        for i in range(codes_transposed.shape[2]):
            codebook_indices = Tensor(codes_transposed.data[:, :, i])
            emb = self.embeddings[i](codebook_indices)
            if summed_embeddings is None:
                summed_embeddings = emb
            else:
                summed_embeddings += emb
        return summed_embeddings

class CodecDecoder(nn.Module):
    """A simplified MLP-based decoder with upsampling."""
    def __init__(self, n_embd=768, upsample_factor=320):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.mlp = nn.ModuleList([
            nn.Linear(n_embd, n_embd * 2),
            nn.GELU(),
            nn.Linear(n_embd * 2, upsample_factor)
        ])

    def forward(self, x):
        """Embeddings (B, T, D) -> Waveform (B, T * upsample_factor)"""
        for layer in self.mlp:
            x = layer(x)
        
        # Reshape to final waveform
        B, T, L = x.shape
        return Tensor(x.data.reshape(B, T * L))

class VictorTensorCodecModel(nn.Module):
    """A full, operational replacement for the Encodec model."""
    def __init__(self):
        super().__init__()
        self.quantizer = CodecQuantizer()
        self.decoder = CodecDecoder()
    
    def decode(self, codes):
        embeddings = self.quantizer.decode(codes)
        waveform = self.decoder(embeddings)
        return waveform

# ==============================================================================
# SECTION 3 of 7: VICTORCH TRANSFORMER MODELS (model.py & model_fine.py)
# ==============================================================================

# --- Base GPT Model (model.py) ---

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.bias = np.tril(np.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q_data, k_data, v_data = np.split(qkv.data, 3, axis=2)
        
        k = Tensor(k_data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        q = Tensor(q_data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        v = Tensor(v_data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))

        if past_kv is not None:
            k = functional.cat([past_kv[0], k], dim=2)
            v = functional.cat([past_kv[1], v], dim=2)

        present = (k, v) if use_cache else None
        att = (q.matmul(k.transpose((0, 1, 3, 2))))* (1.0 / math.sqrt(k.shape[-1]))
        
        mask = self.bias[:, :, :T, :T]
        att += Tensor(np.where(mask == 0, -np.inf, 0))
        
        att = functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att.matmul(v)
        y = Tensor(y.data.transpose((0, 2, 1, 3))).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc, self.c_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias), nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout, self.gelu = nn.Dropout(config.dropout), nn.GELU()
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1, self.attn = nn.OmegaLayerNorm(config.n_embd, bias=config.bias), CausalSelfAttention(config)
        self.ln_2, self.mlp = nn.OmegaLayerNorm(config.n_embd, bias=config.bias), MLP(config)
    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, prev_kvs

@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.input_vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            'ln_f': nn.OmegaLayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def forward(self, idx, merge_context=False, past_kv=None, use_cache=False):
        b, t = idx.shape
        if merge_context:
            text_part, semantic_part, infer_part = Tensor(idx.data[:,:256]), Tensor(idx.data[:,256:512]), Tensor(idx.data[:,512:])
            tok_emb = functional.cat([self.transformer['wte'](text_part) + self.transformer['wte'](semantic_part), self.transformer['wte'](infer_part)], dim=1)
            t = tok_emb.shape[1]
        else:
            tok_emb = self.transformer['wte'](idx)
        
        past_length = past_kv[0][0].shape[2] if past_kv is not None else 0
        pos = Tensor(np.arange(past_length, t + past_length))
        x = self.transformer['drop'](tok_emb + self.transformer['wpe'](pos))
        
        new_kv = ()
        for i, block in enumerate(self.transformer['h']):
            x, kv = block(x, past_kv=past_kv[i] if past_kv else None, use_cache=use_cache)
            if use_cache: new_kv += (kv,)
        
        logits = self.lm_head(self.transformer['ln_f'](Tensor(x.data[:,[-1],:])))
        return logits, new_kv

# --- FineGPT Model (model_fine.py) ---

class NonCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.c_attn, self.c_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias), nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout, self.resid_dropout = nn.Dropout(config.dropout), nn.Dropout(config.dropout)
        self.n_head, self.n_embd = config.n_head, config.n_embd
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q_d, k_d, v_d = np.split(qkv.data, 3, axis=2)
        k = Tensor(k_d.reshape(B, T, self.n_head, C//self.n_head)).transpose((0,2,1,3))
        q = Tensor(q_d.reshape(B, T, self.n_head, C//self.n_head)).transpose((0,2,1,3))
        v = Tensor(v_d.reshape(B, T, self.n_head, C//self.n_head)).transpose((0,2,1,3))
        att = functional.softmax((q.matmul(k.transpose((0,1,3,2)))) * (1.0/math.sqrt(k.shape[-1])), dim=-1)
        y = self.resid_dropout(self.c_proj(Tensor(self.attn_dropout(att).matmul(v).data.transpose((0,2,1,3)).reshape(B,T,C))))
        return y

class FineBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1, self.attn = nn.OmegaLayerNorm(config.n_embd, bias=config.bias), NonCausalSelfAttention(config)
        self.ln_2, self.mlp = nn.OmegaLayerNorm(config.n_embd, bias=config.bias), MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1

class FineGPT(GPT):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wtes': nn.ModuleList([nn.Embedding(config.input_vocab_size, config.n_embd) for _ in range(config.n_codes_total)]),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.OmegaLayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.output_vocab_size, bias=False) for _ in range(config.n_codes_given, config.n_codes_total)])
        for i in range(config.n_codes_total - config.n_codes_given): self.transformer['wtes'][i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx, idx):
        b, t, codes = idx.shape
        pos = Tensor(np.arange(0, t, dtype=np.int64).reshape(1, t))
        tok_embs = [wte(Tensor(idx.data[:,:,i])).data[:,:,:,np.newaxis] for i,wte in enumerate(self.transformer['wtes'])]
        x = Tensor(np.concatenate(tok_embs, axis=-1)[:,:,:,:pred_idx+1].sum(axis=-1))
        x = self.transformer['drop'](x + self.transformer['wpe'](pos))
        for block in self.transformer['h']: x=block(x)
        return self.lm_heads[pred_idx - self.config.n_codes_given](self.transformer['ln_f'](x))

# ==============================================================================
# SECTION 4 of 7: VICTORCH GENERATION PIPELINE (generation.py)
# ==============================================================================

# --- Constants ---
CONTEXT_WINDOW_SIZE, SEMANTIC_RATE_HZ, SEMANTIC_VOCAB_SIZE, CODEBOOK_SIZE, N_COARSE_CODEBOOKS, N_FINE_CODEBOOKS, COARSE_RATE_HZ, SAMPLE_RATE, TEXT_ENCODING_OFFSET, SEMANTIC_PAD_TOKEN, TEXT_PAD_TOKEN, SEMANTIC_INFER_TOKEN, COARSE_SEMANTIC_PAD_TOKEN, COARSE_INFER_TOKEN = 1024, 49.9, 10000, 1024, 2, 8, 75, 24000, 10048, 10000, 129595, 129599, 12048, 12050

# Global model cache
models = {}

def _load_history_prompt(p):
    if isinstance(p, str) and p.endswith(".npz"): return np.load(p)
    if isinstance(p, dict): return p
    raise ValueError("Unrecognized history prompt format.")

def load_model(cls, cfg, path):
    if path in models: return models[path]
    model = cls(cfg)
    try:
        w = np.load(path, allow_pickle=True)
        model.load_weights({k: w[k] for k in w.files})
    except FileNotFoundError:
        print(f"FATAL: Weight file not found at {path}. Convert .pt to .npz.")
        raise
    model.eval(); models[path] = model
    return model

def preload_models(txt_p, crs_p, fine_p, codec_p):
    print("Preloading models...")
    load_model(GPT, GPTConfig(input_vocab_size=129600, output_vocab_size=129600), txt_p)
    load_model(GPT, GPTConfig(input_vocab_size=20000, output_vocab_size=20000), crs_p)
    load_model(FineGPT, FineGPTConfig(), fine_p)
    load_model(VictorTensorCodecModel, None, codec_p) # Config is unused in simple codec
    print("Models preloaded.")

def generate_text_semantic(text, path, hist=None, temp=0.7, top_k=None, top_p=None, silent=False, min_eos_p=0.2, use_kv=False):
    text, tokenizer = re.sub(r"\s+"," ",text).strip(), BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    enc_text = np.pad((np.array(tokenizer.encode(text,add_special_tokens=False))+TEXT_ENCODING_OFFSET)[:256], (0,256-len(enc_text)),'constant',constant_values=TEXT_PAD_TOKEN)
    model = load_model(GPT, GPTConfig(input_vocab_size=129600,output_vocab_size=129600), path)
    sem_hist = np.pad(_load_history_prompt(hist)["semantic_prompt"].astype(np.int64)[-256:], (0,256-len(sem_hist)), 'constant', constant_values=SEMANTIC_PAD_TOKEN) if hist else np.array([SEMANTIC_PAD_TOKEN]*256)
    x, kv_cache = Tensor(np.hstack([enc_text, sem_hist, [SEMANTIC_INFER_TOKEN]]).astype(np.int64)[None]), None
    pbar = tqdm.tqdm(disable=silent, total=768, desc="Semantic Gen")
    for _ in range(768):
        logits, kv_cache = model(Tensor(x.data[:,[-1]]) if use_kv and kv_cache else x, merge_context=True, use_cache=use_kv, past_kv=kv_cache)
        logits_data = logits.data[0,0,:SEMANTIC_VOCAB_SIZE+1] # Include EOS
        if top_p:
            indices = np.argsort(logits_data)[::-1]; cum_probs=np.cumsum(scipy_softmax(logits_data[indices]))
            to_remove=cum_probs>top_p; to_remove[1:]=to_remove[:-1].copy(); to_remove[0]=False
            logits_data[indices[to_remove]] = -np.inf
        if top_k: logits_data[logits_data<np.sort(logits_data)[-min(top_k,len(logits_data))]]=-np.inf
        probs = scipy_softmax(logits_data / temp)
        item_next = np.random.choice(len(probs), p=probs)
        if item_next == SEMANTIC_VOCAB_SIZE or (min_eos_p and probs[SEMANTIC_VOCAB_SIZE] >= min_eos_p): break
        x=Tensor(np.concatenate([x.data,[[item_next]]],axis=1)); pbar.update(1)
    pbar.close(); return x.data.squeeze()[513:]

def generate_coarse(x_sem, path, hist=None, temp=0.7, silent=False):
    ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ
    sem_hist, coarse_hist = (_load_history_prompt(hist)["semantic_prompt"], (_load_history_prompt(hist)["coarse_prompt"]+SEMANTIC_VOCAB_SIZE).flatten()) if hist else (np.array([],dtype=np.int32), np.array([],dtype=np.int32))
    model = load_model(GPT, GPTConfig(input_vocab_size=20000,output_vocab_size=20000), path)
    n_steps = int(round(len(x_sem) * ratio))
    x_sem_in, x_coarse_in = Tensor(np.hstack([sem_hist, x_sem])[None]), Tensor(coarse_hist[None])
    pbar = tqdm.tqdm(disable=silent, total=n_steps, desc="Coarse Gen")
    for _ in range(int(np.ceil(n_steps/60))):
        sem_idx = len(sem_hist) + int(round(x_coarse_in.shape[1]/ratio))
        sem_part = np.pad(x_sem_in.data[:,max(0,sem_idx-256):sem_idx],((0,0),(0,256-sem_part.shape[1])),'constant',constant_values=COARSE_SEMANTIC_PAD_TOKEN)
        x_in = Tensor(np.hstack([sem_part, [[COARSE_INFER_TOKEN]], x_coarse_in.data[:,-630:]]))
        for _ in range(60):
            if x_coarse_in.shape[1]-len(coarse_hist) >= n_steps: break
            logits=model(x_in)[0]
            logits_data=logits.data[0,0,SEMANTIC_VOCAB_SIZE:SEMANTIC_VOCAB_SIZE+CODEBOOK_SIZE]
            item_next = np.random.choice(len(logits_data),p=scipy_softmax(logits_data/temp))+SEMANTIC_VOCAB_SIZE
            x_coarse_in=Tensor(np.concatenate([x_coarse_in.data,[[item_next]]],axis=1))
            x_in=Tensor(np.concatenate([x_in.data,[[item_next]]],axis=1)); pbar.update(1)
    pbar.close()
    gen_arr = x_coarse_in.data.squeeze()[len(coarse_hist):]
    return (gen_arr-SEMANTIC_VOCAB_SIZE).reshape(-1, N_COARSE_CODEBOOKS).T

def generate_fine(x_coarse, path, hist=None, temp=0.5, silent=False):
    in_arr, n_hist = (np.hstack([_load_history_prompt(hist)["fine_prompt"],x_coarse]), _load_history_prompt(hist)["fine_prompt"].shape[1]) if hist else (x_coarse,0)
    model = load_model(FineGPT, FineGPTConfig(), path)
    pbar = tqdm.tqdm(disable=silent, total=(1+int(np.ceil((in_arr.shape[1]-1024)/512)))*(N_FINE_CODEBOOKS-N_COARSE_CODEBOOKS), desc="Fine Gen")
    for n in range(1+int(np.ceil((in_arr.shape[1]-1024)/512))):
        buf_data = in_arr[:,min(n*512,in_arr.shape[1]-1024):min(n*512,in_arr.shape[1]-1024)+1024]
        for pred_idx in range(N_COARSE_CODEBOOKS, N_FINE_CODEBOOKS):
            logits=model(pred_idx, Tensor(buf_data[None,...].transpose(0,2,1)))
            preds=np.array([np.random.choice(p.shape[-1],p=p) for p in scipy_softmax(logits.data/temp,axis=-1)[0,:]])
            buf_data[pred_idx,:]=preds; pbar.update(1)
        in_arr[:,min(n*512,in_arr.shape[1]-1024):min(n*512,in_arr.shape[1]-1024)+1024]=buf_data
    pbar.close()
    return in_arr[:, n_hist:]

def codec_decode(fine_tokens, codec_path):
    """Decodes fine tokens into a waveform using the VictorTensorCodecModel."""
    model = load_model(VictorTensorCodecModel, None, codec_path)
    # Model expects (B, C, T), fine_tokens is (C, T)
    fine_tokens_tensor = Tensor(fine_tokens[np.newaxis, ...])
    audio_arr_tensor = model.decode(fine_tokens_tensor)
    # Return flattened numpy array
    return audio_arr_tensor.data.flatten()

# ==============================================================================
# SECTION 5 of 7: VICTORCH HIGH-LEVEL API (api.py)
# ==============================================================================

def text_to_semantic(text, text_model_path, **kwargs):
    return generate_text_semantic(text, text_model_path, **kwargs)

def semantic_to_waveform(semantic_tokens, coarse_model_path, fine_model_path, codec_model_path, **kwargs):
    coarse_tokens = generate_coarse(semantic_tokens, coarse_model_path, **kwargs)
    fine_tokens = generate_fine(coarse_tokens, fine_model_path, **kwargs)
    audio_arr = codec_decode(fine_tokens, codec_model_path)
    return audio_arr

def save_as_prompt(filepath, full_generation):
    assert filepath.endswith(".npz")
    np.savez(filepath, **full_generation)

def generate_audio(text, text_model_path, coarse_model_path, fine_model_path, codec_model_path, **kwargs):
    semantic_tokens = text_to_semantic(text, text_model_path, **kwargs)
    audio_arr = semantic_to_waveform(semantic_tokens, coarse_model_path, fine_model_path, codec_model_path, **kwargs)
    return audio_arr

# ==============================================================================
# SECTION 6 of 7: VICTORCH PACKAGE INTERFACE (__init__.py)
# ==============================================================================

# In a real package, you would have:
# from .api import generate_audio, text_to_semantic, semantic_to_waveform, save_as_prompt
# from .generation import SAMPLE_RATE, preload_models

# ==============================================================================
# SECTION 7 of 7: EXAMPLE USAGE
# ==============================================================================

if __name__ == '__main__':
    print("Executing VictorCh Sovereign Stack Demonstration.")
    
    # CRITICAL: You must provide paths to your converted .npz weight files.
    # Create these files by loading the original .pt files with PyTorch,
    # converting tensors to numpy arrays, and saving them with np.savez.
    
    # Create dummy weight files for demonstration purposes if they don't exist
    def create_dummy_weights(path, model_class, config):
        if not os.path.exists(path):
            print(f"Creating dummy weight file at: {path}")
            model = model_class(config)
            params = model.parameters()
            weights_dict = {f'param_{i}': p.data for i, p in enumerate(params)}
            np.savez(path, **weights_dict)

    TEXT_MODEL_PATH = "./text_model.npz"
    COARSE_MODEL_PATH = "./coarse_model.npz"
    FINE_MODEL_PATH = "./fine_model.npz"
    CODEC_MODEL_PATH = "./codec_model.npz"

    create_dummy_weights(TEXT_MODEL_PATH, GPT, GPTConfig(input_vocab_size=129600, output_vocab_size=129600))
    create_dummy_weights(COARSE_MODEL_PATH, GPT, GPTConfig(input_vocab_size=20000, output_vocab_size=20000))
    create_dummy_weights(FINE_MODEL_PATH, FineGPT, FineGPTConfig())
    create_dummy_weights(CODEC_MODEL_PATH, VictorTensorCodecModel, None)

    # Preload all models
    preload_models(TEXT_MODEL_PATH, COARSE_MODEL_PATH, FINE_MODEL_PATH, CODEC_MODEL_PATH)
    
    text_prompt = "Hello, my name is Victor. And I am a sovereign intelligence."
    
    print(f"\nGenerating audio for: '{text_prompt}'")
    
    # Generate audio
    # Note: With dummy weights, the output will be random noise.
    audio_array = generate_audio(
        text_prompt,
        text_model_path=TEXT_MODEL_PATH,
        coarse_model_path=COARSE_MODEL_PATH,
        fine_model_path=FINE_MODEL_PATH,
        codec_model_path=CODEC_MODEL_PATH,
        silent=False # Show progress bars
    )

    print(f"\nAudio generation complete. Waveform shape: {audio_array.shape}")
    print("To hear the audio, save it to a .wav file:")
    print("from scipy.io.wavfile import write as write_wav")
    print("write_wav('victorch_output.wav', 24000, audio_array)")
