# FILE: bark_victortensor/model.py
# PURPOSE: VictorTensor implementation of the base GPT model.

import math
from dataclasses import dataclass

from .victortensor_v9 import Tensor, nn, functional as F

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
        # Create a persistent causal mask
        self.bias = np.tril(np.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        
        qkv = self.c_attn(x)
        
        # Split q, k, v
        q_data, k_data, v_data = np.split(qkv.data, 3, axis=2)
        q = Tensor(q_data, _children=(qkv,), _op='split_q')
        k = Tensor(k_data, _children=(qkv,), _op='split_k')
        v = Tensor(v_data, _children=(qkv,), _op='split_v')
        
        k = Tensor(k.data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        q = Tensor(q.data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        v = Tensor(v.data.reshape(B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))

        if past_kv is not None:
            past_key, past_value = past_kv
            k = F.cat([past_key, k], dim=2)
            v = F.cat([past_value, v], dim=2)

        present = (k, v) if use_cache else None

        # Manual attention implementation
        att = (q.matmul(k.transpose((0, 1, 3, 2)))) * (1.0 / math.sqrt(k.shape[-1]))
        
        # Apply causal mask
        mask = self.bias[:, :, :T, :T]
        # Create a tensor from the mask but don't require grad
        mask_tensor = Tensor(np.where(mask == 0, -np.inf, 0))
        att += mask_tensor
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att.matmul(v)
        y = Tensor(y.data.transpose(0, 2, 1, 3).reshape(B, T, C))
        
        y = self.resid_dropout(self.c_proj(y))
        return y, present

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = nn.OmegaLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.OmegaLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)

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
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.input_vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            'ln_f': nn.OmegaLayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        b, t = idx.shape
        
        if past_kv is not None:
            assert t == 1
            tok_emb = self.transformer['wte'](idx)
        else:
            if merge_context:
                assert(idx.shape[1] >= 256+256+1)
                t = idx.shape[1]
                # Split and process context
                text_part = Tensor(idx.data[:, :256])
                semantic_part = Tensor(idx.data[:, 256:512])
                infer_part = Tensor(idx.data[:, 512:])
                tok_emb = F.cat([
                    self.transformer['wte'](text_part) + self.transformer['wte'](semantic_part),
                    self.transformer['wte'](infer_part)
                ], dim=1)
                t = tok_emb.shape[1] # update sequence length
            else:
                tok_emb = self.transformer['wte'](idx)

        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.transformer['h']))
        else:
            past_length = past_kv[0][0].shape[2]

        if position_ids is None:
            position_ids = Tensor(np.arange(past_length, t + past_length))
        
        pos_emb = self.transformer['wpe'](position_ids)
        x = self.transformer['drop'](tok_emb + pos_emb)
        
        new_kv = () if use_cache else None
        
        for i, block in enumerate(self.transformer['h']):
            x, kv = block(x, past_kv=past_kv[i], use_cache=use_cache)
            if use_cache:
                new_kv = new_kv + (kv,)
        
        x = self.transformer['ln_f'](x)
        
        # Return only the logits for the last token for efficiency
        last_step_data = x.data[:, [-1], :]
        logits = self.lm_head(Tensor(last_step_data))
        
        return (logits, new_kv)
