# FILE: bark_victortensor/model.py
# PURPOSE: VictorTensor implementation of the base GPT model.
# VERSION: 3.0.0 "Omega"
# NOTES: This version has been significantly upgraded to use the OmegaLlamaLayers model.

import math
import json
import time
from dataclasses import dataclass, asdict

import numpy as np

from omega_model import TransformerOmega, LlamaModelArgs
from OmegaTensor import Tensor, nn, functional as F


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        
        self.config = LlamaModelArgs(
            dim=config.n_embd,
            n_layers=config.n_layer,
            n_heads=config.n_head,
            n_kv_heads=config.n_head, # MHA
            vocab_size=config.output_vocab_size,
            max_seq_len=config.block_size,
        )

        self.model = TransformerOmega(self.config)

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        # The new model doesn't support these arguments directly.
        # This is a simple adaptation.
        
        if isinstance(idx, Tensor):
            idx = idx.data.astype(np.int64)

        return self.model(tokens=idx)
