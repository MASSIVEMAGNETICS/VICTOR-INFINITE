# VERSION: v7.0.0-PRIMECORE-ΩSIGMA
# FILE: victorch/core/loss_v7.py

import numpy as np
from .tensor_v7 import OmegaTensor

class MSELoss:
    def __call__(self, pred: OmegaTensor, target: OmegaTensor):
        diff = pred.data - target.data
        loss_val = np.mean(diff ** 2)
        out = OmegaTensor(loss_val, requires_grad=True)
        if pred.requires_grad:
            grad = 2 * (pred.data - target.data) / pred.data.size
            out.set_creator(self, pred)
            pred.backward(grad)
        return out

class CrossEntropyLoss:
    def __call__(self, logits: OmegaTensor, target: OmegaTensor):
        exps = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        batch_indices = np.arange(target.data.shape[0])
        loss_val = -np.log(probs[batch_indices, target.data.astype(int)]).mean()
        out = OmegaTensor(loss_val, requires_grad=True)
        if logits.requires_grad:
            grad = probs
            grad[batch_indices, target.data.astype(int)] -= 1
            grad /= target.data.shape[0]
            out.set_creator(self, logits)
            logits.backward(grad)
        return out
