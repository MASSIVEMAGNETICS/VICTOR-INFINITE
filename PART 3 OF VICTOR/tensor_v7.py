# VERSION: v7.0.0-PRIMECORE-ΩSIGMA
# FILE: victorch/core/tensor_v7.py

import numpy as np

class OmegaTensor:
    def __init__(self, data, requires_grad=False, device='cpu'):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = None
        self.device = device
        self.graph_id = id(self)

    def set_creator(self, op, *parents):
        self.creator = (op, parents)
        if self.requires_grad:
            for p in parents:
                p.requires_grad = True

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        grad = grad or np.ones_like(self.data)
        self.grad = grad if self.grad is None else self.grad + grad

        if self.creator:
            op, parents = self.creator
            grads = op.backward(self, grad)
            for parent, g in zip(parents, grads):
                parent.backward(g)

    def __repr__(self):
        return f"ΩTensor(shape={self.data.shape}, grad={self.grad is not None})"

    # Core ops (add, mul, etc.)...
    def __add__(self, other):
        return OpRegistry['add'](self, other)

    def __mul__(self, other):
        return OpRegistry['mul'](self, other)

    def matmul(self, other):
        return OpRegistry['matmul'](self, other)

    def sum(self):
        return OpRegistry['sum'](self)

    def mean(self):
        return OpRegistry['mean'](self)

# === Global Op Registry ===
class Op:
    def forward(self): raise NotImplementedError
    def backward(self, output, grad_output): raise NotImplementedError

OpRegistry = {}
def register_op(name, op_cls):
    OpRegistry[name] = lambda *args: op_cls().forward(*args)

# AddOp example:
class AddOp(Op):
    def forward(self, a, b):
        out = OmegaTensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
        out.set_creator(self, a, b)
        return out
    def backward(self, output, grad_output):
        return [grad_output, grad_output]

register_op('add', AddOp)
