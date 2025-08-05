# VERSION: v7.0.0-PRIMECORE-ΩSIGMA
# FILE: victorch/core/ops_v7.py

from .tensor_v7 import register_op, Op, OmegaTensor

class MulOp(Op):
    def forward(self, a, b):
        out = OmegaTensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
        out.set_creator(self, a, b)
        return out
    def backward(self, output, grad_output):
        a, b = output.creator[1]
        return [grad_output * b.data, grad_output * a.data]

class MatMulOp(Op):
    def forward(self, a, b):
        out = OmegaTensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
        out.set_creator(self, a, b)
        return out
    def backward(self, output, grad_output):
        a, b = output.creator[1]
        return [grad_output @ b.data.T, a.data.T @ grad_output]

# Register ops
register_op('mul', MulOp)
register_op('matmul', MatMulOp)
