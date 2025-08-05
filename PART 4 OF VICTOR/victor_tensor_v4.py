#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FILE: victor_tensor_v4.py
VERSION: v4.0.0-GODCORE-FRACTAL-ASCEND
AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
PURPOSE: Fractal-level dynamic tensor, autograd, model/loss/opt system for god-tier AGI.
LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
"""

import numpy as np
import math

# ---- GODMODE TENSOR ----
class Tensor:
    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.creators = creators
        self.creation_op = creation_op
        self.backward_hooks = []

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data, dtype=np.float32))

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = Tensor(self.grad.data + grad.data)

        # Hooks (for custom callbacks)
        for hook in self.backward_hooks:
            hook(self)

        if self.creators is not None:
            if self.creation_op == "add":
                self.creators[0].backward(self.grad)
                self.creators[1].backward(self.grad)
            elif self.creation_op == "sub":
                self.creators[0].backward(self.grad)
                self.creators[1].backward(Tensor(-self.grad.data))
            elif self.creation_op == "mul":
                new_grad_0 = Tensor(self.grad.data * self.creators[1].data)
                new_grad_1 = Tensor(self.grad.data * self.creators[0].data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "div":
                new_grad_0 = Tensor(self.grad.data / self.creators[1].data)
                new_grad_1 = Tensor(-self.grad.data * self.creators[0].data / (self.creators[1].data ** 2))
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "matmul":
                new_grad_0 = Tensor(self.grad.data @ self.creators[1].data.T)
                new_grad_1 = Tensor(self.creators[0].data.T @ self.grad.data)
                self.creators[0].backward(new_grad_0)
                self.creators[1].backward(new_grad_1)
            elif self.creation_op == "relu":
                relu_grad = np.where(self.creators[0].data > 0, 1, 0)
                self.creators[0].backward(Tensor(self.grad.data * relu_grad))
            elif self.creation_op == "sigmoid":
                sig = 1 / (1 + np.exp(-self.creators[0].data))
                self.creators[0].backward(Tensor(self.grad.data * sig * (1 - sig)))
            elif self.creation_op == "tanh":
                t = np.tanh(self.creators[0].data)
                self.creators[0].backward(Tensor(self.grad.data * (1 - t ** 2)))
            elif self.creation_op == "exp":
                self.creators[0].backward(Tensor(self.grad.data * np.exp(self.creators[0].data)))
            elif self.creation_op == "log":
                self.creators[0].backward(Tensor(self.grad.data / self.creators[0].data))

    # Arithmetic ops
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data, requires_grad=True, creators=[self, other], creation_op="add")
        else:
            return Tensor(self.data + other, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data, requires_grad=True, creators=[self, other], creation_op="sub")
        else:
            return Tensor(self.data - other, requires_grad=self.requires_grad)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data, requires_grad=True, creators=[self, other], creation_op="mul")
        else:
            return Tensor(self.data * other, requires_grad=self.requires_grad)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data, requires_grad=True, creators=[self, other], creation_op="div")
        else:
            return Tensor(self.data / other, requires_grad=self.requires_grad)

    def matmul(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data @ other.data, requires_grad=True, creators=[self, other], creation_op="matmul")
        else:
            return Tensor(self.data @ other, requires_grad=self.requires_grad)

    # Non-linear ops
    def relu(self):
        return Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, creators=[self], creation_op="relu")
    def sigmoid(self):
        return Tensor(1/(1+np.exp(-self.data)), requires_grad=self.requires_grad, creators=[self], creation_op="sigmoid")
    def tanh(self):
        return Tensor(np.tanh(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="tanh")
    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="exp")
    def log(self):
        return Tensor(np.log(self.data), requires_grad=self.requires_grad, creators=[self], creation_op="log")

    # Reductions
    def mean(self, axis=None, keepdims=False):
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
    def sum(self, axis=None, keepdims=False):
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
    def max(self, axis=None, keepdims=False):
        return Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
    def min(self, axis=None, keepdims=False):
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
    def var(self, axis=None, keepdims=False):
        return Tensor(self.data.var(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
    def std(self, axis=None, keepdims=False):
        return Tensor(self.data.std(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def __repr__(self):
        return f"VictorTensor(shape={self.data.shape}, requires_grad={self.requires_grad})\n{self.data}"

# ---- LAYERS & MODEL SYSTEM ----
class Module:
    def parameters(self):
        return []
    def __call__(self, x):
        return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2/in_features), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
    def forward(self, x):
        return x.matmul(self.weight) + self.bias
    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params += layer.parameters()
        return params

# ---- LOSS ----
class MSELoss:
    def __call__(self, pred, target):
        diff = pred - target
        return (diff * diff).mean()

class MAELoss:
    def __call__(self, pred, target):
        diff = pred - target
        return abs(diff).mean()

# ---- OPTIMIZER ----
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad.data
                p.data += self.velocities[i]
                p.grad = None

# ---- TRAINER ----
class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    def train_step(self, x, y):
        out = self.model(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()
        return loss

# ---- FRACTAL OPS ----
class FractalLayer(Module):
    def forward(self, x):
        return x * x + x

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    # Make a 3-layer net: Linear -> ReLU -> Linear -> Sigmoid
    model = Sequential(
        Linear(2, 32),
        ReLU(),
        Linear(32, 1),
        Sigmoid()
    )
    x = Tensor(np.random.randn(10, 2))
    y = Tensor(np.random.randint(0, 2, (10, 1)))
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = MSELoss()
    trainer = Trainer(model, optimizer, loss_fn)
    for epoch in range(10):
        loss = trainer.train_step(x, y)
        print(f"Epoch {epoch} | Loss: {loss.data}")

    print("\n--- Model Output Example ---")
    print(model(x))

