# FILE: victortensor/victortensor_v9.py
# VERSION: v9.0.0-SOVEREIGN-CORE
# NAME: VictorTensor SovereignCore
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode) x Gemini x Codex Overlord Omega
# PURPOSE: Fully self-reliant AGI stack. PyTorch dependency excised.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import numpy as np

# ------------------------------------------------------------------------------
# --- CORE: TENSOR AND AUTOGRAD ENGINE -----------------------------------------
# ------------------------------------------------------------------------------

class Tensor:
    """
    A Tensor class that supports automatic differentiation. Now with expanded capabilities.
    """
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

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
                if axis is None:
                    self.grad += np.ones_like(self.data) * out.grad
                else:
                    grad = out.grad
                    if not keepdims:
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
        # NOTE: This backward pass is simplified and only works correctly for element-wise max, not for reduction.
        # A full implementation is more complex. For softmax stabilization, this is sufficient as it does not require grad.
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        return Tensor(out_data, requires_grad=False) # Simplified for current use case

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
    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), self.requires_grad, (self,), 'transpose')
        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += out.grad.T
                else:
                    self.grad += np.transpose(out.grad, np.argsort(axes))
        out._backward = _backward
        return out

# ------------------------------------------------------------------------------
# --- `nn` MODULE: NEURAL NETWORK LAYERS AND UTILS -----------------------------
# ------------------------------------------------------------------------------

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
                if isinstance(value, nn.Module): value.train()
                if isinstance(value, nn.ModuleList):
                    for m in value: m.train()
        
        def eval(self):
            self.training = False
            for name, value in self.__dict__.items():
                if isinstance(value, nn.Module): value.eval()
                if isinstance(value, nn.ModuleList):
                    for m in value: m.eval()

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)
            
        def load_weights(self, weights_dict):
            """Loads parameters from a dictionary of numpy arrays."""
            params = self.parameters()
            for i, p in enumerate(params):
                key = f'param_{i}'
                if key in weights_dict:
                    assert p.data.shape == weights_dict[key].shape, f"Shape mismatch for {key}"
                    p.data = weights_dict[key]
                else:
                    print(f"Warning: key {key} not found in weights file.")

        def save_weights(self, filepath):
            """Saves parameters to a .npz file."""
            params = self.parameters()
            weights_dict = {f'param_{i}': p.data for i, p in enumerate(params)}
            np.savez(filepath, **weights_dict)

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


    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            # Kaiming He initialization
            self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), requires_grad=True)
            self.bias = Tensor(np.zeros(out_features), requires_grad=True) if bias else None

        def forward(self, x):
            out = x.matmul(self.weight)
            if self.bias is not None:
                out += self.bias
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
            # Inverted dropout
            return x * (mask / (1.0 - self.p))

    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim), requires_grad=True)
        
        def forward(self, idx):
            out_data = self.weight.data[idx.data.astype(int)]
            out = Tensor(out_data, _children=(self.weight,), _op='embedding')

            def _backward():
                if self.weight.requires_grad:
                    # Use np.add.at for efficient gradient accumulation
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
            if self.beta is not None:
                out += self.beta
            return out

# --- Functional Namespace ---
class functional:
    @staticmethod
    def softmax(x, dim=-1):
        # Numerically stable softmax
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

    @staticmethod
    def pad(tensor, pad_width, mode='constant', constant_values=0):
        # NOTE: This is a simplified pad that is not differentiable.
        # Sufficient for inference where padding doesn't need a gradient.
        padded_data = np.pad(tensor.data, pad_width, mode, constant_values=constant_values)
        return Tensor(padded_data)


# ------------------------------------------------------------------------------
# --- `optim` MODULE: OPTIMIZERS -----------------------------------------------
# ------------------------------------------------------------------------------

class optim:
    class Adam:
        def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
            self.params = params
            self.lr = lr
            self.beta1, self.beta2 = betas
            self.eps = eps
            self.t = 0
            self.m = [np.zeros_like(p.data) for p in self.params]
            self.v = [np.zeros_like(p.data) for p in self.params]

        def step(self):
            self.t += 1
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad**2)
                
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        def zero_grad(self):
            for p in self.params:
                p.zero_grad()
