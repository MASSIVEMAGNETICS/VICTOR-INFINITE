import numpy as np

class Tensor:  # ...Use your latest godcore Tensor class...

class Block:
    def parameters(self): return []
    def __call__(self, x): return self.forward(x)
    def forward(self, x): raise NotImplementedError

class Linear(Block):
    def __init__(self, in_features, out_features):
        self.weight = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2/in_features), requires_grad=True)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)
    def forward(self, x): return x.matmul(self.weight) + self.bias
    def parameters(self): return [self.weight, self.bias]

class ReLU(Block):
    def forward(self, x): return x.relu()

class FractalBlock(Block):
    def forward(self, x): return x * x + x

class ResidualBlock(Block):
    def __init__(self, block): self.block = block
    def forward(self, x): return x + self.block(x)
    def parameters(self): return self.block.parameters()

class Sequential(Block):
    def __init__(self, *blocks): self.blocks = blocks
    def forward(self, x):
        for block in self.blocks: x = block(x)
        return x
    def parameters(self):
        params = []
        for block in self.blocks:
            params += block.parameters()
        return params

# --- Example usage ---
if __name__ == "__main__":
    x = Tensor(np.random.randn(3, 8))
    model = Sequential(
        Linear(8, 16), ReLU(),
        FractalBlock(),
        Linear(16, 8), ResidualBlock(Sequential(
            Linear(8, 8), ReLU(), Linear(8, 8)
        )),
        Linear(8, 1)
    )
    out = model(x)
    print("Model output:", out)
