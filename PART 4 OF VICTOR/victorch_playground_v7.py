# VERSION: v7.0.0-PRIMECORE-ΩSIGMA
# FILE: victorch/victorch_playground_v7.py

from core.tensor_v7 import OmegaTensor
from core.ops_v7 import *
from core.loss_v7 import MSELoss, CrossEntropyLoss

if __name__ == "__main__":
    print("🔁 VictorCH PrimeCore Playground — ΩSIGMA Bootstrapped")

    a = OmegaTensor([2.0], requires_grad=True)
    b = OmegaTensor([3.0], requires_grad=True)

    c = a * b
    d = c + b

    print("Forward output:", d.data)
    d.backward()

    print("∇a:", a.grad)
    print("∇b:", b.grad)

    logits = OmegaTensor([[2.0, 1.0, 0.1]], requires_grad=True)
    target = OmegaTensor([0])

    cel = CrossEntropyLoss()
    loss = cel(logits, target)

    print("CrossEntropy Loss:", loss.data)
    print("Logits Gradient:", logits.grad)
