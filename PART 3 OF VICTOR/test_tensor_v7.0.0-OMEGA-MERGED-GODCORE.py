# FILE: tests/test_tensor_v7.0.0-OMEGA-MERGED-GODCORE.py
# PURPOSE: Unit tests for core Tensor operations.

import numpy as np
from victortensor.tensor_v7 import Tensor

def almost_eq(a, b, eps=1e-6):
    """Checks if two numpy arrays are almost equal."""
    return np.allclose(a, b, atol=eps)

def test_add():
    """Tests tensor addition."""
    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    assert almost_eq((a + b).data, np.array([5,7,9]))

def test_sub():
    """Tests tensor subtraction."""
    a = Tensor([7,8,9])
    b = Tensor([1,2,3])
    assert almost_eq((a - b).data, np.array([6,6,6]))

def test_mul():
    """Tests element-wise tensor multiplication."""
    a = Tensor([2,3,4])
    b = Tensor([5,6,7])
    assert almost_eq((a * b).data, np.array([10,18,28]))

def test_div():
    """Tests element-wise tensor division."""
    a = Tensor([10,20,30])
    b = Tensor([2,4,5])
    assert almost_eq((a / b).data, np.array([5,5,6]))

def test_matmul():
    """Tests matrix multiplication."""
    a = Tensor(np.ones((2,3)))
    b = Tensor(np.ones((3,2)))
    c = a.matmul(b)
    assert c.data.shape == (2,2)
    assert almost_eq(c.data, np.full((2,2), 3))

def test_core_stats():
    """Tests statistical operations."""
    a = Tensor([1,2,3,4])
    assert abs(a.mean().data - 2.5) < 1e-6
    assert abs(a.sum().data - 10) < 1e-6
    assert abs(a.var().data - 1.25) < 1e-6
    assert abs(a.std().data - np.std([1,2,3,4])) < 1e-6

def test_transpose():
    """Tests tensor transposition."""
    a = Tensor([[1,2],[3,4]])
    at = a.transpose()
    assert almost_eq(at.data, np.array([[1,3],[2,4]]))
    at_prop = a.T
    assert almost_eq(at_prop.data, np.array([[1,3],[2,4]]))
