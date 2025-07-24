"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # c = a ** b (ewise)
        c = node.op.compute(a, b)  # c = a ** b

        partial_ac = b * divide(c, a)  # d(c)/d(a) = b * a ** (b - 1)

        partial_bc = c * log(a)  # d(c)/d(b) = log(a) * a ** b
        
        return partial_ac * out_grad, partial_bc * out_grad
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * self.scalar * a ** (self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        partial_ac = power_scalar(b, -1)
        partial_bc = negate(power_scalar(b, -2)) * a
        return out_grad * partial_ac, out_grad * partial_bc
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (divide_scalar(out_grad, self.scalar), )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node): 
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]

        # cases when len(a.shape) < len(out_grad.shape)
        axes = tuple(i for i in range(len(out_grad.shape) - len(a.shape)))
        if axes:
            # We need to sum over the leading dimensions that were broadcasted
            out_grad = summation(out_grad, axes=axes)
        # cases when a.shape has dimensions of size 1 but out_grad.shape does not
        axes = [i for i in range(len(a.shape)) if a.shape[i] == 1 and self.shape[i] != 1]
        if axes:
            # We need to sum over the axes where a.shape is 1 but self.shape is not 1
            out_grad = summation(out_grad, axes=tuple(axes))

        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        a_shape = a.shape
        if self.axes is not None:
            reshape_shape = []
            for i in range(len(a_shape)):
                if i in self.axes:
                    reshape_shape.append(1)
                else:
                    reshape_shape.append(a_shape[i])
            out_grad = reshape(out_grad, tuple(reshape_shape))
            # Now we need to broadcast out_grad to the shape of a
        return broadcast_to(out_grad, a_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        '''
        - If a.shape is (m, n) and b.shape is (n, p), then out_grad should be (m, p).
        
        Subtlties when broadcasting happens:
        - Leading dimensions
            - Example: a.shape is (6, 6, 5, 4) and b.shape is (4, 3), then out_grad should be (6, 6, 5, 3).
            - Let c = a @ b, then c.shape is (6, 6, 5, 3). dc/d(b) should be a.T @ (6, 6, 5, 3).
            - dc/d(b) = (6, 6, 4, 3), but it should have shape (4, 3).
            - Need to sum over the dimensions that were broadcasted, i.e. 6, 6.
        - If there are dimensions of size 1
            - Example: a.shape = (3, 1, 5), b.shape = (1, 5, 5)
            - out_grad.shape = (3, 5, 5)
            - grad_a.shape = (3, 5, 5)
            - Need to sum over axis where a.shape is 1 but grad_a.shape > 1.
        
        '''

        a, b = node.inputs
        # print(f"a shape: {a.shape} \n b shape: {b.shape} \n out_grad shape: {out_grad.shape}")
        grad_a = matmul(out_grad, transpose(b, axes=(-1, -2)))
        grad_b = matmul(transpose(a, axes=(-1, -2)), out_grad)

        if grad_a.shape != a.shape:
            # Sum over leading dimensions that were broadcasted
            axes = tuple(i for i in range(len(grad_a.shape) - len(a.shape)))
            if axes:
                grad_a = summation(grad_a, axes=axes)
            # Sum over dimensions where a.shape is 1 but grad_a.shape > 1
            axes = tuple(i for i, (ga, aa) in enumerate(zip(grad_a.shape, a.shape)) if aa == 1 and ga != 1)
            if axes:
                grad_a = summation(grad_a, axes=axes)
            grad_a = reshape(grad_a, a.shape)

        if grad_b.shape != b.shape:
            axes = tuple(i for i in range(len(grad_b.shape) - len(b.shape)))
            if axes:
                grad_b = summation(grad_b, axes=axes)
            axes = tuple(i for i, (gb, bb) in enumerate(zip(grad_b.shape, b.shape)) if bb == 1 and gb != 1)
            if axes:
                grad_b = summation(grad_b, axes=axes)
            grad_b = reshape(grad_b, b.shape)

        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.max(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

