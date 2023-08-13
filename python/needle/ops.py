"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a**self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return a ** (self.scalar - 1) * self.scalar * out_grad
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a / b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a, b = node.inputs
        return (out_grad / b, out_grad * (-a) / (b**2))
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a / self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # order = list(range(len(a.shape)))
        # if self.axes is None:
        #     order[-1] = order[-2]
        #     order[-2] = len(order) - 1
        # else:
        #     order[self.axes[0]] = self.axes[1]
        #     order[self.axes[1]] = self.axes[0]
        # return a.permute(tuple(order))
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        new_axes = list(range(a.ndim))
        new_axes[ax0], new_axes[ax1] = new_axes[ax1], new_axes[ax0]
        return a.permute(new_axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        if self.axes:
            return transpose(out_grad, self.axes)
        else:
            return transpose(out_grad)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # return array_api.reshape(out_grad, node.inputs[0].shape)
        shape = node.inputs[0].shape
        return reshape(out_grad, shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # shape = list(node.inputs[0].shape)  # (10, ) -> (2, 10)
        # axes = []
        # shape = [1] * (len(self.shape) - len(shape)) + shape
        # for i, s in enumerate(self.shape):
        #     if i >= len(shape) or s != shape[i]:
        #         axes.append(i)
        # return reshape(summation(out_grad, tuple(axes)), node.inputs[0].shape)
        ori_shape = node.inputs[0].shape
        if ori_shape == self.shape:
            return out_grad
        shrink_dims = []
        for i, (ori, cur) in enumerate(zip(ori_shape, self.shape)):
            if ori != cur:
                shrink_dims.append(i)
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        if isinstance(self.axes, (list, tuple)) and len(self.axes) > 1:
            # multiple axes case
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        return array_api.summation(a, axis=self.axes)
        # if self.axes is None:
        #     return a.sum()
        # else:
        #     # NOTE self.axes maybe int
        #     if isinstance(self.axes, int):
        #         return a.sum(self.axes)
        #     # NOTE only support sum in a single dim
        #     for i, axis in enumerate(sorted(list(self.axes))):
        #         # NOTE -i because each sum() operation will reduce the dimension number
        #         a = a.sum(axis-i)
        #     return a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # shape = node.inputs[0].shape
        # shape_out = [1] * len(shape)
        # # (5, 4, 3, 2) (0, 2) -> (4, 2)
        #
        # if self.axes is not None:
        #     if isinstance(self.axes, int):
        #         s = set([self.axes])
        #     else:
        #         s = set(self.axes)
        # else:
        #     s = set(range(len(shape)))
        # j = 0
        # for i in range(len(shape)):
        #     if i not in s:
        #         shape_out[i] = out_grad.shape[j]
        #         j += 1
        # result = broadcast_to(reshape(out_grad, tuple(shape_out)), shape)
        # # print(self.axes, out_grad.shape, shape_out, shape)
        # return result
        new_shape = list(node.inputs[0].shape)
        if self.axes is None:
            axes = range(len(new_shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            raise ValueError("Unsupported axes type, must be int, tuple or None!")
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return a @ b
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # lhs, rhs = node.inputs
        # grad_a = matmul(out_grad, transpose(rhs))
        # grad_b = matmul(transpose(lhs), out_grad)
        # if grad_a.shape != lhs.shape:
        #     length = len(grad_a.shape) - len(lhs.shape)
        #     grad_a = summation(grad_a, axes=tuple(range(length)))
        # if grad_b.shape != rhs.shape:
        #     length = len(grad_b.shape) - len(rhs.shape)
        #     grad_b = summation(grad_b, axes=tuple(range(length)))
        # return grad_a, grad_b
        lhs, rhs = node.inputs
        lgrad, rgrad = out_grad @ rhs.transpose(), lhs.transpose() @ out_grad

        # X(B, M, K) matmul W(K, N)
        # out(B, M, N)
        # dX = out matmul W_transpose
        # dW = X_transpose matmul out -> B, K, N，然后在B维度上reduce-> K, N

        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple(range(len(rgrad.shape) - len(rhs.shape))))
        return (lgrad, rgrad)
        # raise NotImplementedError()
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # return -out_grad
        return mul_scalar(out_grad, -1)
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0, device=out_grad.device, dtype=out_grad.dtype, requires_grad=False)
        # END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        # z_max_origindim = Z.max(self.axes, keepdims=True)
        # z_max_reducedim = Z.max(self.axes)
        # return array_api.log(array_api.exp(Z - z_max_origindim.broadcast_to(Z.shape)).sum(self.axes)) + z_max_reducedim
        maxz = Z.max(self.axes, keepdims=True)
        ret = array_api.log(array_api.exp(Z - maxz.broadcast_to(Z.shape)).sum(axis=self.axes, keepdims=True)) + maxz
        if self.axes is None:
            axes = list(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = list(self.axes)

        if self.axes is not None:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in axes]
        else:
            out_shape = [1]

        return ret.reshape(tuple(out_shape))
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # z = node.inputs[0]
        # max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        # exp_z = array_api.exp(z.realize_cached_data() - max_z.broadcast_to(z.shape))
        # sum_exp_z = exp_z.sum(self.axes)
        # grad_sum_exp_z = out_grad / sum_exp_z
        #
        # #   [2.132311 4.359107 2.888913] ->
        # #  [[[2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]]
        # #
        # #  [[2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]]
        # #
        # #  [[2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]
        # #   [2.132311 4.359107 2.888913]]]
        # if self.axes is None:
        #     return grad_sum_exp_z.broadcast_to(z.shape) * exp_z
        #
        # #  self.axes = (1, 2)
        # # [2.132311 4.359107 2.888913] ->
        # # [[[2.132311 2.132311 2.132311]
        # #   [2.132311 2.132311 2.132311]
        # #  [2.132311 2.132311 2.132311]]
        # #
        # # [[4.359107 4.359107 4.359107]
        # #  [4.359107 4.359107 4.359107]
        # #  [4.359107 4.359107 4.359107]]
        # #
        # # [[2.888913 2.888913 2.888913]
        # #  [2.888913 2.888913 2.888913]
        # #  [2.888913 2.888913 2.888913]]]
        # expand_shape = list(z.shape)
        # for axis in self.axes:
        #     expand_shape[axis] = 1
        # grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        # return grad_exp_z * exp_z
        Z = node.inputs[0]
        if self.axes is not None:
            shape = [1] * len(Z.shape)
            if isinstance(self.axes, int):
                s = set([self.axes])
            else:
                s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        return grad_new.broadcast_to(Z.shape) * exp(Z - node_new.broadcast_to(Z.shape))
        # END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * (1 - tanh(a) ** 2)
        # END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        # BEGIN YOUR SOLUTION
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
        new_shape = list(shape)
        new_shape.insert(self.axis, len(args))
        out = args[0].device.empty(new_shape)

        slices = [slice(0, s) for s in new_shape]
        for i, arg in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arg
        return out
        # shape = args[0].shape
        # new_shape = list(shape)
        # new_shape.insert(self.axis, len(args))
        #
        # out = array_api.empty(
        #     new_shape, dtype=args[0].dtype, device=args[0].device)
        #
        # slices = []
        # for i in range(len(new_shape)):
        #     if i != self.axis:
        #         slices.append(slice(new_shape[i]))
        #     else:
        #         slices.append(0)
        # for i in range(len(args)):
        #     slices[self.axis] = i
        #     # NOTE reshape
        #     out[tuple(slices)] = args[i].reshape((1, ) + shape)
        # return out
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        # END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # BEGIN YOUR SOLUTION
        new_shape = list(A.shape)
        new_shape.pop(self.axis)

        out = []
        slices = [slice(0, s) for s in A.shape]
        for i in range(A.shape[self.axis]):
            slices[self.axis] = slice(i, i + 1)
            out.append(A[tuple(slices)].compact().reshape(tuple(new_shape)))
        return tuple(out)
        # slices = [slice(0, A.shape[i], 1) if i != self.axis else 0 for i in range(len(A.shape))]
        # tensors = []
        # new_shape = tuple([A.shape[s] for s in range(len(A.shape)) if s != self.axis])
        # for i in range(A.shape[self.axis]):
        #     slices[self.axis] = i
        #     tensors.append(A[tuple(slices)].reshape(new_shape))
        # return tuple(tensors)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        # END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        # END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # for ax in self.axes:
        #     if ax >= len(a.shape):
        #         return a
        #
        # new_shape = list(a.shape)
        # for ax in self.axes:
        #     new_shape[ax] += self.dilation * new_shape[ax]
        # # NOTE device
        # ret = init.zeros(*new_shape, device=a.device)
        # slices = [
        #     # NOTE +1
        #     slice(0, new_shape[ax], self.dilation+1) if ax in self.axes
        #     else slice(0, new_shape[ax], 1)
        #     for ax in range(len(a.shape))
        # ]
        # ret.cached_data[tuple(slices)] = a
        # return ret.cached_data
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] *= self.dilation + 1
        new_shape = tuple(new_shape)
        out = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(0, s) for s in new_shape]
        for axis in self.axes:
            slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        out[tuple(slices)] = a
        return out
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        # slices = [
        #     slice(0, a.shape[ax], self.dilation+1) if ax in self.axes
        #     else slice(0, a.shape[ax])
        #     for ax in range(len(a.shape))
        # ]
        # return a[tuple(slices)]
        slices = [slice(0, s) for s in a.shape]
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        out = a[tuple(slices)]
        return out
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # BEGIN YOUR SOLUTION
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        # NOTE stride trick https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb
        # NOTE why must compact?
        A = A.as_strided((N, H - K + 1, W - K + 1, K, K, C_in), (Ns, Hs, Ws, Hs, Ws, Cs)).compact()
        A = A.reshape((N * (H - K + 1) * (W - K + 1), K * K * C_in))
        out = A @ (B.reshape((K * K * C_in, C_out)))
        return out.reshape((N, H - K + 1, W - K + 1, C_out))[:, :: self.stride, :: self.stride, :]
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # X, Weight = node.inputs[0], node.inputs[1]
        # '''
        # out_grad N, H-K+1+2*p, W-K+1+2*p, C_out
        # X N, H, W, C_in
        # W K, K, C_in, C_out
        # '''
        # _, H, W, _ = X.shape
        # K = Weight.shape[0]
        # W_flip = flip(Weight, (0, 1)).transpose((2, 3))
        # if self.stride > 1:
        #     out_grad = dilate(out_grad, (1, 2), self.stride-1)
        # dX = conv(out_grad, W_flip, padding=K-1)
        # # dX = Tensor(dX.cached_data[:, K-1:H+K-1, K-1:W+K-1, :])
        # # NOTE: begin with self.padding!
        # # NOTE: device
        # dX = Tensor(dX.cached_data[:, self.padding:H+self.padding, self.padding:W +
        #             self.padding, :], device=dX.device, dtype=dX.dtype)
        # '''
        # X, out_grad
        # C_in, H, W, N
        # H-K+1+2*p, W-K+1+2*p, N, C_out
        # C_in, K, K, C_out
        # '''
        # X = X.transpose((0, 3))
        # out_grad = out_grad.transpose((0, 2)).transpose((0, 1))
        # dW = conv(X, out_grad, padding=self.padding)
        # dW = dW.transpose((0, 2)).transpose((0, 1))
        # return dX, dW
        X, W = node.inputs
        K, _, _, _ = W.shape

        # out_grad: # N * ((H-K+1+2P)/self.stride) * ((W-K+1+2P)/self.stride) * C_out

        out_grad = dilate(out_grad, (1, 2), self.stride - 1)  # N * (H-K+1+2P) * (W-K+1+2P) * C_out
        W_T = transpose(flip(W, (0, 1)), (2, 3))  # K * K * C_out * C_in
        X_grad = conv(out_grad, W_T, padding=K - 1 - self.padding)

        # The gradients of W must be accumulated over the batches.
        # Consider turning batches into channels to make the conv operator itself do this accumulation.
        X_permute = transpose(X, (0, 3))  # C_in * H * W * N
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))  # (H-K+1+2P) * (W-K+1+2P) * N * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding)  # C_in * H * W * C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))  # K * K * C_in * C_out

        return X_grad, W_grad
        # END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
