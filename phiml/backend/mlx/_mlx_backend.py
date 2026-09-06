
from typing import TypeVar, Union, Optional, Callable, Sequence, Tuple, Any, override
import numbers
import numpy as np
from numpy import ndarray
import mlx.core as mx
import mlx.nn as nn

from phiml.backend._dtype import DType
from .. import Backend, NUMPY, ComputeDevice, ML_LOGGER

TensorType = TypeVar('TensorType')
TensorOrArray = Union[TensorType, ndarray]


class MLXBackend(Backend):

    def __init__(self):
        cpu = NUMPY.get_default_device()
        self.devices = [ComputeDevice(self, "CPU", 'CPU', cpu.memory, cpu.processor_count, cpu.description, ref='cpu')]
        Backend.__init__(self, 'mlx', self.devices, self.devices[-1])

    sqrt = mx.sqrt
    exp = mx.exp
    erf = mx.erf
    sin = mx.sin
    arcsin = mx.arcsin
    cos = mx.cos
    arccos = mx.arccos
    tan = mx.tan
    arctan = mx.arctan
    sinh = mx.sinh
    arcsinh = mx.arcsinh
    cosh = mx.cosh
    arccosh = mx.arccosh
    tanh = mx.tanh
    arctanh = mx.arctanh
    log = mx.log
    log2 = mx.log2
    log10 = mx.log10
    sigmoid = mx.sigmoid
    isfinite = mx.isfinite
    isnan = mx.isnan
    isinf = mx.isinf
    abs = mx.abs
    sign = mx.sign
    round = mx.round
    ceil = mx.ceil
    floor = mx.floor
    log_gamma = nn.linear.math.lgamma

    @override
    def prefers_channels_last(self) -> bool:
        return False

    @override
    def nn_library(self):
        from . import nets
        return nets

    @override
    def get_device(self, tensor: TensorType) -> ComputeDevice:
        return self.devices[0]

    @override
    def allocate_on_device(self, tensor: TensorType, device: ComputeDevice) -> TensorType:
        return tensor

    @override
    def seed(self, seed: int):
        mx.random.seed(seed)

    @override
    def is_module(self, obj) -> bool:
        return False

    @override
    def is_tensor(self, x, only_native=False):
        if isinstance(x, nn.Module):
            return True
        if isinstance(x, mx.array):
            return True
        if only_native:
            return False 
        if isinstance(x, (numbers.Number, np.bool_)):
            return True
        if isinstance(x, np.ndarray) and x.dtype != np.object_:
            return True
        return False

    @override
    def is_sparse(self, x: mx.array) -> bool:
        return x.diag is not None

    @override
    def get_sparse_format(self, x) -> str:
        return None

    @override
    def disassemble(self, x) -> Tuple[Callable[..., Any], Sequence[TensorType]]:
        return None, []

    @override
    def as_tensor(self, x, convert_external=True):
        if isinstance(x, mx.array):
            return x
        if isinstance(x, np.ndarray):
            return mx.array(x)
        if isinstance(x, (numbers.Number, np.bool_)):
            return mx.array(x)
        raise ValueError(f"Cannot convert {x} to a tensor")

    @override
    def is_available(self, tensor) -> bool:
        return True

    @override
    def numpy(self, tensor) -> ndarray:
        return np.array(tensor)

    @override
    def to_dlpack(self, tensor):
        return None

    @override
    def from_dlpack(self, capsule):
        return mx.array(capsule)

    @override
    def copy(self, tensor, only_mutable=False):
        return tensor

    @override
    def jacobian(self, f: Callable[..., Any], wrt: Tuple | list, get_output: bool, is_f_scalar: bool):
        return mx.jvp(f, wrt, get_output)

    @override
    def hessian(self, f: Callable[..., Any], wrt: Tuple | list, get_output: bool, get_gradient: bool) -> Tuple:
        return None

    @override
    def custom_gradient(self, f: Callable[..., Any], gradient: Callable[..., Any], get_external_cache: Callable[..., Any] = None, on_call_skipped: Callable[..., Any] = None) -> Callable[..., Any]:
        return f

    @override
    def jit_compile_grad(self, f: Callable[..., Any], wrt: Tuple | list, get_output: bool, is_f_scalar: bool):
        return None

    @override
    def jit_compile_hessian(self, f: Callable[..., Any], wrt: Tuple | list, get_output: bool, get_gradient: bool):
        return None

    @override
    def transpose(self, tensor, axes):
        return tensor

    @override
    def random_uniform(self, shape, low, high, dtype: DType | None):
        return None

    @override
    def random_normal(self, shape, dtype: DType):
        return None

    @override
    def concat(self, values, axis):
        return None

    @override
    def pad(self, value, pad_width, mode: str = 'constant', constant_values=0):
        return None

    @override
    def reshape(self, value, shape):
        return None

    @override
    def sum(self, value, axis=None, keepdims=False):
        return None

    @override
    def prod(self, value, axis=None):
        return None

    @override
    def divide_no_nan(self, x, y):
        return None

    @override
    def where(self, condition, x=None, y=None):
        return None

    @override
    def nonzero(self, values, length=None, fill_value=-1):
        return None

    @override
    def mean(self, value, axis=None, keepdims=False):
        return None

    @override
    def range(self, start, limit=None, delta=1, dtype: DType = ...):
        return None

    @override
    def zeros(self, shape, dtype: DType = None):
        return None

    @override
    def zeros_like(self, tensor):
        return None

    @override
    def ones(self, shape, dtype: DType = None):
        return mx.ones(shape)

    @override
    def ones_like(self, tensor):
        return mx.ones_like(tensor)

    @override
    def meshgrid(self, *coordinates):
        coords = [self.as_tensor(c) for c in coordinates]
        return mx.meshgrid(*coords)

    @override
    def linspace(self, start, stop, number):
        return mx.linspace(start, stop, number)

    @override
    def tensordot(self, a, a_axes: Union[tuple, list], b, b_axes: Union[tuple, list]):
        a, b = self.auto_cast(a, b)
        return mx.tensordot(a, a_axes, b, b_axes)

    @override
    def mul_matrix_batched_vector(self, A, b):
        return None

    @override
    def einsum(self, equation, *tensors):
        return mx.einsum(equation, *tensors)

    @override
    def cumsum(self, x, axis: int):
        return mx.cumsum(x, axis)

    @override
    def abs(self, x):
        return mx.abs(x)

    @override
    def sign(self, x):
        return mx.sign(x)

    @override
    def round(self, x):
        raise mx.round(x)

    @override
    def ceil(self, x):
        return mx.ceil(x)

    @override
    def floor(self, x):
        return mx.floor(x)

    @override
    def max(self, x, axis=None, keepdims=False):
        return mx.max(x, axis, keepdims)

    @override
    def min(self, x, axis=None, keepdims=False):
        return mx.min(x, axis, keepdims)

    @override
    def maximum(self, a, b):
        return mx.maximum(a, b)

    @override
    def minimum(self, a, b):
        return mx.minimum(a, b)

    @override
    def clip(self, x, minimum, maximum):
        return mx.clip(x, minimum, maximum)

    @override
    def argmax(self, x, axis: int, keepdims=False):
        return mx.argmax(x, axis, keepdims)

    @override
    def argmin(self, x, axis: int, keepdims=False):
        return mx.argmin(x, axis, keepdims)

    @override
    def sqrt(self, x):
        return mx.sqrt(x)

    @override
    def exp(self, x):
        return mx.exp(x)

    @override
    def erf(self, x):
        return mx.erf(x)

    @override
    def softplus(self, x):
        return nn.softplus(x)

    @override
    def log_gamma(self, x):
        return nn.log

    @override
    def gamma_inc_l(self, a, x):
        return None

    @override
    def gamma_inc_u(self, a, x):
        return None

    @override
    def conv(self, value, kernel, zero_padding=True):
        return None

    @override
    def expand_dims(self, a, axis=0, number=1):
        return None

    @override
    def shape(self, tensor):
        return None

    @override
    def staticshape(self, tensor) -> tuple:
        if isinstance(tensor, ndarray):
            return tensor.shape
        if isinstance(tensor, mx.array):
            return tensor.shape
        else:
            return NUMPY.staticshape(tensor)

    @override
    def cast(self, x, dtype: DType):
        return None

    @override
    def std(self, x, axis=None, keepdims=False):
        return None

    @override
    def boolean_mask(self, x, mask, axis=0, new_length=None, fill_value=0):
        return None

    @override
    def isfinite(self, x):
        return None

    @override
    def isnan(self, x):
        return None

    @override
    def isinf(self, x):
        return None

    @override
    def scatter(self, base_grid, indices, values, mode: str):
        return None

    @override
    def histogram1d(self, values, weights, bin_edges):
        return None

    @override
    def bincount(self, x, weights: Optional[TensorType], bins: int, x_sorted=False):
        return None

    @override
    def unique(self, x: TensorType, return_inverse: bool, return_counts: bool, axis: int) -> Tuple[TensorType, ...]:
        return None

    @override
    def any(self, boolean_tensor, axis=None, keepdims=False):
        return None

    @override
    def all(self, boolean_tensor, axis=None, keepdims=False):
        return None

    @override
    def quantile(self, x, quantiles):
        return None

    @override
    def argsort(self, x, axis=-1):
        return None

    @override
    def sort(self, x, axis=-1):
        return None

    @override
    def searchsorted(self, sorted_sequence, search_values, side: str, dtype=DType(int, 32)):
        return None

    @override
    def fft(self, x, axes: Union[tuple, list]):
        return None

    @override
    def ifft(self, k, axes: Union[tuple, list]):
        return None

    @override
    def imag(self, x):
        return None

    @override
    def real(self, x):
        return None

    @override
    def conj(self, x):
        return None

    @override
    def sin(self, x):
        return None

    @override
    def arcsin(self, x):
        return None

    @override
    def cos(self, x):
        return None

    @override
    def arccos(self, x):
        return None

    @override
    def tan(self, x):
        return None

    @override
    def arctan(self, x):
        return None

    @override
    def arctan2(self, y, x):
        return None

    @override
    def sinh(self, x):
        return None

    @override
    def arcsinh(self, x):
        return None

    def cosh(self, x):
        return None

    @override
    def arccosh(self, x):
        return None

    @override
    def tanh(self, x):
        return None

    @override
    def arctanh(self, x):
        return None

    @override
    def log(self, x):
        return None

    @override
    def log2(self, x):
        return None

    @override
    def log10(self, x):
        raise NotImplementedError(self)

    @override
    def sigmoid(self, x):
        return 1 / (1 + self.exp(-x))

    @override
    def dtype(self, array) -> DType:
        return None

    @override
    def tile(self, value, multiples):
        return None

    @override
    def repeat(self, x, repeats, axis: int, new_length=None):
        return None

    @override
    def get_diagonal(self, matrices, offset=0):
        return None

    @override
    def indexed_segment_sum(self, x, indices, axis: int):
        return None

    @override
    def sparse_coo_tensor(self, indices: TensorType, values: TensorType, shape: tuple):
        return None

    @override
    def sparse_coo_tensor_batched(self, indices: Union[tuple, list], values, shape: tuple):
        return None

    @override
    def csr_matrix(self, column_indices: TensorOrArray, row_pointers: TensorOrArray, values: TensorOrArray, shape: Tuple[int, int]):
        return None

    @override
    def csr_matrix_batched(self, column_indices, row_pointers, values, shape: Tuple[int, int]):
        return None

    @override
    def csc_matrix(self, column_pointers, row_indices, values, shape: Tuple[int, int]):
        return None

    @override
    def csc_matrix_batched(self, column_pointers, row_indices, values, shape: Tuple[int, int]):
        return None

    @override
    def matrix_solve_least_squares(self, matrix: TensorType, rhs: TensorType) -> Tuple[TensorType]:
        return None

    @override
    def solve_triangular_dense(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        return None

    @override
    def solve_triangular_sparse(self, matrix, rhs, lower: bool, unit_diagonal: bool):
        return None

    @override
    def matrix_rank_dense(self, matrix, hermitian=False) -> TensorType:
        return None

    @override
    def eigvals(self, matrix: TensorType) -> TensorType:
        return None

    @override
    def eig(self, matrix: TensorType) -> TensorType:
        return None

    @override
    def stop_gradient(self, value):
        return None

    @override
    def grid_sample(self, grid, coordinates, extrapolation: str):
        return None

    @override
    def variable(self, value):
        return None

    @override
    def equal(self, x, y):
        return None
