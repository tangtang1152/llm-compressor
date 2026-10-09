"""
Core components for the FlatQuant transform: Kronecker-factored learnable
transformation matrices and fake quantizers used during calibration training.

The transformation ``A`` applied to activations is factored as
``A = kron(L, R) * diag(d)``, where ``L`` and ``R`` are orthogonal
(Cayley-parameterized) or directly-learnable invertible matrices and ``d``
is a learnable diagonal. Applying ``A`` to the input of a linear layer and
absorbing ``A^-T`` into its weights leaves the layer output unchanged, while
reshaping the input distribution so that it is easier to quantize.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "ActivationQuantizer",
    "DiagonalTransMatrix",
    "GeneralMatrixTrans",
    "InvSingleTransMatrix",
    "SVDSingleTransMatrix",
    "SingleTransMatrix",
    "WeightQuantizer",
    "get_decompose_dim",
    "get_init_scale",
    "get_init_orthogonal",
    "invert_matrix",
    "round_ste",
]


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def get_init_scale(
    w_max: torch.Tensor, x_max: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """
    Compute the initialization for the learnable diagonal of a transform,
    interpolating between weight magnitudes (alpha=0) and activation
    magnitudes (alpha=1).
    """
    return (w_max.pow(1 - alpha) / x_max.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n: int) -> tuple[int, int] | None:
    """
    Factor ``n`` as ``n = (a - b) * (a + b) = a^2 - b^2`` such that the two
    factors are as close as possible to ``sqrt(n)``. This decomposition is
    what makes the Kronecker product ``kron(L, R)`` have size ``n`` with two
    small matrices.

    A solution exists iff ``n`` is odd or divisible by 4. Returns None when
    no factorization exists (``n == 2 (mod 4)``) so callers may skip the
    mapping rather than looping forever.

    :param n: dimension of the transformation to decompose
    :return: (left_size, right_size) with left * right == n, or None
    """
    if n <= 0:
        return None
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    # bounded search: a solution, when it exists, has a <= (n + 1) // 2
    limit = max(2 * math.isqrt(n) + 16, (n + 1) // 2)
    while a <= limit:
        tmp = a * a - n
        b = math.isqrt(tmp)
        if b * b == tmp:
            return a - b, a + b
        a += 1
    return None


def get_init_orthogonal(size: int) -> torch.Tensor:
    """
    Draw a random orthogonal matrix via QR decomposition with sign correction
    so that the result is uniformly distributed (Haar measure).
    """
    h = torch.randn(size, size)
    q, r = torch.linalg.qr(h)
    # correct signs so that diagonal of r is positive
    d = torch.sign(torch.diagonal(r))
    d[d == 0] = 1.0
    return q * d.unsqueeze(0)


def invert_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    Invert a matrix in float64 for numerical stability, returning the result
    in the original dtype and on the original device.
    """
    dtype = matrix.dtype
    device = matrix.device
    return matrix.to(torch.float64).inverse().to(dtype=dtype, device=device)


class SingleTransMatrix(nn.Module):
    """
    Base class for a single transformation matrix.
    This matrix can be applied from the left or right to an input tensor.

    :param size: size of the square transformation matrix
    :param direction: "right" applies the matrix to the last dim of the
        input (x @ M); "left" applies it to the second-to-last dim
        (M.T @ x reshaped to (-1, size, n // size))
    """

    def __init__(self, size: int, direction: str = "right"):
        super().__init__()
        if direction not in ("right", "left"):
            raise ValueError(f"Invalid direction: {direction}")
        self.size = size
        self.direction = direction
        self._eval_mode = False
        # buffers populated by reparameterize() for use in eval mode
        self.register_buffer("matrix", torch.empty(0))
        self.register_buffer("matrix_inv_t", torch.empty(0))

    def get_matrix(self, inv_t: bool = False) -> torch.Tensor:
        """
        Return the transformation matrix, or its inverse-transpose if
        ``inv_t`` is True. Must be implemented by subclasses.
        """
        raise NotImplementedError

    def reparameterize(self):
        """
        Freeze any learnable parameterization of the matrix into the
        ``matrix`` and ``matrix_inv_t`` buffers. Must be implemented by
        subclasses.
        """
        raise NotImplementedError

    def to_eval_mode(self):
        """
        Switch the module to evaluation mode, freezing the matrix into
        buffers and deleting learnable parameters.
        """
        if not self._eval_mode:
            with torch.no_grad():
                self.reparameterize()
            self._eval_mode = True

    def forward(self, inp: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        """
        Apply the transformation matrix to the input tensor.

        :param inp: input tensor whose last dim(s) match the matrix shape
        :param inv_t: if True, apply the inverse-transpose of the matrix
        """
        if self.direction == "right":
            init_shape = inp.shape
            matrix = self.get_matrix(inv_t=inv_t).to(inp)
            output = inp.reshape(-1, matrix.shape[0]) @ matrix
            return output.reshape(init_shape)
        else:
            if self.size == 0:
                raise ValueError("size can not be zero.")
            init_shape = inp.shape
            matrix = self.get_matrix(inv_t=inv_t).T.to(inp)
            inp = inp.reshape(-1, self.size, init_shape[-1] // self.size)
            output = matrix @ inp
            return output.reshape(init_shape)

    def get_save_params(self) -> dict:
        return {self.direction + "_trans": self.get_matrix()}


class SVDSingleTransMatrix(SingleTransMatrix):
    """
    A single transformation matrix parameterized by SVD:
    ``M = U @ diag(S) @ V.T`` where U and V are orthogonal matrices
    (Cayley parameterization) and S is a vector of singular values which
    are kept positive with a softplus activation when ``diag_relu`` is set.
    """

    def __init__(self, size: int, direction: str = "right", diag_relu: bool = False):
        super().__init__(size, direction)
        self.linear_u = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            self.linear_u.weight.copy_(get_init_orthogonal(size))
        self.linear_u = nn.utils.parametrizations.orthogonal(
            self.linear_u, orthogonal_map="cayley", use_trivialization=False
        )

        self.linear_v = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            self.linear_v.weight.copy_(get_init_orthogonal(size))
        self.linear_v = nn.utils.parametrizations.orthogonal(
            self.linear_v, orthogonal_map="cayley", use_trivialization=False
        )

        if diag_relu:
            beta = 1.0
            init_diag = math.log(math.exp(beta) - 1.0) / beta
            self.linear_diag = nn.Parameter(
                init_diag * torch.ones(size), requires_grad=True
            )
            self._diag_relu = nn.Softplus(beta=beta, threshold=20)
        else:
            self.linear_diag = nn.Parameter(torch.ones(size), requires_grad=True)
            self._diag_relu = None

    def get_diag(self) -> torch.Tensor:
        if self._diag_relu is not None:
            return self._diag_relu(self.linear_diag)
        return self.linear_diag

    def get_matrix(self, inv_t: bool = False) -> torch.Tensor:
        if not self._eval_mode:
            orthog_u, orthog_v = self.linear_u.weight, self.linear_v.weight
            linear_diag = self.get_diag()
            orthog_u = orthog_u.to(linear_diag)
            orthog_v = orthog_v.to(linear_diag)
            if inv_t:
                # (U diag(S) V.T)^-T = U diag(1/S) V.T
                linear_diag = 1 / linear_diag
            return orthog_u @ torch.diag(linear_diag) @ orthog_v.t()
        if inv_t:
            return self.matrix_inv_t
        return self.matrix

    def reparameterize(self):
        self.matrix = self.get_matrix()
        self.matrix_inv_t = self.get_matrix(inv_t=True)
        del self.linear_u
        del self.linear_diag
        del self.linear_v


class InvSingleTransMatrix(SingleTransMatrix):
    """
    A single transformation matrix which is directly learnable (the matrix
    itself is the parameter of an inner Linear layer).
    """

    def __init__(self, size: int, direction: str = "right", **kwargs):
        super().__init__(size, direction)
        trans_linear = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            trans_linear.weight.copy_(get_init_orthogonal(size))
        self.trans_linear = trans_linear

    def get_matrix(self, inv_t: bool = False) -> torch.Tensor:
        if not self._eval_mode:
            matrix = self.trans_linear.weight
            if inv_t:
                matrix = invert_matrix(matrix).T
            return matrix
        if inv_t:
            return self.matrix_inv_t
        return self.matrix

    def reparameterize(self):
        self.matrix = self.trans_linear.weight
        self.matrix_inv_t = invert_matrix(self.trans_linear.weight).T
        del self.trans_linear


class DiagonalTransMatrix(nn.Module):
    """
    A purely diagonal transformation. During calibration the diagonal is
    learnable; after ``to_eval_mode`` it is set to None (expected to have
    been absorbed into the weights of a preceding module).
    """

    def __init__(self, size: int, init_para: torch.Tensor | None = None):
        super().__init__()
        self.size = size
        if init_para is None:
            self.diag_scale = nn.Parameter(torch.ones(size), requires_grad=True)
        else:
            self.diag_scale = nn.Parameter(init_para, requires_grad=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

    def forward(self, inp: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.diag_scale is None:
            return inp
        if inv_t:
            return inp / self.diag_scale.to(inp)
        return inp * self.diag_scale.to(inp)

    def reparameterize(self):
        # diagonal is absorbed into preceding weights by the caller; the
        # online transform no longer applies it
        self.diag_scale = None

    def to_eval_mode(self):
        self.reparameterize()

    def get_save_params(self) -> dict:
        return {}


class GeneralMatrixTrans(nn.Module):
    """
    A general transformation of the form
    ``left_trans(right_trans(diag_trans(x)))``, i.e. mathematically
    ``A = kron(L, R) * diag(d)`` acting on the flattened input.

    :param left_size: size of the left (Kronecker) factor, applied to the
        second-to-last dim of the input
    :param right_size: size of the right (Kronecker) factor, applied to the
        last dim of the input
    :param add_diag: whether to include a learnable diagonal transform
    :param diag_init_para: optional initialization for the diagonal
    :param tran_type: "svd" for orthogonal SVD parameterization of the
        factors, "inv" for a directly learnable invertible matrix
    :param diag_relu: use softplus on the SVD singular values to keep them
        positive
    """

    def __init__(
        self,
        left_size: int,
        right_size: int,
        add_diag: bool = False,
        diag_init_para: torch.Tensor | None = None,
        tran_type: str = "svd",
        diag_relu: bool = False,
    ):
        super().__init__()
        TranMatrix = (
            SVDSingleTransMatrix if tran_type == "svd" else InvSingleTransMatrix
        )
        self.left_trans = TranMatrix(left_size, direction="left", diag_relu=diag_relu)
        self.right_trans = TranMatrix(
            right_size, direction="right", diag_relu=diag_relu
        )

        if add_diag:
            self.diag_trans = DiagonalTransMatrix(
                left_size * right_size, diag_init_para
            )
        else:
            self.diag_trans = None

    def forward(self, inp: torch.Tensor, inv_t: bool = False) -> torch.Tensor:
        if self.diag_trans is not None:
            inp = self.diag_trans(inp, inv_t=inv_t)
        if self.right_trans is not None:
            inp = self.right_trans(inp, inv_t=inv_t)
        if self.left_trans is not None:
            inp = self.left_trans(inp, inv_t=inv_t)
        return inp

    def to_eval_mode(self):
        self.left_trans.to_eval_mode()
        self.right_trans.to_eval_mode()
        if self.diag_trans is not None:
            self.diag_trans.to_eval_mode()

    def get_save_params(self) -> dict:
        return {
            **self.left_trans.get_save_params(),
            **self.right_trans.get_save_params(),
        }


def _get_qmin_qmax(bits: int, sym: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max, q_min = torch.tensor(2**bits - 1), torch.tensor(0)
    return q_max, q_min


def _get_maxq(bits: int, sym: bool) -> torch.Tensor:
    if sym:
        return torch.tensor(2 ** (bits - 1) - 1)
    return torch.tensor(2**bits - 1)


def _sym_quant(x: torch.Tensor, scale: torch.Tensor, bits: int, is_signed: bool = True):
    scale = scale.to(x.device)
    if is_signed:
        q_min, q_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        q_min, q_max = 0, 2**bits - 1
    q = torch.clamp(round_ste(x / scale), q_min, q_max)
    return q, scale


def _sym_dequant(q: torch.Tensor, scale: torch.Tensor):
    return scale * q


def _sym_quant_dequant(x: torch.Tensor, scale: torch.Tensor, bits: int, is_signed=True):
    return _sym_dequant(*_sym_quant(x, scale, bits, is_signed))


def _asym_quant(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, is_signed=True
):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    if is_signed:
        q_min, q_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        q_min, q_max = 0, 2**bits - 1
    q = torch.clamp(round_ste(x / scale) + zero, q_min, q_max)
    return q, scale, zero


def _asym_dequant(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    return scale * (q - zero)


def _asym_quant_dequant(
    x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, is_signed=True
):
    return _asym_dequant(*_asym_quant(x, scale, zero, bits, is_signed))


class ActivationQuantizer(nn.Module):
    """
    Fake quantizer for activations, supporting symmetric or asymmetric
    per-token (dynamic) quantization. When ``lac`` (learnable activation
    clipping) is enabled, a single learnable clip factor scales the min/max
    of each token before computing scales.

    :param bits: number of quantization bits (16 disables quantization)
    :param sym: use symmetric quantization
    :param lac: enable learnable activation clipping
    :param is_signed: quantized integer range includes negatives
    """

    def __init__(self, bits: int, sym: bool = False, lac: bool = False, is_signed=True):
        super().__init__()
        self.bits = bits
        self.q_max, self.q_min = _get_qmin_qmax(bits, sym)
        self.sym = sym
        self.lac = lac
        self.is_signed = is_signed
        if self.lac:
            init_value = 4.0
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor = torch.nn.Parameter(
                torch.ones((1,)) * init_value, requires_grad=True
            )
        self.enable = True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bits={self.bits}, sym={self.sym}, "
            f"lac={self.lac}, is_signed={self.is_signed})"
        )

    def reparameterize(self):
        """
        Freeze the learnable clip factor into a buffer so that it no longer
        appears in model.parameters().
        """
        if self.lac and isinstance(self.clip_factor, torch.nn.Parameter):
            clip_factor = self.clip_factor
            del self.clip_factor
            self.register_buffer("clip_factor", clip_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits == 16 or not self.enable:
            return x
        return self.fake_quant(x)

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return _sym_quant_dequant(x, scale, self.bits, is_signed=self.is_signed).to(
                x_dtype
            )
        return _asym_quant_dequant(
            x, scale, zero, self.bits, is_signed=self.is_signed
        ).to(x_dtype)

    def get_clip_ratio(self) -> torch.Tensor:
        if self.lac:
            return self.sigmoid(self.clip_factor)
        return None

    def get_scale_zero(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_max = self.q_max.to(x)
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

        if self.lac:
            clip_ratio = self.sigmoid(self.clip_factor).to(xmax)
            xmax = xmax * clip_ratio
            xmin = xmin * clip_ratio

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            scale = xmax / q_max
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)
            if self.is_signed:
                zero = zero - 2 ** (self.bits - 1)
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero


class WeightQuantizer(nn.Module):
    """
    Fake quantizer for weights, supporting symmetric or asymmetric
    per-channel quantization. When ``lwc`` (learnable weight clipping) is
    enabled, the effective min/max of each output channel are scaled by
    learnable clip factors prior to scale computation.

    :param in_size: number of input features of the wrapped linear layer
    :param out_size: number of output features of the wrapped linear layer
    :param bits: number of quantization bits (16 disables quantization)
    :param sym: use symmetric quantization
    :param lwc: enable learnable weight clipping
    :param is_signed: quantized integer range includes negatives
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        bits: int = 8,
        sym: bool = True,
        lwc: bool = False,
        is_signed: bool = True,
    ):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(out_size, 1))
        self.register_buffer("zero", torch.zeros(out_size, 1))
        self.in_size = in_size
        self.out_size = out_size
        self.enable = True
        self.enable_find = True
        self.bits = bits
        self.sym = sym
        self.maxq = _get_maxq(self.bits, self.sym)
        self.lwc = lwc
        self.is_signed = is_signed

        if self.lwc:
            init_value = 4.0
            self.clip_factor_w_max = torch.nn.Parameter(
                torch.ones((out_size, 1)) * init_value, requires_grad=True
            )
            self.clip_factor_w_min = torch.nn.Parameter(
                torch.ones((out_size, 1)) * init_value, requires_grad=True
            )
            self.sigmoid = torch.nn.Sigmoid()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(bits={self.bits}, sym={self.sym}, "
            f"lwc={self.lwc}, is_signed={self.is_signed})"
        )

    def reparameterize(self):
        """
        Freeze learnable clip factors into buffers.
        """
        if self.lwc and isinstance(self.clip_factor_w_max, torch.nn.Parameter):
            clip_factor_w_max = self.clip_factor_w_max
            clip_factor_w_min = self.clip_factor_w_min
            del self.clip_factor_w_max
            del self.clip_factor_w_min
            self.register_buffer("clip_factor_w_max", clip_factor_w_max)
            self.register_buffer("clip_factor_w_min", clip_factor_w_min)

    def apply_wclip(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Clamp the weight to the learnable min/max of each output channel.
        """
        wmin, wmax = weight.min(1, keepdim=True)[0], weight.max(1, keepdim=True)[0]
        wmax = wmax * self.sigmoid(self.clip_factor_w_max).to(wmax)
        wmin = wmin * self.sigmoid(self.clip_factor_w_min).to(wmin)
        return torch.clamp(weight, min=wmin, max=wmax)

    def enable_find_params(self, enable: bool = True):
        self.enable_find = enable

    def find_params(self, x: torch.Tensor):
        if self.bits == 16 or not self.enable:
            return
        dev = x.device
        self.maxq = _get_maxq(self.bits, self.sym).to(dev)

        shape = x.shape
        # per-channel statistics over the flattened weight
        x = x.flatten(1)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
            if self.is_signed:
                self.zero = self.zero - 2 ** (self.bits - 1)

        shape_ = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape_)
        self.zero = self.zero.reshape(shape_)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable and self.bits < 16:
            x_dtype = x.dtype
            if self.enable_find:
                self.find_params(x)
            if not self.ready():
                raise ValueError(
                    "WeightQuantizer is not ready, please call find_params first."
                )
            if self.sym:
                return _sym_quant_dequant(
                    x, self.scale, self.bits, self.is_signed
                ).to(x_dtype)
            return _asym_quant_dequant(
                x, self.scale, self.zero, self.bits, self.is_signed
            ).to(x_dtype)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(x)

    def enable_quant(self, enable: bool = True):
        self.enable = enable

    def ready(self) -> bool:
        return torch.all(self.scale != 0)
