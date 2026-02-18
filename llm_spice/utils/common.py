import copy
from dataclasses import asdict, dataclass
import math
import os
import random
import numpy as np
from typing import ClassVar, List, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_spice.op.operators import BaseOp

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class Si:
    f = 1e-15
    p = 1e-12
    n = 1e-9
    u = 1e-6
    m = 1e-3
    K = 1e3
    M = 1e6
    G = 1e9
    T = 1e12
    P = 1e15

    Ki = 1 << 10
    Mi = 1 << 20
    Gi = 1 << 30
    Ti = 1 << 40


class DataType:
    FP4: ClassVar["DataType"]
    FP8: ClassVar["DataType"]
    FP16: ClassVar["DataType"]
    BF16: ClassVar["DataType"]
    FP32: ClassVar["DataType"]
    FP64: ClassVar["DataType"]

    def __init__(self, name: str, size_in_bits: int):
        self.name = name
        self.size_in_bits = size_in_bits
        self.size_in_bytes = float(size_in_bits) / 8
        self.itemsize = self.size_in_bytes

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name, self.size_in_bits))

    def __eq__(self, other: "DataType"):
        return self.name == other.name and self.size_in_bits == other.size_in_bits

    @staticmethod
    def get_all_dtypes() -> list["DataType"]:
        return [
            DataType.FP4,
            DataType.FP8,
            DataType.FP16,
            DataType.BF16,
            DataType.FP32,
            DataType.FP64,
        ]

    @staticmethod
    def from_str(name: str) -> "DataType":
        return next(dtype for dtype in DataType.get_all_dtypes() if dtype.name == name)


DataType.FP4 = DataType("FP4", 4)
DataType.FP8 = DataType("FP8", 8)
DataType.FP16 = DataType("FP16", 16)
DataType.BF16 = DataType("BF16", 16)
DataType.FP32 = DataType("FP32", 32)
DataType.FP64 = DataType("FP64", 64)


class Tensor:
    def __init__(
        self, shape: tuple[int, ...], dtype: DataType = DataType.BF16, name: str = ""
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.num_elements = int(math.prod(self.shape))
        self.size_in_bytes = int(math.prod(self.shape)) * self.dtype.itemsize
        self.name = name
        self.producer: BaseOp | None = None
        self.consumer: BaseOp | None = None

    def __str__(self) -> str:
        return f"Tensor({self.name}, [{self.shape}], dtype={self.dtype})"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def new_like(other: "Tensor", name: str = "") -> "Tensor":
        return Tensor(other.shape, other.dtype, name)

    def _normalize_axis(self, axis: int, ndim: int) -> int:
        if axis < 0:
            axis += ndim
        if not (0 <= axis < ndim):
            raise ValueError(f"axis {axis} out of bounds for tensor of ndim {ndim}")
        return axis

    def reshape(self, new_shape: tuple[int, ...]) -> "Tensor":
        """Return a new Tensor with the same number of elements and dtype."""
        # Support a single -1 (infer size) like NumPy
        if not new_shape:
            raise ValueError("new_shape must be non-empty")
        infer_count = sum(1 for d in new_shape if d == -1)
        if infer_count > 1:
            raise ValueError("only one dimension can be inferred (-1)")

        old_elems = int(math.prod(self.shape))
        if infer_count == 1:
            known = (
                int(math.prod([d for d in new_shape if d != -1]))
                if any(d != -1 for d in new_shape)
                else 1
            )
            if known == 0 or old_elems % known != 0:
                raise ValueError(
                    f"cannot infer dimension; sizes do not align: {self.shape} -> {new_shape}"
                )
            inferred = old_elems // known
            out_shape = tuple(inferred if d == -1 else d for d in new_shape)
        else:
            out_shape = tuple(new_shape)
            if int(math.prod(out_shape)) != old_elems:
                raise ValueError("cannot reshape to a different number of elements")

        # Reshape does not change dtype or bytes
        out = Tensor(out_shape, self.dtype)
        return out

    def concat(self, other: "Tensor", axis: int = -1) -> "Tensor":
        """Concatenate two tensors along `axis` (dtypes must match; other dims equal)."""
        if self.dtype != other.dtype:
            raise TypeError(f"dtype mismatch: {self.dtype} vs {other.dtype}")

        if len(self.shape) != len(other.shape):
            raise ValueError("rank mismatch in concat")

        ax = self._normalize_axis(axis, len(self.shape))
        for i, (a, b) in enumerate(zip(self.shape, other.shape)):
            if i != ax and a != b:
                raise ValueError(f"non-concat dimension {i} mismatch: {a} vs {b}")

        new_dim = self.shape[ax] + other.shape[ax]
        new_shape = list(self.shape)
        new_shape[ax] = new_dim
        return Tensor(tuple(new_shape), self.dtype)

    def split(self, splits: List[int] | int, axis: int = -1) -> list["Tensor"]:
        """Split tensor along `axis` by sizes (list) or into equal chunks (int)."""
        ax = self._normalize_axis(axis, len(self.shape))
        dim = self.shape[ax]

        if isinstance(splits, int):
            if splits <= 0:
                raise ValueError("number of splits must be positive")
            if dim % splits != 0:
                raise ValueError(
                    f"dimension {dim} not divisible into {splits} equal parts"
                )
            part = dim // splits
            sizes = [part] * splits
        else:
            if not splits:
                raise ValueError("splits list must be non-empty")
            if any(s <= 0 for s in splits):
                raise ValueError("all split sizes must be positive")
            if sum(splits) != dim:
                raise ValueError(f"sum(splits)={sum(splits)} != dimension {dim}")
            sizes = list(splits)

        out = []
        for s in sizes:
            new_shape = list(self.shape)
            new_shape[ax] = s
            out.append(Tensor(tuple(new_shape), self.dtype))
        return out

    def transpose(self, axes: tuple[int, ...]) -> "Tensor":
        """Transpose the tensor according to the given axes.

        - Accepts negative axes (normalized to [0, ndim)).
        - Requires `axes` to be a permutation of range(ndim).
        - If `axes` is None or empty, defaults to reversing all axes.
        """
        ndim = len(self.shape)

        # Allow None/empty (treat as reverse), even though type hints say tuple
        if axes is None or len(axes) == 0:
            axes = tuple(range(ndim - 1, -1, -1))

        if len(axes) < ndim:
            # Fill in missing axes
            axes = tuple(range(ndim - len(axes))) + axes

        norm_axes = tuple(self._normalize_axis(a, ndim) for a in axes)
        if len(set(norm_axes)) != ndim:
            raise ValueError(
                "axes must be a permutation of range(ndim) with no repeats"
            )

        new_shape = tuple(self.shape[i] for i in norm_axes)
        return Tensor(new_shape, self.dtype)

    @property
    def T(self) -> "Tensor":
        """Shorthand for reversing all axes (NumPy-style)."""
        return self.transpose(tuple(range(len(self.shape) - 1, -1, -1)))

    def __getitem__(self, key: Any) -> "Tensor":
        """
        NumPy-like basic indexing:
          - integers remove a dimension (with negative index support)
          - slices compute the resulting length along that axis (respect step)
          - Ellipsis fills with full slices as needed
          - None inserts a new axis of length 1
        Advanced/boolean indexing is intentionally not supported.
        """
        shape = self.shape
        ndim = len(shape)

        # Normalize key to a tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Count and expand Ellipsis
        ellipsis_count = sum(1 for k in key if k is Ellipsis)
        if ellipsis_count > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        if ellipsis_count == 1:
            # number of entries excluding None (newaxis)
            non_none = [k for k in key if k is not None]
            # we will replace Ellipsis with as many full slices as needed
            needed = ndim - (
                len(non_none) - 1
            )  # -1 because Ellipsis itself will be replaced
            if needed < 0:
                # More explicit indices than dimensions (ignoring None)
                raise IndexError("too many indices for tensor")
            expanded = []
            for k in key:
                if k is Ellipsis:
                    expanded.extend([slice(None)] * needed)
                else:
                    expanded.append(k)
            key = tuple(expanded)
        else:
            # No ellipsis: we may need to pad with full slices to reach ndim (ignoring None)
            specified_excl_none = sum(1 for k in key if k is not None)
            if specified_excl_none > ndim:
                raise IndexError("too many indices for tensor")
            pad = ndim - specified_excl_none
            key = tuple(list(key) + [slice(None)] * pad)

        # Now build output shape
        out_shape = []
        dim_ptr = 0  # pointer into original dims

        for k in key:
            if k is None:
                # newaxis
                out_shape.append(1)
                continue

            if dim_ptr >= ndim:
                # indices exceed available dimensions
                raise IndexError("too many indices for tensor")

            dim = shape[dim_ptr]

            if isinstance(k, int):
                # normalize negative
                if k < 0:
                    k += dim
                if not (0 <= k < dim):
                    raise IndexError(
                        f"index {k} is out of bounds for axis with size {dim}"
                    )
                # integer indexing removes this dimension
                dim_ptr += 1
                continue

            if isinstance(k, slice):
                # compute slice length using Python semantics
                start, stop, step = k.indices(dim)
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                if step > 0:
                    length = 0 if start >= stop else ((stop - start + step - 1) // step)
                else:
                    length = (
                        0
                        if start <= stop
                        else ((start - stop + (-step) - 1) // (-step))
                    )
                out_shape.append(length)
                dim_ptr += 1
                continue

            raise TypeError(
                f"unsupported index type {type(k).__name__}; "
                "only int, slice, None (newaxis), and Ellipsis are supported"
            )

        # Any leftover dims? (shouldn’t happen, we padded above, but be safe)
        if dim_ptr < ndim:
            out_shape.extend(shape[dim_ptr:])

        return Tensor(tuple(out_shape), self.dtype)


@dataclass
class OpRunStats:
    duration: float = 0
    compute_time: float = 0
    memory_time: float = 0
    flop: float = 0
    memory_access_bytes: float = 0
    flops_utilization: float = 0
    memory_bw_utilization: float = 0
    arithmetic_intensity: float = 0
    extra_info: dict[str, Any] | None = None

    def merge(self, other: "OpRunStats") -> "OpRunStats":
        total_duration = self.duration + other.duration
        if total_duration == 0:
            return self
        total_compute_time = self.compute_time + other.compute_time
        total_memory_time = self.memory_time + other.memory_time
        total_flop = self.flop + other.flop
        total_memory_access_bytes = self.memory_access_bytes + other.memory_access_bytes
        total_flops_utilization = total_compute_time / total_duration
        total_memory_bw_utilization = total_memory_time / total_duration
        total_arithmetic_intensity = total_flop / total_memory_access_bytes
        return OpRunStats(
            duration=total_duration,
            compute_time=total_compute_time,
            memory_time=total_memory_time,
            flop=total_flop,
            memory_access_bytes=total_memory_access_bytes,
            flops_utilization=total_flops_utilization,
            memory_bw_utilization=total_memory_bw_utilization,
            arithmetic_intensity=total_arithmetic_intensity,
            extra_info=self.extra_info,
        )

    def pretty_str(self):
        lines = [
            f"Duration: {format_value(self.duration, 's')}\n"
            f"Compute time: {format_value(self.compute_time, 's')}\n"
            f"Memory time: {format_value(self.memory_time, 's')}\n"
            f"Flop: {format_value(self.flop, 'FLOPS')}\n"
            f"Memory access bytes: {format_value(self.memory_access_bytes, 'B')}\n"
            f"Flops utilization: {format_value(self.flops_utilization, '%')}\n"
            f"Memory bw utilization: {format_value(self.memory_bw_utilization, '%')}\n"
            f"Arithmetic intensity: {format_value(self.arithmetic_intensity, 'FLOPS/B')}"
        ]
        return "\n".join(lines)

    @staticmethod
    def attn_interpolate(
        first: "OpRunStats", last: "OpRunStats", input_tokens: int, output_tokens: int
    ) -> "OpRunStats":
        def interpolate_sum(x0, x1, y0, y1) -> float:
            if x0 == x1:
                # Only one point; sum==mean==that point
                return float(y0)
            n = x1 - x0 + 1
            # Sum over arithmetic progression: y(x0) + ... + y(x1)
            # First term t0 = y0, last term t1 = y1 (since it's linear on endpoints)
            total = (y0 + y1) * n / 2.0
            return total

        total_context_tokens = input_tokens + output_tokens - 1
        assert input_tokens <= total_context_tokens
        return OpRunStats(
            duration=interpolate_sum(
                input_tokens, total_context_tokens, first.duration, last.duration
            ),
            compute_time=interpolate_sum(
                input_tokens,
                total_context_tokens,
                first.compute_time,
                last.compute_time,
            ),
            memory_time=interpolate_sum(
                input_tokens, total_context_tokens, first.memory_time, last.memory_time
            ),
            flop=interpolate_sum(
                input_tokens, total_context_tokens, first.flop, last.flop
            ),
            memory_access_bytes=interpolate_sum(
                input_tokens,
                total_context_tokens,
                first.memory_access_bytes,
                last.memory_access_bytes,
            ),
            flops_utilization=interpolate_sum(
                input_tokens,
                total_context_tokens,
                first.flops_utilization,
                last.flops_utilization,
            )
            / output_tokens,
            memory_bw_utilization=interpolate_sum(
                input_tokens,
                total_context_tokens,
                first.memory_bw_utilization,
                last.memory_bw_utilization,
            )
            / output_tokens,
            arithmetic_intensity=interpolate_sum(
                input_tokens,
                total_context_tokens,
                first.arithmetic_intensity,
                last.arithmetic_intensity,
            )
            / output_tokens,
            extra_info=first.extra_info,
        )

    @staticmethod
    def linear_interpolate(
        first: "OpRunStats", last: "OpRunStats", output_tokens: int
    ) -> "OpRunStats":
        return OpRunStats(
            duration=(first.duration + last.duration) / 2 * output_tokens,
            compute_time=(first.compute_time + last.compute_time) / 2 * output_tokens,
            memory_time=(first.memory_time + last.memory_time) / 2 * output_tokens,
            flop=(first.flop + last.flop) / 2 * output_tokens,
            memory_access_bytes=(first.memory_access_bytes + last.memory_access_bytes)
            / 2
            * output_tokens,
            flops_utilization=(first.flops_utilization + last.flops_utilization) / 2,
            memory_bw_utilization=(
                first.memory_bw_utilization + last.memory_bw_utilization
            )
            / 2,
            arithmetic_intensity=(
                first.arithmetic_intensity + last.arithmetic_intensity
            )
            / 2,
            extra_info=first.extra_info,
        )


@dataclass
class WorkloadStats:
    prefill_stats: OpRunStats
    decode_stats: OpRunStats
    num_users: int
    input_tokens: int
    output_tokens: int
    num_chips: int
    ttft: float
    tpot: float
    tps_per_user: float
    throughput: float
    throughput_per_chip: float
    ttc: float
    total_tco: float
    cost_per_1m_tokens: float

    def to_dict(self):
        return asdict(self)

    def pretty_str(self):
        lines = (
            [
                "=" * 20 + " Workload Stats " + "=" * 20,
                f"Number of users: {self.num_users}",
                f"Input tokens: {self.input_tokens}",
                f"Output tokens: {self.output_tokens}",
                f"Number of chips: {self.num_chips}",
                f"TTFT: {format_value(self.ttft, 's')}",
                f"TPOT: {format_value(self.tpot, 's')}",
                f"TTC: {format_value(self.ttc, 's')}",
                f"TPS per user: {format_value(self.tps_per_user, 'tokens/s')}",
                f"Throughput: {format_value(self.throughput, 'tokens/s')}",
                f"Throughput per chip: {format_value(self.throughput_per_chip, 'tokens/s')}",
                f"Total TCO: {format_value(self.total_tco, '$/hr')}",
                f"Cost/1M tokens: {format_value(self.cost_per_1m_tokens, '$')}",
                "Prefill stats:",
            ]
            + [
                f"{' ' * 2}{line}"
                for line in self.prefill_stats.pretty_str().split("\n")
            ]
            + [
                "Decode stats:",
            ]
            + [
                f"{' ' * 2}{line}"
                for line in self.decode_stats.pretty_str().split("\n")
            ]
        )
        if self.decode_stats.extra_info is not None:
            lines += (
                [
                    f"{' ' * 2}Attn stats:",
                ]
                + [
                    f"{' ' * 4}{line}"
                    for line in self.decode_stats.extra_info["attn_stats"]
                    .pretty_str()
                    .split("\n")
                ]
                + [
                    f"{' ' * 2}FFN stats:",
                ]
                + [
                    f"{' ' * 4}{line}"
                    for line in self.decode_stats.extra_info["ffn_stats"]
                    .pretty_str()
                    .split("\n")
                ]
            )
        return "\n".join(lines)


@dataclass
class ParallelismConfig:
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["tp_size", "dp_size", "pp_size", "ep_size"]:
            if value <= 0 and value != -1:
                raise ValueError(f"{name} must be positive or -1, got {value}")
        super().__setattr__(name, value)

    @property
    def num_chips(self):
        return self.tp_size * self.dp_size * self.pp_size * self.ep_size

    @property
    def num_tp_shards(self):
        return self.dp_size * self.pp_size * self.ep_size

    @property
    def num_dp_shards(self):
        return self.tp_size * self.pp_size * self.ep_size

    @property
    def num_pp_shards(self):
        return self.tp_size * self.dp_size * self.ep_size

    @property
    def num_ep_shards(self):
        return self.tp_size * self.dp_size * self.pp_size

    def moe_pcfg(self, ep_size: int) -> "ParallelismConfig":
        num_tp_shards = self.num_tp_shards
        new_ep = min(ep_size, num_tp_shards)
        new_dp = num_tp_shards // new_ep
        new_pp = num_tp_shards // new_ep // new_dp
        return ParallelismConfig(
            tp_size=self.tp_size, dp_size=new_dp, pp_size=new_pp, ep_size=new_ep
        )

    def to_dict(self):
        return asdict(self)

    def copy(self, deep: bool = True) -> "ParallelismConfig":
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @staticmethod
    def dim_name(dim: int) -> str:
        return ["tp_size", "dp_size", "pp_size", "ep_size"][dim]

    def set_dim(self, dim: int, value: int):
        self.__setattr__(ParallelismConfig.dim_name(dim), value)

    def get_inferred_dim(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, val in enumerate(
                [self.tp_size, self.dp_size, self.pp_size, self.ep_size]
            )
            if val == -1
        )


def format_value(value: float | int, unit: str = "") -> str:
    if unit == "FULL":
        return f"{value:.2f}"
    if unit == "%":
        return f"{value * 100:.2f}%"

    si_prefix = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "": 1,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }

    sib_prefix = {
        "": 1,
        "K": 1 << 10,
        "M": 1 << 20,
        "G": 1 << 30,
        "T": 1 << 40,
        "P": 1 << 50,
    }

    prefix_dict = si_prefix if unit != "iB" else sib_prefix
    for prefix, factor in prefix_dict.items():
        if value < factor * 1000:
            return f"{value / factor:.2f} {prefix}{unit}"
    return f"{value / prefix_dict['P']:.2f} P{unit}"


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
