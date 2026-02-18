import math
import threading
from typing import List
from functools import cached_property

from llm_spice.utils.common import DataType, Tensor, ParallelismConfig as ParCfg

import numpy as np


class BaseOp:
    _id = 0

    def __init__(self, name: str = "", dtype: DataType = DataType.BF16):
        self.name = self.__class__.__name__ if not name else name
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []
        self.weight: Tensor | None = None
        self.dtype = dtype
        self.total_flops = None
        self._id = BaseOp._id
        BaseOp._id += 1
        self._children_cache: list["BaseOp"] | None = None

    def __hash__(self) -> int:
        return hash((self._id, self.name))

    def __str__(self) -> str:
        inputs_str = (
            ", ".join(str(t.shape) for t in self.inputs) if self.inputs else "-"
        )
        outputs_str = (
            ", ".join(str(t.shape) for t in self.outputs) if self.outputs else "-"
        )
        return f"{self.name} [{inputs_str}] -> [{outputs_str}] (AI={self.get_arithmetic_intensity():.2f})"

    def __repr__(self) -> str:
        return self.__str__()

    def pretty_str(self, level: int = 0) -> str:
        return "\n".join(self._to_tree_lines())

    def _to_tree_lines(self, prefix: str = "", is_last: bool = True) -> list[str]:
        label = str(self)
        if prefix == "":
            # Root node
            lines = [label]
        else:
            connector = "└─ " if is_last else "├─ "
            lines = [f"{prefix}{connector}{label}"]

        if self.children:
            child_prefix = f"{prefix}{('   ' if is_last else '│  ')}"
            for index, child in enumerate(self.children):
                last_child = index == len(self.children) - 1
                lines.extend(child._to_tree_lines(child_prefix, last_child))
        return lines

    @property
    def children(self) -> list["BaseOp"]:
        # Return cached value if available
        if self._children_cache is not None:
            return self._children_cache

        discovered: list[BaseOp] = []
        seen: set[int] = set()

        def add_child(candidate: object):
            if isinstance(candidate, BaseOp):
                ident = id(candidate)
                if ident not in seen:
                    discovered.append(candidate)
                    seen.add(ident)

        for key, value in self.__dict__.items():
            # Skip private attributes (including our cache)
            if key.startswith("_"):
                continue
            add_child(value)
            if isinstance(value, (list, tuple)):
                for item in value:
                    add_child(item)
            elif isinstance(value, dict):
                for item in value.values():
                    add_child(item)

        self._children_cache = discovered
        return discovered

    @cached_property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def verify_input(self):
        pass

    def forward_pass_done(self):
        """
        Called after a forward pass is done.
        Clears the input and output tensors.
        """
        self.inputs = []
        self.outputs = []
        for child in self.children:
            child.forward_pass_done()
        # Clear children cache
        self._children_cache = None

    def __iter__(self):
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child

    def all_children(self):
        for child in self.children:
            yield from child.all_children()
        yield self

    def get_total_num_params(self) -> int:
        own = self.weight.num_elements if self.weight is not None else 0
        children = sum(child.get_total_num_params() for child in self.children)
        return own + children

    def get_total_weights_bytes(self) -> int:
        own = int(self.weight.size_in_bytes if self.weight is not None else 0)
        children = sum(child.get_total_weights_bytes() for child in self.children)
        return own + children

    def add_input(self, inp: Tensor):
        self.inputs.append(inp)
        assert inp.consumer is None, (
            f"Input {inp} already has a consumer {inp.consumer}, cannot add it to {self}"
        )
        inp.consumer = self
        inp.name = f"{self.name}.input"

    def add_output(self, out: Tensor):
        self.outputs.append(out)
        assert out.producer is None, (
            f"Output {out} already has a producer {out.producer}, cannot add it to {self}"
        )
        out.producer = self
        out.name = f"{self.name}.output"

    def get_total_flop(self) -> int:
        if self.total_flops is None:
            return 0
        return self.total_flops

    def get_total_memory_access_bytes(self) -> int:
        if (
            not hasattr(self, "total_memory_access_bytes")
            or self.total_memory_access_bytes is None
        ):
            return 0
        return self.total_memory_access_bytes

    def get_arithmetic_intensity(self) -> float:
        memory_access_bytes = self.get_total_memory_access_bytes()
        if memory_access_bytes == 0:
            return 0
        return self.get_total_flop() / memory_access_bytes

    def calc_total_flop(self) -> int:
        raise NotImplementedError("Subclass must implement this method")

    def calc_total_memory_access_bytes(self) -> int:
        return int(
            sum(i.size_in_bytes for i in self.inputs)
            + sum(o.size_in_bytes for o in self.outputs)
            + (self.weight.size_in_bytes if self.weight is not None else 0)
        )

    def __call__(self, *args, **kwargs):
        if len(self.inputs) > 0 or len(self.outputs) > 0:
            raise RuntimeError("Forward can only be called once")
        out = self.forward(*args, **kwargs)
        self.verify_input()  # May raise an error
        self.total_flops = self.calc_total_flop()
        self.total_memory_access_bytes = self.calc_total_memory_access_bytes()
        return out

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement this method")


class NoOp(BaseOp):
    def __init__(self, name: str = ""):
        super().__init__(name)

    def calc_total_flop(self) -> int:
        return 0

    def calc_total_memory_access_bytes(self) -> int:
        return 0

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out = Tensor.new_like(inp)
        self.add_output(out)
        return out


class Split(NoOp):
    def __init__(self, splits: List[int] | int, axis: int = -1, name: str = ""):
        super().__init__(name)
        self.splits = splits
        self.axis = axis

    def forward(self, inp: Tensor) -> list[Tensor]:
        self.add_input(inp)
        outs = inp.split(splits=self.splits, axis=self.axis)
        for out in outs:
            self.add_output(out)
        return outs


class Concat(NoOp):
    def __init__(self, axis: int = -1, name: str = ""):
        super().__init__(name)
        self.axis = axis

    def forward(self, inps: list[Tensor]) -> Tensor:
        for inp in inps:
            self.add_input(inp)
        out = inps[0]
        for inp in inps[1:]:
            out = out.concat(inp, axis=self.axis)
        self.add_output(out)
        return out


class Transpose(NoOp):
    def __init__(self, axes: tuple[int, ...], name: str = ""):
        super().__init__(name)
        self.axes = axes

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out = inp.transpose(axes=self.axes)
        self.add_output(out)
        return out


class Reshape(NoOp):
    def __init__(self, shape: tuple[int, ...], name: str = ""):
        super().__init__(name)
        self.shape = shape

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out = inp.reshape(self.shape)
        self.add_output(out)
        return out


class FanOut(NoOp):
    def __init__(self, num_outputs: int, name: str = ""):
        super().__init__(name)
        self.num_outputs = num_outputs

    def forward(self, inp: Tensor) -> list[Tensor]:
        self.add_input(inp)
        outs = [Tensor.new_like(inp) for _ in range(self.num_outputs)]
        for out in outs:
            self.add_output(out)
        return outs


class ElementWiseOp(BaseOp):
    def __init__(self, name: str = "", dtype: DataType = DataType.BF16):
        super().__init__(name, dtype)

    def calc_total_flop(self) -> int:
        return math.prod(self.inputs[0].shape)

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out = Tensor.new_like(inp)
        self.add_output(out)
        return out


class ElementWiseBinaryOp(ElementWiseOp):
    def __init__(self, name: str = "", dtype: DataType = DataType.BF16):
        super().__init__(name, dtype)

    def forward(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        self.add_input(inp1)
        self.add_input(inp2)
        out = Tensor.new_like(inp1)
        self.add_output(out)
        return out


class Residual(ElementWiseBinaryOp):
    def __init__(self, name: str = "", dtype: DataType = DataType.BF16):
        super().__init__(name, dtype)


class AFTransfer(NoOp):
    _thread_state = threading.local()  # Needed for thread safety

    def __init__(self, old_pcfg: ParCfg, new_pcfg: ParCfg, name: str = ""):
        super().__init__(name)
        self.old_pcfg = old_pcfg
        self.new_pcfg = new_pcfg
        self._get_state().num_users = -1

    @classmethod
    def _get_state(cls) -> threading.local:
        if not hasattr(cls._thread_state, "num_users"):
            cls._thread_state.num_users = -1
        return cls._thread_state

    def forward(self, inps: list[Tensor]) -> list[Tensor]:
        for inp in inps:
            self.add_input(inp)

        num_users = inps[0].shape[0]
        for tensor in inps[1:]:
            assert tensor.shape[0] == num_users, (
                "tensors must share the same logical batch dimension"
            )
        state = self._get_state()
        if state.num_users < 0:
            state.num_users = num_users * self.old_pcfg.num_tp_shards

        new_num_users = math.ceil(state.num_users / self.new_pcfg.num_tp_shards)
        outs = [
            Tensor(shape=(new_num_users, *inp.shape[1:]), dtype=inp.dtype)
            for inp in inps
        ]
        for out in outs:
            self.add_output(out)
        return outs

    def forward_pass_done(self):
        super().forward_pass_done()
        state = self._get_state()
        state.num_users = -1


class RMSNorm(ElementWiseOp):
    def __init__(self, input_dim: int, name: str = "", dtype: DataType = DataType.BF16):
        super().__init__(name, dtype)
        self.input_dim = input_dim
        self.weight = Tensor(shape=(input_dim,), dtype=dtype)

    def calc_total_flop(self) -> int:
        return math.prod(self.inputs[0].shape) * 2

    def verify_input(self):
        assert self.inputs[0].shape[-1] == self.input_dim, (
            f"Input dimension {self.inputs[0].shape[-1]} must match weight dimension {self.input_dim}"
        )

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out = Tensor.new_like(inp)
        self.add_output(out)
        return out


class BMM(BaseOp):
    def __init__(self, name: str = ""):
        super().__init__(name)

    def forward(self, a: Tensor, b: Tensor):
        self.add_input(a)
        self.add_input(b)
        out = Tensor(shape=(a.shape[:-3] + (a.shape[-2], b.shape[-1])), dtype=a.dtype)
        self.add_output(out)
        return out

    def verify_input(self):
        a = self.inputs[0]
        b = self.inputs[1]
        assert a.shape[-1] == b.shape[-2], (
            f"Inner dimension of a {a.shape} and b {b.shape} must match"
        )
        assert a.shape[:-3] == b.shape[:-3], (
            f"Batch dimension of a {a.shape} and b {b.shape} must match"
        )

    def calc_total_flop(self) -> int:
        """
        Return the total number of floating point operations for the operation.
        """
        inp = self.inputs[0]
        out = self.outputs[0]
        inp_outer = math.prod(inp.shape[:-1])
        inner = inp.shape[-1]
        out_outer = out.shape[-1]
        return 2 * inp_outer * inner * out_outer


class GQA(BaseOp):
    def __init__(self, name: str = ""):
        super().__init__(name)

    def verify_input(self):
        q, k, v = self.inputs
        num_user_q, _, Hq, D_q = q.shape
        num_user_k, S_k, H_kv, D_k = k.shape
        num_user_v, S_v, H_v, D_v = v.shape
        assert num_user_q == num_user_k == num_user_v, "num_user mismatch"
        assert S_k == S_v, "seq_len mismatch"
        assert D_q == D_k, "head_dim mismatch"
        assert H_kv == H_v, "num_kv_heads mismatch"
        assert Hq % H_kv == 0, "Hq must be divisible by num_kv_heads for GQA"

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        self.add_input(q)
        self.add_input(k)
        self.add_input(v)

        # Shapes
        num_user, S_q, Hq, D_q = q.shape
        _, _, H_kv, _ = k.shape
        _, _, _, D_v = v.shape

        # Output: (num_user, S_q, Hq, D_v)
        out = Tensor((num_user, S_q, Hq, D_v), dtype=v.dtype, name=f"{self.name}_out")
        self.add_output(out)
        return out

    def calc_total_flop(self) -> int:
        """
        Total FLOPs for attention (excluding projections & softmax):
          - Scores = Q @ K^T : 2 * S_q * S_k * D_q per head
          - Context = Attn @ V: 2 * S_q * S_k * D_v per head
        Generalizes to v_head_dim != q/k_head_dim.
        Shapes:
          q: (num_user, S_q, Hq, D_q)
          k: (num_user, S_k, H_kv, D_q)
          v: (num_user, S_k, H_kv, D_v)
          out: (num_user, S_q, Hq, D_v)
        """
        q, k, v = self.inputs
        num_user, S_q, Hq, D_q = q.shape
        _, S_k, H_kv, _ = k.shape
        _, _, _, D_v = v.shape

        gqa_factor = Hq // H_kv
        total_heads = H_kv * gqa_factor  # == Hq

        # QK^T term uses D_q; Attn·V term uses D_v
        flops_per_head = 2 * S_q * S_k * (D_q + D_v)
        total_flops = num_user * total_heads * flops_per_head
        return int(total_flops)


class Linear(BaseOp):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(name, dtype)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Tensor(shape=(input_dim, output_dim), dtype=dtype)

    def verify_input(self):
        assert self.inputs[0].shape[-1] == self.input_dim, (
            f"Inner dimension of input {self.inputs[0].shape} must match input dimension {self.input_dim}"
        )

    def forward(self, inp: Tensor):
        weight = self.weight
        assert weight is not None

        self.add_input(inp)
        out_outer = weight.shape[-1]
        out = Tensor((*inp.shape[:-1], out_outer), dtype=inp.dtype)
        self.add_output(out)
        return out

    def calc_total_flop(self) -> int:
        """
        Return the total number of floating point operations for the operation.
        """
        inp = self.inputs[0]
        out = self.outputs[0]
        inp_outer = math.prod(inp.shape[:-1])
        inner = inp.shape[-1]
        out_outer = out.shape[-1]
        return 2 * inp_outer * inner * out_outer


class MultiOp(BaseOp):
    def __init__(self, name: str = "", dtype: DataType = DataType.BF16):
        super().__init__(name, dtype)

    def pretty_str(self, level: int = 0) -> str:
        return "\n".join(self._to_tree_lines())

    def calc_total_flop(self) -> int:
        return sum(child.get_total_flop() for child in self.children)

    def calc_total_memory_access_bytes(self) -> int:
        return int(
            sum(i.size_in_bytes for i in self.inputs)
            + sum(o.size_in_bytes for o in self.outputs)
            + sum(c.get_total_memory_access_bytes() for c in self.children)
        )

    def add_input(self, inp: Tensor):
        self.inputs.append(inp)

    def add_output(self, out: Tensor):
        self.outputs.append(out)


class FFN(MultiOp):
    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(name, dtype)

        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim

        self.fan_out = FanOut(num_outputs=2, name="ffn_fan_out")
        self.gate_proj = Linear(embed_dim, intermediate_dim, "gate_proj", dtype=dtype)
        self.up_proj = Linear(embed_dim, intermediate_dim, "up_proj", dtype=dtype)
        self.mul = ElementWiseBinaryOp(name="mul")
        self.glu = ElementWiseOp(name="glu")
        self.down_proj = Linear(intermediate_dim, embed_dim, "down_proj", dtype=dtype)

        # Children are discovered dynamically

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        out1, out2 = self.fan_out(inp)
        gate_out = self.gate_proj(out1)
        up_out = self.up_proj(out2)
        mul_out = self.mul(gate_out, up_out)
        glu_out = self.glu(mul_out)
        down_out = self.down_proj(glu_out)
        self.add_output(down_out)
        return down_out


class MoEShuffle(NoOp):
    def __init__(self, name: str = ""):
        super().__init__(name)

    def forward(
        self, inp: Tensor, expert_mapping_table: dict[int, int]
    ) -> dict[int, Tensor]:
        self.add_input(inp)
        outs = {}
        for expert_idx, num_tokens in expert_mapping_table.items():
            out = Tensor(shape=(num_tokens, inp.shape[-1]), dtype=inp.dtype)
            self.add_output(out)
            outs[expert_idx] = out
        return outs


class MoECombine(NoOp):
    def __init__(self, name: str = ""):
        super().__init__(name)

    def forward(self, inps: dict[int, Tensor], batch_dim: tuple[int, int]) -> Tensor:
        for inp in inps.values():
            self.add_input(inp)
        out = Tensor(shape=(*batch_dim, inps[0].shape[-1]), dtype=inps[0].dtype)
        self.add_output(out)
        return out


class MoE(MultiOp):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        embed_dim: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int = 0,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(name, dtype)
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.embed_dim = embed_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        if shared_intermediate_size > 0:
            self.shared_expert = FFN(
                embed_dim, shared_intermediate_size, "shared_expert", dtype=dtype
            )
            self.add_shared = Residual(name="add_shared")

        self.fan_out = FanOut(num_outputs=3, name="moe_fan_out")
        self.gate = Linear(embed_dim, num_experts, "gate", dtype=dtype)
        self.shuffle = MoEShuffle(name="shuffle")
        self.experts = [
            FFN(embed_dim, moe_intermediate_size, f"expert_{i}", dtype=dtype)
            for i in range(num_experts)
        ]
        self.combine = MoECombine(name="combine")

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        inp1, inp2, inp3 = self.fan_out(inp)
        gate_out = self.gate(inp1)
        self.add_output(gate_out)

        ### Original implementation (Exact but slow)
        # token_choices = []
        # for _ in range(math.prod(inp.shape[:-1])):
        #     selected_experts = np.random.choice(
        #         range(self.num_experts), size=self.num_experts_per_token, replace=False
        #     )
        #     token_choices.append(selected_experts)

        # expert_mapping_table = defaultdict(lambda: 0)
        # for _, selected_experts in enumerate(token_choices):
        #     for expert_idx in selected_experts:
        #         expert_mapping_table[expert_idx] += 1

        ### New implementation (Approximation but fast)
        num_tokens = math.prod(inp.shape[:-1])
        expert_mapping_table = np.random.multinomial(
            n=num_tokens * self.num_experts_per_token,
            pvals=np.full(self.num_experts, 1.0 / self.num_experts),
        )
        expert_mapping_table = {
            i: int(expert_mapping_table[i]) for i in range(self.num_experts)
        }

        # [N, embed_dim] -> dict[int, [n, embed_dim]]
        shuffled_inps = self.shuffle(inp2, expert_mapping_table)

        shuffled_outs = {}
        for expert_idx, out in shuffled_inps.items():
            expert_out = self.experts[expert_idx](out)
            shuffled_outs[expert_idx] = expert_out

        out = self.combine(shuffled_outs, inp.shape[:-1])
        if self.shared_intermediate_size > 0:
            shared_out = self.shared_expert(inp3)
            out = self.add_shared(out, shared_out)
        self.add_output(out)
        return out


class MoEFast(BaseOp):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        embed_dim: int,
        moe_intermediate_size: int,
        shared_intermediate_size: int = 0,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        """
        Analytical (leaf) approximation of MoE cost.

        - Collapses routing, experts, and combine into a single op.
        - FLOPs are computed as:
          total = num_tokens * [ 2*E*Ne (gate) + k * 6*E*I + (shared? 6*E*Is : 0) ]
        - Memory accounts for input + output + total weights once per call.
        """
        super().__init__(name if name else "MoEFast", dtype)
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.embed_dim = embed_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size

        # Approximate total weights once (not per-token):
        # gate: E x Ne
        # experts: Ne x (gate_proj ExI + up_proj ExI + down_proj IxE) = Ne x (3*E*I)
        # shared expert (optional): 3*E*Is
        self.gate_weight_elems = embed_dim * num_experts
        self.expert_weight_elems = 3 * embed_dim * moe_intermediate_size
        self.shared_expert_weight_elems = 3 * embed_dim * shared_intermediate_size
        total_weight_elems = (
            self.gate_weight_elems
            + num_experts * self.expert_weight_elems
            + self.shared_expert_weight_elems
        )

        self.weight = Tensor(shape=(total_weight_elems,), dtype=dtype)

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        # Include a gate output for memory parity with MoE
        gate_out = Tensor((*inp.shape[:-1], self.num_experts), dtype=inp.dtype)
        self.add_output(gate_out)

        num_tokens = math.prod(inp.shape[:-1])
        expert_mapping_table = np.random.multinomial(
            n=num_tokens * self.num_experts_per_token,
            pvals=np.full(self.num_experts, 1.0 / self.num_experts),
        )
        self.expert_mapping_table = {
            i: int(expert_mapping_table[i]) for i in range(self.num_experts)
        }

        # Output shape matches input’s last dim (expert-combined representation)
        out = Tensor.new_like(inp)
        self.add_output(out)
        return out

    def calc_total_flop(self) -> int:
        # num_tokens is product of all batch/sequence dims
        inp = self.inputs[0]
        num_tokens = int(math.prod(inp.shape[:-1]))

        Ne = self.num_experts
        k = self.num_experts_per_token
        Is = self.shared_intermediate_size

        # Gate cost: linear [E -> Ne]
        gate_flops = 2 * num_tokens * self.embed_dim * Ne

        # FFN per expert per token (approx): gate_proj ExI + up_proj ExI + down_proj IxE => ~6*E*I
        ffn_per_token = 6 * self.embed_dim * self.moe_intermediate_size
        expert_flops = num_tokens * k * ffn_per_token

        # Shared expert (optional): once per token
        shared_flops = 0
        if Is and Is > 0:
            shared_flops = num_tokens * (6 * self.embed_dim * Is)

        return int(gate_flops + expert_flops + shared_flops)

    def calc_total_memory_access_bytes(self) -> int:
        # Use BaseOp’s default: sum(inputs) + sum(outputs) + weights
        num_active_experts = sum(1 for v in self.expert_mapping_table.values() if v > 0)
        total_weight_elems = (
            self.gate_weight_elems
            + num_active_experts * self.expert_weight_elems
            + self.shared_expert_weight_elems
        )
        active_weight = Tensor(shape=(total_weight_elems,), dtype=self.dtype)
        return int(
            sum(i.size_in_bytes for i in self.inputs)
            + sum(o.size_in_bytes for o in self.outputs)
            + active_weight.size_in_bytes
        )


class LinearGQA(BaseOp):
    """
    Linear attention GQA that computes attention in O(N) complexity.
    Uses kernel trick to avoid materializing the full attention matrix.

    Standard attention: O = softmax(QK^T)V = O(N^2 * D)
    Linear attention: O = φ(Q)(φ(K)^TV) = O(N * D^2)

    For long sequences where N >> D, this is much more efficient.
    """

    def __init__(self, name: str = ""):
        super().__init__(name)

    def verify_input(self):
        q, k, v = self.inputs
        num_user_q, _, Hq, D_q = q.shape
        num_user_k, S_k, H_kv, D_k = k.shape
        num_user_v, S_v, H_v, D_v = v.shape
        assert num_user_q == num_user_k == num_user_v, "num_user mismatch"
        assert S_k == S_v, "seq_len mismatch"
        assert D_q == D_k, "head_dim mismatch"
        assert H_kv == H_v, "num_kv_heads mismatch"
        assert Hq % H_kv == 0, "Hq must be divisible by num_kv_heads for GQA"

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        self.add_input(q)
        self.add_input(k)
        self.add_input(v)

        # Shapes
        num_user, S_q, Hq, D_q = q.shape
        _, S_k, H_kv, _ = k.shape
        _, _, _, D_v = v.shape

        # Output: (num_user, S_q, Hq, D_v)
        out = Tensor((num_user, S_q, Hq, D_v), dtype=v.dtype, name=f"{self.name}_out")
        self.add_output(out)
        return out

    def calc_total_flop(self) -> int:
        """
        Linear attention FLOPs:
        - Compute K^T @ V: 2 * S_k * D_k * D_v per head
        - Compute Q @ (K^T V): 2 * S_q * D_q * D_v per head
        Total: 2 * (S_k + S_q) * D_q * D_v per head

        This is O(N*D^2) instead of O(N^2*D) for standard attention.
        """
        q, k, v = self.inputs
        num_user, S_q, Hq, D_q = q.shape
        _, S_k, H_kv, _ = k.shape
        _, _, _, D_v = v.shape

        gqa_factor = Hq // H_kv
        total_heads = H_kv * gqa_factor

        # K^T @ V: 2 * S_k * D_k * D_v
        # Q @ (K^T V): 2 * S_q * D_q * D_v
        flops_per_head = 2 * (S_k * D_q * D_v + S_q * D_q * D_v)
        total_flops = num_user * total_heads * flops_per_head
        return int(total_flops)


class Attention(MultiOp):
    def __init__(
        self,
        embed_dim: int,
        num_kv_heads: int,
        head_dim: int,
        gqa_factor: int,
        enable_qk_norm: bool = False,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(name, dtype)

        self.embed_dim = embed_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_factor = gqa_factor

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_factor = gqa_factor
        self.enable_qk_norm = enable_qk_norm

        self.fan_out = FanOut(num_outputs=3, name="attention_fan_out")
        self.q_proj = Linear(
            embed_dim, head_dim * num_kv_heads * gqa_factor, "q_proj", dtype=dtype
        )
        self.k_proj = Linear(embed_dim, head_dim * num_kv_heads, "k_proj", dtype=dtype)
        self.v_proj = Linear(embed_dim, head_dim * num_kv_heads, "v_proj", dtype=dtype)
        if enable_qk_norm:
            self.q_norm = RMSNorm(head_dim, name="q_norm")
            self.k_norm = RMSNorm(head_dim, name="k_norm")

        self.gqa = GQA()
        self.out_proj = Linear(
            head_dim * num_kv_heads * gqa_factor, embed_dim, "out_proj", dtype=dtype
        )

        self.kvcache = None

    def get_kvcache(self) -> Tensor | None:
        return self.kvcache

    def set_kvcache(self, kv: Tensor):
        self.kvcache = Tensor.new_like(kv)
        self.kvcache.producer = kv.producer

    def clear_kvcache(self):
        self.kvcache = None

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        self.clear_kvcache()
        self.kvcache = Tensor(
            shape=(2, num_users, seq_len, self.num_kv_heads, self.head_dim), dtype=dtype
        )

    def get_total_memory_access_bytes(self) -> int:
        return int(
            super().get_total_memory_access_bytes() + self.kvcache.size_in_bytes
            if self.kvcache is not None
            else 0
        )

    def forward(self, inp: Tensor) -> Tensor:
        # Expand to 3D [num_user, seq_len, embed_dim]
        num_user, _, _ = inp.reshape((-1, inp.shape[-2], inp.shape[-1])).shape

        self.q_reshape = Reshape(
            (num_user, -1, self.num_kv_heads * self.gqa_factor, self.head_dim),
            name="q_reshape",
        )
        self.k_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="k_reshape"
        )
        self.v_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="v_reshape"
        )

        self.add_input(inp)

        inp1, inp2, inp3 = self.fan_out(inp)
        q = self.q_proj(inp1)
        k = self.k_proj(inp2)
        v = self.v_proj(inp3)

        q = self.q_reshape(q)
        k = self.k_reshape(k)
        v = self.v_reshape(v)

        if self.enable_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        prev_kv = self.get_kvcache()
        if prev_kv is not None:
            self.k_concat = Concat(
                axis=1, name="k_concat"
            )  # Append along sequence axis
            prev_k = prev_kv[0]
            k = self.k_concat([prev_k, k])
            self.v_concat = Concat(
                axis=1, name="v_concat"
            )  # Append along sequence axis
            prev_v = prev_kv[1]
            v = self.v_concat([prev_v, v])

        kv = Tensor(shape=(2, *k.shape), dtype=k.dtype)
        self.set_kvcache(kv)

        qkv = self.gqa(q, k, v)

        qkv_reshape = Reshape(
            (qkv.shape[0], -1, self.head_dim * self.num_kv_heads * self.gqa_factor),
            name="qkv_reshape",
        )
        qkv_r = qkv_reshape(qkv)
        out = self.out_proj(qkv_r)
        self.add_output(out)
        self.add_output(kv)
        return out


class LinearAttention(MultiOp):
    """
    Linear Attention implementation with O(N) complexity instead of O(N^2).

    Uses kernel-based approximation to avoid materializing the full attention matrix.
    Suitable for long sequences where sequence length N >> head dimension D.
    """

    def __init__(
        self,
        embed_dim: int,
        num_kv_heads: int,
        head_dim: int,
        gqa_factor: int,
        enable_qk_norm: bool = False,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(name, dtype)

        self.embed_dim = embed_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_factor = gqa_factor
        self.enable_qk_norm = enable_qk_norm

        self.fan_out = FanOut(num_outputs=3, name="linear_attention_fan_out")
        self.q_proj = Linear(
            embed_dim, head_dim * num_kv_heads * gqa_factor, "q_proj", dtype=dtype
        )
        self.k_proj = Linear(embed_dim, head_dim * num_kv_heads, "k_proj", dtype=dtype)
        self.v_proj = Linear(embed_dim, head_dim * num_kv_heads, "v_proj", dtype=dtype)

        if enable_qk_norm:
            self.q_norm = RMSNorm(head_dim, name="q_norm")
            self.k_norm = RMSNorm(head_dim, name="k_norm")

        # Use LinearGQA instead of standard GQA
        self.linear_gqa = LinearGQA(name="linear_gqa")
        self.out_proj = Linear(
            head_dim * num_kv_heads * gqa_factor, embed_dim, "out_proj", dtype=dtype
        )

        self.kvcache = None

    def get_kvcache(self) -> Tensor | None:
        return self.kvcache

    def set_kvcache(self, kv: Tensor):
        self.kvcache = Tensor.new_like(kv)
        self.kvcache.producer = kv.producer

    def clear_kvcache(self):
        self.kvcache = None

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        self.clear_kvcache()
        self.kvcache = Tensor(
            shape=(2, num_users, seq_len, self.num_kv_heads, self.head_dim), dtype=dtype
        )

    def get_total_memory_access_bytes(self) -> int:
        return int(
            super().get_total_memory_access_bytes() + self.kvcache.size_in_bytes
            if self.kvcache is not None
            else 0
        )

    def forward(self, inp: Tensor) -> Tensor:
        # Expand to 3D [num_user, seq_len, embed_dim]
        num_user, _, _ = inp.reshape((-1, inp.shape[-2], inp.shape[-1])).shape

        self.q_reshape = Reshape(
            (num_user, -1, self.num_kv_heads * self.gqa_factor, self.head_dim),
            name="q_reshape",
        )
        self.k_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="k_reshape"
        )
        self.v_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="v_reshape"
        )

        self.add_input(inp)

        inp1, inp2, inp3 = self.fan_out(inp)
        q = self.q_proj(inp1)
        k = self.k_proj(inp2)
        v = self.v_proj(inp3)

        q = self.q_reshape(q)
        k = self.k_reshape(k)
        v = self.v_reshape(v)

        if self.enable_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        prev_kv = self.get_kvcache()
        if prev_kv is not None:
            self.k_concat = Concat(
                axis=1, name="k_concat"
            )  # Append along sequence axis
            prev_k = prev_kv[0]
            k = self.k_concat([prev_k, k])
            self.v_concat = Concat(
                axis=1, name="v_concat"
            )  # Append along sequence axis
            prev_v = prev_kv[1]
            v = self.v_concat([prev_v, v])

        kv = Tensor(shape=(2, *k.shape), dtype=k.dtype)
        self.set_kvcache(kv)

        # Use linear attention instead of standard attention
        qkv = self.linear_gqa(q, k, v)

        qkv_reshape = Reshape(
            (qkv.shape[0], -1, self.head_dim * self.num_kv_heads * self.gqa_factor),
            name="qkv_reshape",
        )
        qkv_r = qkv_reshape(qkv)
        out = self.out_proj(qkv_r)
        self.add_output(out)
        self.add_output(kv)
        return out
