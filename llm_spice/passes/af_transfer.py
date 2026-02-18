from dataclasses import dataclass
from typing import Any, Literal

from llm_spice.model import Model
from llm_spice.op.operators import AFTransfer, BaseOp, GQA, MultiOp
from llm_spice.utils.common import ParallelismConfig as ParCfg, Tensor


@dataclass
class _GQABinding:
    container: Any
    key: Any
    gqa: GQA
    kind: Literal["attr", "list", "dict"]


class _GQAWithAFTransfer(MultiOp):
    """Wrap a GQA op with AFTransfer ops around it for disaggregated execution."""

    def __init__(self, gqa: GQA, attn_pcfg: ParCfg, ffn_pcfg: ParCfg):
        super().__init__(name=f"{gqa.name}_Wrapper", dtype=gqa.dtype)
        self.pre_transfer = AFTransfer(
            old_pcfg=ffn_pcfg,
            new_pcfg=attn_pcfg,
            name=f"{gqa.name}_AFTransferIn",
        )
        self.gqa = gqa
        self.post_transfer = AFTransfer(
            old_pcfg=attn_pcfg,
            new_pcfg=ffn_pcfg,
            name=f"{gqa.name}_AFTransferOut",
        )
        self._is_af_transfer_wrapper = True

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.add_input(q)
        self.add_input(k)
        self.add_input(v)

        q, k, v = self.pre_transfer([q, k, v])
        gqa_out = self.gqa(q, k, v)

        (gqa_out,) = self.post_transfer([gqa_out])

        self.add_output(gqa_out)
        return gqa_out


class AFTransferPass:
    def __init__(self, attn_pcfg: ParCfg, ffn_pcfg: ParCfg):
        self.attn_pcfg = attn_pcfg
        self.ffn_pcfg = ffn_pcfg
        self._visited: set[int] = set()

    def apply(self, model: Model) -> Model:
        self._visited.clear()
        self._process_container(model)
        return model

    def _process_container(self, obj: Any):
        if isinstance(obj, (GQA, _GQAWithAFTransfer, AFTransfer)):
            return

        obj_id = id(obj)
        if obj_id in self._visited:
            return
        self._visited.add(obj_id)

        if isinstance(obj, BaseOp):
            self._process_op(obj)
        elif isinstance(obj, list):
            for item in obj:
                self._process_container(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                self._process_container(item)

    def _process_op(self, op: BaseOp):
        gqa_bindings: list[_GQABinding] = []
        child_candidates: list[BaseOp] = []

        for attr, value in op.__dict__.items():
            if attr in {"inputs", "outputs", "weight", "_attention_layers"}:
                continue

            self._collect_bindings(
                value,
                container=op,
                key=attr,
                kind="attr",
                gqa_bindings=gqa_bindings,
                child_candidates=child_candidates,
            )

        for binding in gqa_bindings:
            gqa_op = binding.gqa
            if isinstance(gqa_op, _GQAWithAFTransfer):
                continue

            wrapper = _GQAWithAFTransfer(gqa_op, self.attn_pcfg, self.ffn_pcfg)

            if binding.kind == "attr":
                setattr(binding.container, binding.key, wrapper)
            else:
                binding.container[binding.key] = wrapper

        for child in child_candidates:
            self._process_container(child)

    def _collect_bindings(
        self,
        value: Any,
        *,
        container: Any,
        key: Any,
        kind: Literal["attr", "list", "dict"],
        gqa_bindings: list[_GQABinding],
        child_candidates: list[BaseOp],
    ) -> None:
        if isinstance(value, _GQAWithAFTransfer):
            child_candidates.append(value)
        elif isinstance(value, GQA):
            gqa_bindings.append(
                _GQABinding(container=container, key=key, gqa=value, kind=kind)
            )
        elif isinstance(value, BaseOp):
            child_candidates.append(value)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                self._collect_bindings(
                    item,
                    container=value,
                    key=idx,
                    kind="list",
                    gqa_bindings=gqa_bindings,
                    child_candidates=child_candidates,
                )
        elif isinstance(value, dict):
            for dict_key, item in value.items():
                self._collect_bindings(
                    item,
                    container=value,
                    key=dict_key,
                    kind="dict",
                    gqa_bindings=gqa_bindings,
                    child_candidates=child_candidates,
                )
