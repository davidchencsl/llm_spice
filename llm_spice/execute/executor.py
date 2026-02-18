from llm_spice.hardware import Processor
from llm_spice.model import Model
from llm_spice.op.operators import BaseOp, GQA
from llm_spice.utils.common import OpRunStats


class Executor:
    def run_model(self, model: Model) -> OpRunStats:
        raise NotImplementedError

    def run_op(self, op: BaseOp) -> OpRunStats:
        # Switch processor at Residual Op
        processors = self.get_processor()
        if isinstance(op, GQA):
            stats = processors[0].execute_op(op)
            stats.extra_info = {
                "attn_stats": stats,
            }
            return stats

        if op.is_leaf:
            child_stats = processors[1 % len(processors)].execute_op(op)
            child_stats.extra_info = {
                "ffn_stats": child_stats,
            }
            return child_stats

        overall_stats = OpRunStats()
        overall_stats.extra_info = {
            "attn_stats": OpRunStats(),
            "ffn_stats": OpRunStats(),
        }
        for child in op.children:
            child_stats = self.run_op(child)
            overall_stats = overall_stats.merge(child_stats)
            assert overall_stats.extra_info is not None
            assert child_stats.extra_info is not None
            if "attn_stats" in child_stats.extra_info:
                overall_stats.extra_info["attn_stats"] = overall_stats.extra_info[
                    "attn_stats"
                ].merge(child_stats.extra_info["attn_stats"])
            if "ffn_stats" in child_stats.extra_info:
                overall_stats.extra_info["ffn_stats"] = overall_stats.extra_info[
                    "ffn_stats"
                ].merge(child_stats.extra_info["ffn_stats"])
        assert overall_stats.duration >= 0
        return overall_stats

    def get_processor(self) -> list[Processor]:
        raise NotImplementedError


class SimpleExecutor(Executor):
    def __init__(self, processor: Processor):
        self.processor = processor

    def run_model(self, model: Model) -> OpRunStats:
        return self.run_op(model)

    def get_processor(self) -> list[Processor]:
        return [self.processor]


class AFExecutor(Executor):
    def __init__(self, p1: Processor, p2: Processor):
        self.processors = [p1, p2]

    def run_model(self, model: Model) -> OpRunStats:
        return self.run_op(model)

    def get_processor(self) -> list[Processor]:
        return self.processors
