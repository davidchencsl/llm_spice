from typing import Callable
from llm_spice.execute.executor import AFExecutor, SimpleExecutor
from llm_spice.model.model import Model
from llm_spice.hardware.processor import Processor
from .allocator import Allocator
from llm_spice.execute.workload import run_workload_paf, run_workload_pd


def binary_search(
    fn: Callable[[float, dict], tuple[dict, bool]],
    lo: float,
    hi: float,
    eps=1e-3,
    max_iters=-1,
) -> dict:
    state = {}
    iters = 0
    while hi - lo > eps and (max_iters == -1 or iters < max_iters):
        mid = (lo + hi) / 2.0
        state, move_right = fn(mid, state)
        if move_right:
            lo = mid
        else:
            hi = mid
        iters += 1
    return state


class BalanceAllocator(Allocator):
    def __init__(
        self,
        model: Model,
        input_tokens: int,
        output_tokens: int,
        num_users: int,
        total_budget: float,
        alloc_mode: Allocator.Mode = Allocator.Mode.TCO,
    ):
        super().__init__(
            model, input_tokens, output_tokens, num_users, total_budget, alloc_mode
        )

    def allocate_pd(self, processors: list[Processor]) -> list[Processor]:
        assert len(processors) == 2
        prefill_proc, decode_proc = processors

        prefill_inferred_dim = prefill_proc.pcfg.get_inferred_dim()
        decode_inferred_dim = decode_proc.pcfg.get_inferred_dim()
        assert len(prefill_inferred_dim) == 1, (
            "Balance allocator only supports autotuning on one dimension"
        )
        assert len(decode_inferred_dim) == 1, (
            "Balance allocator only supports autotuning on one dimension"
        )
        prefill_inferred_dim, decode_inferred_dim = (
            prefill_inferred_dim[0],
            decode_inferred_dim[0],
        )

        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive for PD allocation")

        prefill_proc.pcfg.set_dim(prefill_inferred_dim, 1)
        decode_proc.pcfg.set_dim(decode_inferred_dim, 1)

        prefill_min_num_chips = prefill_proc.pcfg.num_chips
        decode_min_num_chips = decode_proc.pcfg.num_chips

        prefill_min_cost = self.cost_function(prefill_proc)(prefill_min_num_chips)
        decode_min_cost = self.cost_function(decode_proc)(decode_min_num_chips)

        if prefill_min_cost + decode_min_cost > self.total_budget:
            raise ValueError("Budget insufficient for PD allocation")

        min_ratio = max(0.0, prefill_min_cost / self.total_budget)
        max_ratio = min(1.0, 1.0 - decode_min_cost / self.total_budget)

        if min_ratio > max_ratio:
            raise ValueError("Invalid budget split for PD allocation")

        def evaluate_ratio(ratio: float, state: dict):
            prefill_budget = ratio * self.total_budget
            decode_budget = (1.0 - ratio) * self.total_budget

            prefill_num_chips = self.get_num_chips(prefill_proc, prefill_budget)
            decode_num_chips = self.get_num_chips(decode_proc, decode_budget)

            # Try rounding both
            prefill_inferred_dim_size = round(prefill_num_chips / prefill_min_num_chips)
            decode_inferred_dim_size = round(decode_num_chips / decode_min_num_chips)
            prefill_proc.pcfg.set_dim(prefill_inferred_dim, prefill_inferred_dim_size)
            decode_proc.pcfg.set_dim(decode_inferred_dim, decode_inferred_dim_size)

            if (
                self.cost_function(prefill_proc)(prefill_proc.pcfg.num_chips)
                + self.cost_function(decode_proc)(decode_proc.pcfg.num_chips)
                > self.total_budget
            ):
                # Default to floor
                prefill_inferred_dim_size = prefill_num_chips // prefill_min_num_chips
                decode_inferred_dim_size = decode_num_chips // decode_min_num_chips
                prefill_proc.pcfg.set_dim(
                    prefill_inferred_dim, prefill_inferred_dim_size
                )
                decode_proc.pcfg.set_dim(decode_inferred_dim, decode_inferred_dim_size)

            prefill_executor = SimpleExecutor(prefill_proc)
            decode_executor = SimpleExecutor(decode_proc)

            stats = run_workload_pd(
                self.model,
                self.input_tokens,
                self.output_tokens,
                self.num_users,
                prefill_executor,
                decode_executor,
            )

            if "best_throughput" not in state:
                state["best_throughput"] = stats.throughput
                state["best_prefill_pcfg"] = prefill_proc.pcfg.copy()
                state["best_decode_pcfg"] = decode_proc.pcfg.copy()
            elif stats.throughput > state["best_throughput"]:
                state["best_throughput"] = stats.throughput
                state["best_prefill_pcfg"] = prefill_proc.pcfg.copy()
                state["best_decode_pcfg"] = decode_proc.pcfg.copy()

            should_decrease_ratio = (
                stats.prefill_stats.duration > stats.decode_stats.duration
            )

            return state, should_decrease_ratio

        state = binary_search(evaluate_ratio, min_ratio, max_ratio)
        prefill_proc.set_parallelism(state["best_prefill_pcfg"])
        decode_proc.set_parallelism(state["best_decode_pcfg"])

        return processors

    def allocate_paf(self, processors: list[Processor]) -> list[Processor]:
        assert len(processors) == 3
        p_proc, a_proc, f_proc = processors

        p_inferred_dim = p_proc.pcfg.get_inferred_dim()
        a_inferred_dim = a_proc.pcfg.get_inferred_dim()
        f_inferred_dim = f_proc.pcfg.get_inferred_dim()
        assert len(p_inferred_dim) == 1, (
            "Balance allocator only supports autotuning on one dimension"
        )
        assert len(a_inferred_dim) == 1, (
            "Balance allocator only supports autotuning on one dimension"
        )
        assert len(f_inferred_dim) == 1, (
            "Balance allocator only supports autotuning on one dimension"
        )
        p_inferred_dim, a_inferred_dim, f_inferred_dim = (
            p_inferred_dim[0],
            a_inferred_dim[0],
            f_inferred_dim[0],
        )

        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive for PAF allocation")

        p_proc.pcfg.set_dim(p_inferred_dim, 1)
        a_proc.pcfg.set_dim(a_inferred_dim, 1)
        f_proc.pcfg.set_dim(f_inferred_dim, 1)

        p_min_num_chips = p_proc.pcfg.num_chips
        a_min_num_chips = a_proc.pcfg.num_chips
        f_min_num_chips = f_proc.pcfg.num_chips

        p_min_cost = self.cost_function(p_proc)(p_min_num_chips)
        a_min_cost = self.cost_function(a_proc)(a_min_num_chips)
        f_min_cost = self.cost_function(f_proc)(f_min_num_chips)

        if (p_min_cost + a_min_cost + f_min_cost) > self.total_budget:
            raise ValueError("Budget insufficient for PAF allocation")

        p_min_ratio = max(0.0, p_min_cost / self.total_budget)
        p_max_ratio = min(1.0, 1.0 - (a_min_cost + f_min_cost) / self.total_budget)

        def evaluate_paf_ratio(p_ratio: float, state: dict):
            p_budget = p_ratio * self.total_budget
            af_budget = (1.0 - p_ratio) * self.total_budget

            p_num_chips = self.get_num_chips(p_proc, p_budget)

            a_min_ratio = max(0.0, a_min_cost / af_budget)
            a_max_ratio = min(1.0, 1.0 - f_min_cost / af_budget)

            def evaluate_af_ratio(a_ratio: float, state: dict):
                a_budget = a_ratio * af_budget
                f_budget = af_budget - a_budget

                a_num_chips = self.get_num_chips(a_proc, a_budget)
                f_num_chips = self.get_num_chips(f_proc, f_budget)

                p_inferred_dim_size = max(1, round(p_num_chips / p_min_num_chips))
                a_inferred_dim_size = max(1, round(a_num_chips / a_min_num_chips))
                f_inferred_dim_size = max(1, round(f_num_chips / f_min_num_chips))
                p_proc.pcfg.set_dim(p_inferred_dim, p_inferred_dim_size)
                a_proc.pcfg.set_dim(a_inferred_dim, a_inferred_dim_size)
                f_proc.pcfg.set_dim(f_inferred_dim, f_inferred_dim_size)

                if (
                    self.cost_function(p_proc)(p_proc.pcfg.num_chips)
                    + self.cost_function(a_proc)(a_proc.pcfg.num_chips)
                    + self.cost_function(f_proc)(f_proc.pcfg.num_chips)
                    > self.total_budget
                ):
                    p_inferred_dim_size = max(1, p_num_chips // p_min_num_chips)
                    a_inferred_dim_size = max(1, a_num_chips // a_min_num_chips)
                    f_inferred_dim_size = max(1, f_num_chips // f_min_num_chips)

                    p_proc.pcfg.set_dim(p_inferred_dim, p_inferred_dim_size)
                    a_proc.pcfg.set_dim(a_inferred_dim, a_inferred_dim_size)
                    f_proc.pcfg.set_dim(f_inferred_dim, f_inferred_dim_size)

                p_executor = SimpleExecutor(p_proc)
                af_executor = AFExecutor(a_proc, f_proc)

                stats = run_workload_paf(
                    self.model,
                    self.input_tokens,
                    self.output_tokens,
                    self.num_users,
                    p_executor,
                    af_executor,
                )

                assert stats.decode_stats.extra_info is not None
                attn_duration = stats.decode_stats.extra_info["attn_stats"].duration
                ffn_duration = stats.decode_stats.extra_info["ffn_stats"].duration

                if "best_throughput" not in state:
                    state["best_throughput"] = stats.throughput
                    state["best_p_duration"] = stats.prefill_stats.duration
                    state["best_a_duration"] = attn_duration
                    state["best_f_duration"] = ffn_duration
                    state["best_p_pcfg"] = p_proc.pcfg.copy()
                    state["best_a_pcfg"] = a_proc.pcfg.copy()
                    state["best_f_pcfg"] = f_proc.pcfg.copy()
                elif stats.throughput > state["best_throughput"]:
                    state["best_throughput"] = stats.throughput
                    state["best_p_duration"] = stats.prefill_stats.duration
                    state["best_a_duration"] = attn_duration
                    state["best_f_duration"] = ffn_duration
                    state["best_p_pcfg"] = p_proc.pcfg.copy()
                    state["best_a_pcfg"] = a_proc.pcfg.copy()
                    state["best_f_pcfg"] = f_proc.pcfg.copy()

                should_decrease_ratio = attn_duration > ffn_duration

                return state, should_decrease_ratio

            af_state = binary_search(evaluate_af_ratio, a_min_ratio, a_max_ratio)
            a_proc.set_parallelism(af_state["best_a_pcfg"])
            f_proc.set_parallelism(af_state["best_f_pcfg"])

            if "best_throughput" not in state:
                state["best_throughput"] = af_state["best_throughput"]
                state["best_p_pcfg"] = af_state["best_p_pcfg"].copy()
                state["best_a_pcfg"] = af_state["best_a_pcfg"].copy()
                state["best_f_pcfg"] = af_state["best_f_pcfg"].copy()
            elif af_state["best_throughput"] > state["best_throughput"]:
                state["best_throughput"] = af_state["best_throughput"]
                state["best_p_pcfg"] = af_state["best_p_pcfg"].copy()
                state["best_a_pcfg"] = af_state["best_a_pcfg"].copy()
                state["best_f_pcfg"] = af_state["best_f_pcfg"].copy()

            should_decrease_ratio = (
                af_state["best_p_duration"] > af_state["best_a_duration"]
            )

            return state, should_decrease_ratio

        p_state = binary_search(evaluate_paf_ratio, p_min_ratio, p_max_ratio)
        p_proc.set_parallelism(p_state["best_p_pcfg"])
        a_proc.set_parallelism(p_state["best_a_pcfg"])
        f_proc.set_parallelism(p_state["best_f_pcfg"])

        return processors
