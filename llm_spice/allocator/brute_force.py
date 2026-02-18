from llm_spice.model.model import Model
from llm_spice.execute.executor import AFExecutor, SimpleExecutor
from llm_spice.execute.workload import run_workload_paf, run_workload_pd
from llm_spice.hardware.processor import Processor
from .allocator import Allocator


class BruteForceAllocator(Allocator):
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
            "BruteForce allocator only supports autotuning on one dimension"
        )
        assert len(decode_inferred_dim) == 1, (
            "BruteForce allocator only supports autotuning on one dimension"
        )
        prefill_inferred_dim, decode_inferred_dim = (
            prefill_inferred_dim[0],
            decode_inferred_dim[0],
        )

        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive for PD allocation")

        # Set baseline config with inferred dims = 1 to calculate baselines
        prefill_proc.pcfg.set_dim(prefill_inferred_dim, 1)
        decode_proc.pcfg.set_dim(decode_inferred_dim, 1)

        prefill_min_num_chips = prefill_proc.pcfg.num_chips
        decode_min_num_chips = decode_proc.pcfg.num_chips

        prefill_cost_fn = self.cost_function(prefill_proc)
        decode_cost_fn = self.cost_function(decode_proc)

        prefill_min_cost = prefill_cost_fn(prefill_min_num_chips)
        decode_min_cost = decode_cost_fn(decode_min_num_chips)

        if prefill_min_cost + decode_min_cost > self.total_budget:
            raise ValueError("Budget insufficient for PD allocation")

        # Upper bounds for enumeration
        max_prefill_chips = Allocator.search_for_max_num_chips(
            self.total_budget - decode_min_cost, prefill_cost_fn
        )
        max_decode_chips_global = Allocator.search_for_max_num_chips(
            self.total_budget - prefill_min_cost, decode_cost_fn
        )

        max_prefill_scale = max(1, max_prefill_chips // prefill_min_num_chips)
        max_decode_scale_global = max(
            1, max_decode_chips_global // decode_min_num_chips
        )

        best_throughput = -1.0
        best_prefill_pcfg = prefill_proc.pcfg.copy()
        best_decode_pcfg = decode_proc.pcfg.copy()

        for prefill_scale in range(1, max_prefill_scale + 1):
            prefill_chips = prefill_scale * prefill_min_num_chips
            prefill_cost = prefill_cost_fn(prefill_chips)
            remaining_budget_for_decode = self.total_budget - prefill_cost

            if remaining_budget_for_decode < decode_min_cost:
                continue

            max_decode_chips = Allocator.search_for_max_num_chips(
                remaining_budget_for_decode, decode_cost_fn
            )
            max_decode_scale = max(1, max_decode_chips // decode_min_num_chips)
            # Further clamp by global bound (optional but safe)
            if max_decode_scale > max_decode_scale_global:
                max_decode_scale = max_decode_scale_global

            for decode_scale in range(1, max_decode_scale + 1):
                # Apply pcfgs for evaluation
                prefill_proc.pcfg.set_dim(prefill_inferred_dim, prefill_scale)
                decode_proc.pcfg.set_dim(decode_inferred_dim, decode_scale)

                # Validate budget guard (defensive)
                total_cost = prefill_cost_fn(
                    prefill_proc.pcfg.num_chips
                ) + decode_cost_fn(decode_proc.pcfg.num_chips)
                if total_cost > self.total_budget:
                    continue

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

                if stats.throughput > best_throughput:
                    best_throughput = stats.throughput
                    best_prefill_pcfg = prefill_proc.pcfg.copy()
                    best_decode_pcfg = decode_proc.pcfg.copy()

        prefill_proc.set_parallelism(best_prefill_pcfg)
        decode_proc.set_parallelism(best_decode_pcfg)

        return processors

    def allocate_paf(self, processors: list[Processor]) -> list[Processor]:
        assert len(processors) == 3
        p_proc, a_proc, f_proc = processors

        p_inferred_dim = p_proc.pcfg.get_inferred_dim()
        a_inferred_dim = a_proc.pcfg.get_inferred_dim()
        f_inferred_dim = f_proc.pcfg.get_inferred_dim()
        assert len(p_inferred_dim) == 1, (
            "BruteForce allocator only supports autotuning on one dimension"
        )
        assert len(a_inferred_dim) == 1, (
            "BruteForce allocator only supports autotuning on one dimension"
        )
        assert len(f_inferred_dim) == 1, (
            "BruteForce allocator only supports autotuning on one dimension"
        )
        p_inferred_dim, a_inferred_dim, f_inferred_dim = (
            p_inferred_dim[0],
            a_inferred_dim[0],
            f_inferred_dim[0],
        )

        if self.total_budget <= 0:
            raise ValueError("Total budget must be positive for PAF allocation")

        # Baseline with inferred dims = 1
        p_proc.pcfg.set_dim(p_inferred_dim, 1)
        a_proc.pcfg.set_dim(a_inferred_dim, 1)
        f_proc.pcfg.set_dim(f_inferred_dim, 1)

        p_min_num_chips = p_proc.pcfg.num_chips
        a_min_num_chips = a_proc.pcfg.num_chips
        f_min_num_chips = f_proc.pcfg.num_chips

        p_cost_fn = self.cost_function(p_proc)
        a_cost_fn = self.cost_function(a_proc)
        f_cost_fn = self.cost_function(f_proc)

        p_min_cost = p_cost_fn(p_min_num_chips)
        a_min_cost = a_cost_fn(a_min_num_chips)
        f_min_cost = f_cost_fn(f_min_num_chips)

        if (p_min_cost + a_min_cost + f_min_cost) > self.total_budget:
            raise ValueError("Budget insufficient for PAF allocation")

        # Upper bounds for enumeration
        max_p_chips = Allocator.search_for_max_num_chips(
            self.total_budget - (a_min_cost + f_min_cost), p_cost_fn
        )
        max_p_scale = max(1, max_p_chips // p_min_num_chips)

        best_throughput = -1.0
        best_p_pcfg = p_proc.pcfg.copy()
        best_a_pcfg = a_proc.pcfg.copy()
        best_f_pcfg = f_proc.pcfg.copy()

        for p_scale in range(1, max_p_scale + 1):
            p_chips = p_scale * p_min_num_chips
            p_cost = p_cost_fn(p_chips)
            remaining_af_budget = self.total_budget - p_cost
            if remaining_af_budget < (a_min_cost + f_min_cost):
                continue

            # For attention under remaining budget while reserving minimal FFN
            max_a_chips = Allocator.search_for_max_num_chips(
                remaining_af_budget - f_min_cost, a_cost_fn
            )
            max_a_scale = max(1, max_a_chips // a_min_num_chips)

            for a_scale in range(1, max_a_scale + 1):
                a_chips = a_scale * a_min_num_chips
                a_cost = a_cost_fn(a_chips)
                remaining_f_budget = remaining_af_budget - a_cost
                if remaining_f_budget < f_min_cost:
                    continue

                max_f_chips = Allocator.search_for_max_num_chips(
                    remaining_f_budget, f_cost_fn
                )
                max_f_scale = max(1, max_f_chips // f_min_num_chips)

                for f_scale in range(1, max_f_scale + 1):
                    # Apply pcfgs for evaluation
                    p_proc.pcfg.set_dim(p_inferred_dim, p_scale)
                    a_proc.pcfg.set_dim(a_inferred_dim, a_scale)
                    f_proc.pcfg.set_dim(f_inferred_dim, f_scale)

                    total_cost = (
                        p_cost_fn(p_proc.pcfg.num_chips)
                        + a_cost_fn(a_proc.pcfg.num_chips)
                        + f_cost_fn(f_proc.pcfg.num_chips)
                    )
                    if total_cost > self.total_budget:
                        continue

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

                    if stats.throughput > best_throughput:
                        best_throughput = stats.throughput
                        best_p_pcfg = p_proc.pcfg.copy()
                        best_a_pcfg = a_proc.pcfg.copy()
                        best_f_pcfg = f_proc.pcfg.copy()

        p_proc.set_parallelism(best_p_pcfg)
        a_proc.set_parallelism(best_a_pcfg)
        f_proc.set_parallelism(best_f_pcfg)

        return processors
