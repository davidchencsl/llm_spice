from dataclasses import dataclass
from enum import Enum
from typing import Callable

from llm_spice import DataType
from llm_spice.hardware.processor import Processor
from llm_spice.model.model import Model
from llm_spice.utils.common import format_value


@dataclass
class MemoryUsage:
    used_bytes: int | float
    available_bytes: int | float

    @property
    def does_fit(self) -> bool:
        return self.used_bytes <= self.available_bytes

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.available_bytes

    def __repr__(self) -> str:
        return f"MemoryUsage {format_value(self.used_bytes, 'iB')} / {format_value(self.available_bytes, 'iB')} ({format_value(self.utilization, '%')})"


class Allocator:
    class Mode(Enum):
        TCO = 1
        DIE_COUNT = 2
        DIE_AREA = 3
        PACKAGE_AREA = 4

    Mode = Mode

    def __init__(
        self,
        model: Model,
        input_tokens: int,
        output_tokens: int,
        num_users: int,
        total_budget: float = 0.0,
        alloc_mode: Mode = Mode.TCO,
    ):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.num_users = num_users
        self.total_budget = total_budget
        self.alloc_mode = alloc_mode

    @staticmethod
    def search_for_max_num_chips(
        budget: float, criteria: Callable[[int], float]
    ) -> int:
        """
        Return the maximum number of chips whose cost is <= `budget`.
        Strategy:
          1) Handle edge cases (budget <= 0).
          2) If even 1 chip exceeds budget, return 0.
          3) Exponential search to find an upper bound hi with cost(hi) > budget.
          4) Binary search in (lo, hi] to find the largest n with cost(n) <= budget.
        Assumes cost is monotonic non-decreasing in num_chips.
        """
        if budget <= 0:
            return 0

        # If even a single chip is too expensive, budget buys 0 chips.
        cost_one = criteria(1)
        if cost_one > budget:
            return 0

        # Exponential search to bracket the solution.
        lo = 1
        hi = 1
        while True:
            hi *= 2
            c = criteria(hi)
            if c > budget:
                break
            lo = hi

            # Optional safety: bail out if hi grows too large in extreme models.
            # if hi > 1_000_000_000:
            #     break

        # Binary search for max n with cost(n) <= budget in (lo, hi]
        # Invariant: cost(lo) <= budget, cost(hi) > budget
        ans = lo
        left, right = lo + 1, hi
        while left <= right:
            mid = (left + right) // 2
            c_mid = criteria(mid)
            if c_mid <= budget:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans

    def cost_function(self, processor: Processor) -> Callable[[int], float]:
        match self.alloc_mode:
            case Allocator.Mode.TCO:
                return processor.get_tco
            case Allocator.Mode.DIE_COUNT:
                return lambda x: x
            case Allocator.Mode.DIE_AREA:
                return lambda x: processor.die_area() * x
            case Allocator.Mode.PACKAGE_AREA:
                return lambda x: processor.total_area() * x
            case _:
                raise ValueError(f"Invalid allocation mode: {self.alloc_mode}")

    def get_min_budget(self, processor: Processor) -> float:
        return self.cost_function(processor)(1)

    def get_num_chips(self, processor: Processor, budget: float) -> int:
        return Allocator.search_for_max_num_chips(budget, self.cost_function(processor))

    def memory_usage(self, processors: list[Processor]) -> dict[str, MemoryUsage]:
        match len(processors):
            case 1:
                return self.memory_usage_normal(processors)
            case 2:
                return self.memory_usage_pd(processors)
            case 3:
                return self.memory_usage_paf(processors)
            case _:
                raise ValueError(f"Invalid number of processors: {len(processors)}")

    def memory_usage_normal(
        self, processors: list[Processor]
    ) -> dict[str, MemoryUsage]:
        assert len(processors) == 1
        proc = processors[0]

        # Model parameters:
        total_weights_bytes = self.model.get_total_weights_bytes()

        # KV-cache size:
        self.model.insert_kvcache(
            self.num_users, self.input_tokens + self.output_tokens, dtype=DataType.BF16
        )
        total_kvcache_bytes = self.model.get_total_kvcache_bytes()
        self.model.clear_kvcache()

        # Total available memory:
        total_available_bytes = proc.get_effective_memory_capacity()

        return {
            "Total": MemoryUsage(
                total_weights_bytes + total_kvcache_bytes, total_available_bytes
            )
        }

    def memory_usage_pd(self, processors: list[Processor]) -> dict[str, MemoryUsage]:
        assert len(processors) == 2
        prefill_proc, decode_proc = processors

        # Model parameters:
        total_weights_bytes = self.model.get_total_weights_bytes()

        # Prefill KV-cache size:
        self.model.insert_kvcache(
            self.num_users, self.input_tokens, dtype=DataType.BF16
        )
        total_prefill_kvcache_bytes = self.model.get_total_kvcache_bytes()
        self.model.clear_kvcache()

        # Decode KV-cache size:
        self.model.insert_kvcache(
            self.num_users, self.input_tokens + self.output_tokens, dtype=DataType.BF16
        )
        total_kvcache_bytes = self.model.get_total_kvcache_bytes()
        self.model.clear_kvcache()

        # Prefill available memory:
        prefill_available_bytes = prefill_proc.get_effective_memory_capacity()

        # Decode available memory:
        decode_available_bytes = decode_proc.get_effective_memory_capacity()

        return {
            "Prefill": MemoryUsage(
                total_weights_bytes + total_prefill_kvcache_bytes,
                prefill_available_bytes,
            ),
            "Decode": MemoryUsage(
                total_weights_bytes + total_kvcache_bytes, decode_available_bytes
            ),
        }

    def memory_usage_paf(self, processors: list[Processor]) -> dict[str, MemoryUsage]:
        assert len(processors) == 3
        prefill_proc, attn_proc, ffn_proc = processors

        # Model parameters:
        total_weights_bytes = self.model.get_total_weights_bytes()

        # Prefill KV-cache size:
        self.model.insert_kvcache(
            self.num_users, self.input_tokens, dtype=DataType.BF16
        )
        total_prefill_kvcache_bytes = self.model.get_total_kvcache_bytes()
        self.model.clear_kvcache()

        # Decode KV-cache size:
        self.model.insert_kvcache(
            self.num_users, self.input_tokens + self.output_tokens, dtype=DataType.BF16
        )
        total_kvcache_bytes = self.model.get_total_kvcache_bytes()
        self.model.clear_kvcache()

        # Prefill available memory:
        prefill_available_bytes = prefill_proc.get_effective_memory_capacity()

        # Attn available memory:
        attn_available_bytes = attn_proc.memory_capacity() * attn_proc.pcfg.num_chips

        # FFN available memory:
        ffn_available_bytes = ffn_proc.get_effective_memory_capacity()

        return {
            "Prefill": MemoryUsage(
                total_weights_bytes + total_prefill_kvcache_bytes,
                prefill_available_bytes,
            ),
            "Attention": MemoryUsage(total_kvcache_bytes, attn_available_bytes),
            "FFN": MemoryUsage(total_weights_bytes, ffn_available_bytes),
        }

    def check_memory_capacity(self, processors: list[Processor]) -> bool:
        memory_usage = self.memory_usage(processors)
        return all(usage.does_fit for usage in memory_usage.values())

    def allocate(self, processors: list[Processor]) -> list[Processor]:
        match len(processors):
            case 1:
                return self.allocate_normal(processors)
            case 2:
                return self.allocate_pd(processors)
            case 3:
                return self.allocate_paf(processors)
            case _:
                raise ValueError(f"Invalid number of processors: {len(processors)}")

    def allocate_normal(self, processors: list[Processor]) -> list[Processor]:
        assert len(processors) == 1
        proc = processors[0]
        assert len(proc.pcfg.get_inferred_dim()) == 1, (
            "Allocator only supports autotuning on one dimension"
        )
        inferred_dim = proc.pcfg.get_inferred_dim()[0]
        proc.pcfg.set_dim(inferred_dim, 1)
        proc_min_num_chips = proc.pcfg.num_chips
        total_num_chips = self.get_num_chips(proc, self.total_budget)
        proc.pcfg.set_dim(inferred_dim, max(1, total_num_chips // proc_min_num_chips))
        return processors

    def allocate_pd(self, processors: list[Processor]) -> list[Processor]:
        raise NotImplementedError("PD allocation is not implemented for this allocator")

    def allocate_paf(self, processors: list[Processor]) -> list[Processor]:
        raise NotImplementedError(
            "PAF allocation is not implemented for this allocator"
        )
