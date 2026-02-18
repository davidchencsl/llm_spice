from collections import defaultdict
import math
import random
import heapq
import logging
from llm_spice import (
    Model,
    OpRunStats,
    Trace,
    Executor,
    Tensor,
    DataType,
)
from llm_spice.execute.scheduler import Scheduler, BatchConfig, SchedulerMode
from llm_spice.utils.trace import Request
from llm_spice.utils.common import format_value
from llm_spice.utils.experiments import mp_tqdm


from dataclasses import dataclass
from typing import List, Callable, Literal

import numpy as np


@dataclass
class DynamicWorkloadStats:
    stats: OpRunStats
    requests: List[Request]
    executors_active_ratio: List[float]
    req_throughput: float
    throughput: float
    total_tco: float
    cost_per_1m_tokens: float
    prefill_stats: OpRunStats | None = None
    decode_stats: OpRunStats | None = None

    def apply_warmup(self, num_warmup_reqs: int | float = 0.1):
        if isinstance(num_warmup_reqs, float):
            assert 0 <= num_warmup_reqs < 1, "num_warmup_reqs must be between 0 and 1"
            num_warmup_reqs = int(num_warmup_reqs * len(self.requests))
        assert 0 <= num_warmup_reqs < len(self.requests), (
            "num_warmup_reqs must be between 0 and len(requests)"
        )
        self.requests = self.requests[num_warmup_reqs:]

    def get_distribution(self, key: str):
        values = np.array([getattr(req, key) for req in self.requests])
        result = list(np.percentile(values, [25, 50, 75, 90, 95, 99])) + [
            np.min(values),
            np.max(values),
            np.mean(values),
        ]
        names = ["P25", "P50", "P75", "P90", "P95", "P99", "Min", "Max", "Avg"]
        return {name: float(x) for name, x in zip(names, result)}

    def pretty_str(self):
        ttft_distribution = self.get_distribution("ttft")
        ttc_distribution = self.get_distribution("ttc")
        tpot_distribution = self.get_distribution("tpot")
        if self.prefill_stats is not None and self.decode_stats is not None:
            assert self.decode_stats.extra_info is not None
            pd_stats = (
                ["Prefill Stats:"]
                + [
                    f"{' ' * 2}{line}"
                    for line in self.prefill_stats.pretty_str().split("\n")
                ]
                if self.prefill_stats is not None
                else []
            ) + (
                ["Decode Stats:"]
                + [
                    f"{' ' * 2}{line}"
                    for line in self.decode_stats.pretty_str().split("\n")
                ]
                if self.decode_stats is not None
                else []
            )

            if self.decode_stats.extra_info is not None:
                pd_stats += (
                    [f"{' ' * 2}Attn Stats:"]
                    + [
                        f"{' ' * 4}{line}"
                        for line in self.decode_stats.extra_info["attn_stats"]
                        .pretty_str()
                        .split("\n")
                    ]
                    + [f"{' ' * 2}FFN Stats:"]
                    + [
                        f"{' ' * 4}{line}"
                        for line in self.decode_stats.extra_info["ffn_stats"]
                        .pretty_str()
                        .split("\n")
                    ]
                )
        else:
            pd_stats = []

        lines = (
            [
                "Dynamic Workload Stats:",
                f"Executors Active Ratio: {[format_value(x, '%') for x in self.executors_active_ratio]}",
                f"Request Throughput: {format_value(self.req_throughput, 'req/s')}",
                f"Throughput: {format_value(self.throughput, 'tokens/s')}",
                f"Total TCO: {format_value(self.total_tco, '$/hr')}",
                f"Cost/1M tokens: {format_value(self.cost_per_1m_tokens, '$')}",
            ]
            + [f"{' ' * 2}{line}" for line in self.stats.pretty_str().split("\n")]
            + pd_stats
            + [f"Total # Requests: {len(self.requests)}", "TTFT:"]
            + [
                f"{' ' * 2}{name}: {format_value(value, 's')}"
                for name, value in ttft_distribution.items()
            ]
            + [
                "TTC:",
            ]
            + [
                f"{' ' * 2}{name}: {format_value(value, 's')}"
                for name, value in ttc_distribution.items()
            ]
            + [
                "TPOT:",
            ]
            + [
                f"{' ' * 2}{name}: {format_value(value, 's')}"
                for name, value in tpot_distribution.items()
            ]
        )
        return "\n".join(lines)


class Bucket:
    def __init__(
        self,
        size: int,
        mode: Literal["Linear", "Exp", "Hybrid"],
        threshold: int = 1024,  # Switch point
        large_bucket_size: int = 256,  # Linear size for large values
    ):
        self.size = size
        self.mode = mode
        self.threshold = threshold
        self.large_bucket_size = large_bucket_size

    def fit(self, val: int):
        if val == 0:
            return 0
        if self.mode == "Linear":
            return math.ceil(val / self.size) * self.size
        elif self.mode == "Exp":
            return self.size ** math.ceil(math.log(val, self.size))
        elif self.mode == "Hybrid":
            # Use exponential up to threshold, then linear
            if val <= self.threshold:
                return self.size ** math.ceil(math.log(val, self.size))
            else:
                # Round to nearest large_bucket_size above threshold
                return math.ceil(val / self.large_bucket_size) * self.large_bucket_size

        else:
            raise ValueError(f"Invalid mode: {self.mode}")


def run_continuous_batching(
    model: Model,
    batch_config: BatchConfig,
    executor: Executor,
    context_bucket: Bucket = Bucket(size=2, mode="Hybrid"),
):
    processor_list = executor.get_processor()
    ffn_dp_size = processor_list[len(processor_list) - 1].pcfg.dp_size
    attn_dp_size = processor_list[0].pcfg.dp_size

    # First compute the cost of batchable FFNs
    model.clear_kvcache()
    inp = Tensor(
        shape=(
            max(1, math.ceil(batch_config.total_tokens() / ffn_dp_size)),
            1,
            model.hf_config.hidden_size,
        )
    )
    model(inp)
    stats = executor.run_model(model)
    model.forward_pass_done()
    assert stats.extra_info is not None
    ffn_stats: OpRunStats = stats.extra_info["ffn_stats"]

    # Then compute the attention part
    overall_attn_stats = OpRunStats()
    users_bucket = defaultdict(lambda: defaultdict(list))

    # Group users by context length (bucketize and pad align)
    for u in batch_config.user_configs:
        users_bucket[u.seq_len][context_bucket.fit(u.context_len)].append(u)

    # Process each group together
    for seq_len, users_by_seq_len in users_bucket.items():
        for context_len, user_configs in users_by_seq_len.items():
            num_users = len(user_configs)
            model.clear_kvcache()
            model.insert_kvcache(num_users, context_len, dtype=DataType.BF16)
            inp = Tensor(shape=(num_users, seq_len, model.hf_config.hidden_size))
            model(inp)
            stats = executor.run_model(model)
            model.forward_pass_done()
            assert stats.extra_info is not None
            attn_stats = stats.extra_info["attn_stats"]
            overall_attn_stats = overall_attn_stats.merge(attn_stats)

    overall_attn_stats.compute_time /= attn_dp_size
    overall_attn_stats.memory_time /= attn_dp_size
    overall_attn_stats.duration /= attn_dp_size
    overall_attn_stats.flop /= attn_dp_size
    overall_attn_stats.memory_access_bytes /= attn_dp_size

    final_stats = ffn_stats.merge(overall_attn_stats)
    final_stats.extra_info = {}
    final_stats.extra_info["ffn_stats"] = ffn_stats
    final_stats.extra_info["attn_stats"] = overall_attn_stats

    return final_stats


def run_continuous_prefill(
    model: Model,
    batch_config: BatchConfig,
    executor: Executor,
    attention_batch_bucket: Bucket,
):
    processor = executor.get_processor()[0]
    model.clear_kvcache()
    inp = Tensor(
        shape=(
            max(1, math.ceil(batch_config.total_tokens() / processor.pcfg.dp_size)),
            1,
            model.hf_config.hidden_size,
        )
    )
    model(inp)
    stats = executor.run_model(model)
    model.forward_pass_done()
    return stats


def run_continuous_decode(
    model: Model,
    batch_config: BatchConfig,
    executor: Executor,
    attention_batch_bucket: Bucket,
):
    assert all(user_config.seq_len == 1 for user_config in batch_config.user_configs)
    stats = run_continuous_batching(
        model, batch_config, executor, attention_batch_bucket
    )
    return stats


# Event types
ARRIVAL = "arrival"
EXECUTOR_FREE = "executor_free"
PREFILL_FREE = "prefill_free"
DECODE_FREE = "decode_free"


@dataclass(order=True)
class Event:
    time: float
    order: int
    kind: str


@dataclass
class SimExecutorConfig:
    event_name: str
    mode: SchedulerMode
    executor: Executor
    run_fn: Callable[[Model, BatchConfig, Executor, Bucket], OpRunStats]


def run_event_driven_simulation(
    model: Model,
    trace: Trace,
    scheduler: Scheduler,
    req_rate: float,
    max_num_reqs: int | None,
    executor_configs: List[SimExecutorConfig],
    init_num_reqs: int,
    attention_batch_bucket: Bucket,
) -> DynamicWorkloadStats:
    """
    Core event-driven simulation loop.

    Args:
        executor_configs: List of (event_name, mode, executor, run_fn) tuples
            - event_name: name of the FREE event (e.g., "EXECUTOR_FREE", "PREFILL_FREE")
            - mode: scheduler mode string (e.g., "PD", "P", "D")
            - executor: Executor instance
            - run_fn: function(model, batch_config, executor) -> OpRunStats
    """
    assert req_rate >= 0.0, "req_rate must be non-negative"

    # Reset upstream state
    trace.reset()
    trace_iter = iter(trace)

    pbar = mp_tqdm(total=max_num_reqs)

    pbar_stats = {
        "current_time": 0.0,
        "HOL_output_tokens_left": 0,
        "inflight_reqs": 0,
        "queued_reqs": 0,
    }

    def update_pbar_stats(**kwargs):
        pbar_stats.update(kwargs)
        pbar.set_postfix(pbar_stats)

    # Initialize stats for each executor
    executor_stats = [OpRunStats() for _ in executor_configs]
    executors_busy_until = [0.0 for _ in executor_configs]
    executors_active_duration = [0.0 for _ in executor_configs]

    current_time = 0.0
    events: list[Event] = []
    order_counter = 0

    def push_event(t: float, kind: str):
        nonlocal order_counter
        heapq.heappush(events, Event(time=t, order=order_counter, kind=kind))
        order_counter += 1

    def schedule_next_arrival(start_t: float):
        if req_rate <= 0.0:
            return
        inter = random.expovariate(req_rate)
        push_event(start_t + inter, ARRIVAL)

    def make_try_launch(
        idx: int,
        event_name: str,
        mode: SchedulerMode,
        executor: Executor,
        run_fn: Callable[[Model, BatchConfig, Executor, Bucket], OpRunStats],
    ):
        def try_launch(now_t: float):
            nonlocal executors_busy_until, executor_stats, executors_active_duration
            if executors_busy_until[idx] > now_t:
                return
            batch_config = scheduler.step(now_t, mode)
            if batch_config is not None:
                batch_reqs_info = {
                    f"{mode.value}_batch_reqs": len(batch_config.user_configs),
                }
                update_pbar_stats(
                    **batch_reqs_info,
                    inflight_reqs=len(scheduler.in_flight_reqs),
                    queued_reqs=len(scheduler.queue),
                    HOL_output_tokens_left=next(
                        iter(scheduler.in_flight_reqs.values())
                    )["output_tokens"],
                )
                stats = run_fn(model, batch_config, executor, attention_batch_bucket)
                assert stats.duration >= 0, (
                    f"Running {model.name} on {executor.get_processor()} got negative duration {stats.duration} at time {now_t}"
                )
                executors_busy_until[idx] = now_t + stats.duration
                executors_active_duration[idx] += stats.duration
                executor_stats[idx] = executor_stats[idx].merge(stats)
                if stats.extra_info is not None:
                    if executor_stats[idx].extra_info is None:
                        executor_stats[idx].extra_info = stats.extra_info
                    else:
                        executor_stats[idx].extra_info["attn_stats"] = (  # type: ignore
                            executor_stats[idx]  # type: ignore
                            .extra_info["attn_stats"]
                            .merge(stats.extra_info["attn_stats"])
                        )
                        executor_stats[idx].extra_info["ffn_stats"] = (  # type: ignore
                            executor_stats[idx]  # type: ignore
                            .extra_info["ffn_stats"]
                            .merge(stats.extra_info["ffn_stats"])
                        )
                push_event(executors_busy_until[idx], event_name)

        return try_launch

    # Create launch functions and seed initial events
    launch_fns = {}
    for idx, config in enumerate(executor_configs):
        launch_fns[config.event_name] = make_try_launch(
            idx, config.event_name, config.mode, config.executor, config.run_fn
        )
        push_event(current_time, config.event_name)

    num_arrivals = 0
    trace_exhausted = False

    def spawn_request():
        nonlocal num_arrivals, trace_exhausted
        if not trace_exhausted:
            try:
                request = next(trace_iter)
                scheduler.enqueue(request, current_time)
                num_arrivals += 1
            except StopIteration:
                trace_exhausted = True

    for _ in range(init_num_reqs):
        spawn_request()

    schedule_next_arrival(current_time)

    while events:
        ev = heapq.heappop(events)
        current_time = ev.time
        update_pbar_stats(
            current_time=current_time,
            inflight_reqs=len(scheduler.in_flight_reqs),
            queued_reqs=len(scheduler.queue),
        )

        num_completed_reqs = max(0, len(scheduler.completed_reqs) - init_num_reqs)
        pbar.update(num_completed_reqs - pbar.n)

        # Check termination
        if max_num_reqs is not None and num_completed_reqs >= max_num_reqs:
            pbar.close()
            break

        if ev.kind == ARRIVAL:
            spawn_request()
            schedule_next_arrival(current_time)
            # Try to launch all executors
            for launch_fn in launch_fns.values():
                launch_fn(current_time)

        elif ev.kind in launch_fns:
            for launch_fn in launch_fns.values():
                launch_fn(current_time)

    logging.info(f"Simulation finished at time {current_time:.6f}.")

    # Combine stats
    combined_stats = OpRunStats()
    for stats in executor_stats:
        combined_stats = combined_stats.merge(stats)

    # TODO: How to get rid of warmup time?
    total_duration = current_time - 0.0

    req_throughput = len(scheduler.completed_reqs) / total_duration

    throughput = (
        sum(
            request.input_tokens + request.output_tokens
            for request in scheduler.completed_reqs
        )
        / total_duration
    )

    total_tco = sum(
        sum(p.get_tco(p.pcfg.num_chips) for p in e.executor.get_processor())
        for e in executor_configs
    )

    cost_per_1m_tokens = total_tco / (throughput * 3600 / 1e6)

    # Return appropriate stats based on number of executors
    if len(executor_configs) == 1:
        return DynamicWorkloadStats(
            requests=scheduler.completed_reqs,
            stats=combined_stats,
            executors_active_ratio=[
                d / total_duration for d in executors_active_duration
            ],
            req_throughput=req_throughput,
            throughput=throughput,
            total_tco=total_tco,
            cost_per_1m_tokens=cost_per_1m_tokens,
        )
    else:
        return DynamicWorkloadStats(
            requests=scheduler.completed_reqs,
            stats=combined_stats,
            prefill_stats=executor_stats[0] if len(executor_stats) > 0 else None,
            decode_stats=executor_stats[1] if len(executor_stats) > 1 else None,
            executors_active_ratio=[
                d / total_duration for d in executors_active_duration
            ],
            req_throughput=req_throughput,
            throughput=throughput,
            total_tco=total_tco,
            cost_per_1m_tokens=cost_per_1m_tokens,
        )


def run_online_workload(
    model: Model,
    trace: Trace,
    executor: Executor,
    scheduler: Scheduler,
    req_rate: float = 5.0,  # requests per second (Poisson arrivals)
    max_num_reqs: int | None = None,  # cap on number of arrivals consumed from `trace`
    init_num_reqs: int = 0,  # number of requests to enqueue initially
    attention_batch_bucket: Bucket = Bucket(size=2, mode="Hybrid"),
) -> DynamicWorkloadStats:
    """
    Event-driven online workload simulation with a single executor.

    Events:
      - ARRIVAL: draw interarrival time ~ Exp(req_rate), pull next request from `trace`,
                 enqueue into `scheduler`.
      - EXECUTOR_FREE: executor finished the last batch; try to schedule & launch next.

    Time advances by jumping directly to the next event timestamp.
    """
    executor_configs = [
        SimExecutorConfig(
            EXECUTOR_FREE, SchedulerMode.PD, executor, run_continuous_batching
        )
    ]
    return run_event_driven_simulation(
        model,
        trace,
        scheduler,
        req_rate,
        max_num_reqs,
        executor_configs,
        init_num_reqs,
        attention_batch_bucket,
    )


def run_online_workload_disaggregated(
    model: Model,
    trace: Trace,
    prefill_executor: Executor,
    decoder_executor: Executor,
    scheduler: Scheduler,
    req_rate: float = 5.0,  # requests per second (Poisson arrivals)
    max_num_reqs: int | None = None,  # cap on number of arrivals consumed from `trace`
    init_num_reqs: int = 0,  # number of requests to enqueue initially
    attention_batch_bucket: Bucket = Bucket(size=2, mode="Hybrid"),
) -> DynamicWorkloadStats:
    """
    Event-driven online workload simulation with disaggregated prefill/decode executors.

    Events:
      - ARRIVAL: draw interarrival time ~ Exp(req_rate), pull next request from `trace`,
                 enqueue into `scheduler`.
      - PREFILL_FREE: prefill executor finished; try to schedule & launch next prefill batch.
      - DECODE_FREE: decode executor finished; try to schedule & launch next decode batch.

    Time advances by jumping directly to the next event timestamp.
    """
    executor_configs = [
        SimExecutorConfig(
            PREFILL_FREE, SchedulerMode.P, prefill_executor, run_continuous_prefill
        ),
        SimExecutorConfig(
            DECODE_FREE, SchedulerMode.D, decoder_executor, run_continuous_decode
        ),
    ]
    return run_event_driven_simulation(
        model,
        trace,
        scheduler,
        req_rate,
        max_num_reqs,
        executor_configs,
        init_num_reqs,
        attention_batch_bucket,
    )
