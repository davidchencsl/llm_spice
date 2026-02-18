from functools import cached_property
from llm_spice.utils.trace import Request
from dataclasses import dataclass
from typing import List
from enum import Enum
from collections import deque


@dataclass(slots=True)
class UserConfig:
    request_id: int
    context_len: int  # KV cache length
    seq_len: int  # > 1 for Prefill, = 1 for Decode


@dataclass(slots=True)
class BatchConfig:
    user_configs: List[UserConfig]

    def total_tokens(self) -> int:
        # Local binding avoids global lookup in tight loops
        uc = self.user_configs
        return sum(x.seq_len for x in uc)


class SchedulerMode(Enum):
    PD = "PD"
    P = "P"
    D = "D"

    @cached_property
    def has_prefill(self) -> bool:
        return self == SchedulerMode.PD or self == SchedulerMode.P

    @cached_property
    def has_decode(self) -> bool:
        return self == SchedulerMode.PD or self == SchedulerMode.D


class Scheduler:
    def __init__(self):
        self.queue: deque[Request] = deque()
        self.in_flight_reqs: dict[Request, dict] = {}
        self.completed_reqs: List[Request] = []

    def enqueue(self, request: Request, current_time: float):
        request.enqueue_time = current_time
        self.queue.append(request)

    def has_pending(self):
        return len(self.queue) > 0

    def step(self, current_time: float, mode: SchedulerMode) -> BatchConfig | None:
        raise NotImplementedError


class ASAPScheduler(Scheduler):
    def __init__(
        self,
        max_num_seqs: int | float = float("inf"),
        max_num_batched_tokens: int | float = float("inf"),
        decode_chunk_size: int = 1,
    ):
        """
        Parameters:
            max_num_seqs: maximum number of sequences allowed
            max_num_batched_tokens: maximum number of total tokens allowed
            decode_chunk_size: number of speculative tokens that need to be verified per decode pass
        """
        super().__init__()
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.decode_chunk_size = decode_chunk_size

        assert max_num_seqs <= max_num_batched_tokens, (
            "max_num_seqs must be less than or equal to max_num_batched_tokens"
        )

    def has_pending(self):
        return len(self.in_flight_reqs) > 0 or len(self.queue) > 0

    def step(
        self,
        current_time: float,
        mode: SchedulerMode,
    ) -> BatchConfig | None:
        if not self.has_pending():
            return None

        # Shove everything from queue to in_flight_reqs within the max_num_batched_tokens limit
        while self.queue and len(self.in_flight_reqs) < self.max_num_seqs:
            req = self.queue.popleft()
            self.in_flight_reqs[req] = {
                "input_tokens": req.input_tokens,
                "output_tokens": req.output_tokens,
                "context_len": 0,
            }

        num_batched_tokens = 0

        user_configs = []
        done_reqs: List[Request] = []
        decode_reqs: List[tuple[Request, dict]] = []
        prefill_reqs: List[tuple[Request, dict]] = []

        for req, info in self.in_flight_reqs.items():
            # First, check if the request is done
            if info["input_tokens"] <= 0 and info["output_tokens"] <= 0:
                done_reqs.append(req)
            # Then decode
            elif (
                info["input_tokens"] == 0
                and info["output_tokens"] > 0
                and mode.has_decode
            ):
                decode_reqs.append((req, info))
            # Then prefill
            elif info["input_tokens"] > 0 and mode.has_prefill:
                prefill_reqs.append((req, info))

        # Process all decode requests
        for req, info in decode_reqs:
            if num_batched_tokens >= self.max_num_seqs:
                break
            num_batched_tokens += self.decode_chunk_size
            if req.prefill_finish_time == 0:
                req.prefill_finish_time = current_time
            decode_chunk_size = min(info["output_tokens"], self.decode_chunk_size)
            user_configs.append(
                UserConfig(
                    request_id=req.id,
                    context_len=info["context_len"],
                    seq_len=decode_chunk_size,
                )
            )
            info["output_tokens"] -= decode_chunk_size
            info["context_len"] += decode_chunk_size

        # Process all prefill requests
        for req, info in prefill_reqs:
            if num_batched_tokens >= self.max_num_batched_tokens:
                break
            processed_tokens = int(
                min(
                    info["input_tokens"],
                    self.max_num_batched_tokens - num_batched_tokens,
                )
            )
            num_batched_tokens += processed_tokens
            user_configs.append(
                UserConfig(request_id=req.id, context_len=0, seq_len=processed_tokens)
            )
            if req.prefill_start_time == 0:
                req.prefill_start_time = current_time
            info["context_len"] += processed_tokens
            info["input_tokens"] -= processed_tokens

        for req in done_reqs:
            req.dequeue_time = current_time
            self.completed_reqs.append(req)
            del self.in_flight_reqs[req]

        return BatchConfig(user_configs=user_configs) if user_configs else None
