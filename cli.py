from llm_spice import (
    Model,
    Processor,
    DataType,
    ParallelismConfig,
    SimpleExecutor,
    Tensor,
    Trace,
)
from llm_spice.execute.executor import AFExecutor
from llm_spice.execute.online_workload import (
    run_online_workload,
    run_online_workload_disaggregated,
)
from llm_spice.execute.scheduler import ASAPScheduler
from llm_spice.execute.workload import (
    run_workload,
    run_workload_paf,
    run_workload_pd,
)
from llm_spice.hardware.memory import Memory
from llm_spice.utils.common import Si, format_value

from fire import Fire
from tabulate import tabulate
import pandas as pd


def parse_parallelism(parallelism: tuple[int, ...]) -> tuple[int, int, int]:
    tp_size = int(parallelism[0]) if len(parallelism) >= 1 else 1
    dp_size = int(parallelism[1]) if len(parallelism) >= 2 else 1
    pp_size = int(parallelism[2]) if len(parallelism) >= 3 else 1
    return tp_size, dp_size, pp_size


def _create_parallelism_config(
    tp_size: int, dp_size: int, pp_size: int
) -> ParallelismConfig:
    """Create a parallelism config"""
    return ParallelismConfig(
        tp_size=tp_size,
        dp_size=dp_size,
        pp_size=pp_size,
    )


def _create_processor(chip: str, parallelism: tuple[int, ...]) -> Processor:
    """Create and configure a processor with the given parallelism."""
    tp_size, dp_size, pp_size = parse_parallelism(parallelism)
    processor = Processor.create(chip)
    processor.set_parallelism(_create_parallelism_config(tp_size, dp_size, pp_size))
    return processor


def _create_model(model_name: str, dtype: str) -> Model:
    """Create a model with the given dtype."""
    hf_config = Model.get_pretrained_config(model_name)
    return Model.create(model_name, hf_config=hf_config, dtype=DataType.from_str(dtype))


class App:
    class run_static:
        def normal(
            self,
            model="llama3_70b",
            dtype="BF16",
            chip="H100",
            parallelism=(1, 1, 1),
            input_tokens=1024,
            output_tokens=1024,
            num_users=1,
        ):
            """Run static batching workload on a model."""
            model = _create_model(model, dtype)
            processor = _create_processor(chip, parallelism)
            executor = SimpleExecutor(processor)
            stats = run_workload(
                model, input_tokens, output_tokens, num_users, executor
            )
            print(stats.pretty_str())

        def pd(
            self,
            model="llama3_70b",
            dtype="BF16",
            prefill_chip="H100",
            decode_chip="H100",
            prefill_parallelism=(1, 1, 1),
            decode_parallelism=(1, 1, 1),
            input_tokens=1024,
            output_tokens=1024,
            num_users=1,
        ):
            """Run static batching workload on a model, with PD disaggregation."""
            model = _create_model(model, dtype)
            prefill_executor = SimpleExecutor(
                _create_processor(prefill_chip, prefill_parallelism)
            )
            decode_executor = SimpleExecutor(
                _create_processor(decode_chip, decode_parallelism)
            )
            stats = run_workload_pd(
                model,
                input_tokens,
                output_tokens,
                num_users,
                prefill_executor,
                decode_executor,
            )
            print(stats.pretty_str())

        def paf(
            self,
            model="llama3_70b",
            dtype="BF16",
            prefill_chip="H100",
            attn_chip="H100",
            ffn_chip="H100",
            prefill_parallelism=(1, 1, 1),
            attn_parallelism=(1, 1, 1),
            ffn_parallelism=(1, 1, 1),
            input_tokens=1024,
            output_tokens=1024,
            num_users=1,
        ):
            """Run static batching workload on a model, with PAF disaggregation."""
            model = _create_model(model, dtype)
            prefill_executor = SimpleExecutor(
                _create_processor(prefill_chip, prefill_parallelism)
            )
            decode_executor = AFExecutor(
                _create_processor(attn_chip, attn_parallelism),
                _create_processor(ffn_chip, ffn_parallelism),
            )
            stats = run_workload_paf(
                model,
                input_tokens,
                output_tokens,
                num_users,
                prefill_executor,
                decode_executor,
            )
            print(stats.pretty_str())

    class run_online:
        def normal(
            self,
            model="llama3_70b",
            dtype="BF16",
            chip="H100",
            parallelism=(1, 1, 1),
            trace="AzureLLMChat2024",
            req_rate=5.0,
            max_num_reqs=10,
        ):
            """Run online workload on a model."""
            model = _create_model(model, dtype)
            executor = SimpleExecutor(_create_processor(chip, parallelism))
            trace = Trace.create(trace)
            scheduler = ASAPScheduler()
            stats = run_online_workload(
                model, trace, executor, scheduler, req_rate, max_num_reqs
            )
            print(stats.pretty_str())

        def pd(
            self,
            model="llama3_70b",
            dtype="BF16",
            prefill_chip="H100",
            decode_chip="H100",
            prefill_parallelism=(1, 1, 1),
            decode_parallelism=(1, 1, 1),
            trace="AzureLLMChat2024",
            req_rate=5.0,
            max_num_reqs=10,
        ):
            """Run online workload on a model with PD disaggregation."""
            model = _create_model(model, dtype)
            prefill_executor = SimpleExecutor(
                _create_processor(prefill_chip, prefill_parallelism)
            )
            decode_executor = SimpleExecutor(
                _create_processor(decode_chip, decode_parallelism)
            )
            trace = Trace.create(trace)
            scheduler = ASAPScheduler()
            stats = run_online_workload_disaggregated(
                model,
                trace,
                prefill_executor,
                decode_executor,
                scheduler,
                req_rate,
                max_num_reqs,
            )
            print(stats.pretty_str())

        def paf(
            self,
            model="llama3_70b",
            dtype="BF16",
            prefill_chip="H100",
            attn_chip="H100",
            ffn_chip="H100",
            prefill_parallelism=(1, 1, 1),
            attn_parallelism=(1, 1, 1),
            ffn_parallelism=(1, 1, 1),
            trace="AzureLLMChat2024",
            req_rate=5.0,
            max_num_reqs=10,
        ):
            """Run online workload on a model with PAF disaggregation."""
            model = _create_model(model, dtype)
            prefill_executor = SimpleExecutor(
                _create_processor(prefill_chip, prefill_parallelism)
            )
            decode_executor = AFExecutor(
                _create_processor(attn_chip, attn_parallelism),
                _create_processor(ffn_chip, ffn_parallelism),
            )
            trace = Trace.create(trace)
            scheduler = ASAPScheduler()
            stats = run_online_workload_disaggregated(
                model,
                trace,
                prefill_executor,
                decode_executor,
                scheduler,
                req_rate,
                max_num_reqs,
            )
            print(stats.pretty_str())

    class show:
        def processor(self, name: str, mode: str = "tco", num_chips: int = 32):
            processor = Processor.create(name)
            processor.set_num_chips(num_chips)
            if mode == "tco":
                print(processor.get_tco_breakdown(num_chips))
            elif mode == "capex":
                print(processor.dc.capex_breakdown())
            elif mode == "power":
                print(processor.dc.power_breakdown())
            elif mode == "cost":
                print(processor.dc.cost_breakdown())

        def model(self, name: str, num_layers: int | None = None, input_tokens: int = 1024, num_users: int = 1, dtype: str = "BF16"):
            hf_config = Model.get_pretrained_config(name)
            if num_layers is not None:
                hf_config.num_hidden_layers = num_layers
            model = Model.create(name, hf_config=hf_config, dtype=DataType.from_str(dtype))
            inp = Tensor(shape=(num_users, input_tokens, model.hf_config.hidden_size))
            _ = model.forward(inp)
            print(model.pretty_str())
        
        def memory(self, name: str):
            memory = Memory.create(name)
            print(memory.pretty_str())

    class list:
        def model(self):
            df = []
            for model_name in Model.get_all_models():
                model = Model.create(model_name)
                info = {}
                info["Name"] = model_name
                info["Total # Parameters"] = format_value(model.get_total_num_params())
                info["# Layers"] = format_value(model.hf_config.num_hidden_layers)
                info["Hidden Size"] = format_value(model.hf_config.hidden_size, "FULL")
                info["Intermediate Size"] = format_value(
                    model.hf_config.intermediate_size, "FULL"
                )
                info["Vocab Size"] = format_value(model.hf_config.vocab_size, "FULL")
                df.append(info)
            df = pd.DataFrame(df)
            print(tabulate(df, headers="keys", tablefmt="rounded_grid"))

        def processor(self):
            df = []
            for hw_name in Processor.get_all_processors():
                hw = Processor.create(hw_name)
                info = {}
                info["Name"] = hw_name
                info["BF16"] = format_value(hw.get_hw_flops(DataType.BF16), "FLOP/s")
                info["FP8"] = format_value(hw.get_hw_flops(DataType.FP8), "FLOP/s")
                info["FP4"] = format_value(hw.get_hw_flops(DataType.FP4), "FLOP/s")
                info["Memory Capacity"] = format_value(hw.memory_capacity(), "iB")
                info["Memory Bandwidth"] = format_value(hw.memory_bw(), "B/s")
                info["C/M (FP8)"] = format_value(
                    hw.get_hw_flops(DataType.FP8) / hw.memory_bw(), "FLOP/B"
                )
                info["Area"] = format_value(hw.total_area(), "mm^2")
                info["Die Yield"] = format_value(hw.chip_yield_rate(), "%")
                info["Power"] = format_value(hw.total_power(), "W")
                info["Cost"] = format_value(hw.total_cost(), "$")
                df.append(info)
            df = pd.DataFrame(df)
            print(tabulate(df, headers="keys", tablefmt="rounded_grid"))
        
        def memory(self):
            df = []
            for memory_name in Memory.get_all_memory():
                memory = Memory.create(memory_name)
                info = {}
                info["Name"] = memory_name
                info["Capacity"] = format_value(memory.capacity, "iB")
                info["Bandwidth"] = format_value(memory.memory_bw, "iB/s")
                info["Power"] = format_value(memory.power, "W")
                info["Cost"] = format_value(memory.cost, "$")
                info["Cost/GB"] = format_value(memory.cost / (memory.capacity / Si.Gi), "$/GiB")
                info["Cost/GBps"] = format_value(memory.cost / (memory.memory_bw / Si.Gi), "$/GiBps")
                info["W/GB"] = format_value(memory.power / (memory.capacity / Si.Gi), "W/GiB")
                info["W/GBps"] = format_value(memory.power / (memory.memory_bw / Si.Gi), "W/GiBps")
                df.append(info)
            print(tabulate(df, headers="keys", tablefmt="rounded_grid"))


if __name__ == "__main__":
    Fire(App)
