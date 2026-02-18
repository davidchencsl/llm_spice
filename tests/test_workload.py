from llm_spice.execute.executor import AFExecutor, SimpleExecutor
from llm_spice.utils.common import DataType, ParallelismConfig
from llm_spice.model import Model
from llm_spice.hardware import Processor
from llm_spice.execute.workload import run_workload_paf, run_workload_pd
import pytest


def test_pd_asymmetric():
    model = Model.create("llama3_70b")
    h100 = Processor.create("H100")
    h100.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    prefill_executor = SimpleExecutor(h100)
    h200 = Processor.create("H200")
    h200.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    decode_executor = SimpleExecutor(h200)
    stats = run_workload_pd(model, 1024, 1, 6, prefill_executor, decode_executor)
    assert stats.prefill_stats.duration >= 0
    assert stats.decode_stats.duration >= 0


def test_pd_asymmetric_fp4():
    model = Model.create("llama3_70b", dtype=DataType.FP4)
    h100 = Processor.create("H100")
    h100.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    prefill_executor = SimpleExecutor(h100)
    h200 = Processor.create("H200")
    h200.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    decode_executor = SimpleExecutor(h200)
    stats = run_workload_pd(model, 1024, 1, 6, prefill_executor, decode_executor)
    assert stats.prefill_stats.duration >= 0
    assert stats.decode_stats.duration >= 0


def test_pd_asymmetric_moe():
    model = Model.create("qwen3_30b_a3b", dtype=DataType.BF16)
    h100 = Processor.create("H100")
    h100.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    prefill_executor = SimpleExecutor(h100)
    h200 = Processor.create("H200")
    h200.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    decode_executor = SimpleExecutor(h200)
    stats = run_workload_pd(model, 1024, 1, 6, prefill_executor, decode_executor)
    assert stats.prefill_stats.duration >= 0
    assert stats.decode_stats.duration >= 0


def test_paf_stress_test():
    prefill_hw = Processor.create("VR200-CPX")
    prefill_hw.set_parallelism(ParallelismConfig(tp_size=8, dp_size=136//8))
    attn_hw = Processor.create("VR200")
    attn_hw.set_parallelism(ParallelismConfig(tp_size=8, dp_size=80//8))
    ffn_hw = Processor.create("VR200-CPX")
    ffn_hw.set_parallelism(ParallelismConfig(tp_size=8, dp_size=136//8))
    af_executor = AFExecutor(attn_hw, ffn_hw)
    prefill_executor = SimpleExecutor(prefill_hw)
    af_executor = AFExecutor(attn_hw, ffn_hw)
    for model in ["llama3_70b", "llama3_405b"]:
        model = Model.create(model, dtype=DataType.FP8)
        _ = run_workload_paf(model, 1024, 2, 2176, prefill_executor, af_executor)


def test_pd_fast_mode():
    model = Model.create("llama3_70b", dtype=DataType.BF16)
    h100 = Processor.create("H100")
    h100.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    prefill_executor = SimpleExecutor(h100)
    h200 = Processor.create("H200")
    h200.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    decode_executor = SimpleExecutor(h200)
    stats = run_workload_pd(model, 1024, 16, 6, prefill_executor, decode_executor, fast_mode=False)
    model = Model.create("llama3_70b", dtype=DataType.BF16)
    fast_stats = run_workload_pd(model, 1024, 16, 6, prefill_executor, decode_executor, fast_mode=True)

    assert fast_stats.decode_stats.duration == pytest.approx(stats.decode_stats.duration, rel=0.1)
