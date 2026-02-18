from llm_spice.execute.executor import SimpleExecutor, AFExecutor
from llm_spice.execute.workload import run_workload_paf, run_workload_pd
from llm_spice.hardware import Processor
from llm_spice.utils.common import ParallelismConfig as ParCfg
from llm_spice.model import Model
from llm_spice.allocator import Allocator, BalanceAllocator

import pytest

pytestmark = [pytest.mark.slow]

def test_pd_allocator_edge_case():
    model = Model.create("llama3_405b")
    input_tokens = 8192
    output_tokens = 1024
    num_users = 1024
    allocator = BalanceAllocator(
        model=model, 
        input_tokens=input_tokens, 
        output_tokens=output_tokens, 
        num_users=num_users, 
        total_budget=80 * 4 * 858, 
        alloc_mode=Allocator.Mode.DIE_AREA
    )
    prefill = Processor.create("VR200")
    decode = Processor.create("VR200")
    prefill.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    decode.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    prefill, decode = allocator.allocate([prefill, decode])
    prefill_executor = SimpleExecutor(prefill)
    decode_executor = SimpleExecutor(decode)
    stats = run_workload_pd(model, input_tokens, output_tokens, num_users, prefill_executor, decode_executor)
    print(prefill.pcfg)
    print(decode.pcfg)
    print(stats)
    prefill_duration = stats.prefill_stats.duration
    decode_duration = stats.decode_stats.duration
    assert prefill_duration == pytest.approx(decode_duration, rel=0.3)

def test_paf_allocator_edge_case():
    model = Model.create("llama3_405b")
    input_tokens = 8192
    output_tokens = 8192
    num_users = 1024
    allocator = BalanceAllocator(
        model=model, 
        input_tokens=input_tokens, 
        output_tokens=output_tokens, 
        num_users=num_users, 
        total_budget=160 * 4 * 858, 
        alloc_mode=Allocator.Mode.DIE_AREA
    )
    cpx = Processor.create("VR200")
    pnm = Processor.create("6xHBM-PNM-8H")
    ffn = Processor.create("VR200")
    cpx.set_parallelism(ParCfg(tp_size=-1, dp_size=1, pp_size=1, ep_size=1))
    pnm.set_parallelism(ParCfg(tp_size=1, dp_size=-1, pp_size=1, ep_size=1))
    ffn.set_parallelism(ParCfg(tp_size=-1, dp_size=1, pp_size=1, ep_size=1))
    cpx, pnm, ffn = allocator.allocate([cpx, pnm, ffn])
    prefill_executor = SimpleExecutor(cpx)
    decode_executor = AFExecutor(pnm, ffn)
    stats = run_workload_paf(model, input_tokens, output_tokens, num_users, prefill_executor, decode_executor)
    print(cpx.pcfg)
    print(pnm.pcfg)
    print(ffn.pcfg)
    print(stats)
    prefill_duration = stats.prefill_stats.duration
    assert stats.decode_stats.extra_info is not None
    assert stats.decode_stats.extra_info["attn_stats"] is not None
    assert stats.decode_stats.extra_info["ffn_stats"] is not None
    attn_duration = stats.decode_stats.extra_info["attn_stats"].duration
    ffn_duration = stats.decode_stats.extra_info["ffn_stats"].duration
    assert prefill_duration == pytest.approx(attn_duration, rel=0.3)
