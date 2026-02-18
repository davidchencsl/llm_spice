from llm_spice import Allocator, BalanceAllocator
from llm_spice.execute.executor import SimpleExecutor, AFExecutor
from llm_spice.execute.workload import run_workload_paf
from llm_spice.hardware import Processor
from llm_spice.utils.common import ParallelismConfig as ParCfg
from llm_spice.model.model import Model

import pytest

pytestmark = [pytest.mark.slow]

def test_pd_allocator_tco():
    model = Model.create("llama3_405b")
    allocator = BalanceAllocator(model, 128, 16, 1024, 2000.0, Allocator.Mode.TCO)
    h100 = Processor.create("H100")
    h200 = Processor.create("H200")
    h100.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    h200.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    processors = [h100, h200]
    processors = allocator.allocate(processors)
    total_tco = 0
    for proc in processors:
        print(proc.pcfg)
        total_tco += proc.get_tco(proc.pcfg.num_chips)
    print(total_tco)
    assert abs(total_tco - 2000.0) < 200

def test_pd_allocator_die_count():
    model = Model.create("llama3_405b")
    allocator = BalanceAllocator(model, 128, 16, 1024, 288, Allocator.Mode.DIE_COUNT)
    h100 = Processor.create("H100")
    h200 = Processor.create("H200")
    h100.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    h200.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    processors = [h100, h200]
    processors = allocator.allocate(processors)
    total_chips = 0
    for proc in processors:
        print(proc.pcfg)
        total_chips += proc.pcfg.num_chips
    print(total_chips)

def test_paf_allocator_tco():
    model = Model.create("llama3_405b")
    allocator = BalanceAllocator(model, 128, 16, 1024, 2000.0, Allocator.Mode.TCO)
    h100 = Processor.create("H100")
    h200 = Processor.create("H200")
    vr200 = Processor.create("VR200")
    h100.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    h200.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    vr200.set_parallelism(ParCfg(tp_size=8, dp_size=-1, pp_size=1, ep_size=1))
    processors = [h100, h200, vr200]
    processors = allocator.allocate(processors)
    total_tco = 0
    for proc in processors:
        print(proc.pcfg)
        total_tco += proc.get_tco(proc.pcfg.num_chips)
    print(total_tco)
    assert abs(total_tco - 2000.0) < 200

