from llm_spice import Model, DataType, Processor, ParallelismConfig
from llm_spice.execute.online_workload import run_online_workload, run_online_workload_disaggregated
from llm_spice.utils.trace import Trace
from llm_spice.execute.scheduler import Scheduler, ASAPScheduler
from llm_spice.execute.executor import SimpleExecutor, AFExecutor
from llm_spice.utils.common import fix_seed

import pytest
pytestmark = [pytest.mark.slow]

def test_online_workload():
    fix_seed()
    model = Model.create("llama3_70b", dtype=DataType.BF16)
    trace = Trace.create("AzureLLMCode2024")
    chip = Processor.create("H200")
    chip.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    executor = SimpleExecutor(chip)
    scheduler = ASAPScheduler()
    stats = run_online_workload(model, trace, executor, scheduler, max_num_reqs=10)
    print(stats.pretty_str())

def test_online_workload_pd():
    fix_seed()
    model = Model.create("llama3_70b", dtype=DataType.BF16)
    trace = Trace.create("AzureLLMCode2024")
    chip = Processor.create("H200")
    chip.set_parallelism(ParallelismConfig(tp_size=8, dp_size=1, pp_size=1))
    prefill_executor = SimpleExecutor(chip)
    decoder_executor = SimpleExecutor(chip)
    scheduler = ASAPScheduler()
    stats = run_online_workload_disaggregated(model, trace, prefill_executor, decoder_executor, scheduler, max_num_reqs=10)
    print(stats.pretty_str())
