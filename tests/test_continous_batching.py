from llm_spice import Model, Processor, DataType
from llm_spice.execute.online_workload import run_continuous_batching
from llm_spice.execute.scheduler import BatchConfig, UserConfig
from llm_spice.execute.executor import SimpleExecutor

def test_continous_batching():
    model = Model.create("llama3_70b", dtype=DataType.BF16)
    batch_config = BatchConfig(user_configs=[
        UserConfig(request_id=0, context_len=0, seq_len=1024),
        UserConfig(request_id=1, context_len=1024, seq_len=1),
        UserConfig(request_id=2, context_len=1024, seq_len=512),
    ])
    processor = Processor.create("H200")
    executor = SimpleExecutor(processor)
    stats = run_continuous_batching(model, batch_config, executor)
    print(stats.pretty_str())