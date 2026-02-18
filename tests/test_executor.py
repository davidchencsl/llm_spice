from llm_spice.execute.executor import SimpleExecutor, AFExecutor
from llm_spice.hardware import Processor
from llm_spice.utils.common import ParallelismConfig as ParCfg
from llm_spice.model import Model
from llm_spice.op.operators import Tensor

import pytest

def test_af_executor():
    h100 = Processor.create("H100")
    h100.set_parallelism(ParCfg(tp_size=8, dp_size=1, pp_size=1))
    h200 = Processor.create("H200")
    h200.set_parallelism(ParCfg(tp_size=8, dp_size=1, pp_size=1))
    af_executor = AFExecutor(h100, h200)
    model = Model.create("llama4_scout")
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    stats = af_executor.run_model(model)
    assert stats.extra_info is not None
    assert stats.extra_info["attn_stats"] is not None
    assert stats.extra_info["ffn_stats"] is not None
    assert pytest.approx(stats.extra_info["attn_stats"].duration + stats.extra_info["ffn_stats"].duration, rel=1e-3) == stats.duration