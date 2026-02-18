from llm_spice.op.operators import MoEFast, MoE
from llm_spice.hardware import Processor
from llm_spice.execute.executor import SimpleExecutor
from llm_spice.utils.common import Tensor, ParallelismConfig as ParCfg

def test_moe():
    num_experts = 128
    num_experts_per_token = 8
    embed_dim = 4096
    moe_intermediate_size = 1536
    moe = MoE(
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        embed_dim=embed_dim,
        moe_intermediate_size=moe_intermediate_size,
    )
    moe_fast = MoEFast(
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        embed_dim=embed_dim,
        moe_intermediate_size=moe_intermediate_size,
    )
    inp = Tensor(shape=(1, 1024, embed_dim))
    _ = moe(inp)
    inp = Tensor(shape=(1, 1024, embed_dim))
    _ = moe_fast(inp)

    hw = Processor.create("H100")
    hw.set_parallelism(ParCfg(tp_size=8, dp_size=8, pp_size=1, ep_size=1))
    executor = SimpleExecutor(hw)
    print(f"MoE: {executor.run_op(moe)}")
    print(f"MoEFast: {executor.run_op(moe_fast)}")

    print(f"MoE num params: {moe.get_total_num_params()}")
    print(f"MoEFast num params: {moe_fast.get_total_num_params()}")

    assert moe.get_total_num_params() == moe_fast.get_total_num_params()