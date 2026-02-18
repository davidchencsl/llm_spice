from llm_spice.hardware import Processor
from llm_spice.execute.executor import SimpleExecutor
from llm_spice.model import Model
from llm_spice.op.operators import Attention, FFN, Tensor
from llm_spice.utils.common import DataType, ParallelismConfig as ParCfg


def test_simd_ffn():
    inp = Tensor(shape=(1024, 2048))
    ffn = FFN(2048, 2048 * 4)
    _ = ffn(inp)
    chip = Processor.create("H100")
    executor = SimpleExecutor(chip)
    print(ffn.pretty_str())
    print(executor.run_op(ffn))


def test_simd_attention():
    inp = Tensor(shape=(1024, 2048))
    attn = Attention(2048, 16, 128, 1)
    _ = attn(inp)
    chip = Processor.create("H100")
    executor = SimpleExecutor(chip)
    print(attn.pretty_str())
    print(executor.run_op(attn))


def test_simd_attention_batched():
    inp = Tensor(shape=(5, 1024, 2048))
    attn = Attention(2048, 16, 128, 8)
    _ = attn(inp)
    chip = Processor.create("H100")
    executor = SimpleExecutor(chip)
    print(attn.pretty_str())
    print(executor.run_op(attn))


def test_simd_llama3_70b():
    model = Model.create("llama3_70b")
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    h100 = Processor.create("H100")
    pcfg = ParCfg(tp_size=8, dp_size=1, pp_size=1)
    h100.set_parallelism(pcfg)
    executor = SimpleExecutor(h100)
    print(model.pretty_str())
    stats = executor.run_model(model)
    assert stats.arithmetic_intensity >= 400
    model.forward_pass_done()
    inp = Tensor(shape=(1, 1, model.hf_config.hidden_size))
    out = model(inp)
    stats = executor.run_model(model)
    assert 1 <= stats.arithmetic_intensity <= 2

def test_simd_llama3_70b_fp4():
    cfg = Model.get_pretrained_config("llama3_70b")
    cfg.num_hidden_layers = 1
    model = Model.create("llama3_70b", dtype=DataType.FP4, hf_config=cfg)
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    h100 = Processor.create("H100")
    pcfg = ParCfg(tp_size=8, dp_size=1, pp_size=1)
    h100.set_parallelism(pcfg)
    executor = SimpleExecutor(h100)
    print(model.pretty_str())
    stats = executor.run_model(model)
    
def test_simd_qwen3_30b_a3b():
    cfg = Model.get_pretrained_config("qwen3_30b_a3b")
    cfg.num_hidden_layers = 1
    model = Model.create("qwen3_30b_a3b", dtype=DataType.BF16, hf_config=cfg)
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    h100 = Processor.create("H100")
    pcfg = ParCfg(tp_size=8, dp_size=1, pp_size=1)
    h100.set_parallelism(pcfg)
    executor = SimpleExecutor(h100)
    print(model.pretty_str())
    stats = executor.run_model(model)