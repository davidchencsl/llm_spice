from llm_spice.model import Model
from llm_spice.utils.registry import MODEL_REGISTRY
from llm_spice.op.operators import Tensor

def test_qwen3_moe():
    model = Model.create("qwen3_235b_a22b")
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    print(model.get_total_num_params()/1e9)

    cfg = Model.get_pretrained_config("qwen3_30b_a3b")
    model = Model.create("qwen3_30b_a3b", hf_config=cfg)
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    print(model.pretty_str())
    print(model.get_total_num_params()/1e9)

def test_llama4():
    model = Model.create("llama4_scout")
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    print(model.pretty_str())
    print(model.get_total_num_params()/1e9)

def test_llama3_405b():
    model = Model.create("llama3_405b")
    inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
    out = model(inp)
    print(model.pretty_str())
    print(model.get_total_num_params()/1e9)

def test_all_models():
    for model_name in MODEL_REGISTRY.keys():
        model = Model.create(model_name)
        inp = Tensor(shape=(1, 1024, model.hf_config.hidden_size))
        out = model(inp)
        print(model.get_total_num_params()/1e9)