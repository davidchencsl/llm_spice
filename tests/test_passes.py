from llm_spice.model import Model
from llm_spice.passes.af_transfer import AFTransferPass
from llm_spice.op.operators import AFTransfer
from llm_spice.utils.common import ParallelismConfig as ParCfg, Tensor


def test_af_transfer_pass_is_idempotent_on_model():
    hf_config = Model.get_pretrained_config("llama3_70b")
    hf_config.num_hidden_layers = 1
    model = Model.create("llama3_70b", hf_config=hf_config)
    attn_pcfg = ParCfg(tp_size=8, dp_size=2, pp_size=1, ep_size=1)
    ffn_pcfg = ParCfg(tp_size=8, dp_size=1, pp_size=1, ep_size=1)
    AFTransferPass(attn_pcfg, ffn_pcfg).apply(model)
    inp = Tensor(shape=(10, 1024, model.hf_config.hidden_size))
    out = model(inp)
    print(model.pretty_str())


