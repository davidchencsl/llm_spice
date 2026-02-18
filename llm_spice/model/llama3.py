from transformers import PretrainedConfig
from llm_spice.utils.common import DataType
from llm_spice.model.model import Model
from llm_spice.op.operators import (
    FanOut,
    Linear,
    MultiOp,
    Attention,
    RMSNorm,
    Residual,
    Tensor,
    FFN,
)
from llm_spice.utils.registry import register_model


class DecoderLayer(MultiOp):
    def __init__(self, hf_config: PretrainedConfig, name: str, dtype: DataType):
        super().__init__(name, dtype)
        self.hf_config = hf_config

        self.res_fan_out_1 = FanOut(num_outputs=2, name="ResFanOut_1")
        self.input_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )
        self.attention = Attention(
            embed_dim=hf_config.hidden_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
            gqa_factor=hf_config.num_attention_heads // hf_config.num_key_value_heads,
            dtype=dtype,
        )
        self.residual1 = Residual()
        self.res_fan_out_2 = FanOut(num_outputs=2, name="ResFanOut_2")
        self.post_attention_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )
        self.mlp = FFN(
            embed_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            dtype=dtype,
        )
        self.residual2 = Residual()

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        inp1, inp2 = self.res_fan_out_1(inp)
        inp_normed = self.input_layernorm(inp1)
        attn_out = self.attention(inp_normed)
        res = self.residual1(attn_out, inp2)
        res1, res2 = self.res_fan_out_2(res)
        res_normed = self.post_attention_layernorm(res1)
        mlp_out = self.mlp(res_normed)
        out = self.residual2(mlp_out, res2)
        self.add_output(out)
        return out


@register_model("llama3_8b", "meta-llama/Llama-3.1-8B-Instruct")
@register_model("llama3_70b", "meta-llama/Llama-3.1-70B-Instruct")
@register_model("llama3_405b", "meta-llama/Llama-3.1-405B-Instruct")
class Llama3(Model):
    def __init__(
        self,
        model_id: str,
        dtype: DataType,
        hf_config: PretrainedConfig | None = None,
    ):
        super().__init__(model_id, dtype=dtype, hf_config=hf_config)

        self.layers = [
            DecoderLayer(self.hf_config, f"DecoderLayer_{i}", dtype)
            for i in range(self.hf_config.num_hidden_layers)
        ]

        self.unembedding = Linear(
            input_dim=self.hf_config.hidden_size,
            output_dim=self.hf_config.vocab_size,
            dtype=dtype,
        )

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        for layer in self.layers:
            out = layer(inp)
            inp = out
        unembed = self.unembedding(inp)
        self.add_output(unembed)
        return unembed
