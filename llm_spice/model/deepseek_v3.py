from transformers import PretrainedConfig
from llm_spice.utils.common import DataType, Tensor
from llm_spice.model.model import Model
from llm_spice.op.operators import (
    FFN,
    FanOut,
    Linear,
    MultiOp,
    Attention,
    RMSNorm,
    Residual,
    Split,
    Reshape,
    Concat,
    NoOp,
    GQA,
    MoEFast as MoE,
)
from llm_spice.utils.registry import register_model


class RepeatAlongAxis(NoOp):
    def __init__(self, axis: int, repeats: int, name: str = ""):
        super().__init__(name)
        self.axis = axis
        self.repeats = repeats

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        ndim = len(inp.shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for tensor with {ndim} dims"
            )
        if inp.shape[axis] != 1:
            raise ValueError("RepeatAlongAxis requires a singleton dimension to repeat")
        new_shape = list(inp.shape)
        new_shape[axis] = self.repeats
        out = Tensor(tuple(new_shape), dtype=inp.dtype)
        self.add_output(out)
        return out


class MLA(Attention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        v_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super(Attention, self).__init__(name=name, dtype=dtype)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_rope_head_dim + qk_nope_head_dim

        self.kvcache: Tensor | None = None

        self.kvq_fanout = FanOut(num_outputs=2, name="kvq_fanout")

        self.kv_a_proj_with_mqa = Linear(
            embed_dim,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            name="kv_a_proj_with_mqa",
        )

        self.ckv_kpe_split = Split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
            name="ckv_kpe_split",
        )

        self.q_proj = Linear(
            self.embed_dim,
            self.q_head_dim * self.num_heads,
            dtype=dtype,
            name="q_proj",
        )

        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            name="kv_a_layernorm",
        )

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=dtype,
            name="kv_b_proj",
        )

        self.gqa = GQA()

        self.o_proj = Linear(
            self.num_heads * self.v_head_dim,
            self.embed_dim,
            dtype=dtype,
            name="o_proj",
        )

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        self.clear_kvcache()
        self.kvcache = Tensor(
            shape=(num_users, seq_len, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=dtype,
        )

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        # Flatten all batch dimensions into a single logical "user" dimension
        num_user, _, _ = inp.reshape((-1, inp.shape[-2], inp.shape[-1])).shape

        inp_ckv, inp_q = self.kvq_fanout(inp)

        ckv_kpe = self.kv_a_proj_with_mqa(inp_ckv)

        ckv_kpe_split = Split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )
        ckv, kpe = ckv_kpe_split(ckv_kpe)

        ckv = self.kv_a_layernorm(ckv)

        ckv_kpe_concat = Concat(axis=-1)

        ckv_kpe = ckv_kpe_concat([ckv, kpe])

        # Append to KV cache (stores [ckv|kpe] for decode)
        prev_ckv_kpe = self.get_kvcache()
        if prev_ckv_kpe is not None:
            ckv_concat = Concat(axis=1, name="ckv_concat")
            ckv_kpe = ckv_concat([prev_ckv_kpe, ckv_kpe])
        self.set_kvcache(ckv_kpe)

        # Q path
        q = self.q_proj(inp_q)
        q_reshape = Reshape(
            (num_user, -1, self.num_heads, self.q_head_dim), name="q_reshape"
        )
        q = q_reshape(q)
        q_split = Split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
            name="q_split",
        )
        q_pass, q_rot = q_split(q)
        q_concat = Concat(axis=-1, name="q_concat")
        q_full = q_concat([q_pass, q_rot])

        # Split & norm the low-rank content and RoPE component
        ckv_kpe_split = Split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )
        ckv, kpe = ckv_kpe_split(ckv_kpe)

        # Emit (original): project ckv → [K_nope | V], then build/repeat K_rope and concat
        kv = self.kv_b_proj(ckv)

        kv_reshape = Reshape(
            (num_user, -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim),
            name="kv_emit_reshape",
        )
        kv = kv_reshape(kv)

        kv_split = Split(
            [self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
            name="kv_emit_split",
        )
        k_pass, v = kv_split(kv)

        krot_reshape = Reshape(
            (num_user, -1, 1, self.qk_rope_head_dim), name="k_rot_reshape"
        )
        k_rot = krot_reshape(kpe)

        krot_repeat = RepeatAlongAxis(
            axis=2, repeats=self.num_heads, name="k_rot_repeat"
        )
        k_rot = krot_repeat(k_rot)

        k_concat = Concat(axis=-1, name="k_concat")
        k_full = k_concat([k_pass, k_rot])

        # Attention
        attn_out = self.gqa(q_full, k_full, v)

        out_reshape = Reshape((num_user, -1, self.num_heads * self.v_head_dim))
        attn_flat = out_reshape(attn_out)
        out = self.o_proj(attn_flat)

        self.add_output(out)
        self.add_output(ckv)
        return out


class MLAAbsorb(Attention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        v_head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super(Attention, self).__init__(name=name, dtype=dtype)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_rope_head_dim + qk_nope_head_dim

        self.kvcache: Tensor | None = None

        self.kvq_fanout = FanOut(num_outputs=2, name="kvq_fanout")

        self.q_proj = Linear(
            self.embed_dim,
            self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=dtype,
            name="q_proj",
        )

        self.kv_a_proj_with_mqa = Linear(
            embed_dim,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            name="kv_a_proj_with_mqa",
        )

        self.ckv_norm = RMSNorm(
            self.kv_lora_rank,
            name="ckv_norm",
        )

        self.gqa = GQA()

        self.o_proj = Linear(
            self.num_heads * self.kv_lora_rank,
            self.embed_dim,
            dtype=dtype,
            name="o_proj",
        )

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        self.clear_kvcache()
        self.kvcache = Tensor(
            shape=(num_users, seq_len, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=dtype,
        )

    def forward(self, inp: Tensor) -> Tensor:
        self.add_input(inp)
        # Flatten all batch dimensions into a single logical "user" dimension
        num_user, _, _ = inp.reshape((-1, inp.shape[-2], inp.shape[-1])).shape

        inp_ckv, inp_q = self.kvq_fanout(inp)

        ckv_kpe = self.kv_a_proj_with_mqa(inp_ckv)

        ckv_kpe_split = Split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )

        ckv, kpe = ckv_kpe_split(ckv_kpe)

        ckv = self.ckv_norm(ckv)

        ckv_kpe_concat = Concat(axis=-1)

        ckv_kpe = ckv_kpe_concat([ckv, kpe])

        # Append to KV cache (stores [ckv|kpe] for decode)
        prev_ckv_kpe = self.get_kvcache()
        if prev_ckv_kpe is not None:
            ckv_concat = Concat(axis=1, name="ckv_concat")
            ckv_kpe = ckv_concat([prev_ckv_kpe, ckv_kpe])
        self.set_kvcache(ckv_kpe)

        q = self.q_proj(inp_q)

        ckv_kpe_fanout = FanOut(num_outputs=2, name="ckv_kpe_fanout")
        ckv_kpe1, ckv_kpe2 = ckv_kpe_fanout(ckv_kpe)

        ckv_kpe_split = Split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            axis=-1,
        )

        ckv, kpe = ckv_kpe_split(ckv_kpe1)

        q_reshape = Reshape(
            (num_user, -1, self.num_heads, (self.kv_lora_rank + self.qk_rope_head_dim)),
            name="q_reshape",
        )
        q = q_reshape(q)

        k_reshape = Reshape(
            (num_user, -1, 1, (self.kv_lora_rank + self.qk_rope_head_dim)),
            name="k_reshape",
        )
        k = k_reshape(ckv_kpe2)

        v_reshape = Reshape((num_user, -1, 1, self.kv_lora_rank), name="v_reshape")
        v = v_reshape(ckv)

        attn_out = self.gqa(q, k, v)

        out_reshape = Reshape((num_user, -1, self.num_heads * self.kv_lora_rank))
        attn_flat = out_reshape(attn_out)
        out = self.o_proj(attn_flat)

        self.add_output(out)
        self.add_output(ckv_kpe)
        return out


class DecoderLayer(MultiOp):
    def __init__(
        self,
        layer_idx: int,
        hf_config: PretrainedConfig,
        enable_absorb_mode: bool,
        name: str,
        dtype: DataType,
    ):
        super().__init__(name, dtype)
        self.hf_config = hf_config

        self.res_fan_out_1 = FanOut(num_outputs=2, name="ResFanOut_1")
        self.input_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )
        mla_cls = MLAAbsorb if enable_absorb_mode else MLA
        self.attention = mla_cls(
            embed_dim=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            v_head_dim=hf_config.v_head_dim,
            kv_lora_rank=hf_config.kv_lora_rank,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            dtype=dtype,
        )
        self.residual1 = Residual()
        self.res_fan_out_2 = FanOut(num_outputs=2, name="ResFanOut_2")
        self.post_attention_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )
        self.mlp = (
            MoE(
                num_experts=hf_config.n_routed_experts,
                num_experts_per_token=hf_config.num_experts_per_tok,
                embed_dim=hf_config.hidden_size,
                moe_intermediate_size=hf_config.moe_intermediate_size,
                shared_intermediate_size=hf_config.n_shared_experts
                * hf_config.moe_intermediate_size,
                dtype=dtype,
            )
            if (
                hf_config.n_routed_experts is not None
                and layer_idx >= hf_config.first_k_dense_replace
                and layer_idx % hf_config.moe_layer_freq == 0
            )
            else FFN(
                embed_dim=hf_config.hidden_size,
                intermediate_dim=hf_config.intermediate_size,
                dtype=dtype,
            )
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


@register_model("deepseek_r1_absorb", "deepseek-ai/DeepSeek-R1")
@register_model("deepseek_r1", "deepseek-ai/DeepSeek-R1", enable_absorb_mode=False)
class DeepSeekV3(Model):
    def __init__(
        self,
        model_id: str,
        dtype: DataType,
        hf_config: PretrainedConfig | None = None,
        enable_absorb_mode: bool = True,
    ):
        super().__init__(model_id, dtype=dtype, hf_config=hf_config)

        self.layers = [
            DecoderLayer(
                i, self.hf_config, enable_absorb_mode, f"DecoderLayer_{i}", dtype
            )
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
