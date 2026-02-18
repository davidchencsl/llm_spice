from transformers import PretrainedConfig
from llm_spice.utils.common import DataType, Tensor
from llm_spice.model.model import Model
from llm_spice.op.operators import (
    FanOut,
    Linear,
    MultiOp,
    Attention,
    RMSNorm,
    Residual,
    MoEFast as MoE,
    Reshape,
    Concat,
)
from llm_spice.utils.registry import register_model


class SlidingWindowAttention(Attention):
    """
    Sliding Window Attention for GPT-OSS models.

    This attention mechanism restricts attention to a fixed-size local window,
    enhancing efficiency for long contexts. The KV cache is managed to store
    only the tokens within the sliding window, preventing memory overflow.
    """

    def __init__(
        self,
        embed_dim: int,
        num_kv_heads: int,
        head_dim: int,
        gqa_factor: int,
        sliding_window: int | None = None,
        enable_qk_norm: bool = False,
        name: str = "",
        dtype: DataType = DataType.BF16,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            gqa_factor=gqa_factor,
            enable_qk_norm=enable_qk_norm,
            name=name,
            dtype=dtype,
        )
        self.sliding_window = sliding_window

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        """
        Initialize KV cache with proper size accounting for sliding window.

        If sliding_window is set, the cache size is limited to the window size.
        This prevents unbounded memory growth for long sequences.
        """
        self.clear_kvcache()

        # If sliding window is enabled, limit cache size to window size
        if self.sliding_window is not None:
            cache_seq_len = min(seq_len, self.sliding_window)
        else:
            cache_seq_len = seq_len

        self.kvcache = Tensor(
            shape=(2, num_users, cache_seq_len, self.num_kv_heads, self.head_dim),
            dtype=dtype,
        )

    def forward(self, inp: Tensor) -> Tensor:
        # Expand to 3D [num_user, seq_len, embed_dim]
        num_user, current_seq_len, _ = inp.reshape(
            (-1, inp.shape[-2], inp.shape[-1])
        ).shape

        self.q_reshape = Reshape(
            (num_user, -1, self.num_kv_heads * self.gqa_factor, self.head_dim),
            name="q_reshape",
        )
        self.k_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="k_reshape"
        )
        self.v_reshape = Reshape(
            (num_user, -1, self.num_kv_heads, self.head_dim), name="v_reshape"
        )

        self.add_input(inp)

        inp1, inp2, inp3 = self.fan_out(inp)
        q = self.q_proj(inp1)
        k = self.k_proj(inp2)
        v = self.v_proj(inp3)

        q = self.q_reshape(q)
        k = self.k_reshape(k)
        v = self.v_reshape(v)

        if self.enable_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        prev_kv = self.get_kvcache()
        if prev_kv is not None:
            self.k_concat = Concat(
                axis=1, name="k_concat"
            )  # Append along sequence axis
            prev_k = prev_kv[0]
            k = self.k_concat([prev_k, k])

            self.v_concat = Concat(
                axis=1, name="v_concat"
            )  # Append along sequence axis
            prev_v = prev_kv[1]
            v = self.v_concat([prev_v, v])

            # Apply sliding window: keep only the most recent tokens if window size is exceeded
            if self.sliding_window is not None:
                k_seq_len = k.shape[1]
                if k_seq_len > self.sliding_window:
                    # In practice, this would slice k and v to keep only the last sliding_window tokens
                    # For tensor shape tracking, we create new tensors with the windowed shape
                    k = Tensor(
                        shape=(
                            num_user,
                            self.sliding_window,
                            self.num_kv_heads,
                            self.head_dim,
                        ),
                        dtype=k.dtype,
                    )
                    v = Tensor(
                        shape=(
                            num_user,
                            self.sliding_window,
                            self.num_kv_heads,
                            self.head_dim,
                        ),
                        dtype=v.dtype,
                    )

        kv = Tensor(shape=(2, *k.shape), dtype=k.dtype)
        self.set_kvcache(kv)

        qkv = self.gqa(q, k, v)

        qkv_reshape = Reshape(
            (qkv.shape[0], -1, self.head_dim * self.num_kv_heads * self.gqa_factor),
            name="qkv_reshape",
        )
        qkv_r = qkv_reshape(qkv)
        out = self.out_proj(qkv_r)
        self.add_output(out)
        self.add_output(kv)
        return out


class DecoderLayer(MultiOp):
    """
    GPT-OSS Decoder Layer with Sliding Window Attention and Mixture-of-Experts.

    Architecture:
    - Pre-norm with RMSNorm
    - Sliding Window Attention (with optional window size limit)
    - Residual connection
    - Pre-norm with RMSNorm
    - Sparse MoE (128 or 32 experts, 4 active per layer)
    - Residual connection
    """

    def __init__(
        self,
        layer_idx: int,
        hf_config: PretrainedConfig,
        name: str,
        dtype: DataType,
    ):
        super().__init__(name, dtype)
        self.hf_config = hf_config
        self.layer_idx = layer_idx

        self.res_fan_out_1 = FanOut(num_outputs=2, name="ResFanOut_1")
        self.input_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )

        # Get sliding window size from config if available
        sliding_window = getattr(hf_config, "sliding_window", None)

        # Use sliding window attention
        self.attention = SlidingWindowAttention(
            embed_dim=hf_config.hidden_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
            gqa_factor=hf_config.num_attention_heads // hf_config.num_key_value_heads,
            sliding_window=sliding_window,
            enable_qk_norm=True,
            name="SlidingWindowAttention",
            dtype=dtype,
        )

        self.residual1 = Residual()
        self.res_fan_out_2 = FanOut(num_outputs=2, name="ResFanOut_2")
        self.post_attention_layernorm = RMSNorm(
            input_dim=hf_config.hidden_size,
        )

        # Sparse MoE layer
        self.mlp = MoE(
            num_experts=hf_config.num_local_experts,
            num_experts_per_token=hf_config.num_experts_per_tok,
            embed_dim=hf_config.hidden_size,
            moe_intermediate_size=hf_config.intermediate_size,
            shared_intermediate_size=0,  # GPT-OSS doesn't use shared experts
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


@register_model("gpt_oss_120b", "openai/gpt-oss-120b")
@register_model("gpt_oss_20b", "openai/gpt-oss-20b")
class GPTOSS(Model):
    """
    GPT-OSS: OpenAI's Open Source Language Models

    Key features:
    - Mixture-of-Experts (MoE) architecture for efficient inference
    - Sliding Window Attention for handling extended contexts (up to 131k tokens)
    - gpt-oss-120b: 36 layers, 128 experts per layer, 4 active experts per token
    - gpt-oss-20b: 36 layers, 32 experts per layer, 4 active experts per token

    Architecture details:
    - Transformer-based with RMSNorm pre-normalization
    - Grouped Query Attention (GQA) for efficient KV cache
    - Sliding window attention restricts attention span to local context
    - RoPE positional embeddings for extended context support
    - 4-bit expert weight quantization (MXFP4) for memory efficiency

    Performance benefits:
    - Efficient deployment on consumer hardware
    - 120b model runs on single 80GB GPU
    - 20b model runs on devices with 16GB memory
    - Sparse activation reduces computational cost
    - Sliding window attention enables long context processing
    """

    def __init__(
        self,
        model_id: str,
        dtype: DataType,
        hf_config: PretrainedConfig | None = None,
    ):
        super().__init__(model_id, dtype=dtype, hf_config=hf_config)

        self.layers = [
            DecoderLayer(i, self.hf_config, f"DecoderLayer_{i}", dtype)
            for i in range(self.hf_config.num_hidden_layers)
        ]

        self.norm = RMSNorm(
            input_dim=self.hf_config.hidden_size,
        )

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
        normed = self.norm(inp)
        unembed = self.unembedding(normed)
        self.add_output(unembed)
        return unembed

    def get_num_experts(self) -> int:
        """Return the number of experts per MoE layer."""
        if hasattr(self.hf_config, "num_local_experts"):
            return self.hf_config.num_local_experts
        # Fallback defaults based on model name
        if "120b" in self.model_name.lower():
            return 128
        elif "20b" in self.model_name.lower():
            return 32
        return 0
