from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from llm_spice.utils.common import DataType, Tensor
from llm_spice.op.operators import Attention, MultiOp
import os
from llm_spice.utils.registry import MODEL_REGISTRY
from dotenv import load_dotenv

load_dotenv()


class Model(MultiOp):
    def __init__(
        self,
        model_name: str,
        dtype: DataType,
        attn_dtype: DataType = DataType.BF16,
        hf_config: PretrainedConfig | None = None,
    ):
        super().__init__(model_name, dtype)
        self.model_name = model_name
        self.attn_dtype = attn_dtype
        self.hf_config = (
            Model.get_pretrained_config(model_name) if hf_config is None else hf_config
        )
        # Cache attention layers to avoid repeated full-tree scans during decode
        self._attention_layers: list[Attention] | None = None

    @staticmethod
    def get_pretrained_config(model_name: str):
        _, model_id, _ = MODEL_REGISTRY[model_name]
        return AutoConfig.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))

    @staticmethod
    def create(
        model_name: str,
        dtype: DataType = DataType.BF16,
        hf_config: PretrainedConfig | None = None,
    ) -> "Model":
        model_class, _, kwargs = MODEL_REGISTRY[model_name]
        return model_class(model_name, dtype=dtype, hf_config=hf_config, **kwargs)

    @staticmethod
    def get_all_models() -> list[str]:
        return list(MODEL_REGISTRY.keys())

    def _get_attention_layers(self) -> list[Attention]:
        """Return and cache the list of Attention ops in the model graph.

        This avoids traversing the entire op tree on every call during decode.
        """
        if self._attention_layers is None:
            atts: list[Attention] = []
            for child in self.all_children():
                if isinstance(child, Attention):
                    atts.append(child)
            self._attention_layers = atts
        return self._attention_layers

    def get_num_experts(self) -> int:
        if hasattr(self.hf_config, "num_experts"):
            return self.hf_config.num_experts
        if hasattr(self.hf_config, "num_local_experts"):
            return self.hf_config.num_local_experts
        return 0

    def get_kvcache(self) -> Tensor | None:
        all_kvcache: list[Tensor] = []
        for att in self._get_attention_layers():
            kvcache = att.get_kvcache()
            if kvcache is None:
                return None
            all_kvcache.append(kvcache)
        num_layers = len(all_kvcache)
        # Use the first layer's shape/dtype as representative (layers share shape)
        kv0 = all_kvcache[0]
        return Tensor(shape=(num_layers, *kv0.shape), dtype=kv0.dtype)

    def clear_kvcache(self):
        for att in self._get_attention_layers():
            att.clear_kvcache()

    def insert_kvcache(self, num_users: int, seq_len: int, dtype: DataType):
        for att in self._get_attention_layers():
            att.insert_kvcache(num_users, seq_len, dtype)

    def get_total_kvcache_bytes(self) -> int:
        """Fast path: sum k/v cache bytes across attention layers directly.

        Avoids constructing large temporary Tensors and scanning the full tree.
        """
        total = 0
        for att in self._get_attention_layers():
            kvcache = att.get_kvcache()
            if kvcache is None:
                return 0
            total += int(kvcache.size_in_bytes)
        return total

    def get_seq_len(self) -> int:
        kvcache = self.get_kvcache()
        if kvcache is None:
            return 0
        kvcache = kvcache[0]
        if len(kvcache.shape) == 3:
            # ckv
            return kvcache.shape[1]
        elif len(kvcache.shape) == 5:
            # K/V cache
            return kvcache.shape[2]
        else:
            raise ValueError(f"Invalid kvcache shape: {kvcache.shape}")
