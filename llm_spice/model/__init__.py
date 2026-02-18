from __future__ import annotations

# 1) Import model provider modules for their side effects (registration)
#    Keep a simple, explicit list here:
from . import llama3  # noqa: F401  # registers all Llama3 variants
from . import qwen3_moe  # noqa: F401  # registers all Qwen3 MoE variants
from . import qwen3_next  # noqa: F401  # registers all Qwen3 Next variants
from . import llama4  # noqa: F401  # registers all Llama4 variants
from . import deepseek_v3  # noqa: F401  # registers all DeepSeek V3 variants
from . import gpt_oss  # noqa: F401  # registers all GPT-OSS variants
# ...add more providers as you add files

# 2) Finally expose the factory
from .model import Model

__all__ = ["Model"]
