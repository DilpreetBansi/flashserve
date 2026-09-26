"""FlashServe: a from-scratch inference engine for Llama-architecture language models."""

__version__ = "0.2.0"

from flashserve.engine.inference_engine import InferenceEngine
from flashserve.model.config import LlamaConfig

__all__ = ["LlamaConfig", "InferenceEngine"]
