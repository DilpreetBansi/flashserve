"""
FlashServe: High-Performance LLM Inference Engine

A complete, production-ready inference engine optimized for serving large language models
with minimal latency and maximum throughput.
"""

__version__ = "0.1.0"
__author__ = "FlashServe Contributors"

from flashserve.model.config import LlamaConfig
from flashserve.engine.inference_engine import InferenceEngine

__all__ = [
    "LlamaConfig",
    "InferenceEngine",
]
