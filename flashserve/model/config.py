"""
Model configuration for Llama-style architectures.

Provides predefined configs for different model sizes and custom configuration support.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlamaConfig:
    """Configuration for Llama-style transformer models."""

    # Model architecture
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # For Grouped Query Attention
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096

    # Activation and normalization
    hidden_act: str = "silu"  # SwiGLU activation
    initializer_range: float = 0.02
    norm_epsilon: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None

    # Attention configuration
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_cache: bool = True

    # Quantization
    quantization_config: Optional[dict] = None

    # Padding token
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        """Validate configuration."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"
        assert (
            self.num_attention_heads % self.num_key_value_heads == 0
        ), "num_attention_heads must be divisible by num_key_value_heads"

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def tiny(cls) -> "LlamaConfig":
        """
        Tiny model configuration for testing and CPU inference.

        - 15M parameters
        - 2 layers
        - 256 hidden size
        - Can run on CPU without issues
        """
        return cls(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,  # GQA
            intermediate_size=512,
            num_hidden_layers=2,
            vocab_size=32000,
            max_position_embeddings=512,
        )

    @classmethod
    def small(cls) -> "LlamaConfig":
        """
        Small model configuration for testing on GPU.

        - 110M parameters
        - 6 layers
        - 512 hidden size
        """
        return cls(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            intermediate_size=1024,
            num_hidden_layers=6,
            vocab_size=32000,
            max_position_embeddings=2048,
        )

    @classmethod
    def llama2_7b(cls) -> "LlamaConfig":
        """
        Llama-2-7B configuration.

        - 7B parameters
        - 32 layers
        - 4096 hidden size
        - Matches official Llama-2-7B layout
        """
        return cls(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA for inference efficiency
            intermediate_size=11008,
            num_hidden_layers=32,
            vocab_size=32000,
            max_position_embeddings=4096,
        )

    @classmethod
    def llama2_13b(cls) -> "LlamaConfig":
        """
        Llama-2-13B configuration.

        - 13B parameters
        - 40 layers
        - 5120 hidden size
        """
        return cls(
            hidden_size=5120,
            num_attention_heads=40,
            num_key_value_heads=10,  # GQA
            intermediate_size=13824,
            num_hidden_layers=40,
            vocab_size=32000,
            max_position_embeddings=4096,
        )

    @classmethod
    def llama2_70b(cls) -> "LlamaConfig":
        """
        Llama-2-70B configuration.

        - 70B parameters
        - 80 layers
        - 8192 hidden size
        """
        return cls(
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA (more aggressive)
            intermediate_size=28672,
            num_hidden_layers=80,
            vocab_size=32000,
            max_position_embeddings=4096,
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LlamaConfig":
        """Load config from dictionary."""
        return cls(**config_dict)
