"""
Weight loading utilities for HuggingFace model format conversion.
"""

import torch
import torch.nn as nn
from typing import Dict
import json
from pathlib import Path


def load_hf_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """
    Load HuggingFace format weights into a FlashServe model.

    Args:
        model: Target model to load weights into
        state_dict: State dict from HuggingFace model
    """
    converted_state = convert_hf_state_dict(state_dict)
    model.load_state_dict(converted_state, strict=False)


def convert_hf_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace model state dict to FlashServe format.

    Maps HuggingFace weight names to our model parameter names.

    Args:
        state_dict: Original HuggingFace state dict

    Returns:
        Converted state dict compatible with FlashServe
    """
    converted = {}

    for hf_key, tensor in state_dict.items():
        # Skip embeddings scaling
        if "inv_freq" in hf_key:
            continue

        # Convert layer names: model.layers.0.xxx -> layers.0.xxx
        key = hf_key.replace("model.layers", "layers")
        key = key.replace("model.norm", "norm")
        key = key.replace("lm_head", "lm_head")

        # Convert attention layer names
        key = key.replace("self_attn.q_proj", "self_attn.q_proj")
        key = key.replace("self_attn.k_proj", "self_attn.k_proj")
        key = key.replace("self_attn.v_proj", "self_attn.v_proj")
        key = key.replace("self_attn.o_proj", "self_attn.o_proj")

        # Convert MLP layer names
        key = key.replace("mlp.gate_proj", "mlp.gate_proj")
        key = key.replace("mlp.up_proj", "mlp.up_proj")
        key = key.replace("mlp.down_proj", "mlp.down_proj")

        # Convert norm layer names
        key = key.replace("input_layernorm", "input_layernorm")
        key = key.replace("post_attention_layernorm", "post_attention_layernorm")

        converted[key] = tensor

    return converted


def load_from_pretrained(
    model_name_or_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Load weights from a HuggingFace model path or model ID.

    Args:
        model_name_or_path: Model name (e.g., 'meta-llama/Llama-2-7b') or local path

    Returns:
        State dict of the model
    """
    try:
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("Please install safetensors and huggingface_hub")

    # Download model if needed
    if not Path(model_name_or_path).exists():
        model_path = snapshot_download(model_name_or_path)
    else:
        model_path = model_name_or_path

    # Load safetensors if available
    safetensors_file = Path(model_path) / "model.safetensors"
    if safetensors_file.exists():
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        return state_dict

    # Fallback to PyTorch checkpoint
    pytorch_file = Path(model_path) / "pytorch_model.bin"
    if pytorch_file.exists():
        return torch.load(pytorch_file, map_location="cpu")

    raise FileNotFoundError(f"No model weights found in {model_path}")


def estimate_model_size(state_dict: Dict[str, torch.Tensor]) -> float:
    """
    Estimate model size in GB.

    Args:
        state_dict: Model state dict

    Returns:
        Size in GB
    """
    total_params = sum(t.numel() for t in state_dict.values())
    bytes_per_param = 4  # float32
    size_gb = (total_params * bytes_per_param) / (1024 ** 3)
    return size_gb
