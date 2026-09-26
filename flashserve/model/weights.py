"""
Load Hugging Face Llama-architecture checkpoints (e.g. SmolLM2, TinyLlama) into FlashServe.

Hugging Face stores q_proj/k_proj rows in the "rotate_half" RoPE layout, while
FlashServe applies RoPE to interleaved (even, odd) pairs like Meta's reference code.
The two are related by a fixed permutation of the rows within each head, so the
loader permutes those two matrices once at load time and the model code stays simple.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from flashserve.model.config import LlamaConfig
from flashserve.model.llama import LlamaForCausalLM


def config_from_hf(hf: dict) -> LlamaConfig:
    """Build a LlamaConfig from a Hugging Face config.json dict."""
    if hf.get("model_type") not in (None, "llama"):
        raise ValueError(f"Only Llama-architecture checkpoints are supported, got {hf.get('model_type')!r}")
    if hf.get("rope_scaling"):
        raise ValueError("RoPE scaling is not supported yet")
    eos = hf.get("eos_token_id", 2)
    return LlamaConfig(
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_key_value_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
        intermediate_size=hf["intermediate_size"],
        num_hidden_layers=hf["num_hidden_layers"],
        vocab_size=hf["vocab_size"],
        max_position_embeddings=hf.get("max_position_embeddings", 2048),
        norm_epsilon=hf.get("rms_norm_eps", 1e-5),
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        attention_bias=hf.get("attention_bias", False),
        bos_token_id=hf.get("bos_token_id", 1),
        eos_token_id=eos[0] if isinstance(eos, list) else eos,
        pad_token_id=hf.get("pad_token_id") or 0,
    )


def _hf_to_interleaved(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Undo the permutation Hugging Face applies to q/k projections for rotate_half RoPE."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // n_heads
    return w.view(n_heads, 2, head_dim // 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def convert_hf_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: LlamaConfig,
    rope_interleaved: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Map a Hugging Face LlamaForCausalLM state dict onto FlashServe's parameters.

    Parameter names already match (model.layers.N.self_attn.q_proj.weight, ...);
    the only change is the q/k row permutation described in the module docstring.
    """
    converted = {}
    for key, tensor in state_dict.items():
        if key.endswith("rotary_emb.inv_freq"):
            continue
        if not rope_interleaved and key.endswith("self_attn.q_proj.weight"):
            tensor = _hf_to_interleaved(tensor, config.num_attention_heads)
        elif not rope_interleaved and key.endswith("self_attn.k_proj.weight"):
            tensor = _hf_to_interleaved(tensor, config.num_key_value_heads)
        converted[key] = tensor
    return converted


def resolve_checkpoint(model_name_or_path: str) -> Path:
    """Local directory for a model id or path (downloads from the Hub if needed)."""
    path = Path(model_name_or_path)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install huggingface_hub to download checkpoints") from e
    return Path(snapshot_download(model_name_or_path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def load_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    state: Dict[str, torch.Tensor] = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def from_pretrained(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Tuple[LlamaForCausalLM, LlamaConfig]:
    """
    Load a Llama-architecture checkpoint.

    Example:
        model, config = from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    """
    model_dir = resolve_checkpoint(model_name_or_path)
    hf_cfg = json.loads((model_dir / "config.json").read_text())
    config = config_from_hf(hf_cfg)

    model = LlamaForCausalLM(config)
    state = convert_hf_state_dict(load_state_dict(model_dir), config, hf_cfg.get("rope_interleaved", False))
    tied = hf_cfg.get("tie_word_embeddings", True)
    if tied:
        state.pop("lm_head.weight", None)
    else:
        model.lm_head.weight = torch.nn.Parameter(torch.empty_like(model.lm_head.weight))

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not (tied and k == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    return model.to(device=device, dtype=dtype).eval(), config


def estimate_model_size(state_dict: Dict[str, torch.Tensor], bytes_per_param: Optional[int] = None) -> float:
    """Size of a state dict in GB (uses each tensor's dtype unless bytes_per_param is given)."""
    total = sum(t.numel() * (bytes_per_param or t.element_size()) for t in state_dict.values())
    return total / (1024 ** 3)
