"""
Tokenizer wrapper for encoding/decoding text.
"""

from typing import List, Optional
import re


class Tokenizer:
    """Simple tokenizer (fallback when HuggingFace tokenizers not available)."""

    def __init__(self, vocab_size: int = 32000):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}

        # Build basic vocab (in practice would load from file)
        self._build_vocab()

    def _build_vocab(self) -> None:
        """Build basic vocabulary from common words."""
        # Add special tokens
        special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Add basic ASCII characters
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            token_id = len(self.token_to_id)
            self.token_to_id[char] = token_id
            self.id_to_token[token_id] = char

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        # Simple character-level tokenization
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                # Unknown token
                tokens.append(self.token_to_id.get("<unk>", 0))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        text = ""
        for token_id in token_ids:
            if token_id in self.id_to_token:
                text += self.id_to_token[token_id]
            else:
                text += "<unk>"

        return text

    @classmethod
    def from_pretrained(cls, model_name: str) -> "Tokenizer":
        """
        Load a tokenizer.

        Args:
            model_name: "gpt2" for tiktoken's GPT-2 encoding, or a Hugging Face model
                id / local directory containing tokenizer.json.

        Returns:
            Tokenizer instance (falls back to the character tokenizer if nothing loads)
        """
        if model_name == "gpt2":
            try:
                import tiktoken

                return HFTokenizer(tiktoken.get_encoding("gpt2"))
            except ImportError:
                return cls()
        return HubTokenizer.from_pretrained(model_name)


class HFTokenizer(Tokenizer):
    """Wrapper for HuggingFace tokenizers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)


class HubTokenizer(Tokenizer):
    """Tokenizer backed by a Hugging Face tokenizer.json (via the `tokenizers` package)."""

    def __init__(self, backend, chat_template: Optional[str] = None, special: Optional[dict] = None):
        self.backend = backend
        self.vocab_size = backend.get_vocab_size()
        self.chat_template = chat_template
        self.special = special or {}

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "HubTokenizer":
        import json
        from pathlib import Path

        from tokenizers import Tokenizer as _Backend

        from flashserve.model.weights import resolve_checkpoint

        model_dir = resolve_checkpoint(model_name_or_path)
        backend = _Backend.from_file(str(Path(model_dir) / "tokenizer.json"))
        cfg_path = Path(model_dir) / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        special = {k: v for k, v in cfg.items() if k.endswith("_token") and isinstance(v, str)}
        return cls(backend, chat_template=cfg.get("chat_template"), special=special)

    def encode(self, text: str) -> List[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.backend.decode(list(token_ids), skip_special_tokens=True)

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        """Render chat messages with the checkpoint's Jinja chat template."""
        if not self.chat_template:
            return "\n".join(m["content"] for m in messages)
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        return env.from_string(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **self.special
        )
