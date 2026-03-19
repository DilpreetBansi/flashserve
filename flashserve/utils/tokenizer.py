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
        Load tokenizer from pretrained model.

        Args:
            model_name: Model name (e.g., "gpt2", "llama")

        Returns:
            Tokenizer instance
        """
        # Try to load from HuggingFace
        try:
            import tiktoken

            if model_name == "gpt2":
                enc = tiktoken.get_encoding("gpt2")
                return HFTokenizer(enc)
        except ImportError:
            pass

        # Fallback to simple tokenizer
        return cls()


class HFTokenizer(Tokenizer):
    """Wrapper for HuggingFace tokenizers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
