"""Vocabulary loader and manager."""
import json
from typing import Dict, List, Tuple


class Vocabulary:
    """Manages the LLM's tokenizer vocabulary."""

    def __init__(self, vocab_path: str) -> None:
        """Initialize the vocabulary mapping."""
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str) -> None:
        """Load vocabulary from the JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data: Dict[str, int] = json.load(f)

        for token, token_id in vocab_data.items():
            # Handle standard byte-fallback or special tokens if necessary.
            # Qwen models often use byte-level BPE; ensure unicode decoding.
            decoded_token = self._decode_token_string(token)
            self.token_to_id[decoded_token] = token_id
            self.id_to_token[token_id] = decoded_token

    def _decode_token_string(self, token: str) -> str:
        """
        Clean up token representations.
        Note: The actual implementation depends on Qwen's specific vocab format
        (e.g., handling 'Ġ' for spaces in some BPE tokenizers).
        """
        return token.replace('Ġ', ' ').replace('Ċ', '\n')

    def get_all_tokens(self) -> List[Tuple[int, str]]:
        """Return a list of all (id, token_string) pairs."""
        return list(self.id_to_token.items())
