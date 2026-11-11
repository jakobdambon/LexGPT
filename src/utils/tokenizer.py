

class CharacterTokenizer:
    """Convert text to character IDs using a vocabulary learned from sample text."""

    def __init__(self, text: str) -> None:
        """Build a stable character-to-ID mapping from the provided training text.

        Characters are sorted before assigning IDs so the same text always
        produces the same vocabulary order. An extra ID is reserved for any
        character that was not present in the sample text.
        """
        self.characters = tuple(sorted(set(text)))
        self.unknown_id = len(self.characters)
        self.tokens = self.characters + ("<unk>",)
        self._stoi = {character: index for index, character in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        """Return the number of known characters plus the unknown-token entry."""
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Turn each character into its vocabulary ID.

        Characters missing from the training vocabulary are represented by the
        reserved unknown-token ID instead of causing a lookup error.
        """
        return [self._stoi.get(character, self.unknown_id) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Turn vocabulary IDs back into text.

        The unknown-token ID is shown as the Unicode replacement character
        because the original unseen character cannot be recovered. Any other
        ID outside the vocabulary is rejected as invalid input.
        """
        decoded = []
        for token_id in token_ids:
            if token_id == self.unknown_id:
                decoded.append("\ufffd")
            elif 0 <= token_id < self.unknown_id:
                decoded.append(self.characters[token_id])
            else:
                raise ValueError(f"Token ID out of range: {token_id}")
        return "".join(decoded)


def char_token(text):
    """Return the legacy character-tokenizer interface for the given text.

    The result contains the vocabulary (including ``<unk>``), an encoder
    function that maps text to integer IDs, and a decoder function that maps
    those IDs back to text. This keeps older callers working while using the
    ``CharacterTokenizer`` implementation.
    """
    tokenizer = CharacterTokenizer(text)
    return list(tokenizer.tokens), tokenizer.encode, tokenizer.decode
