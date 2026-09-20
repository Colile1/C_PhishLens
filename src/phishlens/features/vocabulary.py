# vocabulary.py
# Purpose: build and persist the token-to-index map for the M1 Text-CNN.
# Author: Colile Sibanda

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from phishlens.config.schema import Config
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise

PAD_INDEX = 0
UNK_INDEX = 1


@dataclass(frozen=True)
class Vocabulary:
    """Token-to-index map with `<pad>` at index 0 and `<unk>` at index 1.

    Inputs: token_to_index : the full mapping including pad and unk;
    pad_token, unk_token : the configured placeholder strings.
    Output: immutable Vocabulary; `encode_token` and `size` for consumers.
    """

    token_to_index: dict[str, int]
    pad_token: str
    unk_token: str

    def size(self) -> int:
        """Return the number of entries in the vocabulary, pad and unk included.

        Inputs: none.
        Output: vocabulary size as an int.
        """
        return len(self.token_to_index)

    def encode_token(self, token: str) -> int:
        """Map a single token to its index, falling back to `<unk>`.

        Inputs: token : a token as produced by tokenise().
        Output: the token's index, or the unk index if token is out of vocabulary.
        """
        return self.token_to_index.get(token, self.token_to_index[self.unk_token])

    def save(self, path: Path) -> None:
        """Write the vocabulary to a JSON file.

        Inputs: path : destination file path.
        Output: none; writes the token-to-index map and placeholder tokens.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_index": self.token_to_index,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Vocabulary":
        """Read a vocabulary back from a JSON file.

        Inputs: path : file previously written by save().
        Output: reconstructed Vocabulary.

        Raises FileNotFoundError naming the missing path.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"No vocabulary file at {path}. Build it with build_vocabulary() "
                "and save() first."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Vocabulary(
            token_to_index=payload["token_to_index"],
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
        )


def build_vocabulary(train_texts: list[str], config: Config) -> Vocabulary:
    """Build a vocabulary from training texts only.

    The parameter name makes misuse awkward, matching fit_tfidf's convention.
    Fitting on validation or test text would leak their token distribution
    into the model. Tokens are ranked by frequency, then capped at
    `max_size` and filtered by `min_frequency`; ties in frequency are broken
    by first appearance so the result is deterministic for fixed input.

    Inputs: train_texts : cleaned text column from the training split only;
    config : validated Config supplying the vocabulary settings.
    Output: Vocabulary with `<pad>` at index 0 and `<unk>` at index 1,
    followed by the retained tokens in descending frequency order.
    """
    settings = config.features.vocabulary
    counts: Counter[str] = Counter()
    for text in train_texts:
        counts.update(tokenise(normalise_text(text, config)))

    # max_size includes pad and unk, so at most max_size - 2 tokens survive.
    budget = max(settings.max_size - 2, 0)
    eligible = [token for token, count in counts.items() if count >= settings.min_frequency]
    eligible.sort(key=lambda token: (-counts[token], token))
    retained = eligible[:budget]

    token_to_index = {settings.padding_token: PAD_INDEX, settings.unknown_token: UNK_INDEX}
    for index, token in enumerate(retained, start=len(token_to_index)):
        token_to_index[token] = index

    return Vocabulary(
        token_to_index=token_to_index,
        pad_token=settings.padding_token,
        unk_token=settings.unknown_token,
    )
