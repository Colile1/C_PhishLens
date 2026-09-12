# encoder.py
# Purpose: turn text into padded integer sequences for the M1 Text-CNN.
# Author: Colile Sibanda

from __future__ import annotations

import numpy as np

from phishlens.config.schema import Config
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise
from phishlens.features.vocabulary import PAD_INDEX, Vocabulary


def encode(texts: list[str], vocabulary: Vocabulary, config: Config) -> np.ndarray:
    """Encode texts into padded, truncated integer sequences.

    Inputs: texts — cleaned text strings, from any split; vocabulary —
    Vocabulary built on training text only; config — validated Config
    supplying `max_sequence_length`.
    Output: int64 array of shape (len(texts), max_sequence_length). Sequences
    shorter than max_sequence_length are right-padded with `<pad>` (index 0);
    longer ones are truncated. Out-of-vocabulary tokens map to `<unk>`.
    """
    max_length = config.features.max_sequence_length
    sequences = np.full((len(texts), max_length), PAD_INDEX, dtype=np.int64)

    for row, text in enumerate(texts):
        tokens = tokenise(normalise_text(text, config))[:max_length]
        for column, token in enumerate(tokens):
            sequences[row, column] = vocabulary.encode_token(token)

    return sequences
