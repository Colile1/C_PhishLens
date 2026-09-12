# __init__.py
# Purpose: public entry point for the features package.
# Author: Colile Sibanda

from phishlens.features.encoder import encode
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise
from phishlens.features.vectoriser import fit_tfidf, transform
from phishlens.features.vocabulary import PAD_INDEX, UNK_INDEX, Vocabulary, build_vocabulary

__all__ = [
    "normalise_text",
    "tokenise",
    "fit_tfidf",
    "transform",
    "Vocabulary",
    "build_vocabulary",
    "PAD_INDEX",
    "UNK_INDEX",
    "encode",
]
