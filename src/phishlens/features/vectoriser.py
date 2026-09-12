# vectoriser.py
# Purpose: fit and apply the TF-IDF feature space for the M0 baseline.
# Author: Colile Sibanda

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from phishlens.config.schema import Config
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise


def _prepare(texts: list[str], config: Config) -> list[str]:
    """Normalise a batch of texts ahead of vectorisation.

    Inputs: texts — raw cleaned strings; config — validated Config.
    Output: list of normalised strings, same order and length as texts.
    """
    return [normalise_text(text, config) for text in texts]


def fit_tfidf(train_texts: list[str], config: Config) -> TfidfVectorizer:
    """Fit a TF-IDF vectoriser on training texts only.

    The parameter name makes misuse awkward: there is no `texts` argument
    that could be handed a validation or test frame by mistake. Fitting on
    anything but the training split leaks the held-out vocabulary and IDF
    weights into the model, invisibly inflating validation and test scores.

    Inputs: train_texts — cleaned text column from the training split only;
    config — validated Config supplying the tfidf settings.
    Output: a fitted TfidfVectorizer.
    """
    settings = config.features.tfidf
    vectoriser = TfidfVectorizer(
        tokenizer=tokenise,
        preprocessor=lambda text: text,
        lowercase=False,
        max_features=settings.max_features,
        ngram_range=tuple(settings.ngram_range),
        min_df=settings.min_df,
        token_pattern=None,
    )
    vectoriser.fit(_prepare(train_texts, config))
    return vectoriser


def transform(vectoriser: TfidfVectorizer, texts: list[str], config: Config):
    """Transform texts into the TF-IDF space of an already-fitted vectoriser.

    Inputs: vectoriser — fitted TfidfVectorizer from fit_tfidf; texts — texts
    to transform, from any split; config — validated Config.
    Output: sparse matrix of shape (len(texts), vocabulary size).
    """
    return vectoriser.transform(_prepare(texts, config))
