# baseline.py
# Purpose: M0 sklearn pipeline factory : TF-IDF into Logistic Regression.
# Author: Colile Sibanda

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishlens.config.schema import Config
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise
from sklearn.feature_extraction.text import TfidfVectorizer


def build_baseline(config: Config) -> Pipeline:
    """Build an unfitted M0 pipeline: TF-IDF into class-weighted LogReg.

    A factory, not a trained object : callers fit it on the training split
    only. Uses the project tokeniser and normaliser so M0's vocabulary is
    built from the same tokens LIME will later report for M1.

    Inputs: config : validated Config supplying tfidf settings and the seed.
    Output: unfitted sklearn Pipeline with steps "tfidf" and "classifier".
    """
    settings = config.features.tfidf
    vectoriser = TfidfVectorizer(
        tokenizer=tokenise,
        preprocessor=lambda text: normalise_text(text, config),
        lowercase=False,
        max_features=settings.max_features,
        ngram_range=tuple(settings.ngram_range),
        min_df=settings.min_df,
        token_pattern=None,
    )
    classifier = LogisticRegression(
        class_weight="balanced" if config.training.class_weighting else None,
        random_state=config.seed,
        max_iter=1000,
    )
    return Pipeline([("tfidf", vectoriser), ("classifier", classifier)])
