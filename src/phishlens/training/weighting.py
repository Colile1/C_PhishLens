# weighting.py
# Purpose: positive-class weight computed from the training split only.
# Author: Colile Sibanda

from __future__ import annotations

import numpy as np
import torch


def positive_class_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute the positive-class weight for BCEWithLogitsLoss.

    The weight is negatives / positives, which is what `pos_weight` expects:
    it scales the loss contribution of positive examples so that a class
    imbalance does not let the model win by predicting the majority class.

    Computed from the **training split only**. Deriving it from validation or
    test labels would leak their class balance into training.

    Inputs: labels : 1-D array of 0/1 training labels.
    Output: a scalar float32 tensor holding negatives / positives.

    Raises ValueError if the array is empty or holds a single class, since no
    meaningful weight exists in either case.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("positive_class_weight() needs a non-empty label array.")

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())

    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Training labels hold only one class ({positives} positive, "
            f"{negatives} negative), so no positive-class weight is defined. "
            f"Check the split in data/processed/."
        )

    return torch.tensor(negatives / positives, dtype=torch.float32)
