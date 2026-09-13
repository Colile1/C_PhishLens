# batching.py
# Purpose: tensor datasets and seeded loaders for the Text-CNN.
# Author: Colile Sibanda

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from phishlens.config.schema import Config


def make_dataset(sequences: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """Pair encoded sequences with their labels as a TensorDataset.

    Inputs: sequences — int64 array of shape (rows, max_sequence_length) from
    features.encoder.encode; labels — 1-D array of 0/1 labels for those rows.
    Output: TensorDataset yielding (int64 sequence, float32 label) pairs. The
    label is float32 because BCEWithLogitsLoss compares against float targets.

    Raises ValueError if the two arrays disagree on row count, which means the
    encoding and the labels came from different frames.
    """
    if len(sequences) != len(labels):
        raise ValueError(
            f"make_dataset() got {len(sequences)} sequences but {len(labels)} "
            f"labels. They must describe the same rows."
        )

    return TensorDataset(
        torch.from_numpy(np.asarray(sequences, dtype=np.int64)),
        torch.from_numpy(np.asarray(labels, dtype=np.float32)),
    )


def make_loader(
    dataset: TensorDataset, config: Config, *, shuffle: bool
) -> DataLoader:
    """Wrap a dataset in a DataLoader with a seeded shuffle order.

    The generator is seeded from config, so batch order is reproducible: two
    runs on the same seed see the same batches in the same order, which is
    what makes a training run repeatable.

    Inputs: dataset — a TensorDataset; config — validated Config supplying the
    batch size and the seed; shuffle — True for the training loader, False for
    validation and test, where order must stay fixed so predictions line up
    with the rows they came from.
    Output: a DataLoader over the dataset.
    """
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(config.seed)

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        generator=generator,
    )
