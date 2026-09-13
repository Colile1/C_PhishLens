# text_cnn.py
# Purpose: the M1 Text-CNN architecture — embedding, parallel convolutions, one logit.
# Author: Colile Sibanda

from __future__ import annotations

import torch
from torch import nn

from phishlens.config.schema import Config
from phishlens.features.vocabulary import PAD_INDEX


class TextCNN(nn.Module):
    """Text-CNN for binary phishing classification from token ids.

    Architecture: embedding -> parallel 1D convolutions of widths 3, 4 and 5 ->
    ReLU -> max-over-time pooling -> concatenate -> dropout -> linear -> one
    logit. Filter widths act as phrase detectors, which is the argument for a
    CNN on this data: phishing lures are phrases ("verify your account within
    24 hours"). Max-over-time pooling makes the model indifferent to where in
    a long email the lure appears.

    The output is a single logit for use with BCEWithLogitsLoss, not a softmax
    over two classes: numerically steadier, and it keeps the decision
    threshold explicit rather than hidden inside an argmax.

    Inputs: vocab_size — number of vocabulary entries including pad and unk;
    embedding_dim — embedding width; filter_sizes — convolution widths;
    filters_per_size — feature maps per width; dropout — dropout probability
    applied to the pooled features.
    Output: an unfitted nn.Module; forward() returns logits of shape (batch,).

    Raises ValueError naming the offending argument if any size is
    non-positive or dropout is outside [0, 1).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        filter_sizes: tuple[int, ...],
        filters_per_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if not filter_sizes:
            raise ValueError("filter_sizes must name at least one convolution width.")
        if any(size <= 0 for size in filter_sizes):
            raise ValueError(f"Every entry in filter_sizes must be positive, got {filter_sizes}.")
        if filters_per_size <= 0:
            raise ValueError(f"filters_per_size must be positive, got {filters_per_size}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.filter_sizes = tuple(filter_sizes)

        # padding_idx keeps the pad embedding fixed at zero, so padding cannot
        # contribute a learned signal to the pooled features.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, filters_per_size, kernel_size=size)
            for size in self.filter_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters_per_size * len(self.filter_sizes), 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return one logit per input sequence.

        Inputs: token_ids — int64 tensor of shape (batch, sequence_length),
        as produced by features.encoder.encode.
        Output: float tensor of shape (batch,) holding raw logits. Apply
        torch.sigmoid to obtain probabilities; the loss function does this
        internally, so the loop must not do it twice.

        Raises ValueError if the sequence is shorter than the widest filter,
        which would make that convolution's output empty.
        """
        widest = max(self.filter_sizes)
        if token_ids.size(-1) < widest:
            raise ValueError(
                f"Sequence length {token_ids.size(-1)} is shorter than the widest "
                f"filter ({widest}). Increase features.max_sequence_length or "
                f"reduce model.filter_sizes."
            )

        # (batch, sequence, embedding) -> (batch, embedding, sequence): Conv1d
        # convolves over the last dimension, so channels must come first.
        embedded = self.embedding(token_ids).transpose(1, 2)

        pooled = []
        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            # Max over time: one value per feature map, whatever the length.
            pooled.append(activated.max(dim=2).values)

        features = self.dropout(torch.cat(pooled, dim=1))
        # (batch, 1) -> (batch,), so the logits match the label shape.
        return self.classifier(features).squeeze(1)


def build_text_cnn(vocab_size: int, config: Config) -> TextCNN:
    """Build a TextCNN from the validated config.

    Inputs: vocab_size — size of the vocabulary built on training text only;
    config — validated Config supplying the model section.
    Output: an unfitted TextCNN on the CPU. Callers move it to a device.
    """
    settings = config.model
    return TextCNN(
        vocab_size=vocab_size,
        embedding_dim=settings.embedding_dim,
        filter_sizes=tuple(settings.filter_sizes),
        filters_per_size=settings.filters_per_size,
        dropout=settings.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters, for the architecture table in report 3.2.

    Inputs: model — any module.
    Output: number of parameters with requires_grad set.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
