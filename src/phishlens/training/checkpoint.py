# checkpoint.py
# Purpose: write the best epoch's weights and load them back before evaluation.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Write a model's weights to disk, replacing any previous checkpoint.

    Called only when the validation metric improves, so the file always holds
    the best epoch's weights rather than the most recent epoch's.

    Weights are moved to the CPU before saving so a checkpoint written on one
    device loads on another.

    Inputs: model — the module whose state to save; path — destination file.
    Output: none; writes the state dict to path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    torch.save(state, path)


def restore_best(model: nn.Module, path: Path) -> None:
    """Load the best epoch's weights back into the model, in place.

    This is the step that decides which model the reported test numbers
    describe. If it is skipped, the model carries the last epoch's weights,
    which are by definition worse than the best epoch's whenever early
    stopping fired — and every number in the report would then belong to a
    model that was never the one selected.

    Inputs: model — the module to restore into; path — a file written by
    save_checkpoint.
    Output: none; the model's parameters are replaced in place.

    Raises FileNotFoundError naming the missing path rather than silently
    continuing with the last epoch's weights.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {path}, so the best epoch's weights cannot be "
            f"restored. Continuing would evaluate the last epoch instead of the "
            f"best one. Re-run training so the checkpoint is written."
        )

    state = torch.load(path, map_location="cpu", weights_only=True)
    # strict=True: a key mismatch means the checkpoint belongs to a different
    # architecture, which must fail loudly rather than load part of a model.
    model.load_state_dict(state, strict=True)
