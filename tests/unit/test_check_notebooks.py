# test_check_notebooks.py
# Purpose: unit tests for the notebook audit in scripts/check_notebooks.py.
# Author: Colile Sibanda

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# scripts/ is not a package, so load the file by path. It must be registered
# in sys.modules before executing: @dataclass resolves its module there.
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_notebooks.py"
_spec = importlib.util.spec_from_file_location("check_notebooks", _SCRIPT)
check_notebooks = importlib.util.module_from_spec(_spec)
sys.modules["check_notebooks"] = check_notebooks
_spec.loader.exec_module(check_notebooks)


def _cell(source: str, outputs: list | None = None) -> dict:
    """Build a minimal code cell.

    Inputs: source : the cell's code; outputs : its stored outputs.
    Output: a cell dict shaped like .ipynb JSON.
    """
    return {"cell_type": "code", "source": source, "outputs": outputs or []}


_STREAM = [{"output_type": "stream", "text": "ok"}]


def test_a_notebook_with_cleared_outputs_is_a_problem():
    audit = check_notebooks.audit_notebook("x.ipynb", {"cells": [_cell("print(1)")]})

    assert audit.cells_with_output == 0
    assert any("no stored outputs" in p for p in check_notebooks.problems_in([audit]))


def test_a_stored_error_is_a_problem():
    error = [{"output_type": "error", "ename": "MemoryError"}]
    audit = check_notebooks.audit_notebook("x.ipynb", {"cells": [_cell("f()", error)]})

    assert audit.error_cells == 1


def test_reassembling_the_pool_is_not_reading_the_test_split():
    """The LOCO notebook concatenates all three splits; that is not an evaluation."""
    pooled = _cell('concat(["train.csv", "validation.csv", "test.csv"])', _STREAM)
    audit = check_notebooks.audit_notebook("05_loco.ipynb", {"cells": [pooled]})

    assert audit.reads_test_split is False


def test_only_the_evaluation_notebook_may_read_the_test_split():
    evaluate = check_notebooks.audit_notebook(
        "04_evaluate.ipynb", {"cells": [_cell('read_csv("test.csv")', _STREAM)]})
    rogue = check_notebooks.audit_notebook(
        "03_train_cnn.ipynb", {"cells": [_cell('read_csv("test.csv")', _STREAM)]})

    assert check_notebooks.problems_in([evaluate]) == []

    # Loading all three splits side by side, without concatenating, is still
    # reading the test split: this is what 04_evaluate.ipynb actually does.
    side_by_side = check_notebooks.audit_notebook("x.ipynb", {"cells": [_cell(
        'read_csv("train.csv"); read_csv("validation.csv"); read_csv("test.csv")',
        _STREAM)]})
    assert side_by_side.reads_test_split is True
    assert any("04_evaluate.ipynb only" in p
               for p in check_notebooks.problems_in([evaluate, rogue]))
