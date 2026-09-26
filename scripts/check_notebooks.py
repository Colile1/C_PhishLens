# check_notebooks.py
# Purpose: assert every notebook has visible outputs and only one evaluates the test split.
# Author: Colile Sibanda

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TEST_FILE = "test.csv"
POOL_FILES = ("train.csv", "validation.csv")
EXPECTED_TEST_NOTEBOOK = "04_evaluate.ipynb"


@dataclass(frozen=True)
class NotebookAudit:
    """What one notebook shows and whether it evaluates the test split.

    Inputs: name : the file name; code_cells : non-empty code cells;
    cells_with_output : how many of them have stored outputs; error_cells :
    how many stored an error; reads_test_split : whether a cell loads
    test.csv on its own, rather than as part of reassembling the full pool.
    Output: immutable NotebookAudit.
    """

    name: str
    code_cells: int
    cells_with_output: int
    error_cells: int
    reads_test_split: bool


def audit_notebook(name: str, notebook: dict) -> NotebookAudit:
    """Audit one decoded notebook. Pure: no file access.

    A cell that concatenates test.csv with train.csv and validation.csv is
    reassembling the whole pool (the LOCO notebook does this to re-split by
    corpus), which is not an evaluation on the reserved test split. Merely
    loading all three side by side, as the evaluation notebook does, still
    counts as reading the test split.

    Inputs: name : the notebook's file name; notebook : the decoded .ipynb JSON.
    Output: a NotebookAudit.
    """
    code = [cell for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code" and "".join(cell.get("source", "")).strip()]
    with_output = sum(1 for cell in code if cell.get("outputs"))
    errors = sum(1 for cell in code
                 if any(out.get("output_type") == "error" for out in cell.get("outputs", [])))
    reads_test = False
    for cell in code:
        source = "".join(cell.get("source", ""))
        reassembles_pool = "concat" in source and all(f in source for f in POOL_FILES)
        if TEST_FILE in source and not reassembles_pool:
            reads_test = True
    return NotebookAudit(name, len(code), with_output, errors, reads_test)


def problems_in(audits: list[NotebookAudit]) -> list[str]:
    """List every rule the audited notebooks break.

    Inputs: audits : one NotebookAudit per notebook.
    Output: readable problem sentences; empty when everything passes.
    """
    problems = []
    for audit in audits:
        if audit.cells_with_output == 0:
            problems.append(f"{audit.name} has no stored outputs: re-run it and save.")
        if audit.error_cells:
            problems.append(f"{audit.name} stored {audit.error_cells} error output(s).")
    readers = [audit.name for audit in audits if audit.reads_test_split]
    if readers != [EXPECTED_TEST_NOTEBOOK]:
        problems.append(
            f"The test split must be evaluated in {EXPECTED_TEST_NOTEBOOK} only; "
            f"found it read in: {readers or 'no notebook'}."
        )
    return problems


def main() -> None:
    """Audit every notebook in notebooks/ and exit non-zero on any problem.

    Inputs: none.
    Output: none; prints `Description: value` lines and exits 1 on failure.
    """
    paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not paths:
        sys.exit(f"No notebooks found in {NOTEBOOK_DIR}.")
    audits = [audit_notebook(path.name, json.loads(path.read_text(encoding="utf-8")))
              for path in paths]
    for audit in audits:
        print(f"{audit.name}: {audit.cells_with_output}/{audit.code_cells} code cells "
              f"with output, {audit.error_cells} errors, "
              f"reads test split: {audit.reads_test_split}")
    problems = problems_in(audits)
    for problem in problems:
        print(f"Problem: {problem}")
    print(f"Result: {'FAIL' if problems else 'PASS'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
