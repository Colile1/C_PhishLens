# email_file.py
# Purpose: turn a saved .eml or .txt file into the subject-and-body text the model takes.
# Author: Colile Sibanda

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path


def read_email_text(path: Path) -> str:
    """Read an email file as the subject and body text the model expects.

    A .eml file is parsed and reduced to its subject and plain-text body:
    headers are dropped on purpose, because the model is trained on subject
    and body only and header residue is the corpus artefact this project
    guards against (see the root README). Any other
    file is read as plain text.

    Inputs: path : the file to read.
    Output: the text to analyse. Raises ValueError naming the file when an
    .eml file has neither a subject nor a readable text body, and OSError
    when the file cannot be read.
    """
    if path.suffix.lower() != ".eml":
        return path.read_text(encoding="utf-8", errors="replace")

    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    subject = str(message.get("subject", "") or "").strip()
    part = message.get_body(preferencelist=("plain", "html"))
    body = part.get_content().strip() if part is not None else ""
    text = "\n\n".join(piece for piece in (subject, body) if piece)
    if not text:
        raise ValueError(f"{path.name} has no subject and no readable text body.")
    return text
