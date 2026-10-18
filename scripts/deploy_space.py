# deploy_space.py
# Purpose: assemble the Hugging Face Space folder and, if asked, publish it.
# Author: Colile Sibanda

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = REPO_ROOT / "deploy"

# Everything the image needs, and nothing else: no data, notebooks or
# working files leave the machine.
TREES = ("src/phishlens", "src/web")
FILES = (
    "run_api.py",
    "configs/default.yaml",
    "artifacts/models/text_cnn_best.pt",
    "artifacts/models/vocabulary.json",
    "artifacts/metrics/m1_text_cnn.json",
)
DEPLOY_FILES = {
    "huggingface/Dockerfile": "Dockerfile",
    "huggingface/README.md": "README.md",
    "requirements-serve.txt": "requirements-serve.txt",
    "huggingface/local.yaml": "configs/local.yaml",
}


def stage(out_dir: Path) -> list[Path]:
    """Copy the files the Space needs into a fresh folder.

    Inputs: out_dir : target folder; emptied first if it exists, and refused
    if it lies inside the repository, so a stray copy cannot be committed.
    Output: the staged file paths, relative to out_dir.

    Raises FileNotFoundError naming any required file that is missing, and
    ValueError if out_dir is inside the repository.
    """
    out_dir = out_dir.resolve()
    if out_dir == REPO_ROOT or REPO_ROOT in out_dir.parents:
        raise ValueError(f"Stage outside the repository, not in {out_dir}.")
    missing = [f for f in FILES if not (REPO_ROOT / f).is_file()]
    missing += [f for f in DEPLOY_FILES if not (DEPLOY_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Required for the Space but missing: {', '.join(missing)}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for tree in TREES:
        shutil.copytree(REPO_ROOT / tree, out_dir / tree, ignore=ignore)
    for name in FILES:
        (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, out_dir / name)
    for source, target in DEPLOY_FILES.items():
        (out_dir / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEPLOY_DIR / source, out_dir / target)
    return sorted(p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file())


def publish(out_dir: Path, repo_id: str) -> str:
    """Create the Space if needed and upload the staged folder to it.

    Uses the token saved by `hf auth login`. Large files (the checkpoint) are
    handled by huggingface_hub, so no git LFS setup is needed.

    Inputs: out_dir : a folder produced by stage(); repo_id : "user/space".
    Output: the Space's public URL.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, repo_type="space",
        commit_message="Deploy PhishLens API and browser UI",
    )
    owner, name = repo_id.split("/")
    return f"https://{owner}-{name}.hf.space".lower().replace("_", "-")


def main() -> None:
    """Parse arguments, stage the Space, and publish it when a repo is given.

    Inputs: command-line arguments (see --help).
    Output: none; prints `Description: value` lines.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="staging folder, outside the repository")
    parser.add_argument("--publish", metavar="USER/SPACE", help="upload to this Hugging Face Space")
    args = parser.parse_args()

    staged = stage(args.out_dir)
    print(f"Staged files: {len(staged)}")
    print(f"Staging folder: {args.out_dir.resolve()}")
    if args.publish:
        print(f"Space URL: {publish(args.out_dir, args.publish)}")


if __name__ == "__main__":
    main()
