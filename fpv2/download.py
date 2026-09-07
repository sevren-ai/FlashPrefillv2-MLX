"""Download a Hugging Face model snapshot beneath the local models directory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face repository ID")
    parser.add_argument("--name", help="Local directory name (defaults to repo basename)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--revision", help="Branch, tag, or commit to download")
    parser.add_argument("--token", help="Hugging Face access token")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    name = args.name or args.repo.rstrip("/").rsplit("/", 1)[-1]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("--name must be a single directory name")
    destination = args.models_dir / name
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        token=args.token,
        local_dir=destination,
    )
    print(path)


if __name__ == "__main__":
    main()
