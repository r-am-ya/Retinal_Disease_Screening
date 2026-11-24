"""Safely and idempotently extract the supplied image archives."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_member_path(destination: Path, member_name: str) -> Path:
    candidate = (destination / member_name).resolve()
    if destination.resolve() not in candidate.parents and candidate != destination.resolve():
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return candidate


def _skip_member(name: str) -> bool:
    parts = Path(name).parts
    return "__MACOSX" in parts or Path(name).name == ".DS_Store"


def _has_images(destination: Path) -> bool:
    return destination.exists() and any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in destination.rglob("*")
    )


def extract_archive(archive: Path, destination: Path) -> str:
    """Extract one zip archive without overwriting a completed extraction."""
    if _has_images(destination):
        return f"Skipping {archive.name}: images already exist in {destination}."
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir() or _skip_member(member.filename):
                continue
            target = _safe_member_path(destination, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    return f"Extracted {extracted} files from {archive.name} into {destination}."


def folder_tree(root: Path, max_depth: int = 3) -> str:
    """Return a compact tree with image counts, avoiding thousands of file names."""
    lines = [str(root)]
    for path in sorted(path for path in root.rglob("*") if path.is_dir()):
        relative = path.relative_to(root)
        if len(relative.parts) > max_depth:
            continue
        count = sum(
            file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            for file in path.rglob("*")
        )
        lines.append(f"{'  ' * len(relative.parts)}{path.name}/ ({count} images)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    archives = {
        "original": args.root / "Original Dataset.zip",
        "augmented": args.root / "Augmented Dataset.zip",
    }
    for name, archive in archives.items():
        if not archive.exists():
            raise FileNotFoundError(f"Required archive not found: {archive}")
        destination = args.root / "data" / "raw" / name
        print(extract_archive(archive, destination))
        print(folder_tree(destination))


if __name__ == "__main__":
    main()
