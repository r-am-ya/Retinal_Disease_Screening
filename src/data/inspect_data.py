"""Inspect archives, detect duplicates, and link augmented samples to originals."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, UnidentifiedImageError

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.common import CLASS_NAMES, discover_images, hamming, phash, sha256

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "central_serous_chorioretinopathy": 101, "diabetic_retinopathy": 1509,
    "disc_edema": 127, "glaucoma": 1349, "healthy": 1024, "macular_scar": 444,
    "myopia": 500, "pterygium": 17, "retinal_detachment": 125,
    "retinitis_pigmentosa": 139,
}
TRANSFORMS = ("identity", "flip_lr", "flip_ud", "rotate_180")


def dimensions(paths: list[Path]) -> tuple[str, list[str]]:
    sizes: list[tuple[int, int]] = []
    corrupt: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                sizes.append(image.size)
        except (OSError, UnidentifiedImageError, ValueError):
            corrupt.append(str(path))
    if not sizes:
        return "n/a", corrupt
    widths, heights = zip(*sizes, strict=True)
    return (
        f"W {min(widths)}/{int(statistics.median(widths))}/{max(widths)}, "
        f"H {min(heights)}/{int(statistics.median(heights))}/{max(heights)}",
        corrupt,
    )


def exact_duplicates(paths: list[Path]) -> int:
    hashes: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        hashes[sha256(path)].append(path)
    return sum(len(group) - 1 for group in hashes.values() if len(group) > 1)


def closest_link(augmented: Path, original_hashes: list[tuple[Path, list[np.uint64]]], bands: dict[tuple[int, int], set[int]]) -> tuple[Path | None, int]:
    aug_hash = phash(augmented)
    paths = [entry[0] for entry in original_hashes]
    candidates = set()
    integer = int(aug_hash)
    for band in range(16):
        candidates.update(bands[(band, (integer >> (band * 4)) & 0xF)])
    if not candidates:
        return None, 65
    candidate_indexes = np.fromiter(candidates, dtype=np.int64)
    values = np.asarray([original_hashes[index][1] for index in candidate_indexes], dtype=np.uint64)
    xor_values = np.bitwise_xor(values, aug_hash).view(np.uint8).reshape(-1, 8)
    distances = np.unpackbits(xor_values, axis=1).sum(axis=1).reshape(len(candidate_indexes), -1).min(axis=1)
    local_index = int(np.argmin(distances))
    return paths[int(candidate_indexes[local_index])], int(distances[local_index])


def link_augmented(original: dict[str, list[Path]], augmented: dict[str, list[Path]]) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Link by same stem first, then a transform-tolerant perceptual hash."""
    links: dict[str, dict[str, object]] = {}
    match_counts: dict[str, int] = {}
    for label in CLASS_NAMES:
        original_paths = original.get(label, [])
        stem_index = {path.stem.lower(): path for path in original_paths}
        hash_entries: list[tuple[Path, list[np.uint64]]] = []
        for path in original_paths:
            hash_entries.append((path, [phash(path, transform) for transform in TRANSFORMS]))
        bands: dict[tuple[int, int], set[int]] = defaultdict(set)
        for index, (_, values) in enumerate(hash_entries):
            for value in values:
                integer = int(value)
                for band in range(16):
                    bands[(band, (integer >> (band * 4)) & 0xF)].add(index)
        matched = 0
        for path in augmented.get(label, []):
            source = stem_index.get(path.stem.lower())
            method = "filename" if source else ""
            distance: int | None = 0 if source else None
            if source is None:
                source, distance = closest_link(path, hash_entries, bands)
                if source is not None and distance <= 10:
                    method = "perceptual_hash"
                else:
                    source, method = None, "unmatched"
            if source is not None:
                matched += 1
            links[str(path.resolve())] = {
                "class": label,
                "source_path": str(source.resolve()) if source else None,
                "source_id": str(source.resolve()) if source else None,
                "method": method,
                "distance": distance,
            }
        match_counts[label] = matched
    return links, match_counts


def plot_counts(original: dict[str, list[Path]], augmented: dict[str, list[Path]], output: Path) -> None:
    labels = CLASS_NAMES
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(positions - 0.2, [len(original.get(label, [])) for label in labels], 0.4, label="original")
    axis.bar(positions + 0.2, [len(augmented.get(label, [])) for label in labels], 0.4, label="augmented")
    axis.set_xticks(positions, [label.replace("_", " ") for label in labels], rotation=35, ha="right")
    axis.set_ylabel("Images")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    original = discover_images(args.root / "data" / "raw" / "original")
    augmented = discover_images(args.root / "data" / "raw" / "augmented")
    report_dir = args.root / "reports"
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_counts(original, augmented, figure_dir / "class_distribution.png")
    all_original = [path for paths in original.values() for path in paths]
    links, match_counts = link_augmented(original, augmented)
    link_path = report_dir / "augmentation_links.json"
    link_path.write_text(json.dumps(links, indent=2), encoding="utf-8")
    total_augmented = sum(len(augmented.get(label, [])) for label in CLASS_NAMES)
    linked = sum(match_counts.values())
    rate = linked / total_augmented if total_augmented else 0.0
    lines = ["# Data inspection report", "", "## Counts", "", "| Class | Original | Expected | Augmented | Status |", "|---|---:|---:|---:|---|"]
    for label in [*CLASS_NAMES, "pterygium"]:
        observed = len(original.get(label, []))
        expected = EXPECTED[label]
        status = "OK" if observed == expected else "DIFFERS"
        lines.append(f"| {label} | {observed} | {expected} | {len(augmented.get(label, []))} | {status} |")
    lines.extend(["", "## Image details", ""])
    corrupt: list[str] = []
    for label in [*CLASS_NAMES, "pterygium"]:
        paths = original.get(label, [])
        size_summary, invalid = dimensions(paths)
        corrupt.extend(invalid)
        extensions = Counter(path.suffix.lower() for path in paths)
        samples = ", ".join(path.name for path in paths[:5])
        lines.append(f"### {label}")
        lines.append(f"- Samples: {samples}")
        lines.append(f"- Extensions: {dict(extensions)}")
        lines.append(f"- Size min/median/max: {size_summary}")
    lines.extend(["", "## Integrity", "", f"- Corrupt/unreadable original images: {len(corrupt)}", f"- Exact duplicate files within original: {exact_duplicates(all_original)}"])
    lines.extend(["", "## Augmentation linkage", "", f"- Linked {linked}/{total_augmented} non-Pterygium augmented images ({rate:.2%}).", "- Link method: exact filename stem first; remaining images use DCT perceptual hashes compared against original flip/rotation variants (Hamming distance ≤10)."])
    for label in CLASS_NAMES:
        count = len(augmented.get(label, []))
        percentage = match_counts[label] / count if count else 0.0
        lines.append(f"- {label}: {match_counts[label]}/{count} ({percentage:.2%}) linked.")
    if rate < 0.95:
        lines.extend(["", "## WARNING", "", "**Augmented-to-original linkage is below 95%; the splitter will exclude augmented files and use on-the-fly augmentation only.**"])
    lines.extend(["", "## Limitations", "", "Near-duplicate checks use transform-tolerant perceptual hashes. Patient identifiers are unavailable, so patient-level separation cannot be verified."])
    (report_dir / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Original images: {len(all_original)}")
    print(f"Augmented linkage: {linked}/{total_augmented} ({rate:.2%})")
    print(f"Report: {report_dir / 'data_report.md'}")


if __name__ == "__main__":
    main()
