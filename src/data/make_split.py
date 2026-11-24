"""Create leakage-aware copies for train/validation/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.data.common import CLASS_NAMES, discover_images, phash

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def duplicate_clusters(paths: list[Path]) -> list[list[Path]]:
    """Group exact perceptual-hash duplicates so they cannot cross a split."""
    clusters: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        clusters[int(phash(path))].append(path)
    return list(clusters.values())


def assign_clusters(clusters: list[list[Path]], seed: int) -> dict[str, list[Path]]:
    """Greedily allocate whole clusters to the closest 70/15/15 target."""
    total = sum(len(cluster) for cluster in clusters)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    assigned: dict[str, list[Path]] = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    shuffled = clusters.copy()
    random.Random(seed).shuffle(shuffled)
    for cluster in sorted(shuffled, key=len, reverse=True):
        split = min(targets, key=lambda name: (counts[name] - targets[name], counts[name]))
        assigned[split].extend(cluster)
        counts[split] += len(cluster)
    return assigned


def select_augmented(links: dict[str, dict[str, object]], train_sources: set[str], label: str, target: int) -> list[tuple[Path, str]]:
    """Round-robin augmented copies across eligible source originals."""
    by_source: dict[str, list[Path]] = defaultdict(list)
    for raw_path, metadata in links.items():
        if metadata["class"] != label or metadata["source_id"] not in train_sources:
            continue
        by_source[str(metadata["source_id"])].append(Path(raw_path))
    selected: list[tuple[Path, str]] = []
    indexes = {source: 0 for source in by_source}
    while len(selected) < target:
        added = False
        for source in sorted(by_source):
            index = indexes[source]
            if index >= len(by_source[source]) or len(selected) >= target:
                continue
            selected.append((by_source[source][index], source))
            indexes[source] += 1
            added = True
        if not added:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    original = discover_images(root / "data" / "raw" / "original")
    links_path = root / "reports" / "augmentation_links.json"
    if not links_path.exists():
        raise FileNotFoundError("Run `python -m src.data.inspect_data` before splitting.")
    links: dict[str, dict[str, object]] = json.loads(links_path.read_text(encoding="utf-8"))
    possible = [value for value in links.values() if value["class"] in CLASS_NAMES]
    reliable = bool(possible) and sum(value["source_id"] is not None for value in possible) / len(possible) >= 0.95
    split_root = root / "data" / "splits"
    if split_root.exists():
        shutil.rmtree(split_root)
    rows: list[dict[str, str]] = []
    clusters_record: dict[str, list[list[str]]] = {}
    summaries: list[tuple[str, int, int, int, int, float]] = []
    for label in CLASS_NAMES:
        clusters = duplicate_clusters(original[label])
        clusters_record[label] = [[str(path.resolve()) for path in cluster] for cluster in clusters if len(cluster) > 1]
        partitions = assign_clusters(clusters, SEED)
        train_sources = {str(path.resolve()) for path in partitions["train"]}
        augmented = select_augmented(links, train_sources, label, len(partitions["train"]) // 3) if reliable else []
        for split, paths in partitions.items():
            target_dir = split_root / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(sorted(paths)):
                target = target_dir / f"orig_{index:05d}_{source.name}"
                shutil.copy2(source, target)
                rows.append({"path": str(target.relative_to(root)), "class": label, "split": split, "is_augmented": "False", "source_id": str(source.resolve()), "raw_path": str(source.resolve())})
        for index, (source, source_id) in enumerate(augmented):
            target = split_root / "train" / label / f"aug_{index:05d}_{source.name}"
            shutil.copy2(source, target)
            rows.append({"path": str(target.relative_to(root)), "class": label, "split": "train", "is_augmented": "True", "source_id": source_id, "raw_path": str(source.resolve())})
        train_count = len(partitions["train"])
        summaries.append((label, train_count + len(augmented), len(partitions["val"]), len(partitions["test"]), len(augmented), len(augmented) / (train_count + len(augmented))))
    with (split_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "class", "split", "is_augmented", "source_id", "raw_path"])
        writer.writeheader()
        writer.writerows(rows)
    (split_root / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    (split_root / "near_duplicate_clusters.json").write_text(json.dumps(clusters_record, indent=2), encoding="utf-8")
    report = ["# Split summary", "", "Original images were stratified per class with seed 42 into approximate 70/15/15 train/validation/test partitions. Exact perceptual-hash duplicate clusters were kept intact. Patient IDs are unavailable, so patient-level leakage cannot be ruled out.", "", f"Augmentation linkage was {'reliable; eligible copies were included only for original train sources.' if reliable else 'below 95%; no augmented image was used.'}", "", "| Class | Train | Validation | Test | Augmented train | Augmented % train |", "|---|---:|---:|---:|---:|---:|"]
    for label, train, val, test, augmented, ratio in summaries:
        report.append(f"| {label} | {train} | {val} | {test} | {augmented} | {ratio:.2%} |")
    (root / "reports" / "split_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Created {len(rows)} copied split images; augmented enabled: {reliable}.")


if __name__ == "__main__":
    main()
