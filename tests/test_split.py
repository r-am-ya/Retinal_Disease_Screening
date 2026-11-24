"""Leakage guards for a generated split; skipped in source-only CI checkouts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "splits" / "manifest.csv"


@pytest.fixture
def rows() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        pytest.skip("Generated dataset split is not available in CI.")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_no_pterygium_or_augmented_evaluation(rows: list[dict[str, str]]) -> None:
    assert all(row["class"] != "pterygium" for row in rows)
    assert all(not (row["is_augmented"] == "True" and row["split"] != "train") for row in rows)


def test_originals_do_not_cross_splits(rows: list[dict[str, str]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["is_augmented"] == "False":
            locations[row["raw_path"]].add(row["split"])
    assert all(len(splits) == 1 for splits in locations.values())


def test_augmented_sources_are_training_originals(rows: list[dict[str, str]]) -> None:
    train_sources = {row["source_id"] for row in rows if row["split"] == "train" and row["is_augmented"] == "False"}
    assert all(row["source_id"] in train_sources for row in rows if row["is_augmented"] == "True")


def test_classes_and_duplicate_clusters_do_not_cross_splits(rows: list[dict[str, str]]) -> None:
    classes = {row["class"] for row in rows}
    for split in ("train", "val", "test"):
        assert {row["class"] for row in rows if row["split"] == split} == classes
    clusters = json.loads((ROOT / "data" / "splits" / "near_duplicate_clusters.json").read_text())
    originals = {row["raw_path"]: row["split"] for row in rows if row["is_augmented"] == "False"}
    for class_clusters in clusters.values():
        for cluster in class_clusters:
            assert len({originals[path] for path in cluster}) == 1
