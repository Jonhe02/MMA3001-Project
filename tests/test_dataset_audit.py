"""Use a tiny artificial dataset to verify read-only auditing and CSV outputs."""

import csv
from pathlib import Path
import sys

import pytest

import cv2
import numpy as np

import data_check as audit
import experiment_config as cfg


def make_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A real, decodable image (unlike the dummy bytes used in converter tests).
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_audit_reports_valid_empty_missing_invalid_and_duplicates(tmp_path, monkeypatch):
    root = tmp_path
    data_root = root / "data" / "4 - two_class_v2"
    for split in ("train", "valid", "test"):
        (data_root / split / "images").mkdir(parents=True)
        (data_root / split / "labels").mkdir()
    (data_root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 2\nnames: {0: packaging-error, 1: unsealed}\n",
        encoding="utf-8",
    )
    make_image(data_root / "train" / "images" / "good.png", 120)
    (data_root / "train" / "labels" / "good.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    make_image(data_root / "train" / "images" / "background.png", 50)
    (data_root / "train" / "labels" / "background.txt").write_text("")

    make_image(data_root / "train" / "images" / "bad_box.png", 100)
    (data_root / "train" / "labels" / "bad_box.txt").write_text("1 0.95 0.5 0.2 0.2\n")

    make_image(data_root / "valid" / "images" / "missing.png", 70)
    # Deliberately no corresponding valid/labels/missing.txt.

    make_image(data_root / "test" / "images" / "copy.png", 120)
    (data_root / "test" / "labels" / "copy.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    monkeypatch.setattr(cfg, "PROJECT_ROOT", root)
    monkeypatch.setattr(audit, "PROJECT_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["data_check.py", "--dataset", "4"])

    audit.main()

    results = root / "results" / "2class_v2_audit"
    summary = csv_rows(results / "summary.csv")
    class_counts = csv_rows(results / "class_counts.csv")
    issues = csv_rows(results / "issues.csv")
    duplicate_candidates = csv_rows(results / "duplicate_candidates.csv")

    assert len(summary) == 3
    train = next(row for row in summary if row["split"] == "train")
    assert int(train["images"]) == 3
    assert int(train["empty_label_files"]) == 1
    assert int(train["valid_annotations"]) == 1
    assert any(row["issue"] == "missing_label_file" for row in issues)
    assert any(row["issue"] == "invalid_bounding_box" for row in issues)
    assert any(row["class_name"] == "packaging-error" for row in class_counts)
    assert any(row["type"] == "exact" and row["split_a"] != row["split_b"]
               for row in duplicate_candidates)

    # Audit must never delete, rewrite or move original labels.
    assert (data_root / "train" / "labels" / "good.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert (data_root / "train" / "labels" / "background.txt").read_text() == ""


def test_audit_reports_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["data_check.py", "--dataset", "4"])
    with pytest.raises(FileNotFoundError):
        audit.main()
