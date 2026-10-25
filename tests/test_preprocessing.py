"""Test actual two-class converter with disposable image/label examples."""

import pytest

import prep_2_classes as prep


def make_split(root, split, samples):
    """Create dummy image bytes and YOLO labels: conversion only copies images."""
    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for stem, text in samples.items():
        (images / f"{stem}.jpg").write_bytes(f"image {stem}".encode())
        (labels / f"{stem}.txt").write_text(text, encoding="utf-8")


@pytest.fixture
def converter_paths(tmp_path, monkeypatch):
    source = tmp_path / "source"
    dest = tmp_path / "destination"
    make_split(source, "train", {
        "packaging": "1 0.5 0.5 0.3 0.3\n",
        "unsealed": "3 0.5 0.5 0.2 0.2\n",
        "mixed": "1 0.5 0.5 0.3 0.3\n0 0.5 0.5 0.1 0.1\n",
        "background": "",
    })
    make_split(source, "valid", {"valid_unsealed": "3 0.3 0.4 0.1 0.2\n"})
    make_split(source, "test", {"test_background": ""})
    monkeypatch.setattr(prep, "SOURCE", source)
    monkeypatch.setattr(prep, "DEST", dest)
    return source, dest


def test_converter_remaps_classes_and_preserves_background(converter_paths):
    source, dest = converter_paths
    prep.main()
    assert (dest / "train" / "labels" / "packaging.txt").read_text().startswith("0 ")
    assert (dest / "train" / "labels" / "unsealed.txt").read_text().startswith("1 ")
    assert (dest / "train" / "labels" / "background.txt").read_text() == ""
    assert (dest / "valid" / "labels" / "valid_unsealed.txt").read_text().startswith("1 ")
    assert (dest / "test" / "labels" / "test_background.txt").read_text() == ""

    # An image containing a class outside the scope must be excluded entirely.
    assert not (dest / "train" / "images" / "mixed.jpg").exists()
    assert not (dest / "train" / "labels" / "mixed.txt").exists()

    # Original labels must remain untouched.
    assert (source / "train" / "labels" / "packaging.txt").read_text().startswith("1 ")
    yaml_text = (dest / "data.yaml").read_text()
    assert "nc: 2" in yaml_text
    assert "packaging-error" in yaml_text
    assert "unsealed" in yaml_text


def test_converter_does_not_overwrite_existing_dataset(converter_paths):
    _, dest = converter_paths
    dest.mkdir()
    sentinel = dest / "keep_me.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError, match="already exists"):
        prep.main()
    assert sentinel.read_text() == "unchanged"


def test_converter_requires_corresponding_label(converter_paths):
    source, _ = converter_paths
    (source / "train" / "labels" / "packaging.txt").unlink()
    with pytest.raises(FileNotFoundError):
        prep.main()
