"""Verify numbered dataset selection and trained-run naming without training YOLO."""

from pathlib import Path
import pytest

import experiment_config as cfg


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """Redirect configuration helpers to a disposable mini-project."""
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


@pytest.mark.parametrize("number,folder,slug", [
    (1, "1 - five_class", "5class_aug"),
    (2, "2 - two_class", "2class_v1"),
    (3, "3 - raw_unaug", "5class_unaug"),
    (4, "4 - two_class_v2", "2class_v2"),
])
def test_select_dataset_resolves_number(isolated_project, number, folder, slug):
    yaml_path = isolated_project / "data" / folder / "data.yaml"
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text("names: [example]\n", encoding="utf-8")

    result_number, result_yaml, result_slug, description = cfg.select_dataset(number)
    assert result_number == number
    assert result_yaml == yaml_path
    assert result_slug == slug
    assert description


@pytest.mark.parametrize("invalid", [0, 5, "nine", "", None])
def test_invalid_dataset_number_is_rejected(isolated_project, invalid, monkeypatch):
    if invalid is None:
        monkeypatch.setattr("builtins.input", lambda _: "nine")
    with pytest.raises(ValueError):
        cfg.select_dataset(invalid)


def test_missing_dataset_is_reported(isolated_project):
    with pytest.raises(FileNotFoundError, match="Dataset YAML not found"):
        cfg.select_dataset(4)


@pytest.mark.parametrize("raw,expected", [("n", "n"), ("S", "s"), ("  s  ", "s")])
def test_model_selection(raw, expected):
    assert cfg.select_model(raw) == expected


def test_invalid_model_selection():
    with pytest.raises(ValueError, match="Model must"):
        cfg.select_model("x")


def test_training_run_name():
    assert cfg.training_run_base("s", "2class_v2") == "yolo11s_2class_v2"


def test_repeat_runs_get_unique_names(isolated_project):
    base = "yolo11s_2class_v2"
    assert cfg.unused_run_name(base) == base
    runs = isolated_project / "runs"
    (runs / base).mkdir(parents=True)
    (runs / f"{base}_run2").mkdir()
    assert cfg.unused_run_name(base) == f"{base}_run3"


def test_checkpoint_lookup_selects_latest_numbered_run(isolated_project):
    base = "yolo11n_2class_v2"
    for run_name in (base, f"{base}_run2", f"{base}_run3"):
        checkpoint = isolated_project / "runs" / run_name / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"pretend model checkpoint")

    run_dir, checkpoint = cfg.find_training_checkpoint(base)
    assert run_dir.name == f"{base}_run3"
    assert checkpoint.name == "best.pt"

    run_dir, checkpoint = cfg.find_training_checkpoint(base, requested_run=base)
    assert run_dir.name == base
    assert checkpoint.is_file()


def test_checkpoint_lookup_rejects_wrong_run(isolated_project):
    with pytest.raises(ValueError, match="does not match"):
        cfg.find_training_checkpoint("yolo11n_2class_v2", requested_run="yolo11s_2class_v2")
    with pytest.raises(ValueError, match="not a path"):
        cfg.find_training_checkpoint("yolo11n_2class_v2", requested_run="../other")


def test_checkpoint_lookup_requires_saved_weights(isolated_project):
    with pytest.raises(FileNotFoundError, match="No trained best.pt"):
        cfg.find_training_checkpoint("yolo11n_2class_v2")
