"""Check model-comparison helpers using generated JSON instead of GPUs."""

import json
import pytest

import compare_models as comparison


def test_run_name_recognises_reruns():
    base = "yolo11s_2class_v2"
    assert comparison.matches_training_run(base, base)
    assert comparison.matches_training_run(f"{base}_run2", base)
    assert not comparison.matches_training_run(f"{base}_validation", base)
    assert not comparison.matches_training_run("yolo11n_2class_v2", base)


def test_load_summary_validates_json(tmp_path):
    path = tmp_path / "evaluation_summary.json"
    path.write_text('{"precision": 0.5}', encoding="utf-8")
    assert comparison.load_summary(path)["precision"] == 0.5
    path.write_text("{ invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read JSON"):
        comparison.load_summary(path)


def test_metric_value_reads_inference_time_and_rejects_missing():
    data = {"precision": 0.55, "speed_ms_per_image": {"inference": 8.5}}
    assert comparison.metric_value(data, "precision") == 0.55
    assert comparison.metric_value(data, "inference_ms") == 8.5
    with pytest.raises(ValueError, match="missing"):
        comparison.metric_value(data, "recall")


def test_summary_selection_ignores_other_datasets_and_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "RUNS_DIR", tmp_path)

    def save(run_folder, training_run, dataset_number, split, timestamp):
        path = tmp_path / run_folder / "evaluation_summary.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "training_run": training_run,
            "dataset_number": dataset_number,
            "model": "yolo11n",
            "evaluation_split": split,
            "created_at_utc": timestamp,
        }), encoding="utf-8")
        return path

    older = save("val_a", "yolo11n_2class_v2", 4, "val", "2026-10-01T01:00:00+00:00")
    newer = save("val_b", "yolo11n_2class_v2_run2", 4, "val", "2026-10-02T01:00:00+00:00")
    save("test_run", "yolo11n_2class_v2", 4, "test", "2026-10-05T01:00:00+00:00")
    save("other_dataset", "yolo11n_2class_v2", 1, "val", "2026-10-07T01:00:00+00:00")

    path, info = comparison.choose_summary(4, "2class_v2", "n", "val", None)
    assert path == newer
    assert info["training_run"] == "yolo11n_2class_v2_run2"
    path, info = comparison.choose_summary(4, "2class_v2", "n", "val", "yolo11n_2class_v2")
    assert path == older
