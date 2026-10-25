"""Test the real optimisation/final-test maths without loading YOLO weights."""

from collections import Counter
import importlib
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture(scope="module")
def logic_modules():
    """Import prediction helpers with a lightweight YOLO placeholder.

    The functions under test are pure geometry/metric functions. This avoids
    downloading or initialising pretrained models during a unit-test run.
    """
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "ultralytics", fake_ultralytics)
        opt = importlib.import_module("optimise_inference")
        # The repository uses src/test.py for its final held-out evaluation.
        # Some earlier copies used src/final_test.py; support either filename.
        src = Path(__file__).resolve().parents[1] / "src"
        module_path = src / "test.py"
        if not module_path.is_file():
            module_path = src / "final_test.py"
        if not module_path.is_file():
            pytest.fail("Missing src/test.py (or src/final_test.py)")
        spec = importlib.util.spec_from_file_location("mma3001_final_test", module_path)
        final = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(final)
        yield opt, final
    # Avoid retaining the fake YOLO import in other Python tests.
    sys.modules.pop("optimise_inference", None)


def test_identical_and_nonoverlapping_boxes(logic_modules):
    opt, _ = logic_modules
    assert opt.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1)
    assert opt.box_iou((0, 0, 10, 10), (11, 11, 20, 20)) == pytest.approx(0)
    assert opt.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_yolo_normalised_annotations_convert_to_pixel_boxes(logic_modules, tmp_path):
    opt, _ = logic_modules
    label = tmp_path / "example.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n1 0.25 0.25 0.1 0.1\n")
    annotations = opt.read_ground_truth(label, width=200, height=100, n_classes=2)
    assert annotations[0][0] == 0
    assert annotations[0][1] == pytest.approx((60, 40, 140, 60))
    assert annotations[1][0] == 1


def test_empty_annotation_is_a_valid_negative_image(logic_modules, tmp_path):
    opt, _ = logic_modules
    label = tmp_path / "background.txt"
    label.write_text("")
    assert opt.read_ground_truth(label, 100, 100, 2) == []


@pytest.mark.parametrize("bad_text", [
    "2 0.5 0.5 0.2 0.2\n",      # Class index outside 0..1
    "0 0.95 0.5 0.2 0.2\n",     # Box extends outside image
    "0 0.5 0.5 -0.1 0.2\n",     # Negative width
    "0 0.5 0.5\n",             # Wrong field count
    "0 NaN 0.5 0.2 0.2\n",      # Non-finite coordinate
])
def test_bad_yolo_annotations_are_rejected(logic_modules, tmp_path, bad_text):
    opt, _ = logic_modules
    path = tmp_path / "bad.txt"
    path.write_text(bad_text)
    with pytest.raises(ValueError):
        opt.read_ground_truth(path, 100, 100, 2)


def test_missing_annotation_file_fails_loudly(logic_modules, tmp_path):
    opt, _ = logic_modules
    with pytest.raises(FileNotFoundError):
        opt.read_ground_truth(tmp_path / "missing.txt", 100, 100, 2)


def test_correct_detection_counts_as_true_positive(logic_modules):
    opt, _ = logic_modules
    truth = [(0, (0, 0, 10, 10))]
    detections = [(0, 0.9, (0, 0, 10, 10))]
    scores = opt.match_detections(truth, detections, .2, .5, 2)
    assert scores[0] == Counter(tp=1, fp=0, fn=0)
    assert scores[1] == Counter(tp=0, fp=0, fn=0)


def test_duplicate_predictions_do_not_both_match_one_defect(logic_modules):
    opt, _ = logic_modules
    truth = [(0, (0, 0, 10, 10))]
    detections = [(0, .9, (0, 0, 10, 10)), (0, .8, (0, 0, 10, 10))]
    scores = opt.match_detections(truth, detections, .2, .5, 2)
    assert scores[0]["tp"] == 1
    assert scores[0]["fp"] == 1
    assert scores[0]["fn"] == 0


def test_wrong_class_is_false_positive_and_false_negative(logic_modules):
    opt, _ = logic_modules
    scores = opt.match_detections(
        [(0, (0, 0, 10, 10))],
        [(1, .9, (0, 0, 10, 10))],
        .2, .5, 2,
    )
    assert scores[0]["fn"] == 1
    assert scores[1]["fp"] == 1
    assert sum(score["tp"] for score in scores.values()) == 0


def test_confidence_threshold_discards_lower_scored_detections(logic_modules):
    opt, _ = logic_modules
    truth = [(0, (0, 0, 10, 10))]
    detection = [(0, .3, (0, 0, 10, 10))]
    at_025 = opt.match_detections(truth, detection, .25, .5, 2)
    at_040 = opt.match_detections(truth, detection, .40, .5, 2)
    assert at_025[0]["tp"] == 1
    assert at_040[0]["fn"] == 1
    assert at_040[0]["tp"] == 0


def test_background_detection_is_false_alarm(logic_modules):
    opt, _ = logic_modules
    scores = opt.match_detections([], [(1, .8, (0, 0, 10, 10))], .2, .5, 2)
    assert scores[1]["fp"] == 1
    assert scores[0]["fn"] == 0


def test_precision_recall_f1_f2_formulas(logic_modules):
    opt, final = logic_modules
    counts = Counter(tp=3, fp=2, fn=1)
    values = opt.measures(counts)
    assert values["precision"] == pytest.approx(3 / 5)
    assert values["recall"] == pytest.approx(3 / 4)
    assert values["f1"] == pytest.approx(2 / 3)
    assert final.f2_score(3, 2, 1) == pytest.approx(15 / 21)


def test_zero_predictions_do_not_divide_by_zero(logic_modules):
    opt, final = logic_modules
    values = opt.measures(Counter())
    assert values["precision"] == 0
    assert values["recall"] == 0
    assert values["f1"] == 0
    assert final.f2_score(0, 0, 0) == 0


def test_final_test_class_names_and_split_are_validated(logic_modules, tmp_path):
    _, final = logic_modules
    split = tmp_path / "test"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "test: test/images\nnames:\n  0: packaging-error\n  1: unsealed\n",
        encoding="utf-8",
    )
    images, classes = final.locate_test_images(yaml_path)
    assert images == split / "images"
    assert classes == ["packaging-error", "unsealed"]
