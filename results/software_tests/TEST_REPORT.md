# MMA3001 Software Verification — pytest Report

**Status:** PASS  
**Executed at (UTC):** 2026-10-25 05:12:21  
**Git commit:** `4625bd8`  
**Python:** `3.12.5` (Windows)  
**Command:** `python -m pytest tests -v --junitxml=results/software_tests/pytest_results.xml`  
**Outcome:** 45 collected; 45 passed; 0 failed; 0 errors; 0 skipped  
**Exit code:** `0`

## Verification scope

The suite exercises **existing source-code functions** using small synthetic inputs and
isolated temporary directories. It does not copy or reimplement the algorithms being tested.

- `src/experiment_config.py` — dataset selection, names, checkpoint lookup
- `src/prep_2_classes.py` — remapping, background samples, overwrite prevention
- `src/data_check.py` — invalid/missing labels, duplicate detection, read-only audit
- `src/compare_models.py` — comparison summary selection and metric extraction
- `src/optimise_inference.py` — bounding boxes, class-aware IoU, TP/FP/FN and metrics
- `src/test.py` (or `src/final_test.py`) — held-out test dataset parsing / F2 formula


## Evidence files

- `pytest_report.txt`: individual test cases and failures, if any.
- `pytest_results.xml`: structured JUnit results suitable for inspection or CI.

## Scope and limitations

Passing tests confirm the specified software behaviours for the tested cases;
they **do not** independently establish YOLO model accuracy, the absence of data leakage,
or industrial reliability. The latter questions are assessed separately by the
project's dataset audit, validation experiments and held-out model test.

This report was generated from the source code and tests present in this
working copy. If any code changes, rerun `python run_tests.py`.
