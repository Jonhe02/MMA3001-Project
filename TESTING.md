# Software tests and test report (MMA3001)

These tests **import the functions in your actual `src/` scripts**. They do not copy
or retrain the YOLO models, require data downloads, or change project files other
than saving reports to `results/software_tests/`.

## Install

Copy the supplied `tests/` folder, `pytest.ini` and `run_tests.py` into the
root of `MMA3001-Project` (alongside `src/`). **Do not replace your existing
`src/` files.**

`pytest` is listed in the project's requirements, otherwise install it:

```powershell
python -m pip install pytest
```

## Run and save your actual GitHub test report

From your VS Code terminal in the repository root:

```powershell
python run_tests.py
```

The runner executes `pytest` and saves:

```
results/
  software_tests/
    TEST_REPORT.md      # readable software test report with local Git commit
    pytest_report.txt   # full text output with individual test results
    pytest_results.xml  # JUnit machine-readable report
```

**Commit these three reports from your OWN repository test run**. Do not use an
older example report as proof that your current working copy passed.

You may also execute the tests without generating a report:

```powershell
python -m pytest tests -v
```

## What is covered

- Correct dataset paths, input rejection, repeat-run naming and checkpoint selection
- Correct two-class ID remapping, negative images, and no accidental overwrite
- Missing/empty label detection, bounding-box validation and duplicate checks
- Metric extraction and validation-summary selection for model comparison
- Class-aware box matching, IoU, false positives, false negatives and F1/F2 maths

The tests operate on synthetic data in temporary folders. They do **not** claim
that the trained model achieves a particular accuracy or that real-world
packaging safety has been established.

## If a test fails

The runner records failures in `pytest_report.txt`. First check that all your
current `src/` scripts are present, then read the failed assertion and traceback.
Some test assumptions reflect the scripts shared during development, so later
API changes may require updating a test; do **not** change production code
just to silence a failing test without understanding why it failed.
