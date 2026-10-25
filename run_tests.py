"""Run lightweight software verification and write assessment-ready pytest reports.

Run from the repository root:
    python run_tests.py

No dataset downloads, model weights, GPU or YOLO training are needed. The
suite uses disposable synthetic inputs and directly tests functions in src/.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "results" / "software_tests"


def current_commit() -> str:
    """Return local Git commit ID or a readable indication when unavailable."""
    try:
        git = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "Unavailable"
    return git.stdout.strip() if git.returncode == 0 else "Unavailable"


def get_counts(xml_file: Path) -> dict[str, int] | None:
    """Read the counts written by pytest's JUnit XML reporter."""
    if not xml_file.is_file():
        return None
    try:
        node = ElementTree.parse(xml_file).getroot()
        if node.tag == "testsuites":
            suites = node.findall("testsuite")
        elif node.tag == "testsuite":
            suites = [node]
        else:
            return None
        keys = ("tests", "failures", "errors", "skipped")
        counts = {key: sum(int(suite.get(key, "0")) for suite in suites) for key in keys}
        counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        return counts
    except (ElementTree.ParseError, ValueError):
        return None


def main() -> int:
    """Execute the tests and generate text, XML and Markdown evidence."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    xml_report = REPORT_DIR / "pytest_results.xml"
    command = [sys.executable, "-m", "pytest", "tests", "-v", f"--junitxml={xml_report}"]
    print("Executing:", "python -m pytest tests -v --junitxml=results/software_tests/pytest_results.xml")
    try:
        result = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", check=False,
        )
        output, code = result.stdout, result.returncode
    except OSError as exc:
        output, code = f"Cannot launch pytest: {exc}\n", 127

    text_report = REPORT_DIR / "pytest_report.txt"
    text_report.write_text(output, encoding="utf-8")
    print(output)

    counts = get_counts(xml_report)
    status = "PASS" if code == 0 and counts and counts["failures"] == counts["errors"] == 0 else "FAIL / NOT VERIFIED"
    results_summary = (
        f"{counts['tests']} collected; {counts['passed']} passed; "
        f"{counts['failures']} failed; {counts['errors']} errors; {counts['skipped']} skipped"
        if counts else "JUnit results were not generated; see the text log"
    )
    checked_files = (
        "- `src/experiment_config.py` — dataset selection, names, checkpoint lookup\n"
        "- `src/prep_2_classes.py` — remapping, background samples, overwrite prevention\n"
        "- `src/data_check.py` — invalid/missing labels, duplicate detection, read-only audit\n"
        "- `src/compare_models.py` — comparison summary selection and metric extraction\n"
        "- `src/optimise_inference.py` — bounding boxes, class-aware IoU, TP/FP/FN and metrics\n"
        "- `src/test.py` (or `src/final_test.py`) — held-out test dataset parsing / F2 formula\n"
    )
    md = f"""# MMA3001 Software Verification — pytest Report

**Status:** {status}  
**Executed at (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}  
**Git commit:** `{current_commit()}`  
**Python:** `{platform.python_version()}` ({platform.system()})  
**Command:** `python -m pytest tests -v --junitxml=results/software_tests/pytest_results.xml`  
**Outcome:** {results_summary}  
**Exit code:** `{code}`

## Verification scope

The suite exercises **existing source-code functions** using small synthetic inputs and
isolated temporary directories. It does not copy or reimplement the algorithms being tested.

{checked_files}

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
"""
    summary_path = REPORT_DIR / "TEST_REPORT.md"
    summary_path.write_text(md, encoding="utf-8")
    print("Report:", summary_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
