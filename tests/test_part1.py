# tests/test_pipeline.py
"""
This module contains unit tests for the overall data processing pipeline, including regression and classification analysis.
The tests ensure that both the regression and classification pipelines run without errors, output the expected files,
and function as intended.

### Functions:
- `ensure_output_folder`: 
    A fixture that automatically creates the output folder before any tests are run, ensuring the output directory exists for saving files.

- `test_regression_analysis_runs`: 
    A smoke test to verify that the regression analysis pipeline runs without errors when applied to test data.

- `test_classification_analysis_runs`: 
    A smoke test to verify that the classification analysis pipeline runs without errors on the test data.

- `test_output_files_exist`: 
    Ensures that at least one output file is generated in the output folder after running the pipelines.

- `test_output_formats_are_png_or_csv`: 
    Verifies that the output files are in PNG or CSV format.

- `test_pipeline_no_save`: 
    Verifies that the classification pipeline runs correctly without saving the model, ensuring the core pipeline works without outputting saved models.
"""

import os
import pytest
from mod.new_engagement_regression import run_regression_analysis
from mod.new_engagement_classifier import run_engagement_classification

@pytest.fixture(scope="session", autouse=True)
def ensure_output_folder():
    os.makedirs("tests/output", exist_ok=True)


def test_regression_analysis_runs():
    """
    Smoke test: ensures regression pipeline runs on test data without error.
    """
    try:
        run_regression_analysis(data_dir="tests/data", save_dir="tests/output")
    except Exception as e:
        pytest.fail(f"❌ Regression analysis failed with exception: {e}")


def test_classification_analysis_runs():
    """
    Smoke test: ensures classification pipeline runs on test data without error.
    """
    try:
        run_engagement_classification(
            data_dir="tests/data", save_dir="tests/output", save_model=False
        )
    except Exception as e:
        pytest.fail(f"❌ Classification analysis failed with exception: {e}")


def test_output_files_exist():
    """
    Check if at least one output file exists in tests/output after pipeline runs.
    """
    output_files = os.listdir("tests/output")
    assert len(output_files) > 0, "❌ No output files found in tests/output"


def test_output_formats_are_png_or_csv():
    """
    Check if there are PNG plots or CSV outputs saved.
    """
    valid_files = [
        f for f in os.listdir("tests/output") if f.endswith((".png", ".csv"))
    ]
    assert len(valid_files) > 0, "❌ No .png or .csv output found in tests/output"


def test_pipeline_no_save():
    """
    Run the classification pipeline without saving models.
    Just ensure it runs.
    """
    try:
        run_engagement_classification(
            save_model=False, data_dir="tests/data", save_dir="tests/output"
        )
    except Exception as e:
        pytest.fail(f"❌ Pipeline (no save) failed with exception: {e}")
