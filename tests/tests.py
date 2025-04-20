import os
import pytest
from mod.engagement_regression import run_regression_analysis
from mod.engagement_classifier import run_engagement_classification



def test_regression_analysis_runs():
    """
    Smoke test: ensures regression pipeline runs on test data without error.
    """
    try:
        run_regression_analysis(data_dir="tests/data", save_dir="tests/output")
    except Exception as e:
        pytest.fail(f"Regression analysis failed with exception: {e}")


def test_classification_analysis_runs():
    """
    Smoke test: ensures classification pipeline runs on test data without error.
    """
    try:
        run_engagement_classification(data_dir="tests/data", save_dir="tests/output")
    except Exception as e:
        pytest.fail(f"Classification analysis failed with exception: {e}")


def test_output_files_exist():
    """
    Check if at least one output file exists in tests/output.
    """
    output_files = os.listdir("tests/output")
    assert len(output_files) > 0, "No output files found in tests/output"


def test_output_formats_are_png():
    """
    Check if there are PNG plots saved from the classifiers/regressors.
    """
    image_files = [f for f in os.listdir("tests/output") if f.endswith(".png")]
    assert len(image_files) > 0, "No .png output found in tests/output"

