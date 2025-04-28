# tests/test_pipeline.py

import os
import pytest
from mod.new_engagement_regression import run_regression_analysis
from mod.new_engagement_classifier import run_engagement_classification

# 自动创建输出文件夹
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
        run_engagement_classification(data_dir="tests/data", save_dir="tests/output", save_model=False)
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
    valid_files = [f for f in os.listdir("tests/output") if f.endswith((".png", ".csv"))]
    assert len(valid_files) > 0, "❌ No .png or .csv output found in tests/output"

def test_pipeline_no_save():
    """
    Run the classification pipeline without saving models.
    Just ensure it runs.
    """
    try:
        run_engagement_classification(save_model=False, data_dir="tests/data", save_dir="tests/output")
    except Exception as e:
        pytest.fail(f"❌ Pipeline (no save) failed with exception: {e}")
