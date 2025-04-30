# tests/test_cat_vs_dog.py
"""
This test module contains unit tests for the `cat_vs_dog` module, specifically testing
the `load_and_prepare_data` and `run_style_model` functions.

The tests ensure the proper functionality of key features such as data loading, preprocessing, 
feature extraction, and model execution. The module uses the `pytest` testing framework.

### Functions:
- `mini_sample_data`: 
    A fixture that generates a small sample dataset for testing purposes, with 5 cat and 5 dog posts.
    
- `test_load_and_prepare_data`: 
    Tests the `load_and_prepare_data` function to ensure data is loaded correctly and contains 
    the expected features and data types.

- `test_run_style_model_with_sample_data`: 
    Tests the `run_style_model` function with a small dataset, ensuring that the output files (plot and CSV)
    are generated correctly.
"""

import sys
import pathlib
import pandas as pd
import pytest

# add src to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mod.cat_vs_dog import load_and_prepare_data, run_style_model

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def mini_sample_data(tmp_path):
    """
    Fixture to generate a small sample dataset of 10 posts (5 cats and 5 dogs) for testing purposes.
    
    The fixture creates two CSV files, one for cats and one for dogs, with the necessary columns like
    `pet_type`, `adjectives`, and `verbs`. These files are returned as paths to be used in the tests.
    
    Args:
        tmp_path (Path): A pytest-provided temporary directory to store the generated CSV files.
        
    Returns:
        tuple: Paths to the generated cat and dog CSV files.
    """
    # make 10 samples
    cat_data = pd.DataFrame(
        {
            "pet_type": ["cat"] * 5,
            "adjectives": [
                "['Fluffy']",
                "['Cute']",
                "['Lazy']",
                "['Playful']",
                "['Quiet']",
            ],
            "verbs": [
                "['purring']",
                "['sleeping']",
                "['lounging']",
                "['snuggling']",
                "['napping']",
            ],
        }
    )
    dog_data = pd.DataFrame(
        {
            "pet_type": ["dog"] * 5,
            "adjectives": [
                "['Loyal']",
                "['Brave']",
                "['Energetic']",
                "['Friendly']",
                "['Strong']",
            ],
            "verbs": [
                "['barking']",
                "['running']",
                "['jumping']",
                "['guarding']",
                "['fetching']",
            ],
        }
    )

    cat_csv = tmp_path / "cats.csv"
    dog_csv = tmp_path / "dogs.csv"
    cat_data.to_csv(cat_csv, index=False)
    dog_data.to_csv(dog_csv, index=False)

    return cat_csv, dog_csv


def test_load_and_prepare_data(mini_sample_data):
    """
    Tests the `load_and_prepare_data` function to ensure the dataset is loaded and prepared correctly.
    
    This test checks the following:
    - The correct number of rows (10 samples total).
    - The presence of expected labels (0 for cat and 1 for dog).
    - The correct data types for adjectives and verbs (should be lists).
    - The generation of `style_text` and `verb_text` columns as strings.

    Args:
        mini_sample_data (tuple): A tuple containing the paths to the generated cat and dog CSV files.
    """
    cat_path, dog_path = mini_sample_data
    df = load_and_prepare_data(cat_path, dog_path)

    assert df.shape[0] == 10, "Should have 10 samples total."
    assert set(df["label"]) == {0, 1}
    assert all(isinstance(val, list) for val in df["adjectives"])
    assert all(isinstance(val, list) for val in df["verbs"])
    assert isinstance(df["style_text"].iloc[0], str)
    assert isinstance(df["verb_text"].iloc[0], str)


def test_run_style_model_with_sample_data():

    """
    Tests the `run_style_model` function to ensure it correctly generates output files when applied
    to a small dataset of 10 posts (5 cats and 5 dogs). This test validates the generation of the
    diverging plot and the CSV file containing top word scores.

    Args:
        None (uses a hardcoded small dataset in the test function).
    """
    df = pd.DataFrame(
        {
            "pet_type": ["cat"] * 5 + ["dog"] * 5,
            "adjectives": [
                ["Fluffy"],
                ["Cute"],
                ["Lazy"],
                ["Playful"],
                ["Quiet"],
                ["Loyal"],
                ["Brave"],
                ["Energetic"],
                ["Friendly"],
                ["Strong"],
            ],
            "verbs": [
                ["purring"],
                ["sleeping"],
                ["lounging"],
                ["snuggling"],
                ["napping"],
                ["barking"],
                ["running"],
                ["jumping"],
                ["guarding"],
                ["fetching"],
            ],
        }
    )
    df["label"] = df["pet_type"].map({"cat": 0, "dog": 1})
    df["style_text"] = df["adjectives"].apply(lambda adj: " ".join(adj))
    df["verb_text"] = df["verbs"].apply(lambda verb: " ".join(verb))

    # Temporarily override IMG_DIR to use a custom output directory
    import mod.cat_vs_dog as cat_vs_dog_module

    cat_vs_dog_module.IMG_DIR = OUTPUT_DIR

    run_style_model(
        df,
        text_col="style_text",
        plot_title="Top Adjectives Test",
        diverging_title="Diverging Adjective Usage Test",
    )
    # Assert that the generated files exist
    assert (OUTPUT_DIR / "style_text_diverging_plot.png").exists()
    assert (OUTPUT_DIR / "top20_word_scores_style_text.csv").exists()
