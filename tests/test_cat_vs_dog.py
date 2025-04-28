# tests/test_cat_vs_dog.py

import sys
import os
import pathlib
import pandas as pd
import pytest

# 加入 src 到 sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from mod.cat_vs_dog import load_and_prepare_data, run_style_model

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def mini_sample_data(tmp_path):
    # 制作10条小样本数据
    cat_data = pd.DataFrame({
        "pet_type": ["cat"] * 5,
        "adjectives": ["['Fluffy']", "['Cute']", "['Lazy']", "['Playful']", "['Quiet']"],
        "verbs": ["['purring']", "['sleeping']", "['lounging']", "['snuggling']", "['napping']"]
    })
    dog_data = pd.DataFrame({
        "pet_type": ["dog"] * 5,
        "adjectives": ["['Loyal']", "['Brave']", "['Energetic']", "['Friendly']", "['Strong']"],
        "verbs": ["['barking']", "['running']", "['jumping']", "['guarding']", "['fetching']"]
    })

    cat_csv = tmp_path / "cats.csv"
    dog_csv = tmp_path / "dogs.csv"
    cat_data.to_csv(cat_csv, index=False)
    dog_data.to_csv(dog_csv, index=False)

    return cat_csv, dog_csv

def test_load_and_prepare_data(mini_sample_data):
    cat_path, dog_path = mini_sample_data
    df = load_and_prepare_data(cat_path, dog_path)

    assert df.shape[0] == 10, "Should have 10 samples total."
    assert set(df["label"]) == {0, 1}
    assert all(isinstance(val, list) for val in df["adjectives"])
    assert all(isinstance(val, list) for val in df["verbs"])
    assert isinstance(df["style_text"].iloc[0], str)
    assert isinstance(df["verb_text"].iloc[0], str)

def test_run_style_model_with_sample_data():
    # 构造10条小DataFrame
    df = pd.DataFrame({
        "pet_type": ["cat"] * 5 + ["dog"] * 5,
        "adjectives": [
            ["Fluffy"], ["Cute"], ["Lazy"], ["Playful"], ["Quiet"],
            ["Loyal"], ["Brave"], ["Energetic"], ["Friendly"], ["Strong"]
        ],
        "verbs": [
            ["purring"], ["sleeping"], ["lounging"], ["snuggling"], ["napping"],
            ["barking"], ["running"], ["jumping"], ["guarding"], ["fetching"]
        ]
    })
    df["label"] = df["pet_type"].map({"cat": 0, "dog": 1})
    df["style_text"] = df["adjectives"].apply(lambda adj: " ".join(adj))
    df["verb_text"] = df["verbs"].apply(lambda verb: " ".join(verb))

    # 临时覆盖 IMG_DIR
    import mod.cat_vs_dog as cat_vs_dog_module
    cat_vs_dog_module.IMG_DIR = OUTPUT_DIR

    run_style_model(df, text_col="style_text", plot_title="Top Adjectives Test", diverging_title="Diverging Adjective Usage Test")

    assert (OUTPUT_DIR / "style_text_diverging_plot.png").exists()
    assert (OUTPUT_DIR / "top20_word_scores_style_text.csv").exists()
