# src/mod/cat_vs_dog.py
"""
This module performs a "Cat vs Dog" analysis on Reddit posts by comparing
the linguistic styles (adjectives and verbs) used in cat-related vs. dog-related
posts. The analysis includes:
1. Data preprocessing and feature extraction.
2. Training a Random Forest model to identify key linguistic features.
3. Plotting diverging word usage based on TF-IDF scores.

### Functions:
- `load_and_prepare_data(cat_path, dog_path)`: 
    Loads and prepares the datasets for cat and dog posts, including feature extraction and text processing.

- `plot_diverging_words(df, tfidf, feature_names, top_indices, column_name, title)`:
    Plots a diverging bar chart of the most important words that differentiate cat and dog posts, based on TF-IDF scores.

- `run_style_model(df, text_col, plot_title, diverging_title)`:
    Trains a Random Forest model to identify important words (adjectives or verbs) for distinguishing between cat and dog posts.
    Also generates a diverging plot for the top features and saves it.

- `run_cat_vs_dog()`: 
    Main function that orchestrates the analysis, loading datasets, running the style models, and generating the required outputs.
"""

import os
import ast
import logging
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Constants
WORDS_TO_EXCLUDE = {"kitten", "puppy"}
IMG_DIR = pathlib.Path(__file__).resolve().parent.parent / "img"
os.makedirs(IMG_DIR, exist_ok=True)


def load_and_prepare_data(cat_path, dog_path):
    """
    Loads and prepares the datasets for cat and dog posts by combining them into a single dataframe,
    performing feature extraction (e.g., extracting adjectives, verbs, and style text),
    and creating a target label column.

    Args:
        cat_path (str): Path to the cat posts CSV file.
        dog_path (str): Path to the dog posts CSV file.

    Returns:
        pd.DataFrame: A dataframe containing combined data for both cat and dog posts.
    """
    logging.info("🔵 Loading and preparing cat and dog datasets...")

    try:
        cat_df = pd.read_csv(cat_path)
        dog_df = pd.read_csv(dog_path)
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        raise e

    df = pd.concat([cat_df, dog_df], ignore_index=True)
    df["label"] = df["pet_type"].map({"cat": 0, "dog": 1})

    df["adjectives"] = df["adjectives"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df["adjectives"] = df["adjectives"].apply(
        lambda words: [w for w in words if w.lower() not in WORDS_TO_EXCLUDE]
    )
    df["style_text"] = df["adjectives"].apply(lambda adj_list: " ".join(adj_list))

    df["verbs"] = df["verbs"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df["verb_text"] = df["verbs"].apply(lambda verb_list: " ".join(verb_list))

    logging.info(f"✅ Dataset prepared with {len(df)} total samples.")
    return df


def plot_diverging_words(df, tfidf, feature_names, top_indices, column_name, title):
    """
    Plots a diverging bar chart of the most important words that differentiate cat and dog posts
    based on TF-IDF scores. The plot visualizes the difference in average TF-IDF scores for the top features.

    Args:
        df (pd.DataFrame): The prepared dataset.
        tfidf (TfidfVectorizer): The fitted TF-IDF vectorizer.
        feature_names (list): List of feature names corresponding to the words.
        top_indices (list): Indices of the top features to be plotted.
        column_name (str): The text column to analyze (e.g., "style_text" or "verb_text").
        title (str): The title for the plot.
    """
    logging.info(f"🖼️ Plotting Diverging Word Usage: {title}")

    X = tfidf.transform(df[column_name])
    df["tfidf_vector"] = list(X.toarray())

    diffs, words = [], []
    for i in top_indices:
        word = feature_names[i]
        cat_avg = df[df["label"] == 0]["tfidf_vector"].apply(lambda row: row[i]).mean()
        dog_avg = df[df["label"] == 1]["tfidf_vector"].apply(lambda row: row[i]).mean()
        words.append(word)
        diffs.append(dog_avg - cat_avg)

    # Plot
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(diffs)
    sorted_words = np.array(words)[sorted_indices]
    sorted_diffs = np.array(diffs)[sorted_indices]
    colors = ["#3498db" if val > 0 else "#e74c3c" for val in sorted_diffs]

    plt.barh(sorted_words, sorted_diffs, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Difference in Average TF-IDF (Dog - Cat)")
    plt.tight_layout()

    img_save_path = IMG_DIR / f"{column_name}_diverging_plot.png"
    plt.savefig(img_save_path, dpi=300)
    plt.close()
    logging.info(f"✅ Saved diverging plot to {img_save_path}")


def run_style_model(df, text_col, plot_title, diverging_title):
    """
    Trains a Random Forest model to identify the top linguistic features (adjectives or verbs)
    that differentiate cat and dog posts based on TF-IDF scores, and generates a diverging plot.

    Args:
        df (pd.DataFrame): The dataset containing cat and dog posts.
        text_col (str): The text column to analyze (e.g., "style_text" or "verb_text").
        plot_title (str): The title for the plot (e.g., "Top Adjectives").
        diverging_title (str): The title for the diverging plot (e.g., "Diverging Adjective Usage (Dog - Cat)").
    """
    logging.info(f"🧠 Training started using column: {text_col}")

    X = df[text_col]
    y = df["label"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    X_tfidf = tfidf.fit_transform(X)
    feature_names = tfidf.get_feature_names_out()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importances_list = []

    for train_idx, test_idx in skf.split(X_tfidf, y):
        X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        importances_list.append(clf.feature_importances_)

    mean_importance = np.mean(importances_list, axis=0)
    top_indices = np.argsort(mean_importance)[-20:]

    X = tfidf.transform(df[text_col])
    df["tfidf_vector"] = list(X.toarray())

    word_score_list = []
    for i in top_indices:
        word = feature_names[i]
        cat_avg = df[df["label"] == 0]["tfidf_vector"].apply(lambda row: row[i]).mean()
        dog_avg = df[df["label"] == 1]["tfidf_vector"].apply(lambda row: row[i]).mean()
        word_score_list.append(
            {
                "word": word,
                "dog_score": round(dog_avg, 4),
                "cat_score": round(cat_avg, 4),
                "diff": round(dog_avg - cat_avg, 4),
            }
        )

    word_scores_df = pd.DataFrame(word_score_list)
    word_scores_df = word_scores_df.sort_values(by="diff", ascending=False)

    csv_save_path = IMG_DIR / f"top20_word_scores_{text_col}.csv"
    word_scores_df.to_csv(csv_save_path, index=False)
    logging.info(f"✅ Saved top word scores to {csv_save_path}")

    plot_diverging_words(
        df, tfidf, feature_names, top_indices, text_col, diverging_title
    )


def run_cat_vs_dog():
    """
    Main function that orchestrates the entire Cat vs Dog analysis. It loads the data, runs the style model,
    and generates the diverging plots for adjectives and verbs.

    This function initiates the entire analysis workflow and saves the generated plots and CSV files.
    """
    logging.info("🚀 Starting Cat vs Dog Diverging Style Analysis...")

    current_path = pathlib.Path(__file__).resolve()
    project_root = current_path.parent.parent
    data_dir = project_root.parent / "data"

    cat_path = data_dir / "cats_all_post_processed.csv"
    dog_path = data_dir / "dogs_all_posts_processed.csv"

    df = load_and_prepare_data(cat_path, dog_path)

    run_style_model(
        df, "style_text", "Top Adjectives", "Diverging Adjective Usage (Dog - Cat)"
    )
    run_style_model(df, "verb_text", "Top Verbs", "Diverging Verb Usage (Dog - Cat)")
