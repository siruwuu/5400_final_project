# src/mod/new_engagement_regression.py

import os
import re
import emoji
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

IMG_DIR = Path(__file__).resolve().parent.parent / "img"
os.makedirs(IMG_DIR, exist_ok=True)

URGENCY_KEYWORDS = ["urgent", "emergency", "last chance", "help", "please"]
PRONOUNS = ["you", "your", "we", "us"]


def run_regression_analysis(data_dir="data", save_dir="src/img"):
    """
    Orchestrates the regression analysis process for both post and comment datasets.
    It loads the data, processes it, runs regression analysis to predict engagement score,
    and saves the results and feature importance plots.

    Args:
        data_dir (str): The directory where the input data (posts and comments) is located.
        save_dir (str): The directory where the output plots and coefficients will be saved.
    """
    logging.info("🔁 Running Regression Analysis...")

    save_dir_absolute = Path(save_dir).resolve()
    os.makedirs(save_dir_absolute, exist_ok=True)

    try:
        cats_posts = pd.read_csv(f"{data_dir}/cats_all_post_processed.csv")
        dogs_posts = pd.read_csv(f"{data_dir}/dogs_all_posts_processed.csv")
        cats_comments = pd.read_csv(f"{data_dir}/cats_all_comments_processed.csv")
        dogs_comments = pd.read_csv(f"{data_dir}/dogs_all_comments_processed.csv")
        logging.info("✅ Successfully loaded posts and comments data.")
    except Exception as e:
        logging.error(f"❌ Failed to load data: {e}")
        raise e

    for df in [cats_posts, dogs_posts]:
        engineer_post_features(df)

    analyze_engagement(cats_posts, label="Cats Posts", save_dir=save_dir_absolute)
    analyze_engagement(dogs_posts, label="Dogs Posts", save_dir=save_dir_absolute)

    cats_comments = preprocess_comments(cats_comments)
    dogs_comments = preprocess_comments(dogs_comments)

    analyze_comment_engagement(
        cats_comments, label="Cats Comments", save_dir=save_dir_absolute
    )
    analyze_comment_engagement(
        dogs_comments, label="Dogs Comments", save_dir=save_dir_absolute
    )


def engineer_post_features(df):
    """
    Engineers additional features for the Reddit posts dataset, such as engagement score,
    number of adjectives, verbs, emojis, urgency-related words, pronouns, and more.

    Args:
        df (pd.DataFrame): The DataFrame containing the Reddit posts.
    """
    df["engagement_score"] = df["score"] + 0.5 * df["num_comments"]
    df["num_adjectives"] = df["adjectives"].apply(
        lambda x: len(eval(x)) if pd.notnull(x) else 0
    )
    df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_emojis"] = df["cleaned_text"].apply(
        lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA)
    )
    df["has_urgency_words"] = df["cleaned_text"].apply(
        lambda x: int(any(word in str(x).lower() for word in URGENCY_KEYWORDS))
    )
    df["has_pronouns"] = df["cleaned_text"].apply(
        lambda x: int(any(p in str(x).lower() for p in PRONOUNS))
    )
    df["title_length"] = df["title"].apply(lambda x: len(str(x)))
    df["contains_money"] = df["cleaned_text"].apply(contains_money)
    df["num_lines"] = df["selftext"].apply(lambda x: str(x).count("\n"))


def preprocess_comments(df):
    """
    Preprocesses the comments data by extracting features such as the number of adjectives,
    verbs, emojis, urgency-related words, pronouns, etc.

    Args:
        df (pd.DataFrame): The DataFrame containing the Reddit comments.

    Returns:
        pd.DataFrame: The processed DataFrame with new features.
    """
    df["num_adjectives"] = df["adjectives"].apply(
        lambda x: len(eval(x)) if pd.notnull(x) else 0
    )
    df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_emojis"] = df["cleaned_text"].apply(
        lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA)
    )
    df["has_urgency_words"] = df["cleaned_text"].apply(
        lambda x: int(any(word in str(x).lower() for word in URGENCY_KEYWORDS))
    )
    df["has_pronouns"] = df["cleaned_text"].apply(
        lambda x: int(any(p in str(x).lower() for p in PRONOUNS))
    )
    return df


def contains_money(text):
    """
    Checks if the text contains any money-related keywords or symbols.

    Args:
        text (str): The text to check for money-related keywords.

    Returns:
        int: 1 if the text contains money-related keywords, 0 otherwise.
    """
    money_keywords = ["donation", "donate", "pledge", "$", "fund", "raise"]
    text = str(text).lower()
    return int(
        any(kw in text for kw in money_keywords) or bool(re.search(r"\$\d+", text))
    )


def analyze_engagement(df, label="Posts", save_dir=Path("src/img")):
    """
    Analyzes the engagement for posts by running a linear regression model to predict
    the engagement score. It saves the feature coefficients and feature importance plots.

    Args:
        df (pd.DataFrame): The DataFrame containing the post data with features and engagement scores.
        label (str): The label for the dataset (e.g., "Cats Posts" or "Dogs Posts").
        save_dir (Path): The directory where the results and plots will be saved.
    """
    feature_columns = [
        "sentiment_score",
        "num_adjectives",
        "num_verbs",
        "num_exclamations",
        "has_question",
        "num_emojis",
        "contains_adopt_keywords",
        "has_urgency_words",
        "has_pronouns",
        "num_words",
        "title_length",
        "contains_money",
        "num_lines",
    ]

    df_model = df.dropna(subset=["engagement_score"] + feature_columns).copy()
    X = df_model[feature_columns]
    y = df_model["engagement_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(
        f"📈 Linear Regression Performance for {label}: RMSE={rmse:.2f}, R²={r2:.3f}"
    )

    coef_df = pd.DataFrame(
        {"Feature": feature_columns, "Coefficient": model.coef_}
    ).sort_values(by="Coefficient", ascending=False)

    coef_csv_path = (
        save_dir / f"{label.lower().replace(' ', '_')}_feature_coefficients.csv"
    )
    coef_df.to_csv(coef_csv_path, index=False)
    logging.info(f"✅ Saved feature coefficients to {coef_csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.xlabel("Coefficient")
    plt.title(f"{label}: Feature Importance (Linear Regression)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = save_dir / f"{label.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"✅ Saved feature importance plot to {plot_path}")


def analyze_comment_engagement(df, label="Comments", save_dir=Path("src/img")):
    """
    Analyzes the engagement for comments by running a linear regression model to predict
    the score (engagement) of the comments. It saves the feature coefficients and feature importance plots.

    Args:
        df (pd.DataFrame): The DataFrame containing the comment data with features and scores.
        label (str): The label for the dataset (e.g., "Cats Comments" or "Dogs Comments").
        save_dir (Path): The directory where the results and plots will be saved.
    """
    feature_columns = [
        "sentiment_score",
        "num_adjectives",
        "num_verbs",
        "num_exclamations",
        "has_question",
        "num_emojis",
        "contains_adopt_keywords",
        "has_urgency_words",
        "has_pronouns",
        "num_words",
    ]

    df_model = df.dropna(subset=["score"] + feature_columns).copy()
    X = df_model[feature_columns]
    y = df_model["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logging.info(
        f"📈 Linear Regression Performance for {label}: RMSE={rmse:.2f}, R²={r2:.3f}"
    )

    coef_df = pd.DataFrame(
        {"Feature": feature_columns, "Coefficient": model.coef_}
    ).sort_values(by="Coefficient", ascending=False)

    coef_csv_path = (
        save_dir / f"{label.lower().replace(' ', '_')}_feature_coefficients.csv"
    )
    coef_df.to_csv(coef_csv_path, index=False)
    logging.info(f"✅ Saved feature coefficients to {coef_csv_path}")

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.xlabel("Coefficient")
    plt.title(f"{label}: Feature Importance (Linear Regression)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = save_dir / f"{label.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"✅ Saved feature importance plot to {plot_path}")
