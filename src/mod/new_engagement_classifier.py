"""
This module performs the engagement classification task on Reddit posts related to cat and dog adoption.
It classifies posts into high and low engagement categories using two machine learning models: 
Logistic Regression and Random Forest.

### Functions:
- `run_engagement_classification(data_dir="data", save_dir="src/img", model_dir="src/gpt_classifier_suggester/model", save_model=True)`:
    Orchestrates the process of data loading, preprocessing, model training, evaluation, and saving the model and plots.
    - Loads cat and dog posts.
    - Preprocesses the data, creating structured features and labels based on engagement scores.
    - Trains and evaluates Logistic Regression and Random Forest models for engagement classification.
    - Saves the models and plots, including ROC curves and feature importance.

- `preprocess_posts(df, pet_type=None)`:
    Preprocesses the Reddit posts by extracting features like engagement score, number of adjectives, verbs, emojis, and more.
    - Adds several new columns for features that are useful for classification.
    - Computes engagement score and labels high/low engagement based on quantiles.

- `label_high_engagement(df)`:
    Labels the posts as high or low engagement based on the 75th and 50th quantiles of the engagement score.

- `run_classifier(df, label="Posts", model="logistic", save_prefix="cats", save_model=True, model_dir_absolute=None)`:
    Runs the selected classifier (Logistic Regression or Random Forest) on the dataset, evaluates performance, and saves the model and results.
    - Performs train-test split.
    - Trains the model using TF-IDF vectorization for text features and additional structured features.
    - Evaluates the model on the test set, displaying classification reports, confusion matrix, and ROC-AUC score.
    - Saves ROC curve and feature importance plots.

### Data Preprocessing:
- Adds features such as sentiment score, number of adjectives, number of verbs, emojis, urgency words, pronouns, title length, etc.
- Labels posts as high engagement if their engagement score is in the top 25% and low engagement if in the bottom 50%.

### Model Training:
- Logistic Regression and Random Forest classifiers are used to predict high vs. low engagement.
- The models are trained using a combination of text features (processed using TF-IDF) and structured features (such as number of emojis, adjectives, etc.).

### Model Evaluation:
- The models are evaluated using:
    - Classification report (precision, recall, F1-score)
    - Confusion matrix
    - ROC-AUC score and ROC curve
    - Feature importance for Random Forest

### Model Saving:
- The trained models are saved as pickled files for later use.
- Plots (ROC curves and feature importance) are saved to the specified directory.

"""

import os
import re
import emoji
import pickle
import pathlib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def run_engagement_classification(
    data_dir="data",
    save_dir="src/img",
    model_dir="src/gpt_classifier_suggester/model",
    save_model=True,
):
    """
    Orchestrates the entire engagement classification process by loading the cat and dog posts datasets,
    preprocessing the data, training Logistic Regression and Random Forest classifiers, evaluating model
    performance, and saving the models and evaluation plots.

    Args:
        data_dir (str): The directory where the input data (cat and dog posts) is stored.
        save_dir (str): The directory where the output plots (e.g., ROC curve) will be saved.
        model_dir (str): The directory to save the trained model pipelines.
        save_model (bool): Whether to save the trained models to disk.

    """
    logging.info("🚀 Starting run_engagement_classification...")

    current_path = pathlib.Path(__file__).resolve()
    project_root = current_path.parents[2]

    model_dir_absolute = (project_root / model_dir).resolve()
    save_dir_absolute = (project_root / save_dir).resolve()

    os.makedirs(model_dir_absolute, exist_ok=True)
    os.makedirs(save_dir_absolute, exist_ok=True)

    try:
        cats_posts = pd.read_csv(f"{data_dir}/cats_all_post_processed.csv")
        dogs_posts = pd.read_csv(f"{data_dir}/dogs_all_posts_processed.csv")
        logging.info("✅ Successfully loaded cat and dog posts.")
    except Exception as e:
        logging.error(f"❌ Failed to load posts: {e}")
        raise e

    def preprocess_posts(df, pet_type=None):
        """
        Preprocesses Reddit posts by extracting structured features (e.g., engagement score, number of adjectives, verbs, emojis)
        and text-based features (e.g., sentiment score, presence of urgency keywords, etc.).

        Args:
            df (pd.DataFrame): The DataFrame containing the posts to preprocess.
            pet_type (str): The type of pet ("cat" or "dog").

        Returns:
            pd.DataFrame: The processed DataFrame with new features.
        """
        logging.info(f"🔄 Preprocessing posts for {pet_type}...")
        df["engagement_score"] = df["score"] + 0.5 * df["num_comments"]
        df["num_adjectives"] = df["adjectives"].apply(
            lambda x: len(eval(x)) if pd.notnull(x) else 0
        )
        df["num_verbs"] = df["verbs"].apply(
            lambda x: len(eval(x)) if pd.notnull(x) else 0
        )
        df["num_emojis"] = df["cleaned_text"].apply(
            lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA)
        )

        urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
        pronouns = ["you", "your", "we", "us"]

        df["has_urgency_words"] = df["cleaned_text"].apply(
            lambda x: int(any(word in str(x).lower() for word in urgency_keywords))
        )
        df["has_pronouns"] = df["cleaned_text"].apply(
            lambda x: int(any(p in str(x).lower() for p in pronouns))
        )
        df["title_length"] = df["title"].apply(lambda x: len(str(x)))

        def contains_money(text):
            """
            Checks if the text contains any keywords related to money or donations.

            Args:
                text (str): The text to check for money-related keywords.

            Returns:
                int: 1 if the text contains money-related keywords, 0 otherwise.
            """
            text = str(text).lower()
            return int(
                any(
                    kw in text
                    for kw in ["donation", "donate", "pledge", "$", "fund", "raise"]
                )
                or bool(re.search(r"\$\d+", text))
            )

        df["contains_money"] = df["cleaned_text"].apply(contains_money)
        df["num_lines"] = df["selftext"].apply(lambda x: str(x).count("\n"))

        if pet_type:
            df["pet_type"] = pet_type

        return df

    cats_posts = preprocess_posts(cats_posts, "cat")
    dogs_posts = preprocess_posts(dogs_posts, "dog")

    def label_high_engagement(df):
        """
        Labels the posts as having high or low engagement based on the engagement score's quantiles.

        Args:
            df (pd.DataFrame): The DataFrame containing posts with an engagement score.

        Returns:
            pd.DataFrame: The DataFrame with a new column 'high_engagement' (1 for high, 0 for low).
        """
        logging.info("🏷️ Labeling high/low engagement...")
        q75 = df["engagement_score"].quantile(0.75)
        q50 = df["engagement_score"].quantile(0.50)
        df["high_engagement"] = df["engagement_score"].apply(
            lambda x: 1 if x >= q75 else (0 if x <= q50 else None)
        )
        return df.dropna(subset=["high_engagement", "cleaned_text"])

    cats_posts = label_high_engagement(cats_posts)
    dogs_posts = label_high_engagement(dogs_posts)

    def run_classifier(
        df,
        label="Posts",
        model="logistic",
        save_prefix="cats",
        save_model=True,
        model_dir_absolute=None,
    ):
        """
        Runs a classifier (Logistic Regression or Random Forest) on the posts dataset to predict high/low engagement.

        Args:
            df (pd.DataFrame): The DataFrame containing the posts and features.
            label (str): The label for the posts (e.g., "Cats" or "Dogs").
            model (str): The model to use ("logistic" for Logistic Regression, "rf" for Random Forest).
            save_prefix (str): Prefix for saving the model and plot files.
            save_model (bool): Whether to save the trained model.
            model_dir_absolute (pathlib.Path): Directory where the model will be saved.

        """
        logging.info(f"🚦 Start running classifier: {label} ({model})")

        structured_features = [
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
        text_feature = "cleaned_text"

        df = df.dropna(subset=["high_engagement", text_feature])
        X = df[[text_feature] + structured_features]
        y = df["high_engagement"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        preprocessor = ColumnTransformer(
            [
                ("tfidf", tfidf, text_feature),
                ("structured", "passthrough", structured_features),
            ]
        )

        if model == "logistic":
            clf = LogisticRegression(max_iter=1000)
        elif model == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model not recognized")

        pipeline = Pipeline([("features", preprocessor), ("clf", clf)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        logging.info(
            f"📊 Classification Report for {label}:\n{classification_report(y_test, y_pred)}"
        )
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

        # === Plot ROC Curve ===
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, y_prob):.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label}")
        plt.legend()
        plt.tight_layout()
        roc_plot_path = save_dir_absolute / f"{save_prefix}_{model}_roc.png"
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        logging.info(f"✅ Saved ROC curve to {roc_plot_path}")

        if model == "rf":
            tfidf_features = (
                pipeline.named_steps["features"]
                .transformers_[0][1]
                .get_feature_names_out()
            )
            all_feature_names = list(tfidf_features) + structured_features
            importances = pipeline.named_steps["clf"].feature_importances_

            importance_df = pd.DataFrame(
                {"feature": all_feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)

            logging.info(f"📌 Top 20 Feature Importances:\n{importance_df.head(20)}")

            plt.figure(figsize=(7, 5))
            importance_df.head(15).plot(
                kind="barh", x="feature", y="importance", legend=False
            )
            plt.gca().invert_yaxis()
            plt.tight_layout()
            feature_plot_path = (
                save_dir_absolute / f"{save_prefix}_{model}_feature_importance.png"
            )
            plt.savefig(feature_plot_path, dpi=300)
            plt.close()
            logging.info(f"✅ Saved feature importance plot to {feature_plot_path}")

        if save_model:
            model_path = model_dir_absolute / f"{save_prefix}_{model}_pipeline.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            logging.info(f"✅ Saved model to {model_path}")
        else:
            logging.warning(f"⚠️ Model not saved (save_model={save_model})")

    # Run all classifiers
    run_classifier(
        cats_posts,
        label="Cats (Logistic)",
        model="logistic",
        save_prefix="cats",
        save_model=save_model,
        model_dir_absolute=model_dir_absolute,
    )
    run_classifier(
        dogs_posts,
        label="Dogs (Logistic)",
        model="logistic",
        save_prefix="dogs",
        save_model=save_model,
        model_dir_absolute=model_dir_absolute,
    )
    run_classifier(
        cats_posts,
        label="Cats (RF)",
        model="rf",
        save_prefix="cats",
        save_model=save_model,
        model_dir_absolute=model_dir_absolute,
    )
    run_classifier(
        dogs_posts,
        label="Dogs (RF)",
        model="rf",
        save_prefix="dogs",
        save_model=save_model,
        model_dir_absolute=model_dir_absolute,
    )
