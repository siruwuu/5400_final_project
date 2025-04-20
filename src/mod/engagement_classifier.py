import os
import re
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def run_engagement_classification(data_dir="data", save_dir="src/img"):
    # === Load and preprocess ===
    cats_posts = pd.read_csv(f"{data_dir}/cats_all_post_processed.csv")
    dogs_posts = pd.read_csv(f"{data_dir}/dogs_all_posts_processed.csv")

    def preprocess_posts(df, pet_type=None):
        df["engagement_score"] = df["score"] + 0.5 * df["num_comments"]
        df["num_adjectives"] = df["adjectives"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
        df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
        df["num_emojis"] = df["cleaned_text"].apply(lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA))

        urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
        df["has_urgency_words"] = df["cleaned_text"].apply(
            lambda x: int(any(word in str(x).lower() for word in urgency_keywords)))

        pronouns = ["you", "your", "we", "us"]
        df["has_pronouns"] = df["cleaned_text"].apply(
            lambda x: int(any(p in str(x).lower() for p in pronouns)))

        df["title_length"] = df["title"].apply(lambda x: len(str(x)))

        def contains_money(text):
            text = str(text).lower()
            return int(any(kw in text for kw in ["donation", "donate", "pledge", "$", "fund", "raise"]) or bool(re.search(r"\$\d+", text)))

        df["contains_money"] = df["cleaned_text"].apply(contains_money)
        df["num_lines"] = df["selftext"].apply(lambda x: str(x).count("\n"))

        if pet_type:
            df["pet_type"] = pet_type

        return df

    cats_posts = preprocess_posts(cats_posts, "cat")
    dogs_posts = preprocess_posts(dogs_posts, "dog")

    # === Labeling ===
    def label_high_engagement(df):
        q75 = df["engagement_score"].quantile(0.75)
        q50 = df["engagement_score"].quantile(0.50)
        df["high_engagement"] = df["engagement_score"].apply(
            lambda x: 1 if x >= q75 else (0 if x <= q50 else None))
        return df.dropna(subset=["high_engagement", "cleaned_text"])

    cats_posts = label_high_engagement(cats_posts)
    dogs_posts = label_high_engagement(dogs_posts)

    # === Classification Function ===
    def run_classifier(df, label="Posts", model="logistic", save_prefix="cats"):
        structured_features = [
            "sentiment_score", "num_adjectives", "num_verbs", 
            "num_exclamations", "has_question", "num_emojis", 
            "contains_adopt_keywords", "has_urgency_words", 
            "has_pronouns", "num_words", "title_length", 
            "contains_money", "num_lines"
        ]
        text_feature = "cleaned_text"

        df = df.dropna(subset=["high_engagement", text_feature])
        X = df[[text_feature] + structured_features]
        y = df["high_engagement"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        preprocessor = ColumnTransformer([
            ("tfidf", tfidf, text_feature),
            ("structured", "passthrough", structured_features)
        ])

        if model == "logistic":
            clf = LogisticRegression(max_iter=1000)
        elif model == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Model not recognized")

        pipeline = Pipeline([
            ("features", preprocessor),
            ("clf", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # === Evaluation ===
        print(f"\n📊 Classification Report for {label}:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label}")
        plt.legend()
        plt.tight_layout()

        os.makedirs("src/img", exist_ok=True)
        plt.savefig(f"{save_dir}/{save_prefix}_{model}_roc.png", dpi=300)
        plt.show()

        # Feature importances (RF only)
        if model == "rf":
            tfidf_features = pipeline.named_steps["features"].transformers_[0][1].get_feature_names_out()
            all_feature_names = list(tfidf_features) + structured_features
            importances = pipeline.named_steps["clf"].feature_importances_

            importance_df = pd.DataFrame({
                "feature": all_feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            print("\n📌 Top 20 Feature Importances:")
            print(importance_df.head(20))

            plt.figure(figsize=(7, 5))
            importance_df.head(15).plot(kind="barh", x="feature", y="importance", legend=False,
                                        title=f"Top 15 Features - {label}")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{save_prefix}_{model}_feature_importance.png", dpi=300)
            plt.show()

    # === Run Models ===
    run_classifier(cats_posts, label="Cats (Logistic)", model="logistic", save_prefix="cats")
    run_classifier(dogs_posts, label="Dogs (Logistic)", model="logistic", save_prefix="dogs")
    run_classifier(cats_posts, label="Cats (RF)", model="rf", save_prefix="cats")
    run_classifier(dogs_posts, label="Dogs (RF)", model="rf", save_prefix="dogs")
