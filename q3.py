import pandas as pd
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import re
import os

cats_posts = pd.read_csv("data/cats_all_post_processed.csv")
cats_comments= pd.read_csv("data/cats_all_comments_processed.csv")
dogs_posts = pd.read_csv("data/dogs_all_posts_processed.csv")
dogs_comments= pd.read_csv("data/dogs_all_comments_processed.csv")

def preprocess_posts(df, pet_type=None):
    # Step 1: 计算 engagement_score
    df["engagement_score"] = df["score"] + 0.5 * df["num_comments"]
    
    # Step 2: 提取形容词/动词数量、emoji 数
    df["num_adjectives"] = df["adjectives"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_emojis"] = df["cleaned_text"].apply(lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA))
    
    # Step 3: 紧急词特征
    urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
    df["has_urgency_words"] = df["cleaned_text"].apply(
        lambda x: int(any(word in str(x).lower() for word in urgency_keywords))
    )
    
    # Step 4: 人称代词
    pronouns = ["you", "your", "we", "us"]
    df["has_pronouns"] = df["cleaned_text"].apply(
        lambda x: int(any(p in str(x).lower() for p in pronouns))
    )
    
    # Step 5: 标题长度
    df["title_length"] = df["title"].apply(lambda x: len(str(x)))
    
    # Step 6: 金额内容
    money_keywords = ["donation", "donate", "pledge", "$", "fund", "raise"]
    def contains_money(text):
        text = str(text).lower()
        return int(any(kw in text for kw in money_keywords) or bool(re.search(r"\$\d+", text)))
    
    df["contains_money"] = df["cleaned_text"].apply(contains_money)
    
    # Step 7: 段落行数
    df["num_lines"] = df["selftext"].apply(lambda x: str(x).count("\n"))
    
    # Step 8: 标注宠物类型（可选）
    if pet_type:
        df["pet_type"] = pet_type
    
    return df

cats_posts = preprocess_posts(cats_posts, pet_type="cat")
dogs_posts = preprocess_posts(dogs_posts, pet_type="dog")

# 构建二分类标签（高/低互动）
# 为猫单独打标签
cat_q75 = cats_posts["engagement_score"].quantile(0.75)
cat_q50 = cats_posts["engagement_score"].quantile(0.50)

cats_posts["high_engagement"] = cats_posts["engagement_score"].apply(
    lambda x: 1 if x >= cat_q75 else (0 if x <= cat_q50 else None)
)
cats_posts = cats_posts.dropna(subset=["high_engagement", "cleaned_text"])
# 为狗单独打标签
dog_q75 = dogs_posts["engagement_score"].quantile(0.75)
dog_q50 = dogs_posts["engagement_score"].quantile(0.50)

dogs_posts["high_engagement"] = dogs_posts["engagement_score"].apply(
    lambda x: 1 if x >= dog_q75 else (0 if x <= dog_q50 else None)
)
dogs_posts = dogs_posts.dropna(subset=["high_engagement", "cleaned_text"])

def run_engagement_classifier(df, label="Posts", save_path=None):
    text_feature = "cleaned_text"
    structured_features = [
        "sentiment_score", "num_adjectives", "num_verbs", 
        "num_exclamations", "has_question", "num_emojis", 
        "contains_adopt_keywords", "has_urgency_words", 
        "has_pronouns", "num_words", "title_length", 
        "contains_money", "num_lines"
    ]
    
    # Drop NA rows
    df = df.dropna(subset=["high_engagement", text_feature])

    # Split data
    X = df[[text_feature] + structured_features]
    y = df["high_engagement"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + structured feature preprocessor
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    preprocessor = ColumnTransformer([
        ("tfidf", tfidf, text_feature),
        ("structured", "passthrough", structured_features)
    ])

    # Classifier pipeline
    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Fit and predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation
    print(f"\n📊 Classification Report for {label}:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # ROC Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

run_engagement_classifier(cats_posts, label="Cats Posts", save_path="q3/cats_lr_roc.png")
run_engagement_classifier(dogs_posts, label="Dogs Posts", save_path="q3/dogs_lr_roc.png")


def run_engagement_rf(df, label="Posts RF", save_path=None):
    structured_features = [
        "sentiment_score", "num_adjectives", "num_verbs", 
        "num_exclamations", "has_question", "num_emojis", 
        "contains_adopt_keywords", "has_urgency_words", 
        "has_pronouns", "num_words", "title_length", 
        "contains_money", "num_lines"
    ]
    text_feature = "cleaned_text"
    
    # Drop NA rows
    df = df.dropna(subset=["high_engagement", text_feature])

    # Split data
    X = df[[text_feature] + structured_features]
    y = df["high_engagement"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessor
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    preprocessor = ColumnTransformer([
        ("tfidf", tfidf, text_feature),
        ("structured", "passthrough", structured_features)
    ])

    # Pipeline with RF
    pipeline = Pipeline([
        ("features", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train + predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    print(f"\n📊 Classification Report for {label}:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y_test, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

    # Feature importance 可视化
    tfidf_features = pipeline.named_steps["features"].transformers_[0][1].get_feature_names_out()
    all_feature_names = list(tfidf_features) + structured_features
    importances = pipeline.named_steps["clf"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\n📌 Top 20 Feature Importances:")
    print(importance_df.head(20))

    # 条形图
    plt.figure(figsize=(7, 5))
    importance_df.head(15).plot(
        kind="barh", x="feature", y="importance", legend=False,
        title=f"Top 15 Features - {label}"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"q3/{label.lower().replace(' ', '_')}_rf_feature_importance.png", dpi=300)
    plt.show()
run_engagement_rf(cats_posts, label="Cats RF", save_path="q3/cats_rf_roc.png")
run_engagement_rf(dogs_posts, label="Dogs RF", save_path="q3/dogs_rf_roc.png")
