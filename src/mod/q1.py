import pandas as pd
import emoji

cats_posts = pd.read_csv("data/cats_all_post_processed.csv")
cats_comments= pd.read_csv("data/cats_all_comments_processed.csv")
dogs_posts = pd.read_csv("data/dogs_all_posts_processed.csv")
dogs_comments= pd.read_csv("data/dogs_all_comments_processed.csv")

#Posts
# Create engagement_score
cats_posts["engagement_score"] = cats_posts["score"] + 0.5 * cats_posts["num_comments"]
dogs_posts["engagement_score"] = dogs_posts["score"] + 0.5 * dogs_posts["num_comments"]

# Feature engineering
for df in [cats_posts, dogs_posts]:
    df["num_adjectives"] = df["adjectives"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_emojis"] = df["cleaned_text"].apply(lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA))

# 添加 has_urgency_words 特征
urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
cats_posts["has_urgency_words"] = cats_posts["cleaned_text"].apply(
    lambda x: int(any(word in str(x).lower() for word in urgency_keywords))
)
dogs_posts["has_urgency_words"] = dogs_posts["cleaned_text"].apply(
    lambda x: int(any(word in str(x).lower() for word in urgency_keywords))
)

# 添加 has_pronouns 特征
pronouns = ["you", "your", "we", "us"]
cats_posts["has_pronouns"] = cats_posts["cleaned_text"].apply(
    lambda x: int(any(p in str(x).lower() for p in pronouns))
)
dogs_posts["has_pronouns"] = dogs_posts["cleaned_text"].apply(
    lambda x: int(any(p in str(x).lower() for p in pronouns))
)

# 添加 title_length 特征
cats_posts["title_length"] = cats_posts["title"].apply(lambda x: len(str(x)))
dogs_posts["title_length"] = dogs_posts["title"].apply(lambda x: len(str(x)))

# 添加 contains_money 特征（匹配 $100、donation）
import re
money_keywords = ["donation", "donate", "pledge", "$", "fund", "raise"]
def contains_money(text):
    text = str(text).lower()
    return int(any(kw in text for kw in money_keywords) or bool(re.search(r"\$\d+", text)))

cats_posts["contains_money"] = cats_posts["cleaned_text"].apply(contains_money)
dogs_posts["contains_money"] = dogs_posts["cleaned_text"].apply(contains_money)

# 添加 num_lines 特征
cats_posts["num_lines"] = cats_posts["selftext"].apply(lambda x: str(x).count("\n"))
dogs_posts["num_lines"] = dogs_posts["selftext"].apply(lambda x: str(x).count("\n"))

# feature set 
feature_columns = [
    "sentiment_score", "num_adjectives", "num_verbs", 
    "num_exclamations", "has_question", "num_emojis", 
    "contains_adopt_keywords", "num_words","has_urgency_words", "has_pronouns", 
    "title_length", "contains_money", "num_lines"
]

# 建立回归模型（先用 Linear Regression）
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def analyze_engagement(posts_df, label="Cats"):
    # 去除缺失值
    df_model = posts_df.dropna(subset=["engagement_score"] + feature_columns).copy()
    X = df_model[feature_columns]
    y = df_model["engagement_score"]

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 建模
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n🔍 Linear Regression Performance for {label} Posts:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")

    # 输出系数表
    coef_df = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print("\n📊 Feature Coefficients:")
    print(coef_df)
    coef_df.to_csv(f"q1/{label.lower()}_feature_coefficients.csv", index=False)


    # 可视化
    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.xlabel("Coefficient")
    plt.title(f"{label} Posts: Feature Importance (Linear Regression)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"q1/{label.lower()}_feature_importance.png", dpi=300)
    plt.show()

analyze_engagement(cats_posts, label="Cats")
analyze_engagement(dogs_posts, label="Dogs")





# Comments
urgency_words = ["urgent", "emergency", "last chance", "help", "please"]
pronouns = ["you", "your", "we", "us"]

def preprocess_comments(df):
    df["num_adjectives"] = df["adjectives"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_verbs"] = df["verbs"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df["num_emojis"] = df["cleaned_text"].apply(lambda x: sum(1 for c in str(x) if c in emoji.EMOJI_DATA))
    df["has_urgency_words"] = df["cleaned_text"].apply(lambda x: int(any(word in str(x).lower() for word in urgency_words)))
    df["has_pronouns"] = df["cleaned_text"].apply(lambda x: int(any(p in str(x).lower() for p in pronouns)))
    return df

cats_comments = preprocess_comments(cats_comments)
dogs_comments = preprocess_comments(dogs_comments)
feature_columns = [
    "sentiment_score", "num_adjectives", "num_verbs", 
    "num_exclamations", "has_question", "num_emojis", 
    "contains_adopt_keywords", "has_urgency_words", 
    "has_pronouns", "num_words"
]

def analyze_comment_engagement(df, label="Comments"):
    df_model = df.dropna(subset=["score"] + feature_columns).copy()
    X = df_model[feature_columns]
    y = df_model["score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n🔍 Linear Regression Performance for {label}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.3f}")

    coef_df = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    print("\n📊 Feature Coefficients:")
    print(coef_df)
    coef_df.to_csv(f"q1/{label.lower()}_comment_feature_coefficients.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.xlabel("Coefficient")
    plt.title(f"{label}: Feature Importance (Linear Regression)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"q1/{label.lower().replace(' ', '_')}_comment_feature_importance.png", dpi=300)
    plt.show()

analyze_comment_engagement(cats_comments, label="Cats Comments")
analyze_comment_engagement(dogs_comments, label="Dogs Comments")
