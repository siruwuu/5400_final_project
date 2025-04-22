# src/mod/predictor.py
import pickle
import pandas as pd
import re
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer

# 初始化情感分析器
sia = SentimentIntensityAnalyzer()

# -----------------------------
# 🧱 1. 构造用于预测的 DataFrame
# -----------------------------
def build_feature_df(text):
    urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
    pronouns = ["you", "your", "we", "us"]

    cleaned = text.strip()

    df = pd.DataFrame({
        "cleaned_text": [cleaned],
        "title": [cleaned],
        "selftext": [cleaned],
        "adjectives": ["[]"],  # 如不做词性标注，这些设为空
        "verbs": ["[]"],
        "score": [0],
        "num_comments": [0],
        "num_exclamations": [cleaned.count("!")],
        "has_question": [int("?" in cleaned)],
        "contains_adopt_keywords": [int("adopt" in cleaned.lower())],
        "num_words": [len(cleaned.split())],
        "sentiment_score": [sia.polarity_scores(cleaned)["compound"]],
        "num_adjectives": [0],
        "num_verbs": [0],
        "num_emojis": [sum(1 for c in cleaned if c in emoji.EMOJI_DATA)],
        "has_urgency_words": [int(any(word in cleaned.lower() for word in urgency_keywords))],
        "has_pronouns": [int(any(p in cleaned.lower() for p in pronouns))],
        "title_length": [len(cleaned)],
        "contains_money": [int(bool(re.search(r"\$\d+|donate|donation|fund|pledge|raise", cleaned.lower())))],
        "num_lines": [cleaned.count("\n")]
    })

    return df

# -----------------------------
# 🧠 2. 加载模型（自动识别猫狗）
# -----------------------------
def load_model(pet_type="dog", model_dir="src/gpt_classifier_suggester/model"):
    if pet_type == "cat":
        model_path = f"{model_dir}/cats_rf_pipeline.pkl"
    else:
        model_path = f"{model_dir}/dogs_rf_pipeline.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# -----------------------------
# 🔮 3. 自动识别猫狗 + 预测概率
# -----------------------------
def full_predict(text, model_dir="src/gpt_classifier_suggester/model"):
    pet_type = "cat" if "cat" in text.lower() or "kitten" in text.lower() else "dog"
    model = load_model(pet_type, model_dir)
    df = build_feature_df(text)
    prob = model.predict_proba(df)[0][1]
    return pet_type, prob
