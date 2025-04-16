import re
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ========== 初始化 ==========
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# ========== 自定义关键词 ==========
ADOPTION_KEYWORDS = [
    "adopt", "adoption", "adopted", "rescue", "rescued", "rehome", "shelter", "foster"
]

# ========== 文本清洗函数 ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # 移除链接
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # 移除 markdown 链接
    text = re.sub(r"\s+", " ", text).strip()    # 移除多余空格
    return text

# ========== 特征提取函数 ==========
def extract_features(row):
    text = row["body"]
    doc = nlp(text)

    adjectives = [token.lemma_ for token in doc if token.pos_ == "ADJ"]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    sentiment = vader.polarity_scores(text)

    return pd.Series({
        "cleaned_text": text,
        "num_words": len(text.split()),
        "num_exclamations": text.count("!"),
        "has_link": int("http" in row["body"]),
        "has_question": int("?" in row["body"]),
        "contains_adopt_keywords": int(any(k in text for k in ADOPTION_KEYWORDS)),
        "sentiment_score": sentiment["compound"],
        "adjectives": adjectives,
        "verbs": verbs
    })

# ========== 主处理函数 ==========
def preprocess_comments(df):
    df["body"] = df["body"].astype(str).apply(clean_text)
    features = df.apply(extract_features, axis=1)
    return pd.concat([df, features], axis=1)
