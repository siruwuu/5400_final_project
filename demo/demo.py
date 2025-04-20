# src/mod/replacement.py
import pandas as pd
import numpy as np
import re
import nltk
import joblib
from collections import Counter
from nltk.corpus import stopwords
from nltk import pos_tag
import spacy

nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

stop_words = set(stopwords.words("english"))

# ✅ 使用更准确的中型词向量模型
nlp = spacy.load("en_core_web_md")


def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


def is_descriptive(word):
    tag = pos_tag([word])[0][1]
    return tag.startswith("JJ") or tag.startswith("RB") or tag.startswith("VB")


def get_classifier_based_replacements(model_path, top_n=30):
    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["features"].transformers_[0][1]
    structured_features = pipeline.named_steps["features"].transformers_[1][2]

    tfidf_features = tfidf.get_feature_names_out()
    importances = pipeline.named_steps["clf"].feature_importances_
    all_feature_names = list(tfidf_features) + structured_features

    feature_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    })

    tfidf_df = feature_df[feature_df["feature"].isin(tfidf_features)]
    tfidf_df = tfidf_df.sort_values("importance", ascending=False)

    filtered = [w for w in tfidf_df["feature"] if is_descriptive(w) and len(w) > 2 and w not in stop_words]
    return filtered[:top_n]


def get_similar_words_by_semantics(word, candidates, top_n=3, threshold=0.5):
    try:
        word_vec = nlp(word)
        sims = [(cand, nlp(cand).similarity(word_vec)) for cand in candidates]
        sims = [item for item in sims if item[1] >= threshold]
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        return [w for w, _ in sims[:top_n]]
    except Exception:
        return []


def get_replacement_suggestions(high_df, low_df, model_path, top_n=30):
    top_words = get_classifier_based_replacements(model_path=model_path, top_n=top_n)

    low_words = Counter()
    for text in low_df["cleaned_text"].dropna():
        low_words.update([w for w in tokenize(text) if w not in stop_words and len(w) > 2])

    common_low_words = [w for w, c in low_words.items() if c >= 5 and w not in top_words]

    suggestions = {}
    for low_word in common_low_words:
        similar = get_similar_words_by_semantics(low_word, top_words, top_n=3, threshold=0.5)
        if similar:
            suggestions[low_word] = similar

    return suggestions
