import re
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialization
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Define keywords
ADOPTION_KEYWORDS = [
    "adopt",
    "adoption",
    "adopted",
    "rescue",
    "rescued",
    "rehome",
    "shelter",
    "foster",
]


# Text data clean
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove hyperlinks
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown syntax
    text = re.sub(r"\s+", " ", text).strip()  # Remove Extra whitespace
    return text


# Feature extraction
def extract_features(row):
    text = row["body"]
    doc = nlp(text)

    adjectives = [token.lemma_ for token in doc if token.pos_ == "ADJ"]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    sentiment = vader.polarity_scores(text)

    return pd.Series(
        {
            "cleaned_text": text,
            "num_words": len(text.split()),
            "num_exclamations": text.count("!"),
            "has_link": int("http" in row["body"]),
            "has_question": int("?" in row["body"]),
            "contains_adopt_keywords": int(any(k in text for k in ADOPTION_KEYWORDS)),
            "sentiment_score": sentiment["compound"],
            "adjectives": adjectives,
            "verbs": verbs,
        }
    )


def preprocess_comments(df):
    df["body"] = df["body"].astype(str).apply(clean_text)
    features = df.apply(extract_features, axis=1)
    return pd.concat([df, features], axis=1)
