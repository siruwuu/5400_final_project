"""
This module processes text data, particularly focusing on comments, by cleaning and extracting relevant features.
It performs the following tasks:

1. **Text Cleaning**: Strips text of unwanted characters such as hyperlinks and markdown syntax.
2. **Feature Extraction**: Extracts various features such as word count, exclamation marks, sentiment score, and the presence of certain keywords.
3. **Text Preprocessing**: Cleans the text and applies the feature extraction on the given dataset.

### Functions:
- `clean_text(text)`:
    Cleans the input text by removing hyperlinks, markdown syntax, and extra whitespace.
    
- `extract_features(row)`:
    Extracts linguistic and sentiment features from the input text, including word count, sentiment score, adjectives, verbs, and the presence of adoption-related keywords.

- `preprocess_comments(df)`:
    Preprocesses a DataFrame of comments, cleaning the text and extracting relevant features, then returning the enhanced DataFrame.
"""
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
    """
    Cleans the input text by removing hyperlinks, markdown syntax, and extra whitespace.

    Args:
        text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove hyperlinks
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown syntax
    text = re.sub(r"\s+", " ", text).strip()  # Remove Extra whitespace
    return text


# Feature extraction
def extract_features(row):
    """
    Extracts various linguistic and sentiment features from the input text.

    Args:
        row (pd.Series): A row from the DataFrame containing a 'body' column with the text.

    Returns:
        pd.Series: A Series containing extracted features, such as word count, sentiment score, adjectives, and verbs.
    """
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
    """
    Preprocesses the comments DataFrame by cleaning the text and extracting relevant features.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'body' column with the comments text.

    Returns:
        pd.DataFrame: The original DataFrame with new columns for extracted features (e.g., sentiment score, number of adjectives).
    """
    df["body"] = df["body"].astype(str).apply(clean_text)
    features = df.apply(extract_features, axis=1)
    return pd.concat([df, features], axis=1)
