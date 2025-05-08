# src/mod/predictor.py
"""
This module provides functions to process Reddit post text, extract relevant features,
and use a pre-trained machine learning model to predict whether the post is related to
a cat or a dog, as well as to calculate the likelihood of high engagement.

### Functions:
- `build_feature_df(text)`: 
    Converts a Reddit post into a DataFrame of extracted features that are used for prediction.
    
- `load_model(pet_type="dog", model_dir="src/gpt_classifier_suggester/model")`: 
    Loads a pre-trained machine learning model for predicting engagement based on pet type (cat or dog).
    
- `full_predict(text, model_dir="src/gpt_classifier_suggester/model")`: 
    Predicts the pet type (cat or dog) and the probability of high engagement for a given Reddit post.

"""

import pickle
import pandas as pd
import re
import emoji
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
    
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize sentiment
sia = SentimentIntensityAnalyzer()



#  build predictive DataFrame

def build_feature_df(text):
    """
    Processes the input text to extract a set of features for prediction, including
    sentiment analysis, presence of specific keywords, punctuation counts, and more.

    Args:
        text (str): The Reddit post content (title + body).

    Returns:
        pd.DataFrame: A DataFrame containing various features extracted from the text.
    """
    urgency_keywords = ["urgent", "emergency", "last chance", "help", "please"]
    pronouns = ["you", "your", "we", "us"]

    cleaned = text.strip()

    df = pd.DataFrame(
        {
            "cleaned_text": [cleaned],
            "title": [cleaned],
            "selftext": [cleaned],
            "adjectives": ["[]"], 
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
            "has_urgency_words": [
                int(any(word in cleaned.lower() for word in urgency_keywords))
            ],
            "has_pronouns": [int(any(p in cleaned.lower() for p in pronouns))],
            "title_length": [len(cleaned)],
            "contains_money": [
                int(
                    bool(
                        re.search(
                            r"\$\d+|donate|donation|fund|pledge|raise", cleaned.lower()
                        )
                    )
                )
            ],
            "num_lines": [cleaned.count("\n")],
        }
    )

    return df


#  use model to identity dogs and cats
def load_model(pet_type="dog", model_dir="src/gpt_classifier_suggester/model"):
    """
    Loads a pre-trained machine learning model to classify Reddit posts as related
    to either a cat or a dog.

    Args:
        pet_type (str): The pet type to load the model for ("dog" or "cat").
        model_dir (str): The directory containing the pre-trained models.

    Returns:
        model: The loaded pre-trained machine learning model.
    """
    if pet_type == "cat":
        model_path = f"{model_dir}/cats_rf_pipeline.pkl"
    else:
        model_path = f"{model_dir}/dogs_rf_pipeline.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model



# Identify pet type and predict engagement probability

def full_predict(text, model_dir="src/gpt_classifier_suggester/model"):
    """
    Identifies the pet type (cat or dog) based on the content of the Reddit post,
    and predicts the probability of high engagement using the appropriate pre-trained model.

    Args:
        text (str): The Reddit post content (title + body).
        model_dir (str): The directory containing the pre-trained models.

    Returns:
        tuple: A tuple containing:
            - pet_type (str): The predicted pet type ("cat" or "dog").
            - prob (float): The predicted probability of high engagement (0 to 1).
    """
    pet_type = "cat" if "cat" in text.lower() or "kitten" in text.lower() else "dog"
    model = load_model(pet_type, model_dir)
    df = build_feature_df(text)
    prob = model.predict_proba(df)[0][1]
    return pet_type, prob
