# streamlit_app.py

import sys, os
import logging
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gpt_classifier_suggester.prediction.predictor import full_predict
from gpt_classifier_suggester.gpt.suggestion import generate_gpt_suggestions

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "streamlit_app.log"), mode='a'),
        logging.StreamHandler()
    ]
)

st.set_page_config(page_title="Reddit Adoption Post Optimizer", page_icon="🐾", layout="centered")

st.title("🐾 Reddit Adoption Post Optimizer 🐱 🐶")

text = st.text_area("✍️ Enter your Reddit post (title + content):", height=200)

if st.button("🔍 Analyze & Get Suggestions"):
    if not text.strip():
        st.warning("Please enter some text.")
        logging.warning("User submitted an empty text input.")
    else:
        pet_type, prob = full_predict(text)
        label = prob >= 0.5  # High engagement if probability >= 50%

        logging.info(f"Prediction completed. Detected pet_type={pet_type}, predicted prob={prob:.4f}")

        st.markdown("---")
        st.markdown(f"### 🐶 Detected Animal Type: `{pet_type}`")
        st.markdown(f"### 🔮 Predicted Engagement: {'🟢 High' if label else '🔴 Low'} ({prob:.2%})")

        st.markdown("---")
        st.markdown("### 💡 Suggestions to Improve Your Post")
        suggestions = generate_gpt_suggestions(text, pet_type, prob)
        st.markdown(suggestions)
        logging.info("Suggestions generated successfully.")
