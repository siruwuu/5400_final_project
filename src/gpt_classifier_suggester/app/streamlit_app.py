# src/gpt_classifier_suggester/app/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import streamlit as st
from gpt_classifier_suggester.prediction.predictor import full_predict
from gpt_classifier_suggester.gpt.suggestion import generate_gpt_suggestions

st.set_page_config(page_title="Reddit Adoption Post Optimizer", page_icon="🐾", layout="centered")

st.title("🐾 Reddit Adoption Post Optimizer🐱 🐶")

text = st.text_area("✍️ Enter your Reddit post (title + content):", height=200)

if st.button("🔍 Analyze & Get Suggestions"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        pet_type, prob = full_predict(text)
        label = prob >= 0.5  # High engagement if probability >= 50%

        # 显示预测结果
        st.markdown("---")
        st.markdown(f"### 🐶 Detected Animal Type: `{pet_type}`")
        st.markdown(f"### 🔮 Predicted Engagement: {'🟢 High' if label else '🔴 Low'} ({prob:.2%})")

        # GPT 建议
        st.markdown("---")
        st.markdown("### 💡 Suggestions to Improve Your Post")
        suggestions = generate_gpt_suggestions(text, pet_type, prob)
        st.markdown(suggestions)
