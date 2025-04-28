# Reddit Engagement Analyzer

Analyze and predict engagement in Reddit pet adoption posts based on linguistic style, sentiment, and word usage.
This project powers two interactive demos designed to help shelters improve adoption post wording and understand language tendencies across cat and dog communities.

---

## Team Members

- Jiahui Liu
- Siru Wu
- Yiqin Zhou
- Jiaqi Wei

---

## Project Overview

We analyze thousands of Reddit posts and comments to understand how language—particularly emotional tone, part-of-speech features, and sentiment—impacts user engagement.
This project focuses on Reddit adoption posts (cats & dogs) and aims to optimize their engagement performance (likes, comments, shares).
It provides three core capabilities:

- Engagement Classification: Predict whether a post will have high or low engagement.
- Engagement Regression: Estimate engagement score directly.
- Cat vs Dog Linguistic Divergence Analysis: Explore language differences between cat and dog posts.

The ultimate goal is to identify actionable linguistic features that enhance post performance, with GPT-4 providing targeted suggestions.

---

## Project Structure

```bash
5400_final_project/
├── data/
│   ├── raw-data/              # Original Reddit post/comment datasets
│   ├── processed-data/        # Cleaned but not feature-engineered
│   └── preprocessed-data/     # Final feature-enriched datasets for modeling
│
├── src/
│   ├── bin/
│   │   └── main.py            # CLI entry point for data & model pipelines
│   ├── mod/
│   │   ├── new_engagement_regression.py
│   │   ├── new_engagement_classifier.py
│   │   └── cat_vs_dog.py     # Word suggestion logic
│   ├── utils/
│   │   ├── data-collection/   # Reddit crawling and filtering scripts
│   │   └── data_preprocessing/
│   │       └── preprocessing.py
│   ├── gpt_classifier_suggester/
│   │   ├── app/streamlit_app.py            # Streamlit app frontend
│   │   ├── gpt/suggestion.py               # Suggestion logic (GPT-powered or custom)
│   │   ├── model/.pkl                      # Trained pipelines (.pkl)
│   │   └── prediction/predictor.py         # Prediction backend
│   └── img/                   # Visuals: ROC curves, feature importance, reports
│
├── tests/
│   ├── test_cat_vs_dog.py               # Unit tests
│   └── test_part1.py                # Test outputs: plots & CSVs
│
├── logs/
├── .env                        # ignored
├── README.md
├── pyproject.toml
├── environment.yml
└── pytest.ini
```

---

## Architecture Overview
[plot]
- src/mod/:
- - new_engagement_classifier.py: Train classifiers to predict high/low engagement.
- - new_engagement_regression.py: Regression analysis for predicting engagement scores.
- - cat_vs_dog.py: Compare linguistic style divergence between cat and dog posts.

- src/gpt_classifier_suggester/:
- - gpt/suggestion.py: Generate GPT suggestions.
- - prediction/predictor.py: Feature engineering + prediction pipeline.
- - app/streamlit_app.py: Interactive web app for Reddit optimization.

- tests/: Unit and smoke tests for all modules.

## Data Source

The project uses preprocessed Reddit data available at:

 https://drive.google.com/drive/folders/1kyAefJXVvzBxXt4_EZ2XVnrMzkHcumhn?dmr=1&ec=wgc-drive-hero-goto

 Please download and place the folder as data/ in your project root directory.

## Installation

### 1. Install environment:

```bash
conda env create -f environment.yml
conda activate reddit-engagement-analyzer
```
### 2. Install project locally:

```bash
pip install -e .
```
## Usage 

### Run specific modules from the command line:

```bash
# Regression analysis
python src/bin/main.py --task regression

# Engagement classification
python src/bin/main.py --task classifier

# Cat vs Dog divergence analysis
python src/bin/main.py --task cat_vs_dog

# Run all tasks
python src/bin/main.py --task all
```

### Launch the interactive Streamlit web app:

```bash
# Run Streamlit app:
streamlit run src/gpt_classifier_suggester/app/streamlit_app.py
```
### Run tests:

```bash
# Rub tests
pytest tests/
```

## Note on API Key for GPT Suggestions

This project uses OpenAI's GPT-4 API to generate language suggestions for improving Reddit adoption posts.
To protect security and comply with best practices, the API Key is not hard-coded in the source code.
Instead, we use a .env file and load the API Key via environment variables.

Before running the Streamlit web app (streamlit_app.py), you must:

- Create a .env file in the project root with the following content: OPENAI_API_KEY=your-openai-api-key-here
- Install the python-dotenv package (already included in environment.yml and pyproject.toml).
- Then you can run: “streamlit run src/gpt_classifier_suggester/app/streamlit_app.py”

**If no API key is provided, the GPT-powered suggestion feature will not work, but the prediction model will still function.**

## Findings & Conclusion

### Key Findings

- Posts containing **urgency keywords** (e.g., "urgent", "help", "last chance") and **community-oriented pronouns** (e.g., "you", "we") are more likely to receive higher engagement.
- Emotional tone, presence of **adjectives**, and **emojis** positively correlate with higher engagement scores.
- Linguistic style analysis revealed that **dog posts** tend to use more action-oriented verbs ("running", "jumping"), while **cat posts** favor calmer adjectives ("fluffy", "lazy").

### Conclusion

Our project successfully demonstrates that Reddit adoption post engagement can be predicted and optimized through linguistic features.
By combining machine learning classification with GPT-based suggestions, we provide shelters and pet advocates a tool to craft more engaging posts,
potentially improving adoption success rates.
