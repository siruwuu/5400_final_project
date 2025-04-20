# 5400_final_project

# Reddit Engagement Analyzer

A Python toolkit for analyzing and predicting engagement (score + number of comments) on cat- and dog-themed Reddit posts.

---

## 📌 Project Overview

This project aims to explore how linguistic features (e.g., sentiment scores, adjectives, emojis, urgency words) relate to Reddit post engagement. It also builds classification models to predict whether a post will receive high engagement.

Supported tasks:
- `regression`: Understand which features are most correlated with engagement scores
- `classifier`: Predict whether a post is highly engaging or not

---

## 🧱 Project Structure

```bash
5400_final_project-2/
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This file
├── src/
│   ├── mod/
│   │   ├── engagement_regression.py     # Regression model module
│   │   └── engagement_classifier.py     # Classification model module
│   └── bin/
│       └── main.py                     # Entry point for running tasks
├── tests/
│   ├── tests.py                        # Unit tests
│   ├── data/                           # 20% sampled test data
│   └── output/                         # Output plots from test run
├── data/                               # Full dataset (local, not pushed)
└── src/img/                            # Output images for models
```

---

## 🚀 How to Use

### 1. Install dependencies (recommended in a virtual environment)
```bash
pip install -e .
```

### 2. Run tasks
```bash
# Run regression analysis
python src/bin/main.py --task regression

# Run classification prediction
python src/bin/main.py --task classifier

# Run both
python src/bin/main.py --task all
```

---

## 🧪 Testing

The project uses `pytest` to test core functions:
```bash
pytest tests/ -v
```

---

## 🔍 Output

- Model coefficient plots, SHAP visualizations, and feature importance charts will be saved in `src/img/`
- Visual outputs from test runs will be stored in `tests/output/` (non-destructive)

---

## 👤 Team Member

- Jiahui Liu

---

## 🧭 Architecture Diagram

> 📌 Please insert the architecture diagram exported from draw.io here, showing the flow: Data Input → Feature Engineering → Modeling → Output.
