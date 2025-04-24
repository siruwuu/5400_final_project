# Reddit Engagement Analyzer

Analyze and predict engagement in Reddit pet adoption content based on linguistic style, sentiment, and word usage.  
This project powers two interactive demos to help shelters improve adoption post wording and understand language tendencies across cat vs. dog communities.

---

## Team Members

- Jiahui Liu
- Siru Wu
- Yiqin Zhou
- Jiaqi Wei

---

## Project Overview

We analyze thousands of Reddit posts and comments to understand how language—particularly emotional tone, part-of-speech features, and sentiment—impacts engagement. Our final output includes:

- A full NLP modeling pipeline (regression + classification)
- Preprocessed feature-rich datasets (for cats and dogs)
- Two interactive demos built with Streamlit:
  - **Engagement Optimizer**: Suggests better language for adoption posts
  - **Cat/Dog Classifier**: Reveals if wording leans more “cat-like” or “dog-like”

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
│   │   ├── engagement_regression.py
│   │   ├── engagement_classifier.py
│   │   └── replacement.py     # Word suggestion logic
│   ├── utils/
│   │   ├── data-collection/   # Reddit crawling and filtering scripts
│   │   └── data_preprocessing/
│   │       └── preprocessing.py
│   ├── gpt_classifier_suggester/
│   │   ├── app/               # Streamlit app frontend
│   │   ├── gpt/               # Suggestion logic (GPT-powered or custom)
│   │   ├── model/             # Trained pipelines (.pkl)
│   │   └── prediction/        # Prediction backend
│   └── img/                   # Visuals: ROC curves, feature importance, reports
│
├── tests/
│   ├── tests.py               # Unit tests
│   └── output/                # Test outputs: plots & CSVs
│
├── README.md
├── pyproject.toml
├── environment.yml
└── pytest.ini
```

---

## How to Use

### 1. Install environment
```bash
conda env create -f environment.yml
conda activate reddit-nlp-engagement
python -m spacy download en_core_web_sm
```

### 2. Run main tasks

```bash
# Run regression & classification on posts/comments
python src/bin/main.py --task all
```

Supported options:
```bash
--task regression         # Run engagement regression
--task classifier         # Run classification model
--task preprocessing      # Run text processing pipeline
```

### 3. Launch the Streamlit Demos

```bash
# Demo 1: Adoption Post Optimizer
streamlit run src/gpt_classifier_suggester/app/streamlit_app.py
```

---

## Outputs

- `src/img/`: Model metrics, feature plots, ROC curves
- `data/preprocessed-data/`: Final modeling datasets
- `tests/output/`: Visualization output for debugging

---

## Run Tests

```bash
pytest tests/ -v
```


