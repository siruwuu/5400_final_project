# src/bin/main.py

"""
This script runs various engagement analysis tasks, including regression analysis, 
classification analysis, and a Cat vs Dog style analysis. The script is designed to be 
executed from the command line, where the user can choose which task to run.

### Tasks:
- **regression**: Runs the regression analysis to predict engagement scores based on various features.
- **classifier**: Runs the classification analysis to classify posts as high or low engagement.
- **cat_vs_dog**: Runs the Cat vs Dog diverging style analysis, comparing linguistic styles in cat and dog posts.
- **all**: Runs all the tasks sequentially.

### Command-Line Arguments:
- `--task`: Specifies the task to run. Choices include:
  - `regression`: Executes the regression analysis task.
  - `classifier`: Executes the engagement classification task.
  - `cat_vs_dog`: Executes the Cat vs Dog style analysis.
  - `all`: Runs all tasks sequentially.

### Logging:
- The script logs important information, including task execution status, errors, and results, into a log file (`main.log`) in the `logs` directory.

"""

import argparse
import os
import sys
import logging

# Set path to make sure import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Use all model
from mod.new_engagement_regression import run_regression_analysis
from mod.new_engagement_classifier import run_engagement_classification
from mod.cat_vs_dog import run_cat_vs_dog

# Set logging
log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "main.log"), mode="a"),
        logging.StreamHandler(),
    ],
)


def main():
    parser = argparse.ArgumentParser(description="Run engagement analysis tasks")
    parser.add_argument(
        "--task",
        choices=["regression", "classifier", "cat_vs_dog", "all"],
        default="all",
        help="Which task to run: regression, classifier, cat_vs_dog, or all",
    )
    args = parser.parse_args()

    if args.task == "regression":
        logging.info("🔁 Running Regression Analysis...")
        run_regression_analysis()

    elif args.task == "classifier":
        logging.info("🔁 Running Engagement Classification...")
        run_engagement_classification()

    elif args.task == "cat_vs_dog":
        logging.info("🔁 Running Cat vs Dog Diverging Style Analysis...")
        run_cat_vs_dog()

    elif args.task == "all":
        logging.info("🔁 Running Regression Analysis...")
        run_regression_analysis()
        logging.info("\n---\n")
        logging.info("🔁 Running Engagement Classification...")
        run_engagement_classification()
        logging.info("\n---\n")
        logging.info("🔁 Running Cat vs Dog Diverging Style Analysis...")
        run_cat_vs_dog()


if __name__ == "__main__":
    main()
