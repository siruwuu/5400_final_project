# src/bin/main.py

import argparse
import os
import sys
import logging

# 设定工程路径，确保可以import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 引入各模块
from mod.new_engagement_regression import run_regression_analysis
from mod.new_engagement_classifier import run_engagement_classification
from mod.cat_vs_dog import run_cat_vs_dog

# logging统一设置
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
