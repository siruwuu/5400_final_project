import argparse
from mod.engagement_regression import run_regression_analysis
from mod.engagement_classifier import run_engagement_classification

def main():
    parser = argparse.ArgumentParser(description="Run engagement analysis tasks")
    parser.add_argument(
        "--task",
        choices=["regression", "classifier", "all"],
        default="all",
        help="Which task to run: regression, classifier, or all"
    )
    args = parser.parse_args()

    if args.task == "regression":
        print("🔁 Running Regression Analysis...")
        run_regression_analysis()

    elif args.task == "classifier":
        print("🔁 Running Engagement Classification...")
        run_engagement_classification()

    elif args.task == "all":
        print("🔁 Running Regression Analysis...")
        run_regression_analysis()
        print("\n---\n")
        print("🔁 Running Engagement Classification...")
        run_engagement_classification()

if __name__ == "__main__":
    main()
