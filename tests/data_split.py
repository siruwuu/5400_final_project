import pandas as pd
from pathlib import Path

# paths
full_data_path = Path("data")
test_data_path = Path("tests/data")
test_data_path.mkdir(parents=True, exist_ok=True)

# list of files to sample
files = [
    "cats_all_post_processed.csv",
    "dogs_all_posts_processed.csv",
    "cats_all_comments_processed.csv",
    "dogs_all_comments_processed.csv",
]

for fname in files:
    full_path = full_data_path / fname
    df = pd.read_csv(full_path)
    sample_df = df.sample(frac=0.2, random_state=42)
    sample_df.to_csv(test_data_path / fname, index=False)

print("✅ Sampled data saved to tests/data/")
