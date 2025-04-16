import os
import sys
import pandas as pd

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing.preprocessing import preprocess_comments

RAW_DIR = "data/raw-data"
OUTPUT_DIR = "data/processed-data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [
    "cats_all_comments.csv",
    "dogs_all_comments.csv",
    "cats_all_post.csv",
    "dogs_all_posts.csv"
]

for file in files:
    input_path = os.path.join(RAW_DIR, file)
    output_file = os.path.join(OUTPUT_DIR, file.replace(".csv", "_processed.csv"))

    if not os.path.exists(input_path):
        print(f"File not found: {file}, skipping.")
        continue

    if os.path.exists(output_file):
        print(f"⏩ Skipping {file} — already processed.")
        continue

    print(f"🔍 Processing {file} ...")
    df = pd.read_csv(input_path)

    # 为 post 文件创建 body 列
    if "body" not in df.columns:
        if "combined_text" in df.columns:
            df["body"] = df["combined_text"]
        elif "title" in df.columns and "selftext" in df.columns:
            df["body"] = df["title"].astype(str) + " " + df["selftext"].astype(str)
        elif "title" in df.columns:
            df["body"] = df["title"].astype(str)
        else:
            print(f"Skipping {file}: no usable text column found.")
            continue

    # 调用通用的预处理函数
    processed_df = preprocess_comments(df)
    processed_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}\n")
