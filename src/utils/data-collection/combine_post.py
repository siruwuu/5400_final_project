"""
combine_post.py - Reddit Pet Post Dataset Merger

This script combines multiple Reddit post datasets while handling duplicate entries.
It processes both cat and dog post collections separately, merging new data with
existing datasets while preserving data integrity. The script performs the following operations:

1. Reads posts from two sources for each pet type:
   - First datasets: cat_posts_first.csv and dog_posts_first.csv
   - Main datasets: cat_posts_5000.csv and dog_posts_4000.csv

2. Identifies duplicate post IDs between the datasets for each pet type

3. Handles duplicates by prioritizing entries from the main datasets:
   - When the same post_id appears in both datasets, the version from the main dataset
     (cat_posts_5000.csv or dog_posts_4000.csv) is kept
   - Entries with unique post_ids from both datasets are preserved

4. Merges the deduplicated datasets to create comprehensive collections

5. Saves the results as cats_all_post.csv and dogs_all_posts.csv

6. Provides detailed statistics about the merging process:
   - Initial record counts from each source
   - Number of duplicates identified and removed
   - Final record count in the merged datasets


Files:
    Input:
        - cat_posts_first.csv: First cat post dataset
        - cat_posts_5000.csv: Main cat post dataset
        - dog_posts_first.csv: First dog post dataset
        - dog_posts_4000.csv: Main dog post dataset

    Output:
        - cats_all_post.csv: Combined cat posts
        - dogs_all_posts.csv: Combined dog posts

"""

import pandas as pd
import os


def combine_csv_files(first_file, main_file, output_file):
    """
    Combine two CSV files, removing duplicate post_ids, and keeping entries from main_file
    when duplicates are found.

    Args:
        first_file (str): Path to the first CSV file
        main_file (str): Path to the main CSV file (entries will be kept when duplicates are found)
        output_file (str): Path to the output CSV file

    Returns:
        dict: Statistics about the merging process
    """
    # Read both CSV files
    print(f"Reading {first_file}...")
    df_first = pd.read_csv(first_file)
    print(f"Reading {main_file}...")
    df_main = pd.read_csv(main_file)

    # Check initial counts
    initial_count_first = len(df_first)
    initial_count_main = len(df_main)

    print(
        f"Initial counts: {initial_count_first} records in first file, {initial_count_main} records in main file"
    )

    # Identify duplicate post_ids
    duplicate_ids = set(df_first["post_id"]).intersection(set(df_main["post_id"]))
    num_duplicates = len(duplicate_ids)

    print(f"Found {num_duplicates} duplicate post_ids")

    # Remove duplicates from the first dataframe
    if num_duplicates > 0:
        df_first = df_first[~df_first["post_id"].isin(duplicate_ids)]

    # Concatenate the dataframes
    df_combined = pd.concat([df_main, df_first], ignore_index=True)

    # Save the combined dataframe to the output file
    df_combined.to_csv(output_file, index=False)

    # Return statistics
    return {
        "initial_count_first": initial_count_first,
        "initial_count_main": initial_count_main,
        "duplicates_removed": num_duplicates,
        "final_count": len(df_combined),
    }


def main():
    # Define the base directory using relative paths
    # Assuming this script is in 5400_final_project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level

    # Define paths relative to the base directory
    data_combined_dir = os.path.join(base_dir, "data_combined")

    # Define input and output file paths
    cat_first_path = os.path.join(data_combined_dir, "cat_posts_first.csv")
    cat_main_path = os.path.join(data_combined_dir, "cat_posts_5000.csv")
    cat_output_path = os.path.join(data_combined_dir, "cats_all_post.csv")

    dog_first_path = os.path.join(data_combined_dir, "dog_posts_first.csv")
    dog_main_path = os.path.join(data_combined_dir, "dog_posts_4000.csv")
    dog_output_path = os.path.join(data_combined_dir, "dogs_all_posts.csv")

    # Combine cat files
    print("\n===== Combining Cat Files =====")
    cat_stats = combine_csv_files(cat_first_path, cat_main_path, cat_output_path)

    print("Cat files combined successfully!")
    print(
        f"Initial counts: {cat_stats['initial_count_first']} (first) + {cat_stats['initial_count_main']} (main)"
    )
    print(f"Removed {cat_stats['duplicates_removed']} duplicates from first file")
    print(f"Final count: {cat_stats['final_count']} records")
    print(f"Output saved to: {cat_output_path}")

    # Combine dog files
    print("\n===== Combining Dog Files =====")
    dog_stats = combine_csv_files(dog_first_path, dog_main_path, dog_output_path)

    print("Dog files combined successfully!")
    print(
        f"Initial counts: {dog_stats['initial_count_first']} (first) + {dog_stats['initial_count_main']} (main)"
    )
    print(f"Removed {dog_stats['duplicates_removed']} duplicates from first file")
    print(f"Final count: {dog_stats['final_count']} records")
    print(f"Output saved to: {dog_output_path}")


if __name__ == "__main__":
    main()
