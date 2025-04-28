"""
filter_comment.py - Comment Data Processing Tool

This script processes Reddit comment data from different sources, filtering and combining them
based on post IDs from categorized post datasets. It performs the following operations:

1. Reads comments from the original source (comment_first.csv)
2. Filters comments based on post IDs found in cat_posts_first.csv and dog_posts_first.csv
3. Transforms the data structure to match the target format (matching cat_comments_5000.csv)
4. Adds appropriate pet_type values ('cat' or 'dog') based on the source post
5. Checks for and removes duplicate comment IDs when combining with existing datasets
6. Merges the filtered comments with existing comment datasets
7. Saves the results as cats_all_comments.csv and dogs_all_comments.csv


Files:
    Input:
        - comment_first.csv: Original comment data
        - cat_posts_first.csv: Categorized cat posts
        - dog_posts_first.csv: Categorized dog posts
        - cat_comments_5000.csv: Existing cat comment dataset
        - dog_comments_4000.csv: Existing dog comment dataset

    Output:
        - cats_all_comments.csv: Combined cat comments
        - dogs_all_comments.csv: Combined dog comments
"""

import pandas as pd
import os


def filter_and_transform_comments(
    comment_file,
    cat_posts_file,
    dog_posts_file,
    cat_comments_file,
    dog_comments_file,
    cat_output_file,
    dog_output_file,
):
    """
    Filter comments based on post_ids from cat and dog post files,
    transform to match the desired format, and combine with existing comment files.

    Args:
        comment_file (str): Path to the original comments file
        cat_posts_file (str): Path to the cat posts file
        dog_posts_file (str): Path to the dog posts file
        cat_comments_file (str): Path to the existing cat comments file
        dog_comments_file (str): Path to the existing dog comments file
        cat_output_file (str): Path to the output cat comments file
        dog_output_file (str): Path to the output dog comments file

    Returns:
        dict: Statistics about the processing
    """
    # Read all the necessary files
    print(f"Reading comments file: {comment_file}")
    comments_df = pd.read_csv(comment_file)

    print(f"Reading cat posts file: {cat_posts_file}")
    cat_posts_df = pd.read_csv(cat_posts_file)

    print(f"Reading dog posts file: {dog_posts_file}")
    dog_posts_df = pd.read_csv(dog_posts_file)

    print(f"Reading existing cat comments file: {cat_comments_file}")
    existing_cat_comments_df = pd.read_csv(cat_comments_file)

    print(f"Reading existing dog comments file: {dog_comments_file}")
    existing_dog_comments_df = pd.read_csv(dog_comments_file)

    # Get the set of post_ids for cats and dogs
    cat_post_ids = set(cat_posts_df["post_id"])
    dog_post_ids = set(dog_posts_df["post_id"])

    print(
        f"Found {len(cat_post_ids)} cat post_ids and {len(dog_post_ids)} dog post_ids"
    )

    # Filter comments for cats and dogs
    cat_comments = comments_df[comments_df["post_id"].isin(cat_post_ids)].copy()
    dog_comments = comments_df[comments_df["post_id"].isin(dog_post_ids)].copy()

    print(
        f"Filtered {len(cat_comments)} cat comments and {len(dog_comments)} dog comments"
    )

    # Transform the cat comments to match the desired format
    cat_comments_transformed = pd.DataFrame()
    cat_comments_transformed["comment_id"] = cat_comments["comment_id"]
    cat_comments_transformed["post_id"] = cat_comments["post_id"]
    cat_comments_transformed["subreddit"] = cat_comments["subreddit"]
    cat_comments_transformed["body"] = cat_comments["body"]
    cat_comments_transformed["score"] = cat_comments["score"]
    cat_comments_transformed["created_utc"] = cat_comments["created_utc"]
    cat_comments_transformed["pet_type"] = "cat"  # Add pet_type column

    # Transform the dog comments to match the desired format
    dog_comments_transformed = pd.DataFrame()
    dog_comments_transformed["comment_id"] = dog_comments["comment_id"]
    dog_comments_transformed["post_id"] = dog_comments["post_id"]
    dog_comments_transformed["subreddit"] = dog_comments["subreddit"]
    dog_comments_transformed["body"] = dog_comments["body"]
    dog_comments_transformed["score"] = dog_comments["score"]
    dog_comments_transformed["created_utc"] = dog_comments["created_utc"]
    dog_comments_transformed["pet_type"] = "dog"  # Add pet_type column

    # Check for duplicate comment_ids in cat comments
    cat_comment_ids = set(cat_comments_transformed["comment_id"])
    existing_cat_comment_ids = set(existing_cat_comments_df["comment_id"])
    cat_duplicates = cat_comment_ids.intersection(existing_cat_comment_ids)
    cat_comments_transformed = cat_comments_transformed[
        ~cat_comments_transformed["comment_id"].isin(cat_duplicates)
    ]

    # Check for duplicate comment_ids in dog comments
    dog_comment_ids = set(dog_comments_transformed["comment_id"])
    existing_dog_comment_ids = set(existing_dog_comments_df["comment_id"])
    dog_duplicates = dog_comment_ids.intersection(existing_dog_comment_ids)
    dog_comments_transformed = dog_comments_transformed[
        ~dog_comments_transformed["comment_id"].isin(dog_duplicates)
    ]

    print(
        f"Removed {len(cat_duplicates)} duplicate cat comments and {len(dog_duplicates)} duplicate dog comments"
    )

    # Combine the transformed comments with the existing comments
    cat_comments_combined = pd.concat(
        [existing_cat_comments_df, cat_comments_transformed], ignore_index=True
    )
    dog_comments_combined = pd.concat(
        [existing_dog_comments_df, dog_comments_transformed], ignore_index=True
    )

    # Save the combined comments to the output files
    cat_comments_combined.to_csv(cat_output_file, index=False)
    dog_comments_combined.to_csv(dog_output_file, index=False)

    print(f"Saved {len(cat_comments_combined)} cat comments to {cat_output_file}")
    print(f"Saved {len(dog_comments_combined)} dog comments to {dog_output_file}")

    # Return statistics
    return {
        "cat_comments_filtered": len(cat_comments),
        "dog_comments_filtered": len(dog_comments),
        "cat_duplicates_removed": len(cat_duplicates),
        "dog_duplicates_removed": len(dog_duplicates),
        "cat_comments_added": len(cat_comments_transformed),
        "dog_comments_added": len(dog_comments_transformed),
        "cat_comments_total": len(cat_comments_combined),
        "dog_comments_total": len(dog_comments_combined),
    }


def main():
    # Define the base directory using relative paths
    # Assuming this script is in 5400_final_project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level

    # Define paths relative to the base directory
    data_combined_dir = os.path.join(base_dir, "data_combined")

    # Define input and output file paths
    comment_file = os.path.join(data_combined_dir, "comment_first.csv")
    cat_posts_file = os.path.join(data_combined_dir, "cat_posts_first.csv")
    dog_posts_file = os.path.join(data_combined_dir, "dog_posts_first.csv")
    cat_comments_file = os.path.join(data_combined_dir, "cat_comments_5000.csv")
    dog_comments_file = os.path.join(data_combined_dir, "dog_comments_4000.csv")
    cat_output_file = os.path.join(data_combined_dir, "cats_all_comments.csv")
    dog_output_file = os.path.join(data_combined_dir, "dogs_all_comments.csv")

    print("\n===== Filtering and Processing Comments =====")
    stats = filter_and_transform_comments(
        comment_file,
        cat_posts_file,
        dog_posts_file,
        cat_comments_file,
        dog_comments_file,
        cat_output_file,
        dog_output_file,
    )

    print("\n===== Processing Summary =====")
    print(f"Cat comments filtered from original file: {stats['cat_comments_filtered']}")
    print(f"Dog comments filtered from original file: {stats['dog_comments_filtered']}")
    print(f"Duplicate cat comments removed: {stats['cat_duplicates_removed']}")
    print(f"Duplicate dog comments removed: {stats['dog_duplicates_removed']}")
    print(f"Cat comments added to existing file: {stats['cat_comments_added']}")
    print(f"Dog comments added to existing file: {stats['dog_comments_added']}")
    print(f"Total cat comments in final file: {stats['cat_comments_total']}")
    print(f"Total dog comments in final file: {stats['dog_comments_total']}")

    print("\nProcessing complete!")
    print(f"Cat comments saved to: {cat_output_file}")
    print(f"Dog comments saved to: {dog_output_file}")


if __name__ == "__main__":
    main()
