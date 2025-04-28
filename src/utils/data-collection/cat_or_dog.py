"""
cat_or_dog.py - Reddit Pet Post Classifier and Format Converter

This script processes Reddit posts from various subreddits, classifies them as either cat
or dog related, and converts them to a standardized format. It performs the following operations:

1. Reads posts from the source CSV file (post_first.csv)
2. Classifies each post as either cat or dog related using the following logic:
   - Direct classification by subreddit:
     * cats, CatAdvice → cat
     * dogs, DogAdvice, rescuedogs → dog
   - For ambiguous subreddits (aww, PetAdvice):
     * Analyzes post content (title and body) for cat/dog-related keywords
     * Examines keyword frequency when both pet types are mentioned
     * Determines the primary subject based on context analysis

3. Transforms the data structure to match the target format with fields:
   - post_id, subreddit, title, selftext, combined_text, score, upvote_ratio,
     num_comments, created_utc, url, pet_type

4. Separates the classified posts into separate datasets for cats and dogs
5. Saves the results as cat_posts_first.csv and dog_posts_first.csv


Files:
    Input:
        - post_first.csv: Original Reddit post data

    Output:
        - cat_posts_first.csv: Classified cat posts
        - dog_posts_first.csv: Classified dog posts

Note: This script handles various forms of pet references (e.g., "cat", "cats", "kitten",
"kitty", "dog", "puppy", etc.) and uses regular expressions to ensure accurate keyword matching.
"""

import pandas as pd
import re
import os


def determine_pet_type(row):
    """
    Determine if a post is about a cat or a dog based on subreddit and content.
    Returns 'cat', 'dog', or None if undetermined.
    """
    # Check subreddit first
    if row["subreddit"].lower() in ["cats", "catadvice"]:
        return "cat"
    elif row["subreddit"].lower() in ["dogs", "dogadvice", "rescuedogs"]:
        return "dog"

    # For ambiguous subreddits, check content
    if row["subreddit"].lower() in ["aww", "petadvice"]:
        # Create a combined text from title and body for searching
        content = (str(row["post_title"]) + " " + str(row["body"])).lower()

        # Look for cat indicators
        cat_keywords = [
            "cat",
            "cats",
            "kitten",
            "kittens",
            "kitty",
            "kitties",
            "feline",
        ]
        has_cat = any(
            re.search(r"\b{}\b".format(word), content) for word in cat_keywords
        )

        # Look for dog indicators
        dog_keywords = ["dog", "dogs", "puppy", "puppies", "pup", "pups", "canine"]
        has_dog = any(
            re.search(r"\b{}\b".format(word), content) for word in dog_keywords
        )

        # Determine type based on keywords
        if has_cat and not has_dog:
            return "cat"
        elif has_dog and not has_cat:
            return "dog"
        elif has_cat and has_dog:
            # If both are mentioned, determine by which one is mentioned more frequently
            cat_count = sum(
                len(re.findall(r"\b{}\b".format(word), content))
                for word in cat_keywords
            )
            dog_count = sum(
                len(re.findall(r"\b{}\b".format(word), content))
                for word in dog_keywords
            )

            if cat_count > dog_count:
                return "cat"
            elif dog_count > cat_count:
                return "dog"

    # If we can't determine, return None
    return None


def process_csv(input_path, output_dir):
    """
    Process the input CSV and convert it to the desired format.
    Split the results into cat and dog files.
    """
    # Read the input CSV
    df = pd.read_csv(input_path)

    # Create the new DataFrame with the desired columns
    new_df = pd.DataFrame(
        columns=[
            "post_id",
            "subreddit",
            "title",
            "selftext",
            "combined_text",
            "score",
            "upvote_ratio",
            "num_comments",
            "created_utc",
            "url",
            "pet_type",
        ]
    )

    # Copy and transform data
    new_df["post_id"] = df["post_id"]
    new_df["subreddit"] = df["subreddit"]
    new_df["title"] = df["post_title"]
    new_df["selftext"] = df["body"]
    new_df["combined_text"] = df["post_title"] + " " + df["body"]
    new_df["score"] = df["score"]
    # Set default values for columns not in the original data
    new_df["upvote_ratio"] = 1.0  # Default value
    new_df["num_comments"] = 0  # Default value
    new_df["created_utc"] = df["created_utc"]
    # Create a placeholder URL using the post_id
    new_df["url"] = "https://www.reddit.com" + df["permalink"]

    # Determine pet type for each row
    new_df["pet_type"] = df.apply(determine_pet_type, axis=1)

    # Remove rows where pet_type couldn't be determined
    new_df = new_df.dropna(subset=["pet_type"])

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split into cat and dog dataframes
    cat_df = new_df[new_df["pet_type"] == "cat"]
    dog_df = new_df[new_df["pet_type"] == "dog"]

    # Save to CSV files
    cat_df.to_csv(os.path.join(output_dir, "cat_posts_first.csv"), index=False)
    dog_df.to_csv(os.path.join(output_dir, "dog_posts_first.csv"), index=False)

    # Return statistics
    return {
        "total_processed": len(df),
        "cat_count": len(cat_df),
        "dog_count": len(dog_df),
        "undetermined": len(df) - len(cat_df) - len(dog_df),
    }


if __name__ == "__main__":
    # Define relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "data_combined", "post_first.csv")
    output_dir = os.path.join(script_dir, "data_combined")

    # Process the CSV
    stats = process_csv(input_path, output_dir)

    # Print statistics
    print("Processing complete!")
    print(f"Total records processed: {stats['total_processed']}")
    print(f"Cat posts: {stats['cat_count']}")
    print(f"Dog posts: {stats['dog_count']}")
    print(f"Undetermined (removed): {stats['undetermined']}")
    print(f"Output files saved to {output_dir}")
