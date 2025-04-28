"""
data_collection.py - Reddit Comment Collection Script

This script collects comments from various Reddit subreddits, focusing on pet-related
communities. It gathers a large dataset of comments for analysis and extracts a subset
of adoption-related content. The script performs the following operations:

1. API Authentication and Connection:
  - Loads Reddit API credentials from an environment file
  - Establishes a connection to the Reddit API using PRAW library

2. Data Collection Configuration:
  - Targets 10 pet-related subreddits including: "cats", "dogs", "CatAdvice",
    "DogAdvice", "PetAdvice", "Adoptapet", "AdoptMyPet", "puppyAdoption",
    "rescuedogs", and "aww"
  - Processes up to 2000 submissions per subreddit
  - Outputs progress updates every 100 posts

3. Comment Collection Process:
  - Retrieves top-level comments from hot posts in each subreddit
  - Extracts metadata including: subreddit name, post ID, post title, comment ID,
    comment body, score, creation time, and permalink
  - Skips removed or deleted comments
  - Implements rate limiting (1 second delay between requests)

4. Adoption Content Filtering:
  - Maintains a list of adoption-related keywords (e.g., "adopt", "rehome", "rescue")
  - Filters collected comments to identify those discussing animal adoption
  - Creates a specialized subset of adoption-related content

5. Data Storage:
  - Creates a directory structure for organizing the collected data
  - Saves the complete comment dataset as a CSV file
  - Saves the filtered adoption-related comments as a separate CSV file

6. Error Handling:
  - Implements exception handling to manage API errors
  - Continues collection despite issues with individual posts or subreddits

Files:
   Output:
       - all_comments.csv: Complete dataset of all comments
       - adoption_comments.csv: Filtered dataset of adoption-related comments

Note: Unlike some other scripts, this collector doesn't separate cat and dog content,
instead gathering all pet-related comments into a unified dataset.
"""

import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time

# ===== Step 1: Load API Credentials from .env =====
load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ===== Step 2: Configuration =====
subreddits = [
    "cats",
    "dogs",
    "CatAdvice",
    "DogAdvice",
    "PetAdvice",
    "Adoptapet",
    "AdoptMyPet",
    "puppyAdoption",
    "rescuedogs",
    "aww",
]
num_submissions_per_sub = 2000
print_interval = 100

# ===== Step 3: Adoption-related keyword list =====
adoption_keywords = [
    "adopt",
    "adoption",
    "adopted",
    "adopting",
    "rehome",
    "re-home",
    "rescue",
    "rescued",
    "foster",
    "forever home",
    "shelter",
]


def is_adoption_related(text):
    text = text.lower()
    return any(keyword in text for keyword in adoption_keywords)


# ===== Step 4: Initialize storage =====
all_comments = []

# ===== Step 5: Begin fetching =====
for sub in subreddits:
    print(f"\nFetching from r/{sub}...")
    subreddit = reddit.subreddit(sub)

    try:
        for idx, post in enumerate(subreddit.hot(limit=num_submissions_per_sub)):

            try:
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if comment.body in ["[removed]", "[deleted]"]:
                        continue
                    all_comments.append(
                        {
                            "subreddit": sub,
                            "post_id": post.id,
                            "post_title": post.title,
                            "comment_id": comment.id,
                            "body": comment.body,
                            "score": comment.score,
                            "created_utc": (
                                datetime.utcfromtimestamp(
                                    comment.created_utc
                                ).isoformat()
                            ),
                            "permalink": f"https://www.reddit.com{post.permalink}",
                        }
                    )
            except Exception as e:
                print(f"Failed to fetch comments for post: {e}")
                continue

            if idx % print_interval == 0:
                print(f"  ... processed {idx} posts")

            time.sleep(1)

    except Exception as e:
        print(f"Error in r/{sub}: {e}")
        continue

# ===== Step 6: Save all comments =====
os.makedirs("data/raw-data", exist_ok=True)
all_df = pd.DataFrame(all_comments)
all_df.to_csv("data/raw-data/all_comments.csv", index=False)
print(f"\nAll comments saved: {len(all_df)} rows → data/raw-data/all_comments.csv")

# ===== Step 7: Filter adoption-related comments =====
adoption_df = all_df[all_df["body"].apply(is_adoption_related)]
adoption_df.to_csv("data/raw-data/adoption_comments.csv", index=False)
print(
    f"Adoption-related comments saved: {len(adoption_df)} rows → data/raw-data/adoption_comments.csv"
)
