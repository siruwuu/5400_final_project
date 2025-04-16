"""
data_v1.py - Reddit Pet Adoption Data Collector

This script collects Reddit posts and comments related to cat and dog adoptions from
specific subreddits. It was initially designed to focus exclusively on dedicated
rescue subreddits (r/rescuecats and r/rescuedogs) for precise data collection,
but was expanded to include the more general r/AdoptMe subreddit due to limited
post volume in the rescue-specific communities.

The script performs the following operations:

1. Establishes connection to Reddit API using PRAW with credentials from .env file

2. Collects adoption-related posts from multiple sources:
   - Primary sources: r/rescuecats and r/rescuedogs (specialized rescue communities)
   - Supplementary source: r/AdoptMe (mixed adoption content requiring filtering)

3. Implements multi-level filtering to ensure relevance:
   - Checks for adoption-related keywords in all posts
   - For posts from r/AdoptMe, applies additional pet-specific filtering to separate
     cat and dog content using extensive keyword lists
   - Skips previously processed posts to avoid duplicates

4. Processes posts across multiple sorting methods (new, hot, top) to maximize data collection

5. For each qualifying post:
   - Extracts post metadata and content
   - Collects up to 10 top comments (sorted by score)
   - Adds pet_type classification ("cat" or "dog")

6. Implements progressive saving:
   - Stores temporary datasets every 500 posts to prevent data loss
   - Creates separate files for cats and dogs
   - Maintains separate post and comment collections

7. Generates final datasets:
   - Complete datasets (all_pet_posts.csv, all_pet_comments.csv)
   - Pet-specific datasets (cat_posts.csv, cat_comments.csv, dog_posts.csv, dog_comments.csv)
   - Provides detailed collection statistics

Note: Due to limited content in the specialized rescue subreddits, the script
relies on keyword filtering of more general adoption subreddits to reach the
target dataset size of 5000 posts per pet type. This approach balances data
volume requirements with content relevance.


Requirements:
    - .env file with Reddit API credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
    - praw, pandas, python-dotenv packages

"""

import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time

# ===== Step 1: Load environment variables =====
load_dotenv()  # Load environment variables from .env

# Create Reddit instance using credentials
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# ===== Step 2: Configuration =====
# "AdoptMe" may contain various animals, requires extra filtering
cat_subreddits = ["rescuecats", "AdoptMe"]
dog_subreddits = ["rescuedogs", "AdoptMe"]
target_posts_per_type = 5000
max_comments_per_post = 10
print_interval = 50  # Print progress every N posts

# ===== Step 3: Adoption-related keywords =====
adoption_keywords = [
    "adopt", "adoption", "adopted", "adopting",
    "rehome", "re-home", "rescue", "rescued",
    "foster", "forever home", "shelter",
    "looking for home", "needs home", "new home",
    "available", "seeking", "help", "please"
]

# Cat-related keywords
cat_keywords = [
    "cat", "cats", "kitten", "kittens", "kitty", 
    "feline", "tabby", "calico", "siamese", "persian",
    "she-cat", "tomcat", "meow"
]

# Dog-related keywords
dog_keywords = [
    "dog", "dogs", "puppy", "puppies", "pup", "pooch",
    "canine", "hound", "retriever", "shepherd", "terrier",
    "labrador", "beagle", "collie", "poodle", "bark", "woof"
]

def is_adoption_related(text):
    """Check if the text is adoption-related (lenient matching)"""
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(keyword in text for keyword in adoption_keywords)

def is_cat_related(text):
    """Check if the text is cat-related"""
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(keyword in text for keyword in cat_keywords)

def is_dog_related(text):
    """Check if the text is dog-related"""
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(keyword in text for keyword in dog_keywords)

# ===== Step 4: Initialize storage =====
cat_posts = []
dog_posts = []
cat_comments = []
dog_comments = []
cat_posts_count = 0
dog_posts_count = 0

# ===== Step 5: Collect cat-related posts =====
print("\nStarting collection of cat-related posts...")
for sub in cat_subreddits:
    print(f"\nScraping cat-related posts from r/{sub}...")
    subreddit = reddit.subreddit(sub)
    posts_collected = 0
    posts_checked = 0

    for sort_method in ["new", "hot", "top"]:
        if cat_posts_count >= target_posts_per_type:
            print(f"Target number of cat posts reached ({target_posts_per_type})")
            break

        print(f"  Using sorting: {sort_method}")

        if sort_method == "new":
            posts_iterator = subreddit.new(limit=None)
        elif sort_method == "hot":
            posts_iterator = subreddit.hot(limit=None)
        elif sort_method == "top":
            posts_iterator = subreddit.top(limit=None)

        try:
            for post in posts_iterator:
                posts_checked += 1

                if any(p.get("post_id") == post.id for p in cat_posts):
                    continue

                full_text = f"{post.title} {post.selftext}".strip()
                is_related_to_cats = sub == "rescuecats" or is_cat_related(full_text)

                if is_adoption_related(full_text) and is_related_to_cats:
                    post_info = {
                        "post_id": post.id,
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext,
                        "combined_text": full_text,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        "url": f"https://reddit.com{post.permalink}",
                        "pet_type": "cat"
                    }
                    cat_posts.append(post_info)
                    posts_collected += 1
                    cat_posts_count += 1

                    try:
                        post.comments.replace_more(limit=0)
                        comment_count = 0

                        for comment in sorted(post.comments.list(), key=lambda x: x.score, reverse=True):
                            if comment_count >= max_comments_per_post:
                                break
                            if comment.body in ["[removed]", "[deleted]"]:
                                continue
                            comment_info = {
                                "comment_id": comment.id,
                                "post_id": post.id,
                                "subreddit": sub,
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                "pet_type": "cat"
                            }
                            cat_comments.append(comment_info)
                            comment_count += 1
                    except Exception as e:
                        print(f"Failed to fetch comments: {e}")

                    if posts_collected % print_interval == 0:
                        print(f"  ... Collected {posts_collected} cat-related posts (Checked {posts_checked})")
                        print(f"      Total: Cats: {cat_posts_count}, Dogs: {dog_posts_count}")

                    if cat_posts_count % 500 == 0:
                        os.makedirs("data", exist_ok=True)
                        pd.DataFrame(cat_posts).to_csv(f"data/cat_posts_temp_{cat_posts_count}.csv", index=False)
                        pd.DataFrame(cat_comments).to_csv(f"data/cat_comments_temp_{cat_posts_count}.csv", index=False)
                        print(f"  Temporary save: {cat_posts_count} cat posts, {len(cat_comments)} comments")

                if cat_posts_count >= target_posts_per_type:
                    break

                time.sleep(1)

        except Exception as e:
            print(f"Error fetching r/{sub} with sorting {sort_method}: {e}")
            continue

# ===== Step 6: Collect dog-related posts =====
print("\nStarting collection of dog-related posts...")
for sub in dog_subreddits:
    print(f"\nScraping dog-related posts from r/{sub}...")
    subreddit = reddit.subreddit(sub)
    posts_collected = 0
    posts_checked = 0

    for sort_method in ["new", "hot", "top"]:
        if dog_posts_count >= target_posts_per_type:
            print(f"Target number of dog posts reached ({target_posts_per_type})")
            break

        print(f"  Using sorting: {sort_method}")

        if sort_method == "new":
            posts_iterator = subreddit.new(limit=None)
        elif sort_method == "hot":
            posts_iterator = subreddit.hot(limit=None)
        elif sort_method == "top":
            posts_iterator = subreddit.top(limit=None)

        try:
            for post in posts_iterator:
                posts_checked += 1

                if any(p.get("post_id") == post.id for p in dog_posts):
                    continue

                full_text = f"{post.title} {post.selftext}".strip()
                is_related_to_dogs = sub == "rescuedogs" or is_dog_related(full_text)

                if is_adoption_related(full_text) and is_related_to_dogs:
                    post_info = {
                        "post_id": post.id,
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext,
                        "combined_text": full_text,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        "url": f"https://reddit.com{post.permalink}",
                        "pet_type": "dog"
                    }
                    dog_posts.append(post_info)
                    posts_collected += 1
                    dog_posts_count += 1

                    try:
                        post.comments.replace_more(limit=0)
                        comment_count = 0

                        for comment in sorted(post.comments.list(), key=lambda x: x.score, reverse=True):
                            if comment_count >= max_comments_per_post:
                                break
                            if comment.body in ["[removed]", "[deleted]"]:
                                continue
                            comment_info = {
                                "comment_id": comment.id,
                                "post_id": post.id,
                                "subreddit": sub,
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                "pet_type": "dog"
                            }
                            dog_comments.append(comment_info)
                            comment_count += 1
                    except Exception as e:
                        print(f"Failed to fetch comments: {e}")

                    if posts_collected % print_interval == 0:
                        print(f"  ... Collected {posts_collected} dog-related posts (Checked {posts_checked})")
                        print(f"      Total: Cats: {cat_posts_count}, Dogs: {dog_posts_count}")

                    if dog_posts_count % 500 == 0:
                        os.makedirs("data", exist_ok=True)
                        pd.DataFrame(dog_posts).to_csv(f"data/dog_posts_temp_{dog_posts_count}.csv", index=False)
                        pd.DataFrame(dog_comments).to_csv(f"data/dog_comments_temp_{dog_posts_count}.csv", index=False)
                        print(f"  Temporary save: {dog_posts_count} dog posts, {len(dog_comments)} comments")

                if dog_posts_count >= target_posts_per_type:
                    break

                time.sleep(1)

        except Exception as e:
            print(f"Error fetching r/{sub} with sorting {sort_method}: {e}")
            continue

# ===== Step 7: Save final merged results =====
all_posts = cat_posts + dog_posts
all_comments = cat_comments + dog_comments

posts_df = pd.DataFrame(all_posts)
comments_df = pd.DataFrame(all_comments)

os.makedirs("data", exist_ok=True)
posts_df.to_csv("data/all_pet_posts.csv", index=False)
comments_df.to_csv("data/all_pet_comments.csv", index=False)

pd.DataFrame(cat_posts).to_csv("data/cat_posts.csv", index=False)
pd.DataFrame(cat_comments).to_csv("data/cat_comments.csv", index=False)
pd.DataFrame(dog_posts).to_csv("data/dog_posts.csv", index=False)
pd.DataFrame(dog_comments).to_csv("data/dog_comments.csv", index=False)

# ===== Step 8: Print summary statistics =====
print("\nData collection complete:")
print(f"- Total posts: {len(all_posts)}")
print(f"  - Cat-related posts: {len(cat_posts)}")
print(f"  - Dog-related posts: {len(dog_posts)}")
print(f"- Total comments: {len(all_comments)}")
print(f"  - Cat comments: {len(cat_comments)}")
print(f"  - Dog comments: {len(dog_comments)}")

cat_avg = len(cat_comments) / len(cat_posts) if len(cat_posts) > 0 else 0
dog_avg = len(dog_comments) / len(dog_posts) if len(dog_posts) > 0 else 0
print("\nAverage comments per post:")
print(f"- Cats: {cat_avg:.2f}")
print(f"- Dogs: {dog_avg:.2f}")
print(f"- Overall: {len(all_comments) / len(all_posts) if len(all_posts) > 0 else 0:.2f}")

print("\nFiles saved in the data folder.")
