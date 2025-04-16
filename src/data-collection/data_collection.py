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
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# ===== Step 2: Configuration =====
subreddits = [
    "cats", "dogs", "CatAdvice", "DogAdvice", "PetAdvice",
    "Adoptapet", "AdoptMyPet", "puppyAdoption", "rescuedogs", "aww"
]
num_submissions_per_sub = 2000  # 控制每个 subreddit 抓多少贴
print_interval = 100  # 每抓几个帖子打印一次进度

# ===== Step 3: Adoption-related keyword list =====
adoption_keywords = [
    "adopt", "adoption", "adopted", "adopting",
    "rehome", "re-home", "rescue", "rescued",
    "foster", "forever home", "shelter"
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
                    all_comments.append({
                        "subreddit": sub,
                        "post_id": post.id,
                        "post_title": post.title,
                        "comment_id": comment.id,
                        "body": comment.body,
                        "score": comment.score,
                        "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat(),
                        "permalink": f"https://www.reddit.com{post.permalink}"
                    })
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
print(f"Adoption-related comments saved: {len(adoption_df)} rows → data/raw-data/adoption_comments.csv")
