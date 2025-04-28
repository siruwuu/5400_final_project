"""
reddit_pet_expanded_scraper.py - Enhanced Reddit Pet Adoption Data Collector

This script extends the previous data collection efforts that focused on rescue-specific
subreddits (r/rescuecats and r/rescuedogs) by incorporating a broader range of communities
to reach the target dataset size. After discovering that the rescue-specific communities
alone provided insufficient data volume, this enhanced version:

1. Builds upon previously collected data, loading existing datasets if available
   and continuing collection to reach the target of 5000 posts for each pet type

2. Expands the collection scope to include:
   - Pet-specific subreddits: r/cats, r/CatAdvice, r/kittens, r/catpictures, r/catcare,
     r/dogs, r/dogpictures, r/dogswithjobs, r/dogcare, r/dogs_getting_dogs
   - Location-based subreddits for major US cities to find local adoption posts

3. Implements a multi-strategy approach to content discovery:
   - Targeted keyword searches using pet and adoption-specific phrases
   - Browse-based collection across multiple sorting methods (new, hot, top)
   - Progressive filtering through multiple classification stages

4. Features a sophisticated classification system:
   - Uses extensive pet-specific adoption phrases for precise matching
   - Employs contextual analysis for ambiguous posts
   - Handles mixed-content posts through hierarchical decision rules

5. Incorporates robust error handling and incremental saving:
   - Saves progress at regular intervals (every 500 posts after initial 1500)
   - Creates temporary checkpoints after each subreddit
   - Handles interruptions and exceptions with automatic data preservation

6. Maintains consistent data structure with the previous collection:
   - Preserves the same fields and format as the rescue-specific datasets
   - Ensures compatibility for merging with previously collected data

This expanded approach enables the collection of a comprehensive dataset that
includes both specialized rescue content and broader pet adoption discussions
across various communities, providing a more diverse and representative sample
while maintaining the focus on pet adoption.


Requirements:
    - .env file with Reddit API credentials (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT)
    - praw, pandas, python-dotenv packages
    - Existing data in 'data' directory (optional)

"""

import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
import time
import random

# ===== Step 1: Load environment variables =====
load_dotenv()

# Create Reddit instance using environment variables
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)


# ===== Step 2: Load existing data if available =====
def load_existing_data():
    try:
        cat_posts_df = pd.read_csv("data/cat_posts.csv")
        dog_posts_df = pd.read_csv("data/dog_posts.csv")
        cat_comments_df = pd.read_csv("data/cat_comments.csv")
        dog_comments_df = pd.read_csv("data/dog_comments.csv")

        cat_posts = cat_posts_df.to_dict("records")
        dog_posts = dog_posts_df.to_dict("records")
        cat_comments = cat_comments_df.to_dict("records")
        dog_comments = dog_comments_df.to_dict("records")

        print("Existing data loaded:")
        print(f"- Cat posts: {len(cat_posts)}")
        print(f"- Dog posts: {len(dog_posts)}")
        print(f"- Cat comments: {len(cat_comments)}")
        print(f"- Dog comments: {len(dog_comments)}")

        return cat_posts, dog_posts, cat_comments, dog_comments
    except:
        print("No existing data found. Starting fresh.")
        return [], [], [], []


cat_posts, dog_posts, cat_comments, dog_comments = load_existing_data()

# Count current data
cat_posts_count = len(cat_posts)
dog_posts_count = len(dog_posts)

# ===== Step 3: Subreddit Configuration =====
cat_subreddits = ["cats", "CatAdvice", "kittens", "catpictures", "catcare"]
dog_subreddits = ["dogs", "dogpictures", "dogswithjobs", "dogcare", "dogs_getting_dogs"]
location_subreddits = [
    "Indianapolis",
    "washingtondc",
    "chicago",
    "nyc",
    "LosAngeles",
    "sanfrancisco",
    "seattle",
    "portland",
    "austin",
    "boston",
    "atlanta",
    "dallas",
    "houston",
    "phoenix",
    "philadelphia",
]

# Target counts
target_cat_posts = 5000
target_dog_posts = 5000
max_comments_per_post = 10
print_interval = 50

# Save every 500 posts after the initial 1500
save_interval = 500
next_cat_save = 1500 + save_interval if cat_posts_count >= 1500 else 1500
next_dog_save = 1500 + save_interval if dog_posts_count >= 1500 else 1500

# ===== Step 4: Keywords =====
# Cat-specific adoption phrases
cat_adoption_phrases = [
    "adopt cat",
    "cat adoption",
    "adopt a cat",
    "adopting cat",
    "adopt kitten",
    "kitten adoption",
    "rescue cat",
    "cat rescue",
    "rehome cat",
    "cat needs home",
    "foster cat",
    "cat fostering",
    "shelter cat",
    "cat shelter",
    "forever home cat",
    "cat looking for home",
    "cat available for adoption",
]

# Dog-specific adoption phrases
dog_adoption_phrases = [
    "adopt dog",
    "dog adoption",
    "adopt a dog",
    "adopting dog",
    "adopt puppy",
    "puppy adoption",
    "rescue dog",
    "dog rescue",
    "rehome dog",
    "dog needs home",
    "foster dog",
    "dog fostering",
    "shelter dog",
    "dog shelter",
    "forever home dog",
    "dog looking for home",
    "dog available for adoption",
    "Great Dog Needs A Home",
]

# General adoption phrases
general_adoption_phrases = [
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
    "looking for home",
    "needs home",
    "new home",
    "For adoption",
    "Need homes",
    "In need of rehoming",
    "available",
    "seeking",
]

# Keywords for identifying cats and dogs
cat_keywords = [
    "cat",
    "cats",
    "kitten",
    "kittens",
    "kitty",
    "feline",
    "tabby",
    "calico",
    "siamese",
    "persian",
    "she-cat",
    "tomcat",
    "meow",
]

dog_keywords = [
    "dog",
    "dogs",
    "puppy",
    "puppies",
    "pup",
    "pooch",
    "canine",
    "hound",
    "retriever",
    "shepherd",
    "terrier",
    "labrador",
    "beagle",
    "collie",
    "poodle",
    "bark",
    "woof",
]


# ===== Step 5: Detection Functions =====
def is_cat_related(text):
    """Check if text is related to cats"""
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(keyword in text for keyword in cat_keywords)


def is_dog_related(text):
    """Check if text is related to dogs"""
    if not isinstance(text, str):
        return False
    text = text.lower()
    return any(keyword in text for keyword in dog_keywords)


def is_adoption_related(text, pet_type=None):
    """Check if text is related to pet adoption"""
    if not isinstance(text, str):
        return False

    text = text.lower()

    # Check against pet-specific adoption phrases first
    if pet_type == "cat":
        if any(phrase in text for phrase in cat_adoption_phrases):
            return True
    elif pet_type == "dog":
        if any(phrase in text for phrase in dog_adoption_phrases):
            return True

    # Then check general adoption phrases
    return any(phrase in text for phrase in general_adoption_phrases)


# ===== Step 6: Main Data Collection Function =====
def collect_pet_posts(subreddit_name, pet_type="both", max_posts=100):
    """Collect pet-related posts from the specified subreddit"""
    print(f"\nCollecting data from r/{subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    posts_collected = 0
    posts_checked = 0
    new_cat_posts = []
    new_dog_posts = []
    new_cat_comments = []
    new_dog_comments = []

    global cat_posts, dog_posts, cat_posts_count, dog_posts_count, next_cat_save, next_dog_save
    collected_post_ids = set(p.get("post_id") for p in cat_posts + dog_posts)

    # Determine which phrases to search based on pet_type
    search_phrases = []
    if pet_type == "cat":
        search_phrases = cat_adoption_phrases + general_adoption_phrases
    elif pet_type == "dog":
        search_phrases = dog_adoption_phrases + general_adoption_phrases
    else:  # both
        search_phrases = (
            cat_adoption_phrases + dog_adoption_phrases + general_adoption_phrases
        )

    # 1. Search using appropriate adoption phrases
    for phrase in search_phrases:
        if (pet_type in ["cat", "both"]) and cat_posts_count + len(
            new_cat_posts
        ) >= target_cat_posts:
            break
        if (pet_type in ["dog", "both"]) and dog_posts_count + len(
            new_dog_posts
        ) >= target_dog_posts:
            break

        try:
            print(f"  Searching '{phrase}' in r/{subreddit_name}...")

            for post in subreddit.search(phrase, sort="relevance", limit=100):
                posts_checked += 1
                if post.id in collected_post_ids:
                    continue

                full_text = f"{post.title} {post.selftext}".strip()
                is_related_to_cats = is_cat_related(full_text)
                is_related_to_dogs = is_dog_related(full_text)

                # Skip if not related to cats or dogs
                if not (is_related_to_cats or is_related_to_dogs):
                    continue

                # Determine post type based on content and search constraints
                post_pet_type = None

                # For cat-specific subreddits or searches
                if pet_type == "cat":
                    if is_related_to_cats:
                        post_pet_type = "cat"
                # For dog-specific subreddits or searches
                elif pet_type == "dog":
                    if is_related_to_dogs:
                        post_pet_type = "dog"
                # For mixed subreddits, assign based on content
                else:
                    if is_related_to_cats and not is_related_to_dogs:
                        post_pet_type = "cat"
                    elif is_related_to_dogs and not is_related_to_cats:
                        post_pet_type = "dog"
                    elif is_related_to_cats and is_related_to_dogs:
                        # If truly ambiguous, classify based on the search phrase
                        if any(
                            cat_phrase in phrase.lower()
                            for cat_phrase in cat_adoption_phrases
                        ):
                            post_pet_type = "cat"
                        elif any(
                            dog_phrase in phrase.lower()
                            for dog_phrase in dog_adoption_phrases
                        ):
                            post_pet_type = "dog"
                        else:
                            # Last resort: random assignment
                            post_pet_type = random.choice(["cat", "dog"])

                # Check if post is adoption related for the specific pet type
                if post_pet_type and is_adoption_related(full_text, post_pet_type):
                    post_info = {
                        "post_id": post.id,
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "selftext": post.selftext,
                        "combined_text": full_text,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": (
                            datetime.fromtimestamp(post.created_utc).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        ),
                        "url": f"https://reddit.com{post.permalink}",
                        "pet_type": post_pet_type,
                    }

                    if post_pet_type == "cat":
                        new_cat_posts.append(post_info)
                    else:
                        new_dog_posts.append(post_info)

                    collected_post_ids.add(post.id)
                    posts_collected += 1

                    # Collect top comments
                    try:
                        post.comments.replace_more(limit=0)
                        comment_count = 0
                        for comment in sorted(
                            post.comments.list(), key=lambda x: x.score, reverse=True
                        ):
                            if comment_count >= max_comments_per_post:
                                break
                            if comment.body in ["[removed]", "[deleted]"]:
                                continue

                            comment_info = {
                                "comment_id": comment.id,
                                "post_id": post.id,
                                "subreddit": subreddit_name,
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": (
                                    datetime.fromtimestamp(
                                        comment.created_utc
                                    ).strftime("%Y-%m-%d %H:%M:%S")
                                ),
                                "pet_type": post_pet_type,
                            }

                            if post_pet_type == "cat":
                                new_cat_comments.append(comment_info)
                            else:
                                new_dog_comments.append(comment_info)

                            comment_count += 1

                    except Exception as e:
                        print(f"Failed to fetch comments: {e}")

                    # Check if we need to save incremental data
                    if post_pet_type == "cat":
                        if cat_posts_count + len(new_cat_posts) >= next_cat_save:
                            temp_cat_posts = cat_posts + new_cat_posts
                            temp_cat_comments = cat_comments + new_cat_comments
                            pd.DataFrame(temp_cat_posts).to_csv(
                                f"data/temp/cat_posts_{cat_posts_count + len(new_cat_posts)}.csv",
                                index=False,
                            )
                            pd.DataFrame(temp_cat_comments).to_csv(
                                f"data/temp/cat_comments_{cat_posts_count + len(new_cat_posts)}.csv",
                                index=False,
                            )
                            print(
                                f"  ** Saved incremental cat data at {cat_posts_count + len(new_cat_posts)} posts **"
                            )
                            next_cat_save += save_interval

                    elif post_pet_type == "dog":
                        if dog_posts_count + len(new_dog_posts) >= next_dog_save:
                            temp_dog_posts = dog_posts + new_dog_posts
                            temp_dog_comments = dog_comments + new_dog_comments
                            pd.DataFrame(temp_dog_posts).to_csv(
                                f"data/temp/dog_posts_{dog_posts_count + len(new_dog_posts)}.csv",
                                index=False,
                            )
                            pd.DataFrame(temp_dog_comments).to_csv(
                                f"data/temp/dog_comments_{dog_posts_count + len(new_dog_posts)}.csv",
                                index=False,
                            )
                            print(
                                f"  ** Saved incremental dog data at {dog_posts_count + len(new_dog_posts)} posts **"
                            )
                            next_dog_save += save_interval

                    if posts_collected % print_interval == 0:
                        total_cats = cat_posts_count + len(new_cat_posts)
                        total_dogs = dog_posts_count + len(new_dog_posts)
                        print(
                            f"  ... {posts_collected} posts collected (checked {posts_checked})"
                        )
                        print(
                            f"      Current totals - Cats: {total_cats}, Dogs: {total_dogs}"
                        )

                # Check if targets have been reached
                if (
                    cat_posts_count + len(new_cat_posts) >= target_cat_posts
                    and dog_posts_count + len(new_dog_posts) >= target_dog_posts
                ):
                    print("Reached all target post counts.")
                    break
                if (
                    pet_type == "cat"
                    and cat_posts_count + len(new_cat_posts) >= target_cat_posts
                ):
                    print(f"Reached target cat post count ({target_cat_posts})")
                    break
                if (
                    pet_type == "dog"
                    and dog_posts_count + len(new_dog_posts) >= target_dog_posts
                ):
                    print(f"Reached target dog post count ({target_dog_posts})")
                    break

                time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Error during search '{phrase}': {e}")
            continue

    # 2. Browse by sort methods (new/hot/top)
    for sort_method in ["new", "hot", "top"]:
        if (pet_type in ["cat", "both"]) and cat_posts_count + len(
            new_cat_posts
        ) >= target_cat_posts:
            break
        if (pet_type in ["dog", "both"]) and dog_posts_count + len(
            new_dog_posts
        ) >= target_dog_posts:
            break

        try:
            print(f"  Browsing r/{subreddit_name} sorted by {sort_method}...")

            if sort_method == "new":
                posts_iterator = subreddit.new(limit=100)
            elif sort_method == "hot":
                posts_iterator = subreddit.hot(limit=100)
            elif sort_method == "top":
                posts_iterator = subreddit.top(limit=100)

            for post in posts_iterator:
                posts_checked += 1
                if post.id in collected_post_ids:
                    continue

                full_text = f"{post.title} {post.selftext}".strip()

                # Skip if not adoption related based on pet type
                if pet_type == "cat" and not is_adoption_related(full_text, "cat"):
                    continue
                elif pet_type == "dog" and not is_adoption_related(full_text, "dog"):
                    continue
                elif pet_type == "both" and not is_adoption_related(full_text):
                    continue

                is_related_to_cats = is_cat_related(full_text)
                is_related_to_dogs = is_dog_related(full_text)

                if not (is_related_to_cats or is_related_to_dogs):
                    continue

                # Determine post type with the same rules as in search
                post_pet_type = None
                if pet_type == "cat":
                    if is_related_to_cats:
                        post_pet_type = "cat"
                elif pet_type == "dog":
                    if is_related_to_dogs:
                        post_pet_type = "dog"
                else:
                    if is_related_to_cats and not is_related_to_dogs:
                        post_pet_type = "cat"
                    elif is_related_to_dogs and not is_related_to_cats:
                        post_pet_type = "dog"
                    elif is_related_to_cats and is_related_to_dogs:
                        # For mixed content in browsing, prioritize based on subreddit type
                        if any(
                            cat_sub in subreddit_name.lower()
                            for cat_sub in ["cat", "kitten"]
                        ):
                            post_pet_type = "cat"
                        elif any(
                            dog_sub in subreddit_name.lower()
                            for dog_sub in ["dog", "puppy"]
                        ):
                            post_pet_type = "dog"
                        else:
                            post_pet_type = random.choice(["cat", "dog"])

                if post_pet_type:
                    post_info = {
                        "post_id": post.id,
                        "subreddit": subreddit_name,
                        "title": post.title,
                        "selftext": post.selftext,
                        "combined_text": full_text,
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": (
                            datetime.fromtimestamp(post.created_utc).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        ),
                        "url": f"https://reddit.com{post.permalink}",
                        "pet_type": post_pet_type,
                    }

                    if post_pet_type == "cat":
                        new_cat_posts.append(post_info)
                    else:
                        new_dog_posts.append(post_info)

                    collected_post_ids.add(post.id)
                    posts_collected += 1

                    try:
                        post.comments.replace_more(limit=0)
                        comment_count = 0
                        for comment in sorted(
                            post.comments.list(), key=lambda x: x.score, reverse=True
                        ):
                            if comment_count >= max_comments_per_post:
                                break
                            if comment.body in ["[removed]", "[deleted]"]:
                                continue

                            comment_info = {
                                "comment_id": comment.id,
                                "post_id": post.id,
                                "subreddit": subreddit_name,
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": (
                                    datetime.fromtimestamp(
                                        comment.created_utc
                                    ).strftime("%Y-%m-%d %H:%M:%S")
                                ),
                                "pet_type": post_pet_type,
                            }

                            if post_pet_type == "cat":
                                new_cat_comments.append(comment_info)
                            else:
                                new_dog_comments.append(comment_info)

                            comment_count += 1

                    except Exception as e:
                        print(f"Failed to fetch comments: {e}")

                    # Check if we need to save incremental data
                    if post_pet_type == "cat":
                        if cat_posts_count + len(new_cat_posts) >= next_cat_save:
                            temp_cat_posts = cat_posts + new_cat_posts
                            temp_cat_comments = cat_comments + new_cat_comments
                            pd.DataFrame(temp_cat_posts).to_csv(
                                f"data/temp/cat_posts_{cat_posts_count + len(new_cat_posts)}.csv",
                                index=False,
                            )
                            pd.DataFrame(temp_cat_comments).to_csv(
                                f"data/temp/cat_comments_{cat_posts_count + len(new_cat_posts)}.csv",
                                index=False,
                            )
                            print(
                                f"  ** Saved incremental cat data at {cat_posts_count + len(new_cat_posts)} posts **"
                            )
                            next_cat_save += save_interval

                    elif post_pet_type == "dog":
                        if dog_posts_count + len(new_dog_posts) >= next_dog_save:
                            temp_dog_posts = dog_posts + new_dog_posts
                            temp_dog_comments = dog_comments + new_dog_comments
                            pd.DataFrame(temp_dog_posts).to_csv(
                                f"data/temp/dog_posts_{dog_posts_count + len(new_dog_posts)}.csv",
                                index=False,
                            )
                            pd.DataFrame(temp_dog_comments).to_csv(
                                f"data/temp/dog_comments_{dog_posts_count + len(new_dog_posts)}.csv",
                                index=False,
                            )
                            print(
                                f"  ** Saved incremental dog data at {dog_posts_count + len(new_dog_posts)} posts **"
                            )
                            next_dog_save += save_interval

                    if posts_collected % print_interval == 0:
                        total_cats = cat_posts_count + len(new_cat_posts)
                        total_dogs = dog_posts_count + len(new_dog_posts)
                        print(
                            f"  ... {posts_collected} posts collected (checked {posts_checked})"
                        )
                        print(
                            f"      Current totals - Cats: {total_cats}, Dogs: {total_dogs}"
                        )

                if (
                    cat_posts_count + len(new_cat_posts) >= target_cat_posts
                    and dog_posts_count + len(new_dog_posts) >= target_dog_posts
                ):
                    break
                if (
                    pet_type == "cat"
                    and cat_posts_count + len(new_cat_posts) >= target_cat_posts
                ):
                    break
                if (
                    pet_type == "dog"
                    and dog_posts_count + len(new_dog_posts) >= target_dog_posts
                ):
                    break

                time.sleep(1)

        except Exception as e:
            print(f"Error browsing sorted by {sort_method}: {e}")
            continue

    print(f"Finished r/{subreddit_name}:")
    print(f"- New cat posts: {len(new_cat_posts)}")
    print(f"- New dog posts: {len(new_dog_posts)}")
    print(f"- New cat comments: {len(new_cat_comments)}")
    print(f"- New dog comments: {len(new_dog_comments)}")

    return new_cat_posts, new_dog_posts, new_cat_comments, new_dog_comments


# ===== Step 7: Main Script =====
os.makedirs("data/temp", exist_ok=True)

try:
    print("\n=== Collecting from cat-related subreddits ===")
    for sub in cat_subreddits:
        if cat_posts_count >= target_cat_posts:
            print(f"Target cat post count reached ({target_cat_posts})")
            break
        new_cats, new_dogs, new_cat_cmts, new_dog_cmts = collect_pet_posts(
            sub, pet_type="cat", max_posts=200
        )
        cat_posts.extend(new_cats)
        dog_posts.extend(new_dogs)
        cat_comments.extend(new_cat_cmts)
        dog_comments.extend(new_dog_cmts)
        cat_posts_count = len(cat_posts)
        dog_posts_count = len(dog_posts)

        if new_cats or new_dogs:
            pd.DataFrame(cat_posts).to_csv(
                f"data/temp/cat_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_posts).to_csv(
                f"data/temp/dog_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(cat_comments).to_csv(
                f"data/temp/cat_comments_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_comments).to_csv(
                f"data/temp/dog_comments_after_{sub}.csv", index=False
            )
            print(
                f"Temporary save complete: Cats: {cat_posts_count}, Dogs: {dog_posts_count}"
            )

    print("\n=== Collecting from dog-related subreddits ===")
    for sub in dog_subreddits:
        if dog_posts_count >= target_dog_posts:
            print(f"Target dog post count reached ({target_dog_posts})")
            break
        new_cats, new_dogs, new_cat_cmts, new_dog_cmts = collect_pet_posts(
            sub, pet_type="dog", max_posts=200
        )
        cat_posts.extend(new_cats)
        dog_posts.extend(new_dogs)
        cat_comments.extend(new_cat_cmts)
        dog_comments.extend(new_dog_cmts)
        cat_posts_count = len(cat_posts)
        dog_posts_count = len(dog_posts)

        if new_cats or new_dogs:
            pd.DataFrame(cat_posts).to_csv(
                f"data/temp/cat_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_posts).to_csv(
                f"data/temp/dog_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(cat_comments).to_csv(
                f"data/temp/cat_comments_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_comments).to_csv(
                f"data/temp/dog_comments_after_{sub}.csv", index=False
            )
            print(
                f"Temporary save complete: Cats: {cat_posts_count}, Dogs: {dog_posts_count}"
            )

    print("\n=== Collecting from location-based subreddits ===")
    for sub in location_subreddits:
        if cat_posts_count >= target_cat_posts and dog_posts_count >= target_dog_posts:
            print("All post targets reached. Stopping.")
            break
        new_cats, new_dogs, new_cat_cmts, new_dog_cmts = collect_pet_posts(
            sub, pet_type="both", max_posts=100
        )
        cat_posts.extend(new_cats)
        dog_posts.extend(new_dogs)
        cat_comments.extend(new_cat_cmts)
        dog_comments.extend(new_dog_cmts)
        cat_posts_count = len(cat_posts)
        dog_posts_count = len(dog_posts)

        if new_cats or new_dogs:
            pd.DataFrame(cat_posts).to_csv(
                f"data/temp/cat_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_posts).to_csv(
                f"data/temp/dog_posts_after_{sub}.csv", index=False
            )
            pd.DataFrame(cat_comments).to_csv(
                f"data/temp/cat_comments_after_{sub}.csv", index=False
            )
            pd.DataFrame(dog_comments).to_csv(
                f"data/temp/dog_comments_after_{sub}.csv", index=False
            )
            print(
                f"Temporary save complete: Cats: {cat_posts_count}, Dogs: {dog_posts_count}"
            )

    # ===== Step 8: Save Final Result =====
    pd.DataFrame(cat_posts).to_csv("data/cat_posts.csv", index=False)
    pd.DataFrame(dog_posts).to_csv("data/dog_posts.csv", index=False)
    pd.DataFrame(cat_comments).to_csv("data/cat_comments.csv", index=False)
    pd.DataFrame(dog_comments).to_csv("data/dog_comments.csv", index=False)

    all_posts = cat_posts + dog_posts
    all_comments = cat_comments + dog_comments

    pd.DataFrame(all_posts).to_csv("data/all_pet_posts.csv", index=False)
    pd.DataFrame(all_comments).to_csv("data/all_pet_comments.csv", index=False)

    # ===== Step 9: Final Stats =====
    print("\nData collection complete:")
    print(f"- Total posts: {len(all_posts)}")
    print(f"  - Cat posts: {cat_posts_count}")
    print(f"  - Dog posts: {dog_posts_count}")
    print(f"- Total comments: {len(all_comments)}")
    print(f"  - Cat comments: {len(cat_comments)}")
    print(f"  - Dog comments: {len(dog_comments)}")

    cat_avg = len(cat_comments) / cat_posts_count if cat_posts_count > 0 else 0
    dog_avg = len(dog_comments) / dog_posts_count if dog_posts_count > 0 else 0
    print("\nAverage comments per post:")
    print(f"- Cat posts: {cat_avg:.2f}")
    print(f"- Dog posts: {dog_avg:.2f}")
    print(
        f"- Overall: {len(all_comments)/len(all_posts) if len(all_posts) > 0 else 0:.2f}"
    )

except KeyboardInterrupt:
    print("\nProcess interrupted by user.")
    pd.DataFrame(cat_posts).to_csv("data/cat_posts_interrupted.csv", index=False)
    pd.DataFrame(dog_posts).to_csv("data/dog_posts_interrupted.csv", index=False)
    pd.DataFrame(cat_comments).to_csv("data/cat_comments_interrupted.csv", index=False)
    pd.DataFrame(dog_comments).to_csv("data/dog_comments_interrupted.csv", index=False)
    print("Progress saved.")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    pd.DataFrame(cat_posts).to_csv("data/cat_posts_error.csv", index=False)
    pd.DataFrame(dog_posts).to_csv("data/dog_posts_error.csv", index=False)
    pd.DataFrame(cat_comments).to_csv("data/cat_comments_error.csv", index=False)
    pd.DataFrame(dog_comments).to_csv("data/dog_comments_error.csv", index=False)
    print("Error progress saved.")
