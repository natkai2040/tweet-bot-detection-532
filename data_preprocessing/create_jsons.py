'''
Requires the Twibot-22 (https://twibot22.github.io/) to be installed and placed inside the /data directory
Given the Twibot-22 dataset (https://twibot22.github.io/),
generate new json files in our desired formatting, 
which stores only the tweet ID, user ID, label (bot/human) and text
'''

import json
from collections import defaultdict

twibot_data_dir = "data/Twibot-22"
# node_new.json contains user and tweet data
with open(twibot_data_dir + "/node_new.json", "r") as f:
    node_new = json.load(f)

# label_new.json contains labels (bot/human) for each user
with open(twibot_data_dir + "/label_new.json", "r") as f:
    label_new = json.load(f)

# ------------------------
# Tweet-level JSON (Each item in json is a single tweet)
# ------------------------
tweet_dict = {}
user_tweets = defaultdict(list)

# Iterate over all nodes (includes both user data nodes AND tweet nodes)
for tweet_id, tweet_data in node_new.items():
    # Skip tweet if it has no author (its a user data node)
    if "author_id" not in tweet_data:
        continue
    author_key = "u" + str(tweet_data["author_id"])

    # Skip tweet if it has no author ID (it cannot be labeled as bot/human)
    if author_key not in label_new:
        continue
    label = label_new[author_key]
    text = tweet_data.get("text", "")

    # Add to tweet dict
    tweet_dict[tweet_id] = {"label": label, "text": text}
    # Also collect for user-level aggregation
    user_tweets[author_key].append(text)

# Write to JSON
with open("data/tweets_with_labels.json", "w", encoding="utf-8") as f:
    json.dump(tweet_dict, f, ensure_ascii=False, indent=4)

print(f"Saved {len(tweet_dict)} tweets to tweets_with_labels.json")

# ------------------------
# User-level JSON (Each item in the JSON is a single user)
# ------------------------
user_dict = {}
for user_id, tweets in user_tweets.items():
    # Only generate object if the user is labeled
    if user_id in label_new:
        user_dict[user_id] = {
            "label": label_new[user_id],
            "tweets": tweets
        }

with open("data/users_with_tweets.json", "w", encoding="utf-8") as f:
    json.dump(user_dict, f, ensure_ascii=False, indent=4)

print(f"Saved {len(user_dict)} users with tweets to users_with_tweets.json")
