'''
Single-cluster data preprocessing to create a single json file containing only the attributes we need.
(tweet_id, label, text), objects are keyed by tweet_id and contain label and text as attributes.
'''
import json
import ijson
import sys
from pathlib import Path

tweets_path = sys.argv[1]       # e.g., Twibot-22/tweet_0.json
labels_path = sys.argv[2]       # e.g., Twibot-22/labels_twibot_22a.json
output_path = sys.argv[3]       # e.g., tweet_0_with_labels.json

# --- Load labels (small file, normal json load) ---
with open(labels_path, "r") as f:
    labels = json.load(f)   # format: {"u123": "bot", ...}

# Normalize keys (strip 'u')
clean_labels = {k.lstrip("u"): v for k, v in labels.items()}

# --- Stream tweets from giant JSON array using ijson ---
result = {}

with open(tweets_path, "rb") as f:
    tweets = ijson.items(f, "item")   # each element of the JSON array

    for tweet in tweets:
        tweet_id = str(tweet.get("id") or tweet.get("tweet_id"))
        author_id = str(tweet.get("author_id"))
        text = tweet.get("text", "")

        label = clean_labels.get(author_id, None)

        result[tweet_id] = {
            "label": label,
            "text": text
        }

# --- Write final output ---
with open(output_path, "w") as out:
    json.dump(result, out, indent=2)
