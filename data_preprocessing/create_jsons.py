from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
import json

with open("Twibot-22/node_new.json", "r") as f:
    node_new = json.load(f)

with open("Twibot-22/label_new.json", "r") as f:
    label_new = json.load(f)

spark = SparkSession.builder.appName("TweetsToUsersJSON").getOrCreate()

# ------------------------
# Prepare tweet records
# ------------------------
tweet_records = []
for tweet_id, tweet_data in node_new.items():
    if "author_id" not in tweet_data:
        continue
    author_key = "u" + str(tweet_data["author_id"])
    if author_key not in label_new:
        continue
    label = label_new[author_key]
    text = tweet_data.get("text", "")
    tweet_records.append((tweet_id, author_key, label, text))  # include tweet_id

df = spark.createDataFrame(tweet_records, ["tweet_id", "user_id", "label", "text"])

# ------------------------
# Tweet-level JSON
# ------------------------
tweet_dict = {row["tweet_id"]: {"label": row["label"], "text": row["text"]} for row in df.collect()}

with open("tweets_with_labels.json", "w", encoding="utf-8") as f:
    json.dump(tweet_dict, f, ensure_ascii=False, indent=4)

print(f"Saved {len(tweet_dict)} tweets to tweets_with_labels.json")

# ------------------------
# User-level JSON
# ------------------------
df_users = df.groupBy("user_id", "label").agg(collect_list("text").alias("tweets"))

user_records = {row["user_id"]: {"label": row["label"], "tweets": tuple(row["tweets"])} for row in df_users.collect()}

with open("users_with_tweets.json", "w", encoding="utf-8") as f:
    json.dump(user_records, f, ensure_ascii=False, indent=4)

print(f"Saved {len(user_records)} users with tweets to users_with_tweets.json")

spark.stop()
