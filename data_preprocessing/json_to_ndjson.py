import ijson
import json
import sys

input_path = sys.argv[1]  # e.g., Twibot-22/tweet_0.json
output_path = sys.argv[2] # e.g., tweet_0_with_labels.json

i = 0
with open(input_path, "r") as f, open(output_path, "w") as out:
    for tweet in ijson.items(f, "item"):   # iterate array items
        out.write(json.dumps(tweet, default=float) + "\n")
        i+=1
        if i%100000 == 0: print("processed",i,"entries")