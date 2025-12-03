import json
import string
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, size, avg, udf
from pyspark.sql.types import ArrayType, StringType
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# ============================================================
# 1. Start Spark
# ============================================================
spark = SparkSession.builder.appName("TweetAnalysisByLabel").getOrCreate()

# ============================================================
# 2. Load JSON (dict-of-dicts format)
# ============================================================
with open("tweets_with_labels.json", "r") as f:
    raw = json.load(f)

records = []
for tweet_id, data in raw.items():
    records.append({
        "tweet_id": tweet_id,
        "label": data["label"],
        "text": data["text"]
    })

df = spark.createDataFrame(records)
df.show(truncate=False)

# ============================================================
# 3. Set up tokenizer & stopwords
# ============================================================
tknzr = TweetTokenizer(preserve_case=False)

stops = set(stopwords.words("english"))
stops.update(string.punctuation)
stops.update({'…', '’', '...', '“', '”', '‘'})

# ============================================================
# 4. Define tokenization UDF that removes stopwords
# ============================================================
def tokenize_remove_stops(text):
    tokens = tknzr.tokenize(text)
    return [t for t in tokens if t not in stops]

tokenize_udf = udf(tokenize_remove_stops, ArrayType(StringType()))

df_tokens = df.withColumn("tokens", tokenize_udf(col("text")))

# ============================================================
# 5. Function to compute ngrams
# ============================================================
def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

make_ngrams_udf = udf(make_ngrams, ArrayType(StringType()))

df_tokens = df_tokens.withColumn("bigrams", make_ngrams_udf(col("tokens"), lit(2)))
df_tokens = df_tokens.withColumn("trigrams", make_ngrams_udf(col("tokens"), lit(3)))

# ============================================================
# 6. Compute stats per label
# ============================================================
labels = [row['label'] for row in df.select("label").distinct().collect()]

for lbl in labels:
    print(f"\n=== Statistics for {lbl} tweets ===")
    df_lbl = df_tokens.filter(col("label") == lbl)
    
    # Dataset size
    size_lbl = df_lbl.count()
    print(f"Dataset size: {size_lbl}")
    
    # Most frequent tokens
    tokens_exploded = df_lbl.select(explode(col("tokens")).alias("token"))
    token_freq = tokens_exploded.groupBy("token").count().orderBy(col("count").desc())
    print("\nTop tokens:")
    token_freq.show(10, truncate=False)
    
    # Most frequent bigrams
    bigrams_exploded = df_lbl.select(explode(col("bigrams")).alias("bigram"))
    bigram_freq = bigrams_exploded.groupBy("bigram").count().orderBy(col("count").desc())
    print("\nTop bigrams:")
    bigram_freq.show(10, truncate=False)
    
    # Most frequent trigrams
    trigrams_exploded = df_lbl.select(explode(col("trigrams")).alias("trigram"))
    trigram_freq = trigrams_exploded.groupBy("trigram").count().orderBy(col("count").desc())
    print("\nTop trigrams:")
    trigram_freq.show(10, truncate=False)
    
    # Mean tweet length
    mean_len = df_lbl.withColumn("tweet_length", size(col("tokens"))).select(avg("tweet_length")).first()[0]
    print(f"Mean tweet length (tokens): {mean_len}")

# ============================================================
# 7. Stop Spark
# ============================================================
spark.stop()
