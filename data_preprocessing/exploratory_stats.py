import json
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, explode, size, avg, udf, expr
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.pipeline import TWEET_REGEX, selective_lowercase_udf

# Start Spark
spark = SparkSession.builder.appName("TweetAnalysisByLabel").getOrCreate()

# Load JSON (dict-of-dicts format)
data_dir = "data"
with open(data_dir + "/tweets_with_labels.json", "r") as f:
    raw = json.load(f)

records = []
for tweet_id, data in raw.items():
    records.append({
        "tweet_id": tweet_id,
        "label": data["label"],
        "text": data["text"]
    })

df = spark.createDataFrame(records)
df = df.withColumn("text", selective_lowercase_udf(col("text")))

# Tokenize using the same RegexTokenizer
regex_tokenizer = RegexTokenizer(
    inputCol="text",
    outputCol="tokens_raw",
    pattern=TWEET_REGEX,
    gaps=False,
    toLowercase=False 
)

df_tokens = regex_tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(
    inputCol="tokens_raw",
    outputCol="tokens_no_stops"
)

df_tokens = remover.transform(df_tokens)
# Additional Regex Filters keeps only Latin words + emojis
df_tokens = df_tokens.withColumn(
    "tokens",
    expr(
        "filter(tokens_no_stops, t -> "
        "t rlike '^[a-z]{2,}$' OR "          # Latin words
        "t rlike '^\\\\p{So}$' OR "          # emojis / symbols
        "t rlike '^#[a-z0-9_]{1,}$' OR "     # hashtags
        "t rlike '^@[a-z0-9_]{1,}$' OR "     # mentions
        "t rlike '^http\\S+$'"                # URLs
        ")"
    )
)

# Function to compute ngrams
def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

make_ngrams_udf = udf(make_ngrams, ArrayType(StringType()))

df_tokens = df_tokens.withColumn("bigrams", make_ngrams_udf(col("tokens"), lit(2)))
df_tokens = df_tokens.withColumn("trigrams", make_ngrams_udf(col("tokens"), lit(3)))

# Compute stats per label
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
    token_freq.show(50, truncate=False)
    
    # Most frequent bigrams
    bigrams_exploded = df_lbl.select(explode(col("bigrams")).alias("bigram"))
    bigram_freq = bigrams_exploded.groupBy("bigram").count().orderBy(col("count").desc())
    print("\nTop bigrams:")
    bigram_freq.show(50, truncate=False)
    
    # Most frequent trigrams
    trigrams_exploded = df_lbl.select(explode(col("trigrams")).alias("trigram"))
    trigram_freq = trigrams_exploded.groupBy("trigram").count().orderBy(col("count").desc())
    print("\nTop trigrams:")
    trigram_freq.show(50, truncate=False)
    
    # Mean tweet length
    mean_len = df_lbl.withColumn("tweet_length", size(col("tokens"))).select(avg("tweet_length")).first()[0]
    print(f"Mean tweet length (tokens): {mean_len}")

spark.stop()
