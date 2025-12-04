import sys
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf

def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit label_tweets.py <tweets.ndjson> <labels.json> <output_name>")
        sys.exit(1)

    tweets_path = sys.argv[1]
    labels_path = sys.argv[2]
    output_name = sys.argv[3]

    spark = SparkSession.builder.appName("TweetLabeler").getOrCreate()

    # -----------------------------
    # Load labels.json
    # -----------------------------
    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Normalize labels: "u123" → "123"
    labels_clean = {k.replace("u", ""): v for k, v in labels.items()}

    # Broadcast to workers
    bc_labels = spark.sparkContext.broadcast(labels_clean)

    # -----------------------------
    # Load tweets NDJSON
    # -----------------------------
    tweets_df = spark.read.json(tweets_path)

    # tweet_id is expected in "tweet_id" field
    # author_id likely numeric; convert to string for lookup
    tweets_df = tweets_df.withColumn("author_id_str", col("author_id").cast("string"))

    # -----------------------------
    # UDF for lookup
    # -----------------------------
    def lookup_label(author_id):
        return bc_labels.value.get(author_id, "unknown")

    lookup_udf = udf(lookup_label)

    # -----------------------------
    # Build labeled df
    # -----------------------------
    labeled = (
        tweets_df
        .withColumn("label", lookup_udf(col("author_id_str")))
        .select(
            col("id"),
            col("label"),
            col("text")
        )
    )

    # -----------------------------
    # Write as NDJSON
    # -----------------------------
    (
        labeled
        .coalesce(1)                          # produce 1 output file
        .write
        .mode("overwrite")
        .json(output_name)                   # Spark writes JSON lines automatically
    )

    spark.stop()


if __name__ == "__main__":
    main()
