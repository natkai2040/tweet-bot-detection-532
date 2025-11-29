"""
This script provides a pipeline for both training and evaluating text with a Classification Model
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import map_from_arrays, array, lit, explode, col, when
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkapp.tweet_tokenizer import TweetTokenizerTransformer

def preprocess_data(spark: SparkSession, input_json):
    '''
    Given a JSON in the format defined in tokenizer.ipynb (key (tweet ID): {label: "human/bot", tweet: "tweetText}),
    generate a Spark DF which organizes the data into the following columns: [tweet_id, text, label]
    '''
    df = spark.read.option("multiline", "true").json(input_json)
    cols = df.columns

    df_map = df.select(
        map_from_arrays(
            array([lit(c) for c in cols]),
            array([col(c) for c in cols])
        ).alias("tweets")
    )

    #Flatten json such that a given tweet's object contains tweet_id and data columns
    df_flat = df_map.select(explode("tweets").alias("tweet_id", "data"))

    #Rename data columns into two seperate columns, text and label.
    df_formatted = df_flat.select(
        col("tweet_id"),
        col("data.text").alias("text"),
        col("data.label").alias("label")
    )

    #Encode the label of the data as follows:
    # HUMAN = 1
    # BOT = 0
    df_final = df_formatted.withColumn("label",
                                         when(col("label") == "human", 1).otherwise(0))

    print()
    print("PREPROCESSED DATA:")
    df_final.show(truncate=True)
    print()
    return df_final

def train_pipeline(spark, train_json):
    df_training = preprocess_data(spark, train_json)

    # First, tokenize to seperate string into list of semantic units
    # TweetTokenizer pays attention to text such as emoticons and arrows, and combines them into a single unit.
    tweet_tokenizer = TweetTokenizerTransformer(inputCol="text", outputCol="tokens_no_punc")

    #Removing stopwords
    # Stopwords = common words which don't denote meaning (i.e. a, the, is)
    # Removing these tokens may boost model performance
    remover = StopWordsRemover(inputCol="tokens_no_punc", outputCol="tokens_no_stops")

    # Convert tokens w/o stopwords to Term Frequency Vector -> Inverse Doc Frequency Vector
    cv = CountVectorizer(inputCol="tokens_no_stops", outputCol="raw_count")
    idf = IDF(inputCol="raw_count", outputCol="idf_features")
    lr = LogisticRegression(featuresCol="idf_features", labelCol="label")

    # Pipeline runs all of the stages inputted in sequence, which processes input string, then inputs it into our model.
    pipeline = Pipeline(stages=[tweet_tokenizer, remover, cv, idf, lr])

    # Training our pipeline model
    # StopWordsRemover is a Transformer, so it doesn't change as a result of training, it always removes a static set of stopwords.
    # CV constructs a vocabulary (a set of all unique tokens present in dataset) and uses it to construct term frequency vectors for each datapoint.
    # IDF learns the static-length vector of IDF weights for each datapoint, which it can create with the vector from CV.
    # LogisticRegression trains a model based on the IDF vector for each datapoint.
    model = pipeline.fit(df_training)
    return model

def inference_pipeline(spark, model, test_json):
    # Running inference on our pipeline model
    df_test = preprocess_data(spark, test_json)
    output_df = model.transform(df_test)
    # Save run to CSV, not optimal for large datasets but acceptable for small datasets.
    output_df.toPandas().to_csv("runs/test_output.csv", index=False)

    print()
    print("PREDICTION: (0.0 = bot, 1.0 = human)")
    output_df.select("text", "label", "prediction").show(truncate=True)
    print()
