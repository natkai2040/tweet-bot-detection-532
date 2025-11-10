"""
This script simulates a pipeline for both training and evaluating text with a LogisticRegression Model
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark_tweet_tokenizer import TweetTokenizerTransformer

spark = SparkSession.builder.appName("BotClassifier").getOrCreate()

#Sourced from https://www.nltk.org/api/nltk.tokenize.casual.html
#label = 0.0 is bot
#label = 1.0 is human
dummy_training_data = [
    (0, "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <-- @remy: This is waaaaayyyy too much for you!!!!!!", 0.0),
    (1, "This is a test datapoint, arrows arrows arrows", 0.0),
    (2, ":-) wowowowowowowowow @remy hello", 1.0)
]
dummy_test_data = [
    (0, "Hey ;) @remy", 0.0)
]
df_training = spark.createDataFrame(dummy_training_data, ["id", "text", "label"])
df_test = spark.createDataFrame(dummy_test_data, ["id", "text", "label"])

# First, tokenize to seperate string into list of semantic units
# TweetTokenizer pays attention to text such as emoticons and arrows, and combines them into a single unit.
tweet_tokenizer = TweetTokenizerTransformer(inputCol="text", outputCol="tokens")

#Removing stopwords
# Stopwords = common words which don't denote meaning (i.e. a, the, is)
# Removing these tokens may boost model performance
remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_stops")

# Convert tokens w/o stopwords to Term Frequency Vector -> Inverse Doc Frequency Vector
cv = CountVectorizer(inputCol="tokens_no_stops", outputCol="raw_count")
idf = IDF(inputCol="raw_count", outputCol="idf_features")
lr = LogisticRegression(featuresCol="idf_features", labelCol="label")

# Pipeline runs all of the stages inputted in sequence, which processes input string, then inputs it into our model.
pipeline = Pipeline(stages=[tweet_tokenizer, remover, cv, idf, lr])

# Training our pipeline model
# StopWordsRemover is a Transformer, so it doesn't change as a result of training, it always removes a static set of stopwords.
# CV constructs a vocabulary (a set of all tokens present in dataset) and uses it to construct term frequency vectors for each datapoint.
# IDF learns the static-length vector of IDF weights for each datapoint, which it can create with the vector from CV.
# LogisticRegression trains a model based on the IDF vector for each datapoint.
model = pipeline.fit(df_training)

# Running inference on our pipeline model
output_df = model.transform(df_test)
output_df.toPandas().to_csv("data/test_output.csv", index=False)

print()
print("PREDICTION: (0.0 = bot, 1.0 = human)")
output_df.select("text", "label", "prediction").show(truncate=False)
print()
