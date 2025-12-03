"""
This script provides a pipeline for both training and evaluating text with a Classification Model
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import map_from_arrays, array, lit, explode, col, when
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkapp.tweet_tokenizer import TweetTokenizerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

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

    # split by labels
    human_df = df_final.filter(col("label") == 1)
    bot_df   = df_final.filter(col("label") == 0)
    human_train, human_test = human_df.randomSplit([0.8, 0.2], seed=42)
    bot_train, bot_test     = bot_df.randomSplit([0.8, 0.2], seed=42)

    # combine dfs
    train_df = human_train.union(bot_train)
    test_df  = human_test.union(bot_test)

    print("PREPROCESSED TRAINING DATA:")
    # train_df.show(truncate=True)
    return train_df, test_df

def train_pipeline(spark, df_training):

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

def inference_pipeline(spark, model, df_test):
    # Running inference on our pipeline model
    output_df = model.transform(df_test)    
    output_df.select("label", "prediction") \
            .write.mode("overwrite").csv("runs/test_output")
    return output_df
    
def calculate_metrics(df, model, labels=None, outfile="confusion_matrix.png"):

    #FEATURE IMPORTANCE
    tokenizer, remover, cv_model, idf_model, lr_model = model.stages
    vocab = cv_model.vocabulary  
    idf_weights = idf_model.idf.toArray()  
    lr_coeffs = lr_model.coefficients.toArray()
    
    importances = np.abs(lr_coeffs * idf_weights)
    feature_importance = list(zip(vocab, importances))
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    feature_importance = sorted(feature_importance,
                                key=lambda x: x[1],
                                reverse=True)

    top_k = 10
    top_features = feature_importance[:top_k]
    words = [w for w, _ in top_features]
    scores = [s for _, s in top_features]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(words))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, words)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Most Important Features")
    plt.tight_layout()
    plt.savefig("feature_importances.png", dpi=300)
    plt.close()


    #ACCURACY, PRECISION, RECALL, CONFUSION MATRIX

    tp = df.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = df.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = df.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = df.filter((col("label") == 1) & (col("prediction") == 0)).count()

    accuracy = df.filter(col("label") == col("prediction")).count() / df.count()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    cm = np.array([[tn, fp],
                   [fn, tp]])

    #plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    labels = ["Bot", "Human"]  # 0, 1

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix"
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }

