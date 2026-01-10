"""
This script provides a pipeline for both training and evaluating text with a Classification Model
"""

import unicodedata
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import map_from_arrays, array, lit, explode, col, when, udf
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import emoji
from pyspark.sql.types import StringType
import pathlib

# Regular Expression pattern which allows for tweet-aware tokenization (i.e. maintain @ mentions, #, URL's)
TWEET_REGEX = r"(?:@[\w_]+)|(?:\#[\w_]+)|(?:(?i:https?://)\S+)|(?:(?i:www\.)\S+)|(?:\w+(?:-\w+)*(?:'\w+)?)|(?:[^\w\s])"

def selective_lowercase_text(text):
    """Lowercase text but preserve URLs"""
    if not text:
        return text
    
    # Import inside function for better serialization
    import re
    
    # Find all URLs (case-insensitive for the protocol)
    url_pattern = r'(?i)(https?://\S+|www\.\S+)'
    matches = list(re.finditer(url_pattern, text))
    
    if not matches:
        # No URLs found, just lowercase everything
        return text.lower()
    
    # Build result by processing text in segments
    result = []
    last_end = 0
    
    for match in matches:
        # Add lowercased text before the URL
        result.append(text[last_end:match.start()].lower())
        # Add the URL with original case
        result.append(match.group(0))
        last_end = match.end()
    
    # Add any remaining text after the last URL
    result.append(text[last_end:].lower())
    
    return ''.join(result)

# Create UDF at module level with explicit function reference
selective_lowercase_udf = udf(selective_lowercase_text, StringType())

def preprocess_data(spark: SparkSession, input_json):
    '''
    Given a JSON in the format defined in data_preprocessing.ipynb (key (tweet ID): {label: "human/bot", tweet: "tweetText}),
    generate a Spark DF which organizes the data into the following columns: [tweet_id, text, label]
    '''
    
    # Define cache paths
    json_name = pathlib.Path(input_json).name
    cache_dir = "data/preprocessed_cache/" + json_name
    cache_train = os.path.join(cache_dir, "train")
    cache_test = os.path.join(cache_dir, "test")
    
    # Check if cached preprocessed data exists
    if os.path.exists(cache_train) and os.path.exists(cache_test):
        print("Loading cached preprocessed data for " + json_name + "...")
        train_df = spark.read.parquet(cache_train)
        test_df = spark.read.parquet(cache_test)
        print(f"Loaded {train_df.count()} training samples and {test_df.count()} test samples from cache")
        return train_df, test_df
    
    # If cache doesn't exist, preprocess all data from scratch and then cache it.
    print("\nPreprocessing data from scratch...")
    
    df = spark.read.option("multiline", "true").json(input_json)
    cols = df.columns

    df_map = df.select(
        map_from_arrays(
            array([lit(c) for c in cols]),
            array([col(c) for c in cols])
        ).alias("tweets")
    )

    # Flatten json such that a given tweet's object contains tweet_id and data columns
    df_flat = df_map.select(explode("tweets").alias("tweet_id", "data"))

    # Rename data columns into two seperate columns, text and label.
    df_formatted = df_flat.select(
        col("tweet_id"),
        col("data.text").alias("text"),
        col("data.label").alias("label")
    )

    # Encode the label of the data as follows:
    # HUMAN = 1
    # BOT = 0
    df_final = df_formatted.withColumn("label",
                                         when(col("label") == "human", 1).otherwise(0))

    # Split by labels
    human_df = df_final.filter(col("label") == 1)
    bot_df   = df_final.filter(col("label") == 0)
    human_train, human_test = human_df.randomSplit([0.8, 0.2], seed=42)
    bot_train, bot_test     = bot_df.randomSplit([0.8, 0.2], seed=42)

    # Combine dfs
    train_df = human_train.union(bot_train)
    test_df  = human_test.union(bot_test)

    # Apply selective lowercasing using UDF
    print("Applying selective lowercasing (preserving URL case)...")
    train_df = train_df.withColumn("text", selective_lowercase_udf(col("text")))
    test_df = test_df.withColumn("text", selective_lowercase_udf(col("text")))

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Cache and materialize to execute the UDF transformation
    print("Materializing transformations...")
    train_df = train_df.cache()
    test_df = test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    # Write to parquet for future runs
    print("Caching preprocessed data for future runs...")
    train_df.write.mode("overwrite").parquet(cache_train)
    test_df.write.mode("overwrite").parquet(cache_test)
    
    # Read back to break lineage and remove UDF dependency
    train_df = spark.read.parquet(cache_train)
    test_df = spark.read.parquet(cache_test)

    print(f"Preprocessed {train_count} training samples and {test_count} test samples")
    return train_df, test_df

def train_pipeline(spark, df_training, plot_training_loss=True):
    '''
    Train a text classification pipeline using logistic regression
    '''
    
    # First, tokenize to seperate string into list of semantic units
    # RegexTokenizer aims to replicate nltk's TweetTokenizer by keeping @mentions, URL's, hashtags and emojis together
    # It differs in its inability to shorten elongated words (i.e. sooooo -> so) and preserve text-based emoticons [i.e. :-)]
    regex_tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", 
                                     pattern=TWEET_REGEX, gaps=False, toLowercase=False)

    # Removing stopwords
    # Stopwords = common words which don't denote meaning (i.e. a, the, is)
    # Removing these tokens may boost model performance
    remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_no_stops")

    # Convert tokens w/o stopwords to Term Frequency Vector -> Inverse Doc Frequency Vector
    cv = CountVectorizer(inputCol="tokens_no_stops", outputCol="raw_count")
    idf = IDF(inputCol="raw_count", outputCol="idf_features")
    lr = LogisticRegression(featuresCol="idf_features", labelCol="label")

    # Pipeline runs all of the stages inputted in sequence, which processes input string, then inputs it into our model.
    pipeline = Pipeline(stages=[regex_tokenizer, remover, cv, idf, lr])

    # Training our pipeline model
    # StopWordsRemover is a Transformer, so it doesn't change as a result of training, it always removes a static set of stopwords.
    # CV constructs a vocabulary (a set of all unique tokens present in dataset) and uses it to construct term frequency vectors for each datapoint.
    # IDF learns the static-length vector of IDF weights for each datapoint, which it can create with the vector from CV.
    # LogisticRegression trains a model based on the IDF vector for each datapoint.
    print("Training pipeline model...")
    model = pipeline.fit(df_training)

    # Extract training history
    lr_model = model.stages[-1]
    training_summary = lr_model.summary
    
    # Plot training metrics
    if plot_training_loss:
        plot_training_metrics(training_summary)

    return model

def plot_training_metrics(training_summary):
    """
    Plot accuracy and loss over training iterations
    """
    # Extract metrics from training summary
    objective_history = training_summary.objectiveHistory
    iterations = list(range(len(objective_history)))
    
    # Create figs directory if it doesn't exist
    os.makedirs("figs", exist_ok=True)
    
    # Figure: Loss over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, objective_history)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (Objective)', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(iterations))
    plt.tight_layout()
    plt.savefig("figs/training_loss.png", dpi=300)
    plt.close()
    
    print(f"Training completed in {len(objective_history)} iterations")
    print(f"Final loss: {objective_history[-1]:.6f}")

def inference_pipeline(spark, model, df_test):
    '''
    Run inference on test data using the trained model
    '''
    # Create runs directory if it doesn't exist
    os.makedirs("runs", exist_ok=True)
    
    # Running inference on our pipeline model
    print("Running inference on test data...")
    output_df = model.transform(df_test)    
    output_df.select("label", "prediction") \
            .write.mode("overwrite").csv("runs/test_output")

    return output_df

def contains_devanagari_unicodedata(text):
    '''Check if text contains Devanagari characters'''
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, '').upper():
            return True
    return False

def calculate_metrics(df, model, labels=None, outfile="figs/confusion_matrix.png"):
    '''
    Calculate and visualize model test performance metrics
    '''
    # Create figs directory if it doesn't exist
    os.makedirs("figs", exist_ok=True)
    
    # FEATURE IMPORTANCE
    tokenizer, remover, cv_model, idf_model, lr_model = model.stages
    vocab = cv_model.vocabulary  
    idf_weights = idf_model.idf.toArray()  
    lr_coeffs = lr_model.coefficients.toArray()
    
    importances = np.abs(lr_coeffs * idf_weights)
    feature_importance = list(zip(vocab, importances))
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    # Filter out Devanagari characters since they can't be displayed in figure
    feature_importance_filtered = [
        (word, score) for word, score in feature_importance 
            if not contains_devanagari_unicodedata(word)
    ]

    top_k = 10
    top_features = feature_importance_filtered[:top_k]
    words = [emoji.demojize(w) for w, _ in top_features]
    scores = [s for _, s in top_features]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(words))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, words)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} Most Important Features")
    plt.tight_layout()
    plt.savefig("figs/feature_importances.png", dpi=300)
    plt.close()

    # ACCURACY, PRECISION, RECALL, CONFUSION MATRIX
    predictions = df.select("label", "prediction").collect()
    y_true = [row.label for row in predictions]
    y_pred = [row.prediction for row in predictions]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Plot confusion matrix
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

    # Print metrics
    print()
    print("MODEL PERFORMANCE METRICS")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }