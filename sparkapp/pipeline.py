"""
This script provides a pipeline for both training and evaluating text with a Classification Model
"""

import json
import os
import unicodedata
from pyspark.sql import SparkSession
from pyspark.sql.functions import map_from_arrays, array, lit, explode, col, when
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkapp.tweet_tokenizer import TweetTokenizerTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import emoji

def preprocess_data(spark: SparkSession, input_json):
    '''
    Given a JSON in the format defined in data_preprocessing.ipynb (key (tweet ID): {label: "human/bot", tweet: "tweetText}),
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

    print("Preprocessed Training Data:")
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

    # Extract training history
    lr_model = model.stages[-1]
    training_summary = lr_model.summary
    
    # Plot training metrics
    plot_training_metrics(training_summary)

    return model

def plot_training_metrics(training_summary):
    """
    Plot accuracy and loss over training iterations
    """
    # Extract metrics from training summary
    objective_history = training_summary.objectiveHistory
    iterations = list(range(len(objective_history)))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss over time
    ax1.plot(iterations, objective_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss (Objective)', fontsize=12)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(iterations))
    
    # Plot 2: Accuracy on training set over time (if available)
    # Note: Logistic Regression doesn't track per-iteration accuracy by default
    # We'll show convergence instead
    if len(objective_history) > 1:
        # Calculate relative improvement
        improvements = [0]
        for i in range(1, len(objective_history)):
            if objective_history[i-1] != 0:
                improvement = abs(objective_history[i] - objective_history[i-1]) / abs(objective_history[i-1])
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        ax2.plot(iterations, improvements, 'g-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Relative Improvement', fontsize=12)
        ax2.set_title('Training Convergence Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max(iterations))
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("figs/training_metrics.png", dpi=300)
    plt.close()
    
    print(f"Training completed in {len(objective_history)} iterations")
    print(f"Final loss: {objective_history[-1]:.6f}")

def inference_pipeline(spark, model, df_test):
    # Running inference on our pipeline model
    output_df = model.transform(df_test)    
    output_df.select("label", "prediction") \
            .write.mode("overwrite").csv("runs/test_output")

    return output_df

def plot_validation_curve(model, df_train, df_test):
    """
    Plot accuracy on train/test sets over different regularization parameters
    This shows overfitting/underfitting
    """
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    
    # Get pipeline stages except LR
    tokenizer, remover, cv_model, idf_model, _ = model.stages
    
    reg_params = [0.001, 0.01, 0.1, 1.0, 10.0]
    train_accuracies = []
    test_accuracies = []
    
    print("\nCalculating validation curve...")
    for reg_param in reg_params:
        # Create new LR with this regularization
        lr = LogisticRegression(
            featuresCol="idf_features",
            labelCol="label",
            maxIter=100,
            regParam=reg_param
        )
        
        # Create pipeline with existing transformers
        pipeline = Pipeline(stages=[tokenizer, remover, cv_model, idf_model, lr])
        temp_model = pipeline.fit(df_train)
        
        # Calculate accuracies
        train_pred = temp_model.transform(df_train)
        test_pred = temp_model.transform(df_test)
        
        train_acc = train_pred.filter(col("label") == col("prediction")).count() / train_pred.count()
        test_acc = test_pred.filter(col("label") == col("prediction")).count() / test_pred.count()
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"RegParam={reg_param}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.plot(reg_params, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2)
    plt.plot(reg_params, test_accuracies, 'r-o', label='Test Accuracy', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Regularization Parameter (λ)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Curve: Accuracy vs Regularization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figs/validation_curve.png", dpi=300)
    plt.close()
    
    print("Validation curve saved to figs/validation_curve.png")

def contains_devanagari_unicodedata(text):
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, '').upper():
            return True
    return False

def calculate_metrics(df, model, labels=None, outfile="figs/confusion_matrix.png"):
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


    #ACCURACY, PRECISION, RECALL, CONFUSION MATRIX

    predictions = df.select("label", "prediction").collect()
    y_true = [row.label for row in predictions]
    y_pred = [row.prediction for row in predictions]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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