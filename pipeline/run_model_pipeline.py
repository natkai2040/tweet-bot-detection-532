'''
Runs pipeline on our entire dataset
For: Ryan <3
'''

import os
import sys
from pyspark.ml import PipelineModel
from pyspark.sql.session import SparkSession
from pipeline import preprocess_data, train_pipeline, inference_pipeline, calculate_metrics

# UNCOMMENT sys.path.insert() if serialization issues arise with multiple worker deployment
# Adds parent directory to path to ensure proper imports

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import preprocess_data, train_pipeline, inference_pipeline, calculate_metrics

spark = SparkSession.builder \
    .appName("BotClassifier") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Data is taken from json file generated from data_preprocessing.ipynb, tweets_with_labels.json
data = "data/tweets_with_labels.json"
save_model_path = "pipeline_model"

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# Preprocess data (uses cache if available)
df_train, df_test = preprocess_data(spark, data)

# Check if model exists, load it if it does, train and save if it doesn't
if os.path.exists(save_model_path) and os.listdir(save_model_path):
    print("\nLoading existing model...")
    model = PipelineModel.load(save_model_path)
else:
    print("\nTraining new model...")
    model = train_pipeline(spark, df_train)
    model.write().overwrite().save(save_model_path)
    print(f"\nModel saved at: {save_model_path}")

# Run inference and calculate metrics
output = inference_pipeline(spark, model, df_test)
metrics = calculate_metrics(output, model, labels=["Human", "Bot"])

print("\nMetrics and visualizations saved to 'figs/' directory")