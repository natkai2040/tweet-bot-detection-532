'''
Runs pipeline on our entire dataset
For: Ryan <3
'''

import os
from pyspark.ml import PipelineModel
from pyspark.sql.session import SparkSession
import pipeline

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

df_train, df_test = pipeline.preprocess_data(spark, data)

if os.path.exists(save_model_path) and os.listdir(save_model_path):
    print("\nLoading Model...")
    model = PipelineModel.load(save_model_path)
else:
    print("\nTraining New Model...")
    model = pipeline.train_pipeline(spark, df_train)
    model.write().overwrite().save(save_model_path)
    print("\nModel saved at: ", save_model_path)

output = pipeline.inference_pipeline(spark, model, df_test)
metrics = pipeline.calculate_metrics(output, model, labels=["Human", "Bot"])

print("Metrics and visualizations saved to 'figs/' directory")
