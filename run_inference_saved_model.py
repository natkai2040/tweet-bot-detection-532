'''
Runs pipeline on our entire dataset
For: Ryan <3
'''

import sparkapp.pipeline as pipeline
from pyspark.sql.session import SparkSession
from pyspark.ml import PipelineModel


spark = SparkSession.builder \
    .appName("BotClassifier") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Data is taken from json file generated from data_preprocessing.ipynb, tweets_with_labels.json
data = "data/tweets_with_labels.json"
save_model_path = "bot_classification_model"

df_train, df_test = pipeline.preprocess_data(spark, data)

model = PipelineModel.load(save_model_path)

output = pipeline.inference_pipeline(spark, model, df_test)
metrics = pipeline.calculate_metrics(output, model, labels=["Human", "Bot"])

print("Metrics and visualizations saved to 'figs/' directory")
