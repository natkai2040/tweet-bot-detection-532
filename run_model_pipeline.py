'''
Runs pipeline on our entire dataset
For: Ryan <3
'''

import sparkapp.pipeline as pipeline
from pyspark.sql.session import SparkSession

spark = SparkSession.builder \
    .appName("BotClassifier") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Data is taken from json file generated from data_preprocessing.ipynb, tweets_with_labels.json
data = "data/tweets_with_labels.json"

df_train, df_test = pipeline.preprocess_data(spark, data)

model = pipeline.train_pipeline(spark, df_train)
output = pipeline.inference_pipeline(spark, model, df_test)

metrics = pipeline.calculate_metrics(output, model, labels=["Human", "Bot"])

print("Metrics and visualizations saved to 'figs/' directory")
