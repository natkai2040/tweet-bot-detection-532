'''
Runs pipeline on a very small subset of our dataset
'''

import sparkapp.pipeline as pipeline
from pyspark.sql.session import SparkSession

spark = SparkSession.builder.appName("BotClassifier").getOrCreate()

#Data is taken from json file generated from tokenizer.ipynb, tweets_with_labels.json
training_data_subset = "Twibot-22/data_subset/training_subset.json"
test_data_subset = "Twibot-22/data_subset/test_subset.json"

model = pipeline.train_pipeline(spark, training_data_subset)
pipeline.inference_pipeline(spark, model, test_data_subset)