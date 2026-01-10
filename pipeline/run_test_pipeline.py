'''
Runs pipeline on a very small subset of our dataset (dummy_data.json)
Saves to json file for ease of viewing in /runs dir.
'''

import pipeline
from pyspark.sql.session import SparkSession

DUMMY_DATA_SUBSET = "data/dummy_data.json"

if __name__ == '__main__':
    spark = SparkSession.builder.appName("TestBotClassifier").getOrCreate()

    df_train, df_test = pipeline.preprocess_data(spark, DUMMY_DATA_SUBSET)

    model = pipeline.train_pipeline(spark, df_train, plot_training_loss=False)
    output_df = pipeline.inference_pipeline(spark, model, df_test)
    pdf = output_df.toPandas()
    
    results_loc = "runs/dummy_inference.json"
    pdf.to_json(results_loc, orient="records", indent=2)
    print("=" * 10)
    print("Inference results saved to:", results_loc)
    print("=" * 10)
