# 🐦 Tweet Bot Detection (tweet-bot-detection-532)

An **End-to-end ML Twitter bot detection pipeline** built with PySpark, which classifies if an author of a tweet is a bot given only text, 
without social media metric data (i.e. likes, retweets, time posted, etc.).

## Overview
The pipeline performs:

1. Ingestion of JSON data to be converted into Spark Dataframes
2. Text preprocessing (selective lowercasing to maintain casing in URL's only)
3. Social-media aware tokenization (i.e. keeping URL's, @ mentions, # hashtags as single tokens)
4. Feature extraction (TF-IDF on tokens)
5. Model training using logistic regression
6. Inference and Calculation of Accuracy Metrics

The project is designed for **scalability** in mind, with additional scripts for conversion of very large json files to .ndjson files 
as well as PySpark's general compatability with multi-cluster deployments for handling big data. Additionally, our pipeline's reliance on only
text data allows for application for other social media networks or any text-based bot detection task.

# ✅ Requirements
```!pip install -r requirements.txt```

[PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/install.html) must also be installed on your machine.

# ⛳️ Getting Started

## Twibot-22 Dataset
Training and evaluating the same model portrayed in this repository's saved figures requires access 
to the [Twibot-22 dataset](https://twibot22.github.io/). Please refer to this link to inquire more about how to gain access.
This dataset must then be placed in the data directory, resulting in the following file paths:
* data/Twibot-22/label_new.json
* data/Twibot-22/node_new.json

## JSON Generation
You may then run **data_preprocessing/create_jsons.py** in order to convert the raw data into a json file containing only the data we need
for analysis, which is tweet_id, label (human/bot) and text (the actual tweet string).

## Exploratory Statistics
**data_preprocessing/data_exploration.ipynb** and **data_preprocessing/exploratory_stats.py** may then be used to gain statistical insights into
the dataset, such as mean tweet length and most frequent grams, bigrams and trigrams for both bots and human users.

## Running the pipeline
**pipeline/run_dummy_pipeline.py** may be ran to test the entire pipeline on the provided **data/dummy_data.json**, if
gaining access to the TwiBot22 dataset proves to be difficult.

If the TwiBot22 dataset is available to you, **pipeline/run_model_pipeline.py** may be ran for a complete run of the pipeline.

# 🚀 Pipeline Functions
## 1. Preprocess Data

Transforms raw labeled tweet JSON into Spark DataFrames, partitioned into a train and test dataframe:

```df_train, df_test = pipeline.preprocess_data(spark, "data/input.json")```

Preprocessed data is cached in Parquet format to greatly expedite future training and inference runs.

## 2. Train Model

Train a logistic regression classifier using Spark ML and save it for future inference runs:

```model = pipeline.train_pipeline(spark, df_train)```


## 3. Inference

Run predictions on the test set:

```output_df = pipeline.inference_pipeline(spark, model, df_test)```


# 📁 Input Data Format

If you wish to utilize this pipeline for your own dataset, please adhere to the following JSON data format:
```
{
  "tweet_id_1": {
    "text": "tweet text here",
    "label": "human"
  },
  "tweet_id_2": {
    "text": "another tweet",
    "label": "bot"
  }
}
```


Labels are encoded as:
* human → 1
* bot → 0
