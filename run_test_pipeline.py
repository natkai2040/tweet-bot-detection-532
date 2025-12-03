'''
Runs pipeline on a very small subset of our dataset
'''

import string
import sparkapp.pipeline as pipeline
from pyspark.sql.session import SparkSession
import json

TRAINING_DATA_SUBSET = "Twibot-22/testing_subset/training_subset.json"
TEST_DATA_SUBSET = "Twibot-22/testing_subset/test_subset.json"

def modify_test_json():
    # Load JSON from file
    with open(TEST_DATA_SUBSET, 'r') as f:
        data = json.load(f)


model = pipeline.train_pipeline(spark, training_data_subset)
pipeline.inference_pipeline(spark, model, test_data_subset)
    # Randomly sampled tweet from json file generated from tokenizer.ipynb, tweets_with_labels.json
    data['sampled_test_tweet'] = {
        'label': 'bot',
        'text': 'Medical collections are likely less of a tail event than many expect--being both more common and more modest in size than implied by some of the popular discourse.\n\nFor example, in 2020 the median medical collection was $310. https://t.co/hKczgDQ9gP'
    }

    # Testing handling of punctuation
    all_punc = string.punctuation + '…' + "’" + '...' + '“' + '”' + '<' + " " + "=" + " " + '>'
    data['punc_test'] = {
        'label': 'bot',
        'text': all_punc
    }

    # Sanity checking tokenized example provided at https://www.nltk.org/api/nltk.tokenize.casual.html
    data['api_test_1'] = {
        'label': 'bot',
        'text': "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    }
    
    # Save back to file
    with open(TEST_DATA_SUBSET, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    modify_test_json()
    spark = SparkSession.builder.appName("BotClassifier").getOrCreate()

    model = pipeline.train_pipeline(spark, TRAINING_DATA_SUBSET)
    pipeline.inference_pipeline(spark, model, TEST_DATA_SUBSET)
