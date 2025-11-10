"""
This file wraps the nltk TweetTokenizer into a pyspark Transformer class, which allows us to directly integrate it into our
pipeline model instead of processing our input text for training/testing seperately from our pipeline.
"""

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from nltk import TweetTokenizer

class TweetTokenizerTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol, outputCol):
        super(TweetTokenizerTransformer, self).__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.tokenizer = TweetTokenizer(preserve_case=False)
        self.udf_tokenize = udf(
            lambda text: self.tokenizer.tokenize(text) if text else [],
            ArrayType(StringType())
            )

    def _transform(self, df):
        return df.withColumn(self.outputCol, self.udf_tokenize(col(self.inputCol)))