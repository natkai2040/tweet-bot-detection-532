import pytest
from pipeline.pipeline import TWEET_REGEX, RegexTokenizer, selective_lowercase_text
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    '''Create SparkSession for testing'''
    return SparkSession.builder.appName("TokenizerTests").getOrCreate()

@pytest.fixture
def tokenizer():
    '''Custom RegexTokenizer'''
    return RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern=TWEET_REGEX,
        gaps=False,
        toLowercase=False
    )

def tokenize_text(spark, tokenizer, text):
    '''Helper function to tokenize a single text string with selective lowercasing'''
    # Apply selective lowercasing first
    preprocessed_text = selective_lowercase_text(text)
    
    df = spark.createDataFrame([(preprocessed_text,)], ["text"])
    result = tokenizer.transform(df)
    tokens = result.select("tokens").collect()[0][0]
    return tokens

class TestBasicTokenization:
    '''Testing basic tokenizer functionality such as simple token splitting, empty inputs, handling varying whitespace sizes, etc.'''
    
    def test_simple_sentence(self, spark, tokenizer):
        text = "hello world"
        expected = ["hello", "world"]
        assert tokenize_text(spark, tokenizer, text) == expected
    
    def test_empty_string(self, spark, tokenizer):
        assert tokenize_text(spark, tokenizer, "") == []
    
    def test_single_word(self, spark, tokenizer):
        assert tokenize_text(spark, tokenizer, "hello") == ["hello"]
    
    def test_multiple_spaces(self, spark, tokenizer):
        text = "hello    world"
        expected = ["hello", "world"]
        assert tokenize_text(spark, tokenizer, text) == expected


class TestTwitterFeatures:
    '''Testing twitter-specific tokenization, such as keeping @user, #hashtags and URL's'''
    
    def test_username(self, spark, tokenizer):
        text = "@user mentioned this"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "@user" in tokens
    
    def test_single_hashtag(self, spark, tokenizer):
        text = "Check out #python"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "#python" in tokens
    
    def test_multiple_hashtags(self, spark, tokenizer):
        text = "#python #nlp #ai"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "#python" in tokens
        assert "#nlp" in tokens
        assert "#ai" in tokens
    
    def test_http_url(self, spark, tokenizer):
        text = "Check http://test.com"
        tokens = tokenize_text(spark, tokenizer, text)

        print(tokens)
        assert "http://test.com" in tokens
        
    def test_https_url(self, spark, tokenizer):
        # Sample tweet from dataset
        text = "Read https://t.co/hKczgDQ9gP"
        tokens = tokenize_text(spark, tokenizer, text)
        print(tokens)
        assert "https://t.co/hKczgDQ9gP" in tokens
    
    def test_www_url(self, spark, tokenizer):
        text = "Visit www.test.com"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "www.test.com" in tokens

class TestEdgeCases:
    '''Testing potentially uncertain cases of tokenization'''
    
    def test_hyphenated_words(self, spark, tokenizer):
        text = "happy-go-lucky personality"
        tokens = tokenize_text(spark, tokenizer, text)
        assert len(tokens) == 2
    
    def test_abbreviations(self, spark, tokenizer):
        text = "I LOVE THE U.S.A."
        tokens = tokenize_text(spark, tokenizer, text)
        assert len(tokens) == 9
    
    def test_multiple_newlines(self, spark, tokenizer):
        text = "Line 1\n\nLine 2"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "line" in tokens and "1" in tokens and "2" in tokens
        assert len(tokens) == 4

    def test_upper_to_lower(self, spark, tokenizer):
        text = "I AM SHOUTING I AM SHOUTING LOUD"
        tokens = tokenize_text(spark, tokenizer, text)
        assert all(token.islower() or not token.isalpha() for token in tokens)


class TestRealWorldExamples:
    """Test with real-world tweet examples"""
    
    def test_sample_medical_tweet(self, spark, tokenizer):
        text = "Medical collections are likely less of a tail event than many expect--being both more common and more modest in size than implied by some of the popular discourse.\n\nFor example, in 2020 the median medical collection was $310. https://t.co/hKczgDQ9gP"
        tokens = tokenize_text(spark, tokenizer, text)
        
        assert "https://t.co/hKczgDQ9gP" in tokens
        assert "2020" in tokens
        assert "$" in tokens
        assert "medical" in tokens
        assert "collections" in tokens
class TestSelectiveLowercasing:
    '''Testing that URLs preserve case while other text is lowercased'''
    
    def test_url_preserves_case(self, spark, tokenizer):
        text = "Check HTTPS://EXAMPLE.COM for info"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "HTTPS://EXAMPLE.COM" in tokens  # URL preserved
        assert "check" in tokens  # Text lowercased
        assert "for" in tokens
        assert "info" in tokens
    
    def test_mixed_urls_and_text(self, spark, tokenizer):
        text = "Visit HTTP://Test.COM and WWW.Example.COM NOW"
        tokens = tokenize_text(spark, tokenizer, text)
        assert "HTTP://Test.COM" in tokens  # Preserved
        assert "WWW.Example.COM" in tokens  # Preserved
        assert "visit" in tokens  # Lowercased
        assert "and" in tokens
        assert "now" in tokens

if __name__ == "__main__":
    pytest.main([__file__, "-v"])