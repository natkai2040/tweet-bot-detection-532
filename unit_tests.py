import pytest
from nltk.tokenize import TweetTokenizer

@pytest.fixture
def tokenizer():
    '''Tokenizer used in tweet_tokenizer.py'''
    return TweetTokenizer(preserve_case=False)

class TestBasicTokenization:
    '''Testing basic tokenizer functionality such as simple token splitting, empty inputs, handling varying whitespace sizes, etc.'''
    
    def test_simple_sentence(self, tokenizer):
        text = "hello world"
        expected = ["hello", "world"]
        assert tokenizer.tokenize(text) == expected
    
    def test_empty_string(self, tokenizer):
        assert tokenizer.tokenize("") == []
    
    def test_single_word(self, tokenizer):
        assert tokenizer.tokenize("hello") == ["hello"]
    
    def test_multiple_spaces(self, tokenizer):
        text = "hello    world"
        expected = ["hello", "world"]
        assert tokenizer.tokenize(text) == expected

class TestTwitterFeatures:
    '''Testing twitter-specific tokenization, such as keeping @user, #hashtags and URL's'''
    
    def test_username(self, tokenizer):
        text = "@user mentioned this"
        tokens = tokenizer.tokenize(text)
        assert "@user" in tokens
    
    def test_single_hashtag(self, tokenizer):
        text = "Check out #python"
        tokens = tokenizer.tokenize(text)
        assert "#python" in tokens
    
    def test_multiple_hashtags(self, tokenizer):
        text = "#python #nlp #ai"
        tokens = tokenizer.tokenize(text)
        assert "#python" in tokens
        assert "#nlp" in tokens
        assert "#ai" in tokens
    
    def test_http_url(self, tokenizer):
        text = "Check http://test.com"
        tokens = tokenizer.tokenize(text)
        assert "http://test.com" in tokens
    
    def test_https_url(self, tokenizer):
        # Sample tweet from dataset
        text = "Read https://t.co/hKczgDQ9gP"
        tokens = tokenizer.tokenize(text)
        assert "https://t.co/hKczgDQ9gP" in tokens
    
    def test_www_url(self, tokenizer):
        text = "Visit www.test.com"
        tokens = tokenizer.tokenize(text)
        assert "www.test.com" in tokens

class TestEmoticons:
    '''Testing how the tokenizer handles emoticons: (i.e. :), :()'''
    
    def test_smiley_face(self, tokenizer):
        text = "yay :)"
        tokens = tokenizer.tokenize(text)
        assert ":)" in tokens
    
    def test_sad_face(self, tokenizer):
        text = "aughhhhhhhh :("
        tokens = tokenizer.tokenize(text)
        assert ":(" in tokens
    
    def test_with_nose(self, tokenizer):
        text = "happy :-) and sad :-("
        tokens = tokenizer.tokenize(text)
        assert ":-)" in tokens
        assert ":-(" in tokens
    
    def test_big_smile(self, tokenizer):
        text = "yessssss :D"
        tokens = tokenizer.tokenize(text)
        assert ":D" in tokens
    
    def test_wink(self, tokenizer):
        text = "heyy ;)"
        tokens = tokenizer.tokenize(text)
        assert ";)" in tokens

class TestEdgeCases:
    '''Testing potentially uncertain cases of tokenization'''
    
    def test_hyphenated_words(self, tokenizer):
        text = "happy-go-lucky personality"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 2
    
    def test_abbreviations(self, tokenizer):
        text = "I LOVE THE U.S.A."
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 9
    
    def test_multiple_newlines(self, tokenizer):
        text = "Line 1\n\nLine 2"
        tokens = tokenizer.tokenize(text)
        assert "line" in tokens and "1" in tokens and "2" in tokens
        assert len(tokens) == 4

    def upper_to_lower(self, tokenizer):
        text = "I AM SHOUTING I AM SHOUTING LOUD"
        tokens = tokenizer.tokenize(text)
        assert all(token.islower() or not token.isalpha() for token in tokens)

class TestRealWorldExamples:
    """Test with real-world tweet examples"""
    
    def test_sample_medical_tweet(self, tokenizer):
        text = "Medical collections are likely less of a tail event than many expect--being both more common and more modest in size than implied by some of the popular discourse.\n\nFor example, in 2020 the median medical collection was $310. https://t.co/hKczgDQ9gP"
        tokens = tokenizer.tokenize(text)
        
        assert "https://t.co/hKczgDQ9gP" in tokens
        assert "2020" in tokens
        assert "$" in tokens
        assert "medical" in tokens
        assert "collections" in tokens

if __name__ == "__main__":
    pytest.main([__file__, "-v"])