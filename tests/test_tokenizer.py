import pytest

from naive_bayes.tokenizer import Tokenizer


@pytest.fixture(scope='module')
def test_string():
    return 'this is a test of the emergency broadcasting system'


class TestTokenizer:

    def test_downcasing(self):
        expected_result = ['this', 'is', 'all', 'caps']
        result = Tokenizer.tokenize('THIS IS ALL CAPS')
        assert result == expected_result

    def test_ngrams(self):
        expected_result = [
            [u'\u0000', 'quick'],
            ['quick', 'brown'],
            ['brown', 'fox'],
        ]
        result = Tokenizer.ngram('quick brown fox', 2)
        assert result == expected_result
