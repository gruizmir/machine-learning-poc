import re


class Tokenizer:
    EMPTY = u'\u0000'

    @staticmethod
    def tokenize(string):
        return re.findall('\w+', string.lower())

    @classmethod
    def pad(cls, tokens, padding):
        padded_tokens = [cls.EMPTY] * padding
        return padded_tokens + tokens

    @classmethod
    def ngram(cls, string, ngram):
        ngrams = []
        tokens = cls.tokenize(string)
        for i in range(1, len(tokens) + 1):
            shift = i - ngram
            padding = max(-shift, 0)
            first_idx = max(shift, 0)
            last_idx = first_idx + ngram - padding

            ngrams.append(cls.pad(tokens[first_idx:last_idx], padding))

        return ngrams
