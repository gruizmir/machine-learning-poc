from collections import defaultdict

from .tokenizer import Tokenizer
from .email_object import EmailObject


class Classification:

    def __init__(self, guess, score):
        self.guess = guess
        self.score = score

    def __eq__(self, other):
        return self.guess == other.guess and self.score == other.score


class SpamTrainer:

    def __init__(self, training_files):
        self.categories = {cat for cat, path in training_files}
        self.totals = defaultdict(float)
        self.training = {cat: defaultdict(float) for cat in self.categories}
        self.to_train = training_files

    @property
    def preference(self):
        return sorted(self.categories, key=lambda cat: self.total_for(cat))

    @property
    def classification(self):
        return self._classification

    @classification.setter
    def classification(self, value):
        self._classification = Classification(value[0], value[1])

    def total_for(self, category):
        return self.totals[category]

    def train(self):
        for category, path in self.to_train:
            with open(path) as f:
                email = EmailObject(content=f.read())
                self.categories.add(category)

                for token in Tokenizer.tokenize(email.body):
                    self.training[category][token] += 1
                    self.totals['_all'] += 1
                    self.totals[category] += 1

        self.to_train = {}

    def score(self, email):
        self.train()
        totals = self.totals
        aggregates = {
            cat: totals[cat] / totals['_all']
            for cat in self.categories
        }
        for token in Tokenizer.tokenize(email.body):
            for cat in self.categories:
                value = self.training[cat][token]
                prob = (value + 1) / (totals[cat] + 1)
                aggregates[cat] *= prob

        return aggregates

    def normalized_score(self, email):
        score = self.score(email=email)
        score_sum = sum(score.values())
        normalized = {cat: (agg / score_sum) for cat, agg in score.items()}
        return normalized

    def classify(self, email):
        score = self.score(email=email)
        max_score = 0.0
        preference = self.preference
        max_key = preference[-1]
        for key, value in score.items():
            if value > max_score:
                max_key = key
                max_score = value
            elif value == max_score and preference.index(key) > preference.index(max_key):
                max_key = key
                max_score = value
        self.classification = (max_key, max_score)
        return self.classification
