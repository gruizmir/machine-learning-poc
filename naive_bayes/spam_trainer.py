from collections import defaultdict

from .tokenizer import Tokenizer
from .email_object import EmailObject


class SpamTrainer:

    def __init__(self, training_files):
        self.categories = {cat for cat, path in training_files}
        self.totals = defaultdict(float)
        self.training = {cat: defaultdict(float) for cat in self.categories}
        self.to_train = training_files

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
