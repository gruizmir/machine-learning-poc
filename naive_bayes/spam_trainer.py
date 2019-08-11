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
