import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .problem import MushroomProblem


class MushroomClassifier(MushroomProblem):

    def validate(self, folds):
        for test, training in self.validation_data(folds):
            yield pd.crosstab(test, training, rownames=['actual'], colnames=['preds'])


class MushroomForest(MushroomClassifier):

    def train(self, training_set, classifier):
        return RandomForestClassifier(n_jobs=2).fit(training_set, classifier)


class MushroomTree(MushroomClassifier):

    def train(self, training_set, classifier):
        return DecisionTreeClassifier().fit(training_set, classifier)
