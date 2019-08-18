import pandas as pd

from sklearn.ensemblem import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from .problem import MushroomProblem


class MushroomClassifier(MushroomProblem):

    def validate(self, folds):
        # confusion_matrices = []
        # for test, training in self.validation_data(folds):
        #     confusion_matrix = pd.crosstab(test, training, rownames=['actual'], colnames=['preds'])
        #     confusion_matrices.append(confusion_matrix)
        # return confusion_matrices
        for test, training in self.validation_data(folds):
            yield pd.crosstab(test, training, rownames=['actual'], colnames=['preds'])


class MushroomForest(MushroomClassifier):

    def train(self, training_set, classifier):
        return RandomForestClassifier(n_jobs=2).fit(training_set, classifier)


class MushroomTree:

    def train(self, training_set, classifier):
        return DecisionTreeClassifier().fit(training_set, classifier)
