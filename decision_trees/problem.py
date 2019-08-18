import pandas as pd
import numpy as np

from abc import abstractmethod
from numpy.random import permutation


class MushroomProblem:

    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        for k in self.df.columns[1:]:
            self.df[k], _ = pd.factorize(self.df[k])

        sorted_categories = sorted(pd.Categorical(self.df['class']).categories)
        self.classes = np.array(sorted_categories)
        self.features = self.df.columns[self.df.columns != 'class']

    def __factorize(self, data):
        y, _ = pd.factorize(pd.Categorical(data['class']), sort=True)
        return y

    def validation_data(self, folds):
        df = self.df
        assert len(df) > folds
        perms = np.array_split(permutation(len(df)), folds)

        # response = []
        for i in range(folds):
            train_idxs = range(folds)
            train_idxs.pop(i)
            train = [perms[idx] for idx in train_idxs]
            train = np.concatenate(train)
            test_idx = perms[i]

            training = df.iloc[train]
            test_data = df.iloc[test_idx]

            y = self.__factorize(training)
            classifier = self.train(training[self.features], y)
            predictions = classifier.predict[test_data[self.features]]

            expected = self.__factorize(test_data)
            yield (predictions, expected)
            # response.append([predictions, expected])

        # return response

    @abstractmethod
    def train(self, training_set, categories):
        raise NotImplementedError
