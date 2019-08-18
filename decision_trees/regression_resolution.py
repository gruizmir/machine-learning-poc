from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from .problem import MushroomProblem


class MushroomRegression(MushroomProblem):

    def train(self, training_set, categories):
        return DecisionTreeRegressor().fit(training_set, categories)

    def validate(self, folds):
        for y_true, y_pred in self.validation_data(folds):
            yield mean_squared_error(y_true, y_pred)
        # responses = []
        # for y_true, y_pred in self.validation_data(folds):
        #     responses.append(mean_squared_error(y_true, y_pred))
        # return responses
