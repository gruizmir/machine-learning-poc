from decission_trees import (
    MushroomForest,
    MushroomRegression,
    MushroomTree,
)


data = './datasets/agaricus-lepiota.data'
folds = 5

items = {
    'Decision Tree': MushroomTree,
    'Random Forest method': MushroomForest,
    'Regression Tree': MushroomRegression,
}

for method_name, method in items.items():
    print(f'Calculating score for {method_name}')
    for result in method(data).validate(folds):
        print(result)
