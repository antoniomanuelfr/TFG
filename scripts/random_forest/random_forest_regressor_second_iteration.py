"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from random_forest_regressor import preprocessing
from sklearn.tree import DecisionTreeRegressor

import tfg_utils.utils as utils

data_path = join(Path(__file__).parent.parent.parent, "data")

if __name__ == '__main__':
    args = utils.argument_parser()

    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90],
                  'max_features': [None, 1/3]}

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing(x_train, y_train,
                                                                                                     x_test, y_test)
    rows_to_delete = utils.regression_under_sampler(x_train_transformed, y_train_transformed, (5, 7), 0.4,
                                                    DecisionTreeRegressor(max_depth=4))
    x_train_transformed.drop(index=rows_to_delete, inplace=True)
    y_train_transformed = np.delete(y_train_transformed, rows_to_delete)

    g_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param_grid, scoring='r2', n_jobs=-1)

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    clf = g_search.best_estimator_
    print(f"Best parameters {clf.get_params()}")
    folder = KFold(n_splits=5, random_state=10, shuffle=True)

    acum_res = np.array([0, 0, 0])
    print('Validation results')
    print('r2, mean poisson deviance, mse')
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_transformed[train_index]
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_transformed[test_index]

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)
        res = utils.calculate_regression_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5


    clf.fit(x_train_transformed, y_train_transformed)
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    train_res = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    test_res = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for Random Forest',
                               'Observations', 'IEMedia', args.save_figures, 'random_forest')

    utils.get_error_hist(y_test_transformed, y_pred, 'Class', 'Count', 'Error count for Random Forest',
                         args.save_figures, 'random_forest')

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(f"{clf.get_params()}")
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Random Forest features',
                                   args.save_figures, 'rf')
