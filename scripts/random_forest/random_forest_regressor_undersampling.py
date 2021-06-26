"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import tfg_utils.utils as utils
import tfg_utils.compare as cmp
from random_forest_regressor import preprocessing

data_path = join(Path(__file__).parent.parent.parent, "data")


if __name__ == '__main__':
    args = utils.argument_parser()
    under_sampler = SVR()

    results = {}
    param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90],
                  'max_features': [None, 1/3]}

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing()
    x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed, y_train_transformed,
                                                                              (4.5, 7), 0.6, under_sampler)
    g_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param_grid, scoring='r2', n_jobs=-1)

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    clf = g_search.best_estimator_
    print(f"Best parameters {clf.get_params()}")

    results['cross_validation'] = utils.cross_validation(x_train_transformed.to_numpy(), y_train_transformed, clf)

    clf.fit(x_train_transformed, y_train_transformed)
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for Random Forest',
                               'Observations', 'IEMedia', args.save_figures, 'random_forest')

    results['hist'] = utils.get_error_hist(y_test_transformed, y_pred, 'Class', 'Count',
                                           'Error count for Random Forest', args.save_figures, 'random_forest')

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(f"{clf.get_params()}")
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Random Forest features',
                                   args.save_figures, 'rf')

    if args.json_output:
        import json
        with open(join(args.json_output, 'random_forest_1.json'), mode='w') as fd:
            json.dump(results, fd)
    if (args.json_output):
        with open(join(args.json_output, 'random_forest_1.json')) as fp:
            json_dtree1 = json.load(fp)
            d = {'regressor': (json_dtree1['hist'], 'Orginal RF regressor'),
                 'under_sampler': (results['hist'], 'RF regressor with under-sampling')
                 }
            cmp.comp_error_hist(d, 'Class', 'Error count', 'RF regressor comparison', args.save_figures,
                                'tree_orig_und')
