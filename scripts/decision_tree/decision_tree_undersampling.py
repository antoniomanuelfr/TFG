"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join

import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

import tfg_utils.utils as utils
import tfg_utils.compare as cmp
from decision_tree_regressor import preprocessing

if __name__ == '__main__':
    results = {}
    under_sampler = DecisionTreeRegressor(max_depth=4, random_state=10)

    # Parse arguments
    args = utils.argument_parser()
    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing()

    x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed, y_train_transformed,
                                                                              (4.5, 7), 0.8, under_sampler)

    # grid search
    param_grid = {'max_depth': [7, 8, 9, 10, 16, 18, 20],
                  'max_leaf_nodes': [2, 4, 8, 16, 20, 22]}
    g_search = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, scoring='r2')

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    best = g_search.best_estimator_
    print(f"Best parameters {best.get_params()}")

    results['cross_validation'] = utils.cross_validation(x_train_transformed.to_numpy(), y_train_transformed, best)

    best.fit(x_train_transformed, y_train_transformed.ravel())
    print("Train score")
    y_pred = best.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = best.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for decission tree',
                               'Observations', 'IEMedia', args.save_figures, 'dtree')

    results['hist'] = utils.get_error_hist(y_test_transformed.ravel(), y_pred, 'Class', 'Count',
                                           'Error count for decission tree', args.save_figures, 'dtree')

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train_transformed.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, 'decision_tree_plot.png'))
        plt.clf()
    else:
        plt.show()

    results['importances'] = pd.Series(data=best.feature_importances_, index=x_train_transformed.columns)

    print(utils.tree_to_code(best, x_train_transformed.columns.to_numpy()))
    utils.plot_feature_importances(results['importances'], 10, 'Variable', 'Importance', 'Decission Tree features',
                                   args.save_figures, 'dtree')
    if (args.json_output):
        with open(join(args.json_output, 'decision_tree_1.json')) as fp:
            json_dtree1 = json.load(fp)
            d = {'regressor': (json_dtree1['hist'], 'Orginal tree regressor'),
                 'under_sampler': (results['hist'], 'Tree regressor with under-sampling')
                 }
            cmp.comp_error_hist(d, 'Class', 'Error count', 'Tree regressor comparison', args.save_figures,
                                'tree_orig_und')
