"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from os.path import join
from pathlib import Path

import tfg_utils.utils as utils
from decision_tree_regressor import preprocessing

data_path = join(Path(__file__).parent.parent.parent, 'data')

if __name__ == '__main__':
    # Parse arguments
    args = utils.argument_parser(parse_feature_selec=False, parse_ranges=True).parse_args()
    name_str = 'dtree_classification'

    if args.undersampling:
        name_str = f'{name_str}_undersamp'
    if args.ranges:
        assert len(args.ranges) == 2, 'Len of ranges parameters must be 2'
        c_str = ''
        for i in args.ranges:
            c_str = f"{c_str}_{str(i).replace('.0', '').replace('.', '-')}"
        name_str = f'{name_str}{c_str}_7'

    x_train, y_train, x_test, y_test = preprocessing(args.undersampling)
    y_train, y_test = utils.categorize_regression(y_train, y_test, args.ranges)
    results = {'name': name_str}

    param_grid = {'max_depth': [4, 5, 6, 7, 8, 9, 10, 16],
                  'max_leaf_nodes': [16, 17, 18, 19, 20]}

    g_search = GridSearchCV(DecisionTreeClassifier(random_state=utils.seed), param_grid=param_grid, scoring='f1_macro')

    # Grid search
    g_search.fit(x_train, y_train)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    best = g_search.best_estimator_
    print(f'Best parameters {best.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train.to_numpy(), y_train, best,
                                                         metric_callback=utils.calculate_classification_metrics)

    best.fit(x_train, y_train.ravel())

    y_pred = best.predict(x_train)
    results['train'] = utils.calculate_classification_metrics(y_train, y_pred,
                                                              best.predict_proba(x_train))

    y_pred = best.predict(x_test)
    results['test'] = utils.calculate_classification_metrics(y_test, y_pred,
                                                             best.predict_proba(x_test))

    print(utils.json_metrics_to_latex(results))

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, f'{name_str}_plot'))
        plt.clf()
    else:
        plt.show()

    feature_importance = pd.Series(data=best.feature_importances_, index=x_train.columns)
    results['feature_importance'] = dict(feature_importance)

    utils.plot_feature_importance(feature_importance, 10, 'Variable', 'Importance',
                                  'Decission Tree features', args.save_figures, name_str)

    utils.plot_confusion_matrix(y_test, y_pred, ['IEBaja', 'IEMedia', 'IEAlta'], 'Confusion matrix',
                                args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
