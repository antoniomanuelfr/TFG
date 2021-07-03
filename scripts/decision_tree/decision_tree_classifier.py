"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

import tfg_utils.utils as utils
from decision_tree_regressor import preprocessing

data_path = join(Path(__file__).parent.parent.parent, "data")

if __name__ == '__main__':
    # Parse arguments
    args = utils.argument_parser().parse_args()
    name_str = 'dtree_classification'

    if args.undersampling:
        name_str = f'{name_str}_undersamp'

    x_train_transform, y_train_transform, x_test_transform, y_test_transform = preprocessing(args.undersampling)
    y_train_transform, y_test_transform = utils.categorize_regression(y_train_transform, y_test_transform)
    results = {'name': name_str}

    param_grid = {'max_depth': [4, 5, 6, 7, 8, 9, 10, 16],
                  'max_leaf_nodes': [16, 17, 18, 19, 20]}

    g_search = GridSearchCV(DecisionTreeClassifier(random_state=utils.seed), param_grid=param_grid, scoring='f1_macro')

    # Grid search
    g_search.fit(x_train_transform, y_train_transform)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    best = g_search.best_estimator_
    print(f"Best parameters {best.get_params()}")

    results['cross_validation'] = utils.cross_validation(x_train_transform.to_numpy(), y_train_transform, best,
                                                         metric_callback=utils.calculate_classification_metrics)

    best.fit(x_train_transform, y_train_transform.ravel())

    print("Train score")
    y_pred = best.predict(x_train_transform)
    results['train'] = utils.calculate_classification_metrics(y_train_transform, y_pred,
                                                              best.predict_proba(x_train_transform))

    print("Test score")
    y_pred = best.predict(x_test_transform)
    results['test'] = utils.calculate_classification_metrics(y_test_transform, y_pred,
                                                             best.predict_proba(x_test_transform))

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train_transform.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, f'{name_str}_plot'))
        plt.clf()
    else:
        plt.show()

    features_importances = pd.Series(data=best.feature_importances_, index=x_train_transform.columns)
    results['feature_importances'] = dict(features_importances)

    utils.plot_feature_importance(features_importances, 10, 'Variable', 'Importance',
                                  'Decission Tree features', args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
