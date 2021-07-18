"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import tfg_utils.utils as utils
from random_forest_regressor import preprocessing

data_path = join(Path(__file__).parent.parent.parent, 'data')


if __name__ == '__main__':
    name_str = 'rf_class'
    args = utils.argument_parser().parse_args()

    param_grid = {'n_estimators': [15, 16, 17, 18, 19, 20],
                  'max_features': [None, 1/3]}

    if args.undersampling:
        name_str = f"{name_str}_{str(args.undersampling).replace('.', '_')}_undersamp"

    if args.ranges:
        assert len(args.ranges) == 3, 'Len of ranges parameters must be 3'
        c_str = ''
        for i in args.ranges:
            c_str = f"{c_str}_{str(i).replace('.0', '').replace('.', '-')}"
        name_str = f"name_str{c_str}"

    if args.undersampling:
        name_str = f"{name_str}_{str(args.undersampling).replace('.', '_')}_undersamp"

    x_train_p, y_train_p, x_test_p, y_test_p = preprocessing(args.undersampling, args.feature_selection)
    y_train_p, y_test_p = utils.categorize_regression(y_train_p, y_test_p, args.ranges)

    results = {'name': name_str}

    g_search = GridSearchCV(RandomForestClassifier(random_state=utils.seed), param_grid=param_grid, scoring='f1_macro',
                            n_jobs=-1)

    g_search.fit(x_train_p, y_train_p)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train_p.to_numpy(), y_train_p, clf,
                                                         metric_callback=utils.calculate_classification_metrics)

    clf.fit(x_train_p, y_train_p)

    y_pred = clf.predict(x_train_p)
    results['train'] = utils.calculate_classification_metrics(y_train_p, y_pred, clf.predict_proba(x_train_p))

    y_pred = clf.predict(x_test_p)
    results['test'] = utils.calculate_classification_metrics(y_test_p, y_pred, clf.predict_proba(x_test_p))

    print(utils.json_metrics_to_latex(results))

    feature_importance = pd.Series(data=clf.feature_importances_, index=x_train_p.columns)
    print(f'{clf.get_params()}')

    utils.plot_feature_importance(feature_importance, 10, 'Variable', 'Importance', 'Random Forest features',
                                  args.save_figures, name_str)

    utils.plot_confusion_matrix(y_test_p, y_pred, ['IEBaja', 'IEMedia', 'IEAlta'], 'Confusion matrix',
                                args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
