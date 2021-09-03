"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import tfg_utils.utils as utils
from svr import preprocessing

data_path = join(Path(__file__).parent.parent.parent, 'data')

if __name__ == '__main__':
    args = utils.argument_parser().parse_args()
    name_str = 'svc'

    if args.undersampling:
        name_str = f"{name_str}_{str(args.undersampling).replace('.', '_')}_undersamp"

    if args.feature_selection:
        name_str = f'{name_str}_{args.feature_selection}_feature_selection'

    if args.ranges:
        assert len(args.ranges) == 2, 'Len of ranges parameters must be 2'
        c_str = ''
        for i in args.ranges:
            c_str = f"{c_str}_{str(i).replace('.0', '').replace('.', '-')}"
        name_str = f'{name_str}{c_str}_7'

    x_train, y_train, x_test, y_test = preprocessing(
        args.undersampling, args.feature_selection)
    y_train, y_test = utils.categorize_regression(y_train, y_test, args.ranges)

    results = {'name': name_str}

    param_grid = {'kernel': ['poly', 'rbf'],
                  'degree': [2, 3, 4, 5],
                  'C': [0.5, 0.75, 1]
                  }

    clf = SVC(probability=True)
    g_search = GridSearchCV(clf, param_grid=param_grid,
                            scoring='f1_macro', n_jobs=8)

    # Grid search
    g_search.fit(x_train, y_train)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')
    results['cross_validation'] = utils.cross_validation(x_train.to_numpy(), y_train, clf,
                                                         metric_callback=utils.calculate_classification_metrics)

    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_train)
    results['train'] = utils.calculate_classification_metrics(y_train, y_pred, clf.predict_proba(x_train))

    y_pred = clf.predict(x_test)
    results['test'] = utils.calculate_classification_metrics(y_test, y_pred, clf.predict_proba(x_test))
    print(utils.json_metrics_to_latex(results))

    print(clf.get_params())
    utils.plot_confusion_matrix(y_test, y_pred, ['IEBaja', 'IEMedia', 'IEAlta'], 'Confusion matrix',
                                args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
