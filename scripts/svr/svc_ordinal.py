"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from svr import preprocessing
from tfg_utils.OrdinalClassification.OrdinalClassifiers import OrdinalSVC
import tfg_utils.utils as utils

data_path = join(Path(__file__).parent.parent.parent, 'data')


if __name__ == '__main__':
    name_str = 'rf'
    args = utils.argument_parser().parse_args()

    param_grid = {'kernel': ['poly', 'rbf'],
                  'degree': [2, 3, 4, 5],
                  'C': [0.5, 0.75, 1]}

    if args.undersampling:
        name_str = f"{name_str}_{str(args.undersampling).replace('.', '_')}_undersamp"
    if args.ranges:
        assert len(args.ranges) == 2, 'Len of ranges parameters must be 2'
        c_str = ''
        for i in args.ranges:
            c_str = f"{c_str}_{str(i).replace('.0', '').replace('.', '-')}"
        name_str = f'{name_str}{c_str}'

    x_train_p, y_train_p, x_test_p, y_test_p = preprocessing(args.undersampling)
    y_train_p, y_test_p = utils.categorize_regression(y_train_p, y_test_p, args.ranges)
    results = {'name': name_str}
    clf = OrdinalSVC()
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro', n_jobs=-1)

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

    utils.plot_confusion_matrix(y_test_p, y_pred, ['IEBaja', 'IEMedia', 'IEAlta'], 'Confusion matrix',
                                args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
