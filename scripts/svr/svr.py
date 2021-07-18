"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import one_hot_encoder, get_columns_type

data_path = join(Path(__file__).parent.parent.parent, 'data')


def preprocessing(undersampling_thr=None, feature_selection=None):
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    c_cols, n_cols = get_columns_type(x_train)

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, n_cols),
                                                   ('categorical', categorical_transformer, c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')

    y_train_transformed = y_imputer.fit_transform(y_train).mean(axis=1)
    y_test_transformed = y_imputer.transform(y_test).mean(axis=1)

    preprocessor.fit(x_train)
    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    if undersampling_thr:
        x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed,
                                                                                  y_train_transformed,
                                                                                  (1, 7), undersampling_thr)
    if feature_selection:
        x_train_transformed, x_test_transformed = utils.feature_selection(x_train_transformed, x_test_transformed,
                                                                          y_train_transformed, feature_selection)

    return x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed


if __name__ == '__main__':
    args = utils.argument_parser().parse_args()
    name_str = 'svr'

    if args.undersampling:
        name_str = f"{name_str}_{str(args.undersampling).replace('.', '_')}_undersamp"

    if args.feature_selection:
        name_str = f'{name_str}_{args.feature_selection}_feature_selection'

    x_train, y_train, x_test, y_test = preprocessing(args.undersampling, args.feature_selection)

    results = {'name': name_str}

    param_grid = {'kernel': ['poly', 'rbf'],
                  'degree': [2, 3, 4, 5],
                  'C': [0.5, 0.75, 1]
                  }

    clf = SVR()
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring='r2', n_jobs=8)

    # Grid search
    g_search.fit(x_train, y_train)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')
    results['cross_validation'] = utils.cross_validation(x_train.to_numpy(), y_train, clf)

    clf.fit(x_train, y_train.ravel())
    y_pred = clf.predict(x_train)
    results['train'] = utils.calculate_regression_metrics(y_train, y_pred)

    y_pred = clf.predict(x_test)
    results['test'] = utils.calculate_regression_metrics(y_test, y_pred)
    print(utils.json_metrics_to_latex(results))

    utils.plot_scattered_error(y_test, y_pred, 'Scattered error plot SVR', 'Observations', 'IEMedia',
                               args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test.ravel(), y_pred, 'Class', 'Count', 'Error count SVR',
                                           args.save_figures, name_str)

    print(clf.get_params())

    utils.save_dict_as_json(args.json_output, name_str, results)
