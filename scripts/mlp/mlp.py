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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import get_columns_type

data_path = join(Path(__file__).parent.parent.parent, 'data')


if __name__ == '__main__':
    args = utils.argument_parser().parse_args()
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = get_columns_type(x_train)
    name_str = 'mlp'
    results = {'name': name_str}

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('encoder', OneHotEncoder())
                                              ])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, n_cols),
                                                   ('categorical', categorical_transformer, c_cols)])

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)
    y_train_end = y_train_transformed.mean(axis=1)
    y_test_end = y_test_transformed.mean(axis=1)

    preprocessor.fit(x_train)
    x_train_transformed = preprocessor.transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    clf = MLPRegressor(random_state=utils.seed, hidden_layer_sizes=(10, 5), max_iter=500)
    param_grid = {'hidden_layer_sizes': [(100,), (25, 25), (50,), (25,)]}
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring='r2', n_jobs=8)
    # Grid search
    g_search.fit(x_train_transformed, y_train_end)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train_transformed, y_train_end, clf)
    clf.fit(x_train_transformed, y_train_end.ravel())

    y_pred = clf.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_end, y_pred)

    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_end, y_pred)

    print(utils.json_metrics_to_latex(results))

    utils.plot_scattered_error(y_test_end, y_pred, 'Scattered error plot for MLP', 'Observations', 'IEMedia',
                               args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test_end.ravel(), y_pred, 'Class', 'Count', 'Error count for MLP',
                                           args.save_figures, name_str)
    utils.save_dict_as_json(args.json_output, name_str, results)
