"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path
import warnings

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV

from tfg_utils.manual_preprocessing import get_columns_type, one_hot_encoder
import tfg_utils.utils as utils

# This warning is raised when creating a DMatrix. It's a normal behavior (https://github.com/dmlc/xgboost/issues/6908)
warnings.filterwarnings(action='ignore', category=UserWarning)

data_path = join(Path(__file__).parent.parent.parent, 'data')

if __name__ == '__main__':
    args = utils.argument_parser().parse_args()
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = get_columns_type(x_train)
    name_str = 'xgboost'
    type_dict = {}
    results = {'name': name_str}

    for col in n_cols:
        type_dict[col] = x_train[col].dtype

    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')

    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)
    y_train_end = y_train_transformed.mean(axis=1)
    y_test_end = y_test_transformed.mean(axis=1)

    preprocessor.fit(x_train)
    p_x_train = preprocessor.transform(x_train)
    p_x_test = preprocessor.transform(x_test)
    x_train_transformed = pd.DataFrame(p_x_train, columns=transformed_cols)
    x_test_transformed = pd.DataFrame(p_x_test, columns=transformed_cols)

    # OneHotEncode for each category separately
    x_final_train, x_final_test = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)
    x_final_train = x_final_train.astype(type_dict)
    x_final_test = x_final_test.astype(type_dict)

    acum_res = np.array([0, 0, 0])
    param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 100]}

    g_search = GridSearchCV(xgb.XGBRegressor(random_state=utils.seed), param_grid=param_grid, scoring='r2')

    g_search.fit(x_final_train, y_train_end)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')
    results['cross_validation'] = utils.cross_validation(x_final_train.to_numpy(), y_train_end, clf)

    clf.fit(x_final_train, y_train_end)

    y_pred = clf.predict(x_final_train)
    results['train'] = utils.calculate_regression_metrics(y_train_end, y_pred)

    y_pred = clf.predict(x_final_test)
    results['test'] = utils.calculate_regression_metrics(y_test_end, y_pred)
    print(utils.json_metrics_to_latex(results))

    utils.plot_scattered_error(y_test_end, y_pred, 'Scattered error plot for XGBoost', 'Observations', 'IEMedia',
                               args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test_end.ravel(), y_pred, 'Class', 'Count', 'Error count for XGBoost',
                                           args.save_figures, name_str)
    print(clf.get_params())

    utils.save_dict_as_json(args.json_output, name_str, results)
