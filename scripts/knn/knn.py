"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import pandas as pd
from os.path import join
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

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
    name_str = 'knn'
    results = {'name': name_str}

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('encoder', OneHotEncoder())
                                              ])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', MinMaxScaler())])
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
    clf = KNeighborsRegressor()

    results['cross_validation'] = utils.cross_validation(x_train_transformed, y_train_end, clf)

    clf.fit(x_train_transformed, y_train_end.ravel())
    y_pred = clf.predict(x_train_transformed)

    results['train'] = utils.calculate_regression_metrics(y_train_end, y_pred)

    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_end, y_pred)

    print(utils.json_metrics_to_latex(results))

    utils.plot_scattered_error(y_test_end, y_pred, 'Scattered error plot for KNN', 'Observations', 'IEMedia',
                               args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test_end.ravel(), y_pred, 'Class', 'Count', 'Error count for KNN',
                                           args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
