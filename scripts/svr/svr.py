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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import get_columns_type

data_path = join(Path(__file__).parent.parent.parent, "data")


def preprocessing(X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame):
    c_cols, n_cols = get_columns_type(X_train)

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('encoder', OneHotEncoder())
                                              ])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, n_cols),
                                                   ('categorical', categorical_transformer, c_cols)])

    y_imputer = SimpleImputer(strategy='median')

    y_train_transformed = y_imputer.fit_transform(Y_train)
    y_test_transformed = y_imputer.transform(Y_test)
    y_train_end = y_train_transformed.mean(axis=1)
    y_test_end = y_test_transformed.mean(axis=1)

    preprocessor.fit(X_train)
    x_train_transformed = preprocessor.transform(X_train)
    x_test_transformed = preprocessor.transform(X_test)

    return x_train_transformed, y_train_end, x_test_transformed, y_test_end


if __name__ == '__main__':
    args = utils.argument_parser()

    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    results = {}

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing(x_train, y_train,
                                                                                                     x_test, y_test)

    clf = SVR()
    results['cross_validation'] = utils.cross_validation(x_train_transformed, y_train_transformed, clf)

    clf.fit(x_train_transformed, y_train_transformed.ravel())
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot SVR', 'Observations', 'IEMedia',
                               args.save_figures, 'svr')

    results['hist'] = utils.get_error_hist(y_test_transformed.ravel(), y_pred, 'Class', 'Count', 'Error count SVR',
                                           args.save_figures, 'svr')

    print(clf.get_params())

    if args.json_output:
        import json
        with open(join(args.json_output, 'svr_1.json'), mode='w') as fd:
            json.dump(results, fd)
