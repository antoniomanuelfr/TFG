"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from packages.manual_preprocessing import data_path, get_columns_type, one_hot_encoder
import packages.utils as utils
import pandas as pd
import numpy as np
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


if __name__ == '__main__':
    args = utils.argument_parser()
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = get_columns_type(x_train)

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
    folder = KFold(n_splits=5, random_state=10, shuffle=True)
    acum_res = np.array([0, 0, 0])

    print('Validation results')
    print('r2, mean poisson deviance, mse')
    for train_index, test_index in folder.split(x_train_transformed, y_train_end):
        fold_train_x, fold_train_y = x_train_transformed[train_index], y_train_end[train_index].ravel()
        fold_test_x, fold_test_y = x_train_transformed[test_index], y_train_end[test_index].ravel()

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)

        res = utils.calculate_regression_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5
    print(','.join(map(str, np.round(acum_res, 3))))

    clf.fit(x_train_transformed, y_train_end.ravel())
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    train_res = utils.calculate_regression_metrics(y_train_end, y_pred)
    print(','.join(map(str, train_res)))

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    test_res = utils.calculate_regression_metrics(y_test_end, y_pred)
    print(','.join(map(str, test_res)))

    utils.plot_scattered_error(y_test_end, y_pred, 'Scattered error plot for KNN', 'Observations', 'IEMedia',
                               args.save_figures, 'knn')

    utils.get_error_hist(y_test_end.ravel(), y_pred, 'Class', 'Count', 'Error count for KNN',
                         args.save_figures, 'knn')
