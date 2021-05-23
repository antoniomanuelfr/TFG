"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import manual_preprocessing as mp
import utils
import pandas as pd
import numpy as np
from os.path import join
from pandas.core.frame import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib as plt

if __name__ == '__main__':
    args = utils.argument_parser()

    x_train = pd.read_csv(join(mp.data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(mp.data_path, 'y_train.csv'), usecols=['IEMedia'], index_col=False)
    x_test = pd.read_csv(join(mp.data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(mp.data_path, 'y_test.csv'), usecols=['IEMedia'], index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = mp.get_columns_type(x_train)
    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)

    preprocessor.fit(x_train)
    x_train_transformed = DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    # OneHotEncode for each category separately
    x_train_transformed, x_test_transformed = mp.one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)
    clf = RandomForestRegressor(random_state=0)
    folder = KFold(n_splits=5, random_state=10, shuffle=True)

    acum_res = np.array([0, 0, 0])
    print('Validation results')
    print('r2, mean poisson deviance, mse')
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_transformed[train_index].ravel()
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_transformed[test_index].ravel()

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)
        res = utils.calculate_regression_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5
    print(','.join(map(str, acum_res)))

    clf.fit(x_train_transformed, y_train_transformed.ravel())
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    train_res = utils.calculate_regression_metrics(y_train_transformed, y_pred)
    print(','.join(map(str, train_res)))

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    test_res = utils.calculate_regression_metrics(y_test_transformed, y_pred)
    print(','.join(map(str, test_res)))

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for Random Forest',
                               'Observations', 'IEMedia', args.save_figures, 'random_forest')

    utils.get_error_hist(y_test_transformed.ravel(), y_pred, 0.5, 'Class', 'Count', 'Error count for Random Forest',
                         args.save_figures, 'random forest')

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(feature_importances.sort_values(ascending=False)[:10])
