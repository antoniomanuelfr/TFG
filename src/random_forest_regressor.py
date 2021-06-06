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
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    args = utils.argument_parser()

    x_train = pd.read_csv(join(mp.data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(mp.data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(mp.data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(mp.data_path, 'y_test.csv'), index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = mp.get_columns_type(x_train)
    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)
    y_train_end = y_train_transformed.mean(axis=1)
    y_test_end = y_test_transformed.mean(axis=1)

    preprocessor.fit(x_train)
    x_train_transformed = DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    # OneHotEncode for each category separately
    x_train_transformed, x_test_transformed = mp.one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    param_grid = {'n_estimators': [450, 475, 500],
                  'max_features': [None, 1/3]}

    g_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param_grid, scoring='r2', n_jobs=-1)

    g_search.fit(x_train_transformed, y_train_end)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    clf = g_search.best_estimator_
    print(f"Best parameters {clf.get_params()}")
    folder = KFold(n_splits=5, random_state=10, shuffle=True)

    acum_res = np.array([0, 0, 0])
    print('Validation results')
    print('r2, mean poisson deviance, mse')
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_end):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_end[train_index]
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_end[test_index]

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)
        res = utils.calculate_regression_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5
    print(','.join(map(str, acum_res)))

    clf.fit(x_train_transformed, y_train_end)
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    train_res = utils.calculate_regression_metrics(y_train_end, y_pred)
    print(','.join(map(str, train_res)))

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    test_res = utils.calculate_regression_metrics(y_test_end, y_pred)
    print(','.join(map(str, test_res)))

    utils.plot_scattered_error(y_test_end, y_pred, 'Scattered error plot for Random Forest',
                               'Observations', 'IEMedia', args.save_figures, 'random_forest')

    utils.get_error_hist(y_test_end, y_pred, 'Class', 'Count', 'Error count for Random Forest',
                         args.save_figures, 'random_forest')

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(f"{clf.get_params()}")
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Random Forest features',
                                   args.save_figures, 'rf')
