"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from pandas.core.frame import DataFrame
import manual_preprocessing as mp
import pandas as pd
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_poisson_deviance


if __name__ == '__main__':
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
    r2_acum = 0
    poi_acum = 0
    mse_acum = 0
    clf = RandomForestRegressor(random_state=0)
    folder = KFold(n_splits=5, random_state=10, shuffle=True)
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_transformed[train_index].ravel()
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_transformed[test_index].ravel()

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)

        r2 = r2_score(fold_test_y, y_pred)
        poi = mean_poisson_deviance(fold_test_y, y_pred)
        mse = mean_squared_error(fold_test_y, y_pred)
        r2_acum = r2_acum + r2
        poi_acum = poi_acum + poi
        mse_acum = mse_acum + mse

        print(f"r2 = {r2}, poi = {poi}, mse = {mse}")

    clf.fit(x_train_transformed, y_train_transformed.ravel())

    print("TRAIN SCORE")
    y_pred = clf.predict(x_train_transformed)
    r2 = r2_score(y_train_transformed, y_pred)
    poi = mean_poisson_deviance(y_train_transformed, y_pred)
    mse = mean_squared_error(y_train_transformed, y_pred)
    print(f"r2 = {r2}, poi = {poi}, mse = {mse}")

    print("TEST SCORE")
    y_pred = clf.predict(x_test_transformed)
    r2 = r2_score(y_test_transformed, y_pred)
    poi = mean_poisson_deviance(y_test_transformed, y_pred)
    mse = mean_squared_error(y_test_transformed, y_pred)
    print(f"r2 = {r2}, poi = {poi}, mse = {mse}")
