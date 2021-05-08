"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import manual_preprocessing as mp
import utils
import numpy as np
import pandas as pd
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings

# This warning is raised when creating a DMatrix. It's a normal behaviour (https://github.com/dmlc/xgboost/issues/6908)
warnings.filterwarnings(action='ignore', category=UserWarning)

if __name__ == '__main__':
    x_train = pd.read_csv(join(mp.data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(mp.data_path, 'y_train.csv'), usecols=['IEMedia'], index_col=False)
    x_test = pd.read_csv(join(mp.data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(mp.data_path, 'y_test.csv'), usecols=['IEMedia'], index_col=False)
    x_cols = x_train.columns
    c_cols, n_cols = mp.get_columns_type(x_train)

    type_dict = {}

    for col in n_cols:
        type_dict[col] = x_train[col].dtype

    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)

    preprocessor.fit(x_train)
    p_x_train = preprocessor.transform(x_train)
    p_x_test = preprocessor.transform(x_test)
    x_train_transformed = pd.DataFrame(p_x_train, columns=transformed_cols)
    x_test_transformed = pd.DataFrame(p_x_test, columns=transformed_cols)

    # OneHotEncode for each category separately
    x_final_train, x_final_test = mp.one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)
    x_final_train = x_final_train.astype(type_dict)
    x_final_test = x_final_test.astype(type_dict)

    acum_res = np.array([0, 0, 0])

    print('Validation results')
    print('r2, mean poisson deviance, mse')
    clf = xgb.XGBRegressor(random_state=0)
    folder = KFold(n_splits=5, random_state=10, shuffle=True)

    for train_index, test_index in folder.split(x_final_train.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_final_train.iloc[train_index], y_train_transformed[train_index].ravel()
        fold_test_x, fold_test_y = x_final_train.iloc[test_index], y_train_transformed[test_index].ravel()

        clf.fit(fold_train_x, fold_train_y)
        y_pred = clf.predict(fold_test_x)

        res = utils.calculate_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5
    print(','.join(map(str, acum_res)))
    clf.fit(x_final_train, y_train_transformed)

    print("Train score")
    y_pred = clf.predict(x_final_train)
    train_res = utils.calculate_metrics(y_train_transformed, y_pred)
    print(','.join(map(str, train_res)))

    print("Test score")
    y_pred = clf.predict(x_final_test)
    test_res = utils.calculate_metrics(y_test_transformed, y_pred)
    print(','.join(map(str, test_res)))
