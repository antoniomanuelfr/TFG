"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
import manual_preprocessing as mp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import pandas as pd
from os.path import join

if __name__ == '__main__':
    x_train = pd.read_csv(join(mp.data_path, 'x_train.csv'))
    y_train = pd.read_csv(join(mp.data_path, 'y_train.csv'))
    x_test = pd.read_csv(join(mp.data_path, 'x_test.csv'))
    y_test = pd.read_csv(join(mp.data_path, 'y_test.csv'))
    y_cols = y_train.columns

    categorical_columns, numeric_columns = mp.get_columns_type(x_train)

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                              ('encoder', OneHotEncoder(handle_unknown='error'))])
    numeric_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                          ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_columns),
                                                   ('cat', categorical_transformer, categorical_columns)])
    y_imputer = SimpleImputer(strategy='median')
    y_train = y_imputer.fit_transform(y_train)
    y_test = y_imputer.transform(y_test)

    y_train_final = pd.DataFrame(y_train, columns=y_cols)
    y_test_final = pd.DataFrame(y_test, columns=y_cols)

    x_train_final = preprocessor.fit_transform(x_train, y_train_final['IEMedia'])
    x_test_final = preprocessor.transform(x_test)

    clf = RandomForestRegressor(random_state=0).fit(x_train_final, y_train_final['IEMedia'])
    y_pred = clf.predict(x_test_final)
    print(r2_score(y_test_final['IEMedia'], y_pred))
