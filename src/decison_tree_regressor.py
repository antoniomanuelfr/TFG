"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import manual_preprocessing as mp
import pandas as pd
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_poisson_deviance


if __name__ == '__main__':
    x_train = pd.read_csv(join(mp.data_path, 'x_train.csv'))
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
    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    # OneHotEncode for each category separately
    x_train_transformed, x_test_transformed = mp.one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    clf = DecisionTreeRegressor(random_state=0).fit(x_train_transformed, y_train_transformed)
    y_pred = clf.predict(x_test_transformed)
    y_train_pred = clf.predict(x_train_transformed)

    print(r2_score(y_train_transformed, y_train_pred))
    print(r2_score(y_test_transformed, y_pred))

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(feature_importances.sort_values(ascending=False)[:10])

    param_grid = {'max_depth': [2, 4, 8, 9, 10, 16],
                  'max_leaf_nodes': [2, 4, 8, 16, 20, 22]}

    g_search = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, scoring='r2')

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    best = g_search.best_estimator_

    r2_acum = 0
    poi_acum = 0
    mse_acum = 0

    folder = KFold(n_splits=5, random_state=10, shuffle=True)
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_transformed[train_index]
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_transformed[test_index]

        best.fit(fold_train_x, fold_train_y)
        y_pred = best.predict(fold_test_x)
        r2 = r2_score(fold_test_y, y_pred)
        poi = mean_poisson_deviance(fold_test_y, y_pred)
        mse = mean_squared_error(fold_test_y, y_pred)
        r2_acum = r2_acum + r2
        poi_acum = poi_acum + poi
        mse_acum = mse_acum + mse
        print(f"r2 = {r2}, poi = {poi}, mse = {mse}")

    best.fit(x_train_transformed, y_train_transformed.ravel())
    print("TRAIN SCORE")
    y_pred = best.predict(x_train_transformed)
    r2 = r2_score(y_train_transformed, y_pred)
    poi = mean_poisson_deviance(y_train_transformed, y_pred)
    mse = mean_squared_error(y_train_transformed, y_pred)
    print(f"r2 = {r2}, poi = {poi}, mse = {mse}")

    print("TEST SCORE")
    y_pred = best.predict(x_test_transformed)
    r2 = r2_score(y_test_transformed, y_pred)
    poi = mean_poisson_deviance(y_test_transformed, y_pred)
    mse = mean_squared_error(y_test_transformed, y_pred)
    print(f"r2 = {r2}, poi = {poi}, mse = {mse}")
