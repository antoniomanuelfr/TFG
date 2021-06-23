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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from tfg_utils.manual_preprocessing import get_columns_type, one_hot_encoder
import tfg_utils.utils as utils

data_path = join(Path(__file__).parent.parent.parent, "data")


def preprocessing(X_train, Y_train, X_test, Y_test):
    c_cols, n_cols = get_columns_type(X_train)
    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(Y_train)
    y_test_transformed = y_imputer.transform(Y_test)

    y_train_transformed = y_train_transformed.mean(axis=1)
    y_test_transformed = y_test_transformed.mean(axis=1)

    preprocessor.fit(X_train)
    x_train_transformed = pd.DataFrame(preprocessor.transform(X_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(X_test), columns=transformed_cols)

    # OneHotEncode for each category separately
    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    return x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed


if __name__ == '__main__':
    args = utils.argument_parser()

    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    results = {}
    param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90],
                  'max_features': [None, 1/3]}

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing(x_train, y_train,
                                                                                                     x_test, y_test)

    g_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param_grid, scoring='r2', n_jobs=-1)

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    clf = g_search.best_estimator_
    print(f"Best parameters {clf.get_params()}")

    results['cross_validation'] = utils.cross_validation(x_train_transformed.to_numpy(), y_train_transformed, clf)

    clf.fit(x_train_transformed, y_train_transformed)
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for Random Forest',
                               'Observations', 'IEMedia', args.save_figures, 'random_forest')

    results['hist'] = utils.get_error_hist(y_test_transformed, y_pred, 'Class', 'Count',
                                           'Error count for Random Forest', args.save_figures, 'random_forest')

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(f"{clf.get_params()}")
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Random Forest features',
                                   args.save_figures, 'rf')

    if args.json_output:
        import json
        with open(join(args.json_output, 'random_forest_1.json'), mode='w') as fd:
            json.dump(results, fd)