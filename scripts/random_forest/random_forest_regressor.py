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


def preprocessing():
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    c_cols, n_cols = get_columns_type(x_train)
    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)

    y_train_transformed = y_train_transformed.mean(axis=1)
    y_test_transformed = y_test_transformed.mean(axis=1)

    preprocessor.fit(x_train)
    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    # OneHotEncode for each category separately
    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    return x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed


if __name__ == '__main__':
    name_str = 'rf'
    args = utils.argument_parser().parse_args()

    param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90],
                  'max_features': [None, 1/3]}

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing()

    if args.undersampling:
        from sklearn.tree import DecisionTreeRegressor
        name_str = 'rf_undersamp'
        under_sampler = DecisionTreeRegressor(max_depth=4, random_state=utils.seed)
        x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed,
                                                                                  y_train_transformed,
                                                                                  (4.5, 7), 0.8, under_sampler)
    results = {'name': name_str}

    g_search = GridSearchCV(RandomForestRegressor(random_state=utils.seed), param_grid=param_grid, scoring='r2', n_jobs=-1)

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
                               'Observations', 'IEMedia', args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test_transformed, y_pred, 'Class', 'Count',
                                           'Error count for Random Forest', args.save_figures, name_str)

    feature_importances = pd.Series(data=clf.feature_importances_, index=x_train_transformed.columns)
    print(f"{clf.get_params()}")
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Random Forest features',
                                   args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
