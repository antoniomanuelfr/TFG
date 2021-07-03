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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import one_hot_encoder, get_columns_type

data_path = join(Path(__file__).parent.parent.parent, "data")


def preprocessing():
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    c_cols, n_cols = get_columns_type(x_train)

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, n_cols),
                                                   ('categorical', categorical_transformer, c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')

    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)
    y_train_end = y_train_transformed.mean(axis=1)
    y_test_end = y_test_transformed.mean(axis=1)

    preprocessor.fit(x_train)
    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    return x_train_transformed, y_train_end, x_test_transformed, y_test_end


if __name__ == '__main__':
    args = utils.argument_parser().parse_args()
    name_str = 'svr'
    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing()

    if args.undersampling:
        from sklearn.tree import DecisionTreeRegressor
        name_str = 'svr_undersamp'
        under_sampler = DecisionTreeRegressor(max_depth=4, random_state=utils.seed)
        x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed,
                                                                                  y_train_transformed,
                                                                                  (4.5, 7), 0.8, under_sampler)
    results = {'name': name_str}
    param_grid = {'kernel': ['linear', 'poly', 'rbf'],
                  'degree': [2, 3, 4, 5],
                  'C': [0.5, 0.75, 1]
                  }

    clf = SVR()
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring='r2', n_jobs=8)

    # Grid search
    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    clf = g_search.best_estimator_
    print(f"Best parameters {clf.get_params()}")
    results['cross_validation'] = utils.cross_validation(x_train_transformed.to_numpy(), y_train_transformed, clf)

    clf.fit(x_train_transformed, y_train_transformed.ravel())
    print("Train score")
    y_pred = clf.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = clf.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot SVR', 'Observations', 'IEMedia',
                               args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test_transformed.ravel(), y_pred, 'Class', 'Count', 'Error count SVR',
                                           args.save_figures, name_str)

    print(clf.get_params())

    utils.save_dict_as_json(args.json_output, name_str, results)
