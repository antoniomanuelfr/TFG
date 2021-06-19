"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from packages.manual_preprocessing import data_path, get_columns_type, one_hot_encoder
import packages.utils as utils
import numpy as np
import pandas as pd
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


def preprocessing(X_train, Y_train, X_test, Y_test):
    """Function to perform the preprocessing for Decision Trees.
    Params:
        X_train: Train dataset.
        Y_train: Labels for training dataset.
        X_test: Test dataset.
        Y_test: Labels for test dataset.
    Retruns:
        tuple: x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed
    """
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

    x_train = pd.read_csv(join(data_path, 'x_train.csv'))
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    param_grid = {'max_depth': [2, 4, 8, 9, 10, 16],
                  'max_leaf_nodes': [2, 4, 8, 16, 20, 22]}

    g_search = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, scoring='r2')

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing(x_train, y_train,
                                                                                                     x_test, y_test)

    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    best = g_search.best_estimator_
    print(f"Best parameters {best.get_params()}")

    acum_res = np.array([0, 0, 0])
    print('Validation results')
    print('r2, mean poisson deviance, mse')
    folder = KFold(n_splits=5, random_state=10, shuffle=True)
    for train_index, test_index in folder.split(x_train_transformed.to_numpy(), y_train_transformed):
        fold_train_x, fold_train_y = x_train_transformed.iloc[train_index], y_train_transformed[train_index]
        fold_test_x, fold_test_y = x_train_transformed.iloc[test_index], y_train_transformed[test_index]

        best.fit(fold_train_x, fold_train_y)
        y_pred = best.predict(fold_test_x)
        res = utils.calculate_regression_metrics(fold_test_y, y_pred)
        acum_res = acum_res + res

        print(','.join(map(str, res)))

    print("Means in validation")
    acum_res = acum_res / 5
    print(','.join(map(str, np.round(acum_res, 3))))

    best.fit(x_train_transformed, y_train_transformed.ravel())
    print("Train score")
    y_pred = best.predict(x_train_transformed)
    train_res = utils.calculate_regression_metrics(y_train_transformed, y_pred)
    print(','.join(map(str, train_res)))

    print("Test score")
    y_pred = best.predict(x_test_transformed)
    test_res = utils.calculate_regression_metrics(y_test_transformed, y_pred)
    print(','.join(map(str, test_res)))

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for decission tree',
                               'Observations', 'IEMedia', args.save_figures, 'dtree')

    utils.get_error_hist(y_test_transformed.ravel(), y_pred, 'Class', 'Count', 'Error count for decission tree',
                         args.save_figures, 'dtree')

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train_transformed.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, 'decision_tree_plot.png'))
        plt.clf()
    else:
        plt.show()

    feature_importances = pd.Series(data=best.feature_importances_, index=x_train_transformed.columns)

    print(utils.tree_to_code(best, x_train_transformed.columns.to_numpy()))
    utils.plot_feature_importances(feature_importances, 10, 'Variable', 'Importance', 'Decission Tree features',
                                   args.save_figures, 'dtree')
