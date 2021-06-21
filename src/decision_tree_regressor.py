"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from packages.manual_preprocessing import data_path, get_columns_type, one_hot_encoder
import packages.utils as utils
import pandas as pd
from os.path import join
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt


def preprocessing():
    """Function to perform the preprocessing for Decision Trees. This function will read the dataset and perform
    the preprocessing steps for decision trees.
    Returns:
        tuple: x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed
    """

    x_train = pd.read_csv(join(data_path, 'x_train.csv'))
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
    results = {}
    # Parse arguments
    args = utils.argument_parser()

    x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed = preprocessing()

    param_grid = {'max_depth': [2, 4, 8, 9, 10, 16],
                  'max_leaf_nodes': [2, 4, 8, 16, 20, 22]}

    g_search = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, scoring='r2')

    # Grid search
    g_search.fit(x_train_transformed, y_train_transformed)
    print(f"Best score {g_search.best_score_} with {g_search.best_estimator_}")
    best = g_search.best_estimator_
    print(f"Best parameters {best.get_params()}")

    results['cross_validation'] = utils.cross_validation(x_train_transformed.to_numpy(), y_train_transformed, best)

    best.fit(x_train_transformed, y_train_transformed.ravel())

    print("Train score")
    y_pred = best.predict(x_train_transformed)
    results['train'] = utils.calculate_regression_metrics(y_train_transformed, y_pred)

    print("Test score")
    y_pred = best.predict(x_test_transformed)
    results['test'] = utils.calculate_regression_metrics(y_test_transformed, y_pred)

    utils.plot_scattered_error(y_test_transformed, y_pred, 'Scattered error plot for decission tree',
                               'Observations', 'IEMedia', args.save_figures, 'dtree')

    results['hist'] = utils.get_error_hist(y_test_transformed.ravel(), y_pred, 'Class', 'Count',
                                           'Error count for decision tree', args.save_figures, 'dtree')

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train_transformed.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, 'decision_tree_plot.png'))
        plt.clf()
    else:
        plt.show()

    features_importances = pd.Series(data=best.feature_importances_, index=x_train_transformed.columns)
    results['feature_importances'] = dict(features_importances)

    print(utils.tree_to_code(best, x_train_transformed.columns.to_numpy()))
    utils.plot_feature_importances(features_importances, 10, 'Variable', 'Importance',
                                   'Decission Tree features', args.save_figures, 'dtree')
    if args.json_output:
        import json
        with open(join(utils.results_path, 'decision_tree_1.json'), mode='w') as fd:
            json.dump(results, fd)
