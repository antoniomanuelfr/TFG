"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree

import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import get_columns_type, one_hot_encoder


data_path = join(Path(__file__).parent.parent.parent, 'data')


def preprocessing(undersampling=False, feature_selection=False):
    """Function to perform the preprocessing for Decision Trees. This function will read the dataset and perform
    the preprocessing steps for decision trees.
    Args:
        undersampling (bool): Flag to specify if the function will do the undersampling process before returning data.
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

    if undersampling:
        x_train_transformed, y_train_transformed = utils.regression_under_sampler(x_train_transformed,
                                                                                  y_train_transformed,
                                                                                  (4.5, 7), 0.8, undersampling)
    if feature_selection:
        x_train_transformed, x_test_transformed = utils.feature_selection(x_train_transformed, x_test_transformed,
                                                                          y_train_transformed, feature_selection)

    return x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed


if __name__ == '__main__':
    # Parse arguments
    args = utils.argument_parser().parse_args()
    name_str = 'dtree'

    if args.undersampling:
        name_str = f'{name_str}_{args.undersampling}_undersamp'

    if args.feature_selection:
        name_str = f'{name_str}_{args.feature_selection}_feature_selection'

    x_train, y_train, x_test, y_test = preprocessing(args.undersampling, args.feature_selection)

    results = {'name': name_str}

    param_grid = {'max_depth': [2, 4, 8, 9, 10, 16],
                  'max_leaf_nodes': [2, 4, 8, 16, 20, 22]}

    g_search = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid, scoring='r2')

    # Grid search
    g_search.fit(x_train, y_train)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    best = g_search.best_estimator_
    print(f'Best parameters {best.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train.to_numpy(), y_train, best)

    best.fit(x_train, y_train.ravel())

    y_pred = best.predict(x_train)
    results['train'] = utils.calculate_regression_metrics(y_train, y_pred)

    y_pred = best.predict(x_test)
    results['test'] = utils.calculate_regression_metrics(y_test, y_pred)

    print(utils.json_metrics_to_latex(results))

    utils.plot_scattered_error(y_test, y_pred, 'Scattered error plot for decission tree',
                               'Observations', 'IEMedia', args.save_figures, name_str)

    results['hist'] = utils.get_error_hist(y_test.ravel(), y_pred, 'Class', 'Count',
                                           'Error count for decision tree', args.save_figures, name_str)

    plt.figure(figsize=(20, 10))
    plot_tree(best, feature_names=x_train.columns, filled=True, fontsize=8)
    if args.save_figures:
        plt.savefig(join(args.save_figures, f'{name_str}_plot'))
        plt.clf()
    else:
        plt.show()

    feature_importance = pd.Series(data=best.feature_importances_, index=x_train.columns)
    results['feature_importance'] = dict(feature_importance)

    print(utils.tree_to_code(best, x_train.columns.to_numpy()))
    utils.plot_feature_importance(feature_importance, 10, 'Variable', 'Importance',
                                  'Decission Tree features', args.save_figures, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
