"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import r2_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import _tree

results_path = os.path.join(Path(__file__).parent.parent, 'results')


def argument_parser():
    """Argument parser to share between all scripts."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')

    parser.add_argument('--json_output', type=str, action='store', default=None,
                        help='Save script output to a json file')

    return parser


def plot_scattered_error(y_true: np.array, y_pred: np.array, title: str, xlabel: str, ylabel: str, save=None,
                         extra=None):
    """Plot an scattered plot with the true and predicted values of a sample.
        Args:
            y_true (Numpy array): Array with the true value of the sample
            y_pred (Numpy array): Array with the predicted value of the sample
            title (str): Title of the plot.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    _, ax = plt.subplots()

    ax.scatter(x=range(0, y_true.size), y=y_true, c='blue', label='Real', alpha=0.3)
    ax.scatter(x=range(0, y_pred.size), y=y_pred, c='red', label='Prediction', alpha=0.3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save, f'scattered_error_{extra}.png'))
        plt.clf()
    else:
        plt.show()


def calculate_regression_metrics(y_true, y_pred, decimals=3):
    """Calculates the regression metrics.
        Args:
            y_true (Numpy array): Array with the true value of the sample.
            y_pred (Numpy array): Array with the predicted value of the sample.
            decimals (int): Number of decimals to round.
        Returns:
            A numpy array with the used metrics (r2, poisson deviance and mse).
    """
    res = {'r2': round(r2_score(y_true, y_pred), decimals),
           'poisson': round(mean_poisson_deviance(y_true, y_pred), decimals),
           'mse': round(mean_squared_error(y_true, y_pred), decimals)
           }
    print(res)
    return res


def get_error_hist(y_true: np.array, y_pred: np.array, xlabel, ylabel, title, save=None, extra=None):
    """Calculates and plot an error bar plot.
       This plot is calculated by checking if the difference between a predicted and a true value of a sample is bigger
       than a threshold. If it's bigger, the sample is consider as an error and increments the error count of the
       predicted class. (We have a regression problem that can be consider as a classification problem)

       Args:
            y_true (Numpy array): Array with the true value of the sample
            y_pred (Numpy array): Array with the predicted value of the sample
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            title (str): Title of the plot.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    res = np.zeros(shape=[len(np.unique(np.round(y_true)))])

    for true, pred in zip(y_true, y_pred):
        if round(true) != round(pred):
            index = int(round(true))
            if index != 0:
                index = index - 1
            res[index] = res[index] + 1

    for index, data in enumerate(res):
        plt.text(x=index+0.75, y=data+1, s=f"{data}")

    plt.bar(np.arange(1, len(res) + 1), res)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(os.path.join(save, f'error_hist_{extra}.png'))
        plt.clf()
    else:
        plt.show()

    return list(res)


def tree_to_code(tree, feature_names):
    """Function to get the decission rules from a SKlearn decission tree.
    This code has been adapted from:
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    Args:
        tree (SKLearn DecissionTree): Decission tree to extract rules.
        feature_names (numpy array): Array with the names of the features.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]

    def expand_branch(node, depth):
        """Recursive function to expand the tree
        Args
            Node (int): Node of the tree.
            depth (int): Depth of the node
        """
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Expand left branch
            lines = f"{indent}if {feature_name[node]} <= {np.round(tree_.threshold[node], 2)}:\n"
            ret = expand_branch(tree_.children_left[node], depth + 1)
            lines = lines + ret
            lines = lines + f"{indent}else:  # if {feature_name[node]} > {np.round(tree_.threshold[node], 2)}\n"
            # Expand right branch
            ret = expand_branch(tree_.children_right[node], depth + 1)
            lines = lines + ret
            return lines
        else:
            return f"{indent}return {np.round(float(tree_.value[node]), 2)}\n"

    return expand_branch(0, 0)


def plot_feature_importances(feature_importances: pd.Series, n: int, xlabel, ylabel, title, save=None, extra=None):
    to_plot = feature_importances.sort_values(ascending=False)[:n]

    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(0, len(to_plot)), to_plot.values, width=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(np.arange(0, len(to_plot)), to_plot.index)

    if save:
        plt.savefig(os.path.join(save, f'feature_importance_{extra}.png'))
        plt.clf()
    else:
        plt.show()


def regression_under_sampler(x_data: pd.DataFrame, y_data: np.array, range: tuple, threshold: float, predictor):
    """A simple under-sample for regression.
    Args:
        x_data(DataFrame): Train data used for training the model.
        y_data(DataFrame): Real values for train data.
        range(tuple): Range from where the under-sampler will remove the data. (min, max)
        threshold(float): If the difference between a prediction and a real value is bigger, the sample will be removed.
        predictor: Model to train. The model need the method fit to train it and preddict to make the predictions.
    Returns:
        List: indexes of the rows to delete.
    """
    rows_to_delete = []
    predictor.fit(x_data, y_data)
    for row, _ in x_data.iterrows():
        if y_data[row] < range[0] or y_data[row] > range[1]:
            continue

        prediction = predictor.predict(x_data.iloc[[row]])
        if abs(prediction - y_data[row]) > threshold:
            rows_to_delete.append(row)

    return x_data.drop(index=rows_to_delete), np.delete(y_data, rows_to_delete)


def cross_validation(x_train: np.array, y_train: np.array, model, splits=5, seed=10, shuffle=True, decimals=3):
    cnt = 0
    results = {}
    acum_res = {'r2': 0,
                'poisson': 0,
                'mse': 0
                }

    folder = KFold(n_splits=splits, random_state=seed, shuffle=shuffle)

    print('Cross validation results')
    print('r2, mean poisson deviance, mse')

    for train_index, test_index in folder.split(x_train, y_train):
        fold_train_x, fold_train_y = x_train[train_index], y_train[train_index]
        fold_test_x, fold_test_y = x_train[test_index], y_train[test_index]

        model.fit(fold_train_x, fold_train_y)
        y_pred = model.predict(fold_test_x)

        res = calculate_regression_metrics(fold_test_y, y_pred)

        results[f'fold_{cnt}'] = res
        acum_res['r2'] = acum_res['r2'] + res['r2']
        acum_res['poisson'] = acum_res['poisson'] + res['poisson']
        acum_res['mse'] = acum_res['mse'] + res['mse']

        cnt += 1

    print("Means in validation")
    acum_res['r2'] = round(acum_res['r2'] / splits, decimals)
    acum_res['poisson'] = round(acum_res['poisson'] / splits, decimals)
    acum_res['mse'] = round(acum_res['mse'] / splits, decimals)
    print(acum_res)

    results['validation_mean'] = acum_res

    return results
