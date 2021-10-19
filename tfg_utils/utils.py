"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_poisson_deviance, mean_squared_error, f1_score, roc_auc_score, \
                            accuracy_score, confusion_matrix, hamming_loss, multilabel_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import _tree
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from scipy.sparse import issparse

palette = 'Set2'
plot_color = 'mediumaquamarine'
seed = 10

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


metric_name_parser = {
        'r2': 'R2',
        'poisson': 'Poisson Deviance',
        'mse': 'MSE',
        'f1': 'F1 Score',
        'auc_score': 'AUC Score',
        'accuracy': 'Accuracy'
    }


def argument_parser():
    """Argument parser to share between all scripts."""

    class Range(object):
        """This class is restricted to the argument function,
           as it use is only to check if a floating number is in a range of values. This looks complex, but it's the
           best way to check for correctness of the parameters.
        """
        def __init__(self, start: float, end: float):
            """Constructor for the class.
            Args:
                start (float): Lower boundary.
                end (float): Higher boundary
            """
            self.start = start
            self.end = end

        def __eq__(self, needle: float):
            """Equal operator for the class. This function will check if a value is in the specific range.
               Args:
                needle(float): Value to check if it's inside the range.
            """
            return self.start <= needle <= self.end

        def __contains__(self, needle: float):
            """Contains method that will be used when argparse checks for the choices. It will call to the __eq__
               methdod.
               Args:
                needle(float): Value to check if it's inside the range.
            """
            return self.__eq__(needle)

        def __iter__(self):
            yield self

        def __repr__(self):
            """Get a string with the data to show an error message when an invalid choice is introduced."""
            return f'[{self.start}, {self.end}]'

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')

    parser.add_argument('--json_output', type=str, action='store', default=None,
                        help='Save script output to a json file')

    parser.add_argument('--undersampling', type=float, action='store', choices=Range(0.1, 1),
                        help='Use undersampler with a given threshold.')

    parser.add_argument('--feature_selection', action='store', choices=['SVR', 'DTREE'], default=False,
                        help='Perform a feature selection using the given model.')

    parser.add_argument('--ranges', action='store', type=float, default=None, nargs='+',
                        help='Ranges used to categorize the regression. This parameters is ignored in regression')
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
            A dictionary where the key is the name of the metric and the value is the value of the metric.
    """
    res = {'r2': round(r2_score(y_true, y_pred), decimals),
           'poisson': round(mean_poisson_deviance(y_true, y_pred), decimals),
           'mse': round(mean_squared_error(y_true, y_pred), decimals)
           }
    return res


def calculate_classification_metrics(y_true, y_pred, proba, decimals=3):
    """Calculates the classification metrics.
        Args:
            y_true (Numpy array): Array with the true value of the sample.
            y_pred (Numpy array): Array with the predicted value of the sample.
            proba (Numpy array): Array of shape (n_samples, n_classes) of probability estimates provided by the
                                 predict_proba method
            decimals (int): Number of decimals to round.
        Returns:
            A dictionary where the key is the name of the metric and the value is the value of the metric.
    """
    y_pred_use = y_pred
    proba_use = proba

    if issparse(y_pred):
        y_pred_use = y_pred.toarray()
    if issparse(proba):
        proba_use = proba.toarray()

    res = {'f1': round(f1_score(y_true, y_pred_use, average='macro'), decimals,),
           'auc_score': round(roc_auc_score(y_true, proba_use, average='macro', multi_class='ovo'), decimals),
           'accuracy': round(accuracy_score(y_true, y_pred_use), decimals)
           }
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
    plot_data = []

    for true, pred in zip(y_true, y_pred):
        if round(true) != round(pred):
            index = int(round(true))
            if index != 0:
                index = index - 1
            res[index] = res[index] + 1

    for _class, err_count in zip(np.arange(1, len(res) + 1), res):
        plot_data.append([_class, err_count])
    plot_data = pd.DataFrame(data=plot_data, columns=['class', 'count'])

    ax = sns.barplot(x='class', y='count', data=plot_data, color=plot_color)
    ax.bar_label(ax.containers[0])

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
        feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined!'
        for i in tree_.feature
    ]
    feature_names = [f.replace(' ', '_')[:-5] for f in feature_names]

    def expand_branch(node, depth):
        """Recursive function to expand the tree
        Args
            Node (int): Node of the tree.
            depth (int): Depth of the node
        """
        indent = '    ' * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # Expand left branch
            lines = f'{indent}if {feature_name[node]} <= {np.round(tree_.threshold[node], 2)}:\n'
            ret = expand_branch(tree_.children_left[node], depth + 1)
            lines = lines + ret
            lines = lines + f'{indent}else:  # if {feature_name[node]} > {np.round(tree_.threshold[node], 2)}\n'
            # Expand right branch
            ret = expand_branch(tree_.children_right[node], depth + 1)
            lines = lines + ret
            return lines
        else:
            return f'{indent}return {np.round(float(tree_.value[node]), 2)}\n'

    return expand_branch(0, 0)


def plot_feature_importance(feature_importance: pd.Series, n: int, xlabel, ylabel, title, save=None, extra=None):
    """ Function to plot the features importance.
        Args:
            feature_importance (Series): Pandas series with the name of the variable and the importance of the variable.
            n (int): Number of variables to plot
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            title (str): Title of the plot.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    to_plot = feature_importance.sort_values(ascending=False)[:n]

    plt.figure(figsize=(10, 8))
    data = []

    for feature, value in zip(to_plot.index, to_plot.values):
        data.append([feature, value])
    data = pd.DataFrame(data=data, columns=['feature', 'value'])

    ax = sns.barplot(x='feature', y='value', data=data, color=plot_color)
    for item in ax.get_xticklabels():
        item.set_rotation(90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save:
        plt.savefig(os.path.join(save, f'feature_importance_{extra}.png'))
        plt.clf()
    else:
        plt.show()


def regression_under_sampler(x_data: pd.DataFrame, y_data: np.array, range: tuple, threshold: float):
    """A simple under-sample for regression.
    Args:
        x_data(DataFrame): Train data used for training the model.
        y_data(DataFrame): Real values for train data.
        range(tuple): Range from where the under-sampler will remove the data. (min, max)
        threshold(float): If the difference between a prediction and a real value is bigger, the sample will be removed.
    Returns:
        List: indexes of the rows to delete.
    """
    from sklearn.tree import DecisionTreeRegressor
    predictor = DecisionTreeRegressor(max_depth=4, random_state=seed)

    rows_to_delete = []
    predictor.fit(x_data, y_data)
    for row, _ in x_data.iterrows():
        if y_data[row] < range[0] or y_data[row] > range[1]:
            continue

        prediction = predictor.predict(x_data.iloc[[row]])
        if abs(prediction - y_data[row]) > threshold:
            rows_to_delete.append(row)
    print(f'Removing {len(rows_to_delete)} samples of the training dataset.')
    return x_data.drop(index=rows_to_delete), np.delete(y_data, rows_to_delete)


def cross_validation(x_train: np.array, y_train: np.array, model, splits=5, custom_seed=seed, shuffle=True, decimals=3,
                     metric_callback=calculate_regression_metrics):
    """Function to perform the cross validation process.
        Args:
            x_train (Numpy array): Training data.
            y_tain (Numpy array): Training data labels.
            model: Model to train.
            splits: Number of splits in cross validation. Defaults to 5.
            custom_seed: seed to use. Defaults to `seed`.
            shuffle: True if the data is going to be shuffled. Defaults to True.
            decimals: Max number of decimals in results. Defaults to 3.
            metric_callback: Function that will calculate the metrics. It must return a dictionary where the key
                             is the name of the metric and the value is the value of the metric.
    """
    cnt = 0
    results = {}
    acum_res = {}
    multi_label = False
    n_labels = 1

    try:
        multi_label = y_train.shape[1] > 1
        n_labels = y_train.shape[1]
    except IndexError:
        pass

    folder = KFold(n_splits=splits, random_state=custom_seed, shuffle=shuffle)

    for train_index, test_index in folder.split(x_train, y_train):
        fold_train_x, fold_train_y = x_train[train_index], y_train[train_index]
        fold_test_x, fold_test_y = x_train[test_index], y_train[test_index]

        model.fit(fold_train_x, fold_train_y)
        y_pred = model.predict(fold_test_x)

        if metric_callback == calculate_regression_metrics:
            res = metric_callback(fold_test_y, y_pred)
        else:
            res = metric_callback(fold_test_y, y_pred, model.predict_proba(fold_test_x))

        if not multi_label:
            for key in res.keys():
                if key not in acum_res.keys():
                    acum_res[key] = res[key]
                else:
                    acum_res[key] = acum_res[key] + res[key]
        else:
            for label in res.keys():
                if label not in acum_res.keys():
                    acum_res[label] = res[label]
                else:
                    for metric in res[label]:
                        acum_res[label][metric] += res[label][metric]

        results[f'fold_{cnt}'] = res
        cnt += 1

    if not multi_label:
        for key in acum_res.keys():
            acum_res[key] = round(acum_res[key] / splits, decimals)

        results['validation_mean'] = acum_res

    else:
        mean_dict = {}
        for label in acum_res:
            label_metrics = acum_res[label]
            for metric in label_metrics:
                if metric not in mean_dict:
                    mean_dict[metric] = (acum_res[label][metric] / splits) / n_labels
                else:
                    mean_dict[metric] = mean_dict[metric] + ((acum_res[label][metric] / splits) / n_labels)

        for it in mean_dict:
            mean_dict[it] = round(mean_dict[it], decimals)

        results['validation_mean'] = mean_dict

    return results


def json_metrics_to_latex(res_dic: dict):
    metrics = list(res_dic['train'].keys())
    for i in range(len(metrics)):
        metrics[i] = metric_name_parser[metrics[i]]

    metrics_str = ' & '.join(metrics)
    res_str = f'FOLD & {metrics_str} \\\\\n'
    cnt = 0
    cross_validation_dict = res_dic['cross_validation']
    for fold in cross_validation_dict:
        res_str = f'{res_str}Fold {cnt}'
        for metric in cross_validation_dict[fold]:
            res_str = f'{res_str} & {cross_validation_dict[fold][metric]}'

        res_str = f'{res_str}\\\\ \n'
        cnt += 1

    res_str = f'{res_str}Train'
    for metric in res_dic['train']:
        res_str = f"{res_str} & {res_dic['train'][metric]}"

    res_str = f'{res_str}\\\\\nTest'
    for metric in res_dic['test']:
        res_str = f"{res_str} & {res_dic['test'][metric]}"

    return f'{res_str}\\\\\n'


def save_dict_as_json(path: str, name_str: str, dict_to_save: dict):
    """Function that saves a dictionary in a JSON file.
    Args:
        path (str): Path where the file will be stored.
        name_str (str): Name of the JSON file (without the extension)
        dict_to_save (dict): Dictionary to save.
    """
    if path is None or name_str is None:
        return

    with open(os.path.join(path, f'{name_str}.json'), mode='w') as fd:
        json.dump(dict_to_save, fd)


def feature_selection(x_train, x_test, y_train, predictor_t):
    from sklearn.svm import LinearSVR
    from sklearn.tree import DecisionTreeRegressor
    clf = DecisionTreeRegressor(max_depth=4, random_state=seed)

    if predictor_t == 'SVR':
        clf = LinearSVR(random_state=seed, max_iter=1500)

    columns = x_train.columns
    clf.fit(x_train, y_train)

    model = SelectFromModel(clf, prefit=True)
    x_train_new = model.transform(x_train)
    x_test_new = model.transform(x_test)
    columns = x_train.columns[model.get_support()]
    print(f'Columns selected: {columns}')

    return pd.DataFrame(data=x_train_new, columns=columns), pd.DataFrame(data=x_test_new, columns=columns)


def categorize_regression(y_train: np.array, y_test: np.array, ranges=(2.5, 5)):
    """Transforms the regression problem into a classification by setting a class for each value in a interval
    Args:
        y_train (Numpy Array): Numpy array with the train values of y.
        y_test (Numpy Array): Numpy array with the test values of y.
    """
    max_value = int(max(max(y_train), max(y_test)))

    y_train = np.where(y_train <= int(ranges[0]), 0, y_train)
    y_test = np.where(y_test <= int(ranges[0]), 0, y_test)

    y_train = np.where((y_train <= int(ranges[1])) & (y_train > 1), 1, y_train)
    y_test = np.where((y_test <= int(ranges[1])) & (y_test > 1), 1, y_test)
    y_train = np.where((y_train <= max_value) & (y_train > 2), 2, y_train)
    y_test = np.where((y_test <= max_value) & (y_test > 2), 2, y_test)

    return y_train, y_test


def plot_confusion_matrix(y_true: np.array, y_pred: np.array, labels, title='', save=None, extra=None):
    matrix = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        plt.savefig(os.path.join(save, f'confusion_matrix_{extra}.png'))
        plt.clf()
    else:
        plt.show()
    return matrix


def calculate_ml_classification_metrics(y_true, y_pred, proba, decimals=3):
    """Calculates the classification metrics.
        Args:
            y_true (Numpy array): Array with the true value of the sample.
            y_pred (Numpy array): Array with the predicted value of the sample.
            proba (Numpy array): Array of shape (n_samples, n_classes) of probability estimates provided by the
                                 predict_proba method
            decimals (int): Number of decimals to round.
        Returns:
            A dictionary where the key is the name of the metric and the value is the value of the metric.
    """
    res = {}
    y_pred_use = y_pred
    proba_use = proba

    if issparse(y_pred):
        y_pred_use = y_pred.toarray()

    if issparse(proba):
        proba_use = proba.toarray()

    for y_true_column, y_pred_column, proba_col, label_id in zip(y_true.transpose(), y_pred_use.transpose(),
                                                                 proba_use.transpose(), range(y_true.shape[1])):
        res[label_id] = {'f1': round(f1_score(y_true_column, y_pred_column, average='macro'), decimals),
                         'auc_score': round(roc_auc_score(y_true_column, proba_col, average='macro', multi_class='ovo'),
                                            decimals),
                         'accuracy': round(accuracy_score(y_true_column, y_pred_column), decimals)
                        }
    return res


def f1_multilabel_mean(y_true, y_pred, average='macro'):
    score = 0
    for y_true_column, y_pred_column in zip(y_true.transpose(), y_pred.transpose()):
        score += f1_score(y_true_column, y_pred_column, average=average)
    return score / y_true.shape[1]


def plot_multilabel_class_metrics(metric_dict, plot_values=False, save_figures=None, metric_names=None, name=None):
    data = []
    for label in metric_dict.keys():
        test_metrict_dict = metric_dict[label]
        for metric_value_name in test_metrict_dict:
            label_name = label if metric_names is None else metric_names[label]
            data.append([label_name, metric_value_name, test_metrict_dict[metric_value_name]])

    data = pd.DataFrame(data, columns=['label', 'metric', 'value'])
    ax = sns.barplot(x='metric', y='value', hue='label', data=data, palette=palette)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)

    if plot_values:
        for container in ax.containers:
            ax.bar_label(container)

    if save_figures:
        plt.savefig(os.path.join(save_figures, f"{name}_compare.png"))
        plt.clf()
    else:
        plt.show()


def ml_feature_importance_ml(ml_classifier, columns_names, label_names, n, xlabel, ylabel, title, save=None, extra=None):
    #calculate the feature importance for each classifier
    importance_acum = np.zeros(ml_classifier.classifiers_[0].feature_importances_.shape)
    result = {}
    data = []
    for classifier, label in zip(ml_classifier.classifiers_, label_names):
        importance_acum += classifier.feature_importances_
        label_importances = pd.Series(data=classifier.feature_importances_, index=columns_names).sort_values(ascending=False)[:n]
        label_importances = label_importances[label_importances > 0]

        for variable in label_importances.index:
            data.append([label, variable, label_importances[variable]])

    importance_acum = importance_acum / ml_classifier.model_count_

    data_to_plot = pd.DataFrame(data, columns=['Label', 'Variable', 'Importance'])
    ax = sns.barplot(x='Variable', y='Importance', hue='Label', data=data_to_plot, palette=palette)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    for item in ax.get_xticklabels():
        item.set_rotation(90)

    if save:
        plt.savefig(os.path.join(save, f"feature_importance_{extra}_compare.png"))
        plt.clf()
    else:
        plt.show()

    return plot_feature_importance(pd.Series(data=importance_acum, index=columns_names), n, xlabel, ylabel,
                                   f"{title}_mean", save, f"{extra}_mean")


def plot_multi_label_confusion_matrix(y_true: np.array, y_pred: np.array, labels, label_names, title='', save=None, extra=None):

    matrixes = multilabel_confusion_matrix(y_true, y_pred)/y_pred.shape[0]
    for (label_matrix, label) in zip(matrixes, label_names):
        sns.heatmap(label_matrix, xticklabels=labels, yticklabels=labels, annot=True)
        plt.title(f'{title}_{label}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if save:
            plt.savefig(os.path.join(save, f'confusion_matrix_{label}_{extra}.png'))
            plt.clf()
        else:
            plt.show()
