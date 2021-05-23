"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import r2_score, mean_poisson_deviance, mean_squared_error
from os.path import join, exists
from os import mkdir


def argument_parser():
    """Argument parser to share between all scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')
    args = parser.parse_args()
    if args.save_figures:
        if not exists(args.save_figures):
            mkdir(args.save_figures)
    return args


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
        plt.savefig(join(save, f'scattered_error_{extra}.png'))
    else:
        plt.show()


def calculate_regression_metrics(y_true, y_pred):
    """Calculates the regression metrics.
        Args:
            y_true (Numpy array): Array with the true value of the sample
            y_pred (Numpy array): Array with the predicted value of the sample
        Returns:
            A numpy array with the used metrics (r2, poisson deviance and mse).
    """
    r2 = r2_score(y_true, y_pred)
    poi = mean_poisson_deviance(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return np.array([r2, poi, mse])


def get_error_hist(y_true: np.array, y_pred: np.array, threshold: float, xlabel, ylabel, title, save=None, extra=None):
    """Calculates and plot an error bar plot.
       This plot is calculated by checking if the difference between a predicted and a true value of a sample is bigger
       than a threshold. If it's bigger, the sample is consider as an error and increments the error count of the
       predicted class. (We have a regression problem that can be consider as a classification problem)

       Args:
            y_true (Numpy array): Array with the true value of the sample
            y_pred (Numpy array): Array with the predicted value of the sample
            threshold (float): If the difference between two values is bigger, then it will consider as an error.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            title (str): Title of the plot.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    res = np.zeros(shape=[len(np.unique(np.round(y_true)))])

    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) < threshold:
            index = int(round(true))
            if index != 0:
                index = index - 1
            res[index] = res[index] + 1

    plt.bar(np.arange(1, len(res) + 1), res)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(join(save, f'error_hist_{extra}.png'))
    else:
        plt.show()
