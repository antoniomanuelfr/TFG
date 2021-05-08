"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_poisson_deviance, mean_squared_error


def get_scattered_error_plot(y_true: np.array, y_pred: np.array, title, xlabel, ylabel):
    _, ax = plt.subplots()

    ax.scatter(x=range(0, y_true.size), y=y_true, c='blue', label='Real', alpha=0.3)
    ax.scatter(x=range(0, y_pred.size), y=y_pred, c='red', label='Prediction', alpha=0.3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    return ax


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    poi = mean_poisson_deviance(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return np.array([r2, poi, mse])


def get_error_hist(y_true: np.array, y_pred: np.array, offset: float):
    res = np.zeros(shape=[len(np.unique(np.round(y_true))) + 1])

    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) < offset:
            index = int(round(true))
            res[index] = res[index] + 1

    return res
