"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
# Functions that will be used for comparing the algorithm behavior
import os
import matplotlib.pyplot as plt
import numpy as np


def comp_error_hist(models: dict, xlabel: str, ylabel: str, title: str, save, extra: str):
    """
        Args:
            model_1 (tuple): (Error per class, label)
            model_2 (tuple): (Error per class, label)
    """
    for key in models.keys():
        plt.bar(np.arange(1, len(models[key][0]) + 1), models[key][0], label=models[key][1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save, f'error_hist_{extra}.png'))
        plt.clf()
    else:
        plt.show()
