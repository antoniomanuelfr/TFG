"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import os
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    """Function to load a JSON from a file.
    Args:
        path (str): Path to the JSON file
    Returns:
        dict: Dictionary with the contents of the JSON file
    """
    with open(path) as fp:
        return json.load(fp)


def comp_error_ranges(models: dict, xlabel: str, ylabel: str, title: str, save, extra: str):
    """Compare the error ranges between two algorithms.
        Args:
            models (dict): Dictionary where the key is the label and the value is the error counts.
            model_2 (tuple): (Error per class, label)
            title (str): Title of the plot.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    for key in models.keys():
        plt.bar(np.arange(1, len(models[key]) + 1), models[key], label=key)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save, f'error_hist_{extra}.png'))
        plt.clf()
    else:
        plt.show()


def compare_metrics(models: dict,  xlabel: str, ylabel: str, title: str, save, extra: str):
    """Compare the metrics between two models.
        Args:
            models (dict): Dictionary where the key is the label and the value is a dict with the metrics resutls.
            title (str): Title of the plot.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
   """
    for key in models.keys():
        values = []
        for value in models[key]:
            values.append(models[key][value])
        plt.bar(list(models[key].keys()), values, label=key)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save, f'error_hist_{extra}.png'))
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--results1', type=str, action='store', required=True,
                        help='Path with the json results of the first algorithm')

    parser.add_argument('--results2', type=str, action='store', required=True,
                        help='Path with the json results of the second algorithm')

    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')

    args = parser.parse_args()

    alg_1 = load_json(args.results1)
    alg_2 = load_json(args.results2)

    cmp_str = f"{alg_1['name']}_{alg_2['name']}"

    d = {alg_1['name']: alg_1['hist'],
         alg_2['name']: alg_2['hist'],
         }
    comp_error_ranges(d, 'Class', 'Error count', 'Error plot', args.save_figures, f'{cmp_str}_error_hist')

    d[alg_1['name']] = alg_1['cross_validation']['validation_mean']
    d[alg_2['name']] = alg_2['cross_validation']['validation_mean']

    compare_metrics(d, 'Metric', 'Metric value', 'Validations mean comparison', args.save_figures, f'{cmp_str}_metrics')

    d[alg_1['name']] = alg_1['test']
    d[alg_2['name']] = alg_2['test']

    compare_metrics(d, 'Metric', 'Metric value', 'Test comparison', args.save_figures, f'{cmp_str}_metrics')
