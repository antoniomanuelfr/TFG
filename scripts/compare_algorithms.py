"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import os
import argparse
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_label_str(def_str: str):
    return def_str.replace('0_', '0.').replace('_', ' ').replace('-', '.')


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
        plt.bar(np.arange(1, len(models[key]) + 1), models[key], label=generate_label_str(key))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save, f'{extra}.png'))
        plt.clf()
    else:
        plt.show()


def compare_metrics(models: dict,  xlabel: str, ylabel: str, title: str, save, extra: str):
    """Compare the metrics between two models.
        Args:
            models (dict): Dictionary where the key is the label and the value is a dict with the metrics results.
            title (str): Title of the plot.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
   """
    max_metric_dic = {}
    for model in models.keys():
        values = []
        for metric in models[model]:
            metric_value = models[model][metric]

            if metric not in max_metric_dic:
                max_metric_dic[metric] = metric_value
            else:
                if max_metric_dic[metric] < metric_value:
                    max_metric_dic[metric] = metric_value

            values.append(metric_value)

        model_str = model.replace('0_', '0.').replace('_', ' ')
        plt.bar(list(models[model].keys()), values, label=generate_label_str(model_str))

    for index, metric in enumerate(max_metric_dic):
        plt.text(x=index-0.25, y=max_metric_dic[metric]+0.01, s=f'{max_metric_dic[metric]}')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='center')
    if save:
        plt.savefig(os.path.join(save, f'{extra}.png'))
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    alg_list = []
    histogram_dict = {}
    validation_dict = {}
    test_dict = {}
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', dest='algorithms', action='append', required=True,
                        help='Path with the json results of the first algorithm')

    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')
    parser.add_argument('--output_name', type=str, action='store', default=None,
                        help='Name of the output files. If None, the name will be a hash of the names of all the values'
                        )

    arguments = parser.parse_args()

    if arguments.save_figures is not None and arguments.output_name is None:
        parser.error('Argument "output_name" required if saving figures.')

    assert len(arguments.algorithms) >= 2, 'Need at least to algorithms to compare'

    cmp_str = arguments.output_name
    for res in arguments.algorithms:
        alg_list.append(load_json(res))

    for res in alg_list:
        alg_name = res['name']
        if 'hist' in res:
            histogram_dict[alg_name] = res['hist']
        validation_dict[alg_name] = res['cross_validation']['validation_mean']
        test_dict[alg_name] = res['test']
    if histogram_dict:
        comp_error_ranges(histogram_dict, 'Class', 'Error count', 'Error plot', arguments.save_figures,
                          f'{cmp_str}_error_hist')

    compare_metrics(validation_dict, 'Metric', 'Metric value', 'Validations mean comparison', arguments.save_figures,
                    f'{cmp_str}_val_metrics')

    compare_metrics(test_dict, 'Metric', 'Metric value', 'Test comparison', arguments.save_figures,
                    f'{cmp_str}_test_metrics')
