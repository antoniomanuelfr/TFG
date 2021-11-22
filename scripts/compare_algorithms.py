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
import tfg_utils.utils as utils


def generate_label_str(def_str: str):
    return def_str.replace('0_', '0.').replace('_', ' ').replace('-', '.').replace('undersamp', '')\
           .replace('classification', '')


def load_json(path: str):
    """Function to load a JSON from a file.
    Args:
        path (str): Path to the JSON file
    Returns:
        dict: Dictionary with the contents of the JSON file
    """
    with open(path) as fp:
        return json.load(fp)


def comp_error_ranges(models: dict, xlabel: str, ylabel: str, save, extra: str):
    """Compare the error ranges between two algorithms.
        Args:
            models (dict): Dictionary where the key is the label and the value is the error counts.
            model_2 (tuple): (Error per class, label)
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
    """
    data = []
    for model in models.keys():
        row = []
        for i in range(len(models[model])):
            row = [generate_label_str(model), i, models[model][i]]
            data.append(row)

    data = pd.DataFrame(data, columns=['model', 'class', 'value'])

    sns.barplot(x='class', y='value', hue='model', data=data, palette=utils.palette)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save:
        plt.savefig(os.path.join(save, f'{extra}.png'))
        plt.clf()
    else:
        plt.show()


def compare_metrics(models: dict,  xlabel: str, ylabel: str, save, extra: str, plot_values=False):
    """Compare the metrics between two models.
        Args:
            models (dict): Dictionary where the key is the label and the value is a dict with the metrics results.
            xlabel (str): Label of the X axis.
            ylabel (str): Label of the y axis.
            save (str): Folder where the images will be saved. If None, the image will be shown.
            extra (str): Extra name to append at the end of the file name if save is not none.
   """
    data = []
    for model in models.keys():
        row = []
        for metric in models[model]:
            row = [generate_label_str(model), utils.metric_name_parser[metric], models[model][metric]]
            data.append(row)

    data = pd.DataFrame(data, columns=['model', 'metric', 'value'])

    ax = sns.barplot(x='metric', y='value', hue='model', data=data, palette=utils.palette)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_values:
        for container in ax.containers:
            ax.bar_label(container)

    if save:
        plt.savefig(os.path.join(save, f'{extra}.png'))
        plt.clf()
    else:
        plt.show()

def compare_feature_importance(feature_importance_dict: dict, xlabel, ylabel, n=10, save=None, extra=None, plot_values=False):
    data =  []
    for algorithm_name in feature_importance_dict:
        important_features = pd.Series(feature_importance_dict[algorithm_name]).sort_values(ascending=False)[:n]
        important_features = important_features[important_features > 0.01]
        for f_name in important_features.index:
            data.append([generate_label_str(algorithm_name), f_name, important_features[f_name]])

    data = pd.DataFrame(data, columns=['Name', 'Feature', 'Importance'])
    ax = sns.barplot(x='Feature', y='Importance', hue='Name', data=data, palette=utils.palette)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0.)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_values:
        for container in ax.containers:
            ax.bar_label(container)

    for item in ax.get_xticklabels():
        item.set_rotation(45)

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
    feature_importance_dict = {}
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', dest='algorithms', action='append', required=True,
                        help='Path with the json results of the first algorithm')
    parser.add_argument('--save_figures', type=str, action='store', default=None,
                        help='Save figures in the specified folder')
    parser.add_argument('--output_name', type=str, action='store', default=None,
                        help='Name of the output files. '
                             'If None, the name will be a hash of the names of all the values')
    parser.add_argument('--plot_values', action='store_true', default=False,
                        help="Specify if the values will be added to the bar plots")
    parser.add_argument('--only_feature', action='store_true', default=False,
                        help="Calculate only the feature importance differece.")

    arguments = parser.parse_args()

    if arguments.save_figures is not None and arguments.output_name is None:
        parser.error('Argument "output_name" required if saving figures.')

    assert len(arguments.algorithms) >= 2, 'Need at least to algorithms to compare'

    cmp_str = arguments.output_name
    for res in arguments.algorithms:
        json_results = load_json(res)
        ## Calculate histogram dict
        alg_name = json_results['name']
        if 'hist' in res:
            histogram_dict[alg_name] = json_results['hist']

        # Calculate validation metrics dict
        validation_dict[alg_name] = json_results['cross_validation']['validation_mean']
        # Calculate test metrics dict
        test_dict[alg_name] = json_results['test']

        if 'feature_importance' in json_results:
            feature_importance_dict[alg_name] = json_results['feature_importance']

    # plot feature importance
    compare_feature_importance(feature_importance_dict, 'Features', 'Importance', 10, arguments.save_figures,
                               f'{cmp_str}_features')
    if arguments.only_feature:
        exit(0)

    if histogram_dict:
        comp_error_ranges(histogram_dict, 'Class', 'Error count', arguments.save_figures,
                          f'{cmp_str}_error_hist')

    # Plot metrics

    compare_metrics(validation_dict, 'Metric', 'Metric value', arguments.save_figures, f'{cmp_str}_val_metrics',
                    arguments.plot_values)

    compare_metrics(test_dict, 'Metric', 'Metric value', arguments.save_figures, f'{cmp_str}_test_metrics',
                    arguments.plot_values)
