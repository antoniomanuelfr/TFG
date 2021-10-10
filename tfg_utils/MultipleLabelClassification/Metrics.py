"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
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
    for y_true_column, y_pred_column, proba_col, label_id in zip(y_true.transpose(), y_pred.transpose(), proba.transpose(), range(y_true.shape[1])):
        metrics = {'metrics': {'f1': round(f1_score(y_true_column, y_pred_column, average='macro'), decimals),
                               'auc_score': round(roc_auc_score(y_true_column, proba_col, average='macro',
                                                                multi_class='ovo'), decimals),
                               'accuracy': round(accuracy_score(y_true_column, y_pred_column), decimals)
                              }
                  }
        res[label_id] = metrics
    return res


def f1_multilabel_mean(y_true, y_pred, average='macro'):
    score = 0
    for y_true_column, y_pred_column in zip(y_true.transpose(), y_pred.transpose()):
        score += f1_score(y_true_column, y_pred_column, average=average)
    return score / y_true.shape[1]
