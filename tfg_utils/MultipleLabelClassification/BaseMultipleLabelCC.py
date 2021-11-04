"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
from abc import ABC
from sklearn.tree import DecisionTreeClassifier


class BaseMultipleLabelCC(BaseEstimator, ClassifierMixin, ABC):
    """Abstract class to implement a simple Multible Label Classifier. It will train a given classifier per label.

    Attributes:
        classifiers (dict): Dictionay where each key will be the label and it's value will be the trained classifier.
        clf_ (BaseEstimator): Classifier to use.
        classes_ (Array): Array holding the labels to predict
    """
    def fit(self, X, y):
        """Fit method that will train each classifier.
        Args:
            X (array): Array holding the train dataset.
            y (array): Array holding the true values of the dataset.
        Returns:
            A `BaseMultipleLabelCC` trained model.
        """
        self.classifiers = {}
        self.classes_ = np.arange(y.shape[1])
        self.clf_ = DecisionTreeClassifier()

        for y_column, label_id in zip(y.transpose(), self.classes_):
            X, y_column = check_X_y(X, y_column)
            check_classification_targets(y)
            classifier = clone(self.clf_)
            classifier.fit(X, y_column)
            self.classifiers[label_id] = classifier

        return self

    def predict(self, X):
        """Method that will predict the values for a dataset.
        Args:
            X (array): Set to predict.
        Returns:
            Array with the predicted values.
        """
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'classifiers'])
        labels_prediction = []
        for column in self.classes_:
            prediction = self.classifiers[column].predict(X)
            labels_prediction.append(prediction)

        return np.array(labels_prediction).transpose()

    def predict_proba(self, X):
    """Method that will predict the probability of each label of having a given valuew.
        Args:
            X (array): Set to predict.
        Returns:
            Array with the predicted values.
    """
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'classifiers'])
        labels_prediction = []
        for column in self.classes_:
            prediction = self.classifiers[column].predict_proba(X)
            labels_prediction.append(prediction)

        return np.array(labels_prediction).transpose()
