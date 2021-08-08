from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
import numpy as np
from abc import ABC


class BaseOrdinalClassifier(BaseEstimator, ClassifierMixin, ABC):
    def fit(self, X, y):
        self.binary_classifiers = {}
        self.classes_ = None
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.sort(np.unique(y))

        assert len(self.classes_) > 2, 'At least 2 or more classes are needed'

        # Train k - 1 binary models
        for i in range(len(self.classes_) - 1):
            class_ = self.classes_[i]
            binary_y = (y > self.classes_[i]).astype(np.uint8)
            clf = clone(self.clf_)
            clf.fit(X, binary_y)
            self.binary_classifiers[class_] = clf

        return self

    def predict_proba(self, X):
        """Function that will predict the class probability for an input dataset.
           For each sample, predict_proba (for each trained model) will return a 2D array where the first column is
           the probability of that sample to be false and the second the probability of that sample of be true. As we
           are only interested in the true value, we discard the first column
        Args:
            X (array): Dataset to predict.
        Returns:
            Array: 2D array with the probabilities of each sample to belong a class.
        """
        predicted = []
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'binary_classifiers'])

        clfs_predict = {class_: self.binary_classifiers[class_].predict_proba(
            X) for class_ in self.binary_classifiers}
        for class_index, class_ in enumerate(self.classes_):
            if class_index == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[class_][:, 1])

            elif class_ in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(
                    clfs_predict[class_-1][:, 1] - clfs_predict[class_][:, 1])

            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[class_-1][:, 1])

        return np.vstack(predicted).T

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, ['classes_', 'clf_', 'binary_classifiers'])
        return np.argmax(self.predict_proba(X), axis=1)
