"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier

import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import get_columns_type, one_hot_encoder, get_missing_values, classification_predictor

data_path = join(Path(__file__).parent.parent.parent, 'data')


def preprocessing(undersampling_thr=None, bin_thr=5):
    """Function to perform the preprocessing for Decision Trees. This function will read the dataset and perform
    the preprocessing steps for decision trees.
    Args:
        undersampling_thr (float): Threshold to use when performing a undersampling. If None, the undersampling won't
                                   be applied.
        feature_selection (str): Model to use for doing a feature selection. If None, the feature selection process
                                 won't be applied.
    Returns:
        tuple: x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed
    """
    x_train = pd.read_csv(join(data_path, 'x_train.csv'))
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)
    type_dict = {}


    c_cols, n_cols = get_columns_type(x_train)
    preprocessor = ColumnTransformer(transformers=[('numerical',  KNNImputer(n_neighbors=2, weights='uniform'), n_cols),
                                                   ('categorical', SimpleImputer(strategy='most_frequent'), c_cols)])
    transformed_cols = n_cols + c_cols

    for col in n_cols:
        type_dict[col] = x_train[col].dtype

    y_imputer = SimpleImputer(strategy='median')
    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)

    y_train_final = np.where(y_train_transformed < bin_thr, 0, y_train_transformed)
    y_train_final = np.where(y_train_final != 0, 1, y_train_final)

    y_test_final = np.where(y_test_transformed < bin_thr, 0, y_test_transformed)
    y_test_final = np.where(y_test_final != 0, 1, y_test_final)

    preprocessor.fit(x_train)

    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)
    # OneHotEncode for each category separately

    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)
    x_train_transformed = x_train_transformed.astype(type_dict)
    x_test_transformed = x_test_transformed.astype(type_dict)

    return x_train_transformed, y_train_final, x_test_transformed, y_test_final


if __name__ == '__main__':
    name_str = 'rf_br'
    args = utils.argument_parser().parse_args()

    param_grid = {'classifier': [RandomForestClassifier(random_state=utils.seed)],
                  'classifier__n_estimators': [15, 16, 17, 18, 19, 20],
                  'classifier__max_features': [None, 1/3]}


    x_train_p, y_train_p, x_test_p, y_test_p = preprocessing()
    results = {'name': name_str}

    clf = BinaryRelevance(classifier=RandomForestClassifier(random_state=utils.seed))
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1_macro')
    # Grid search
    g_search.fit(x_train_p.to_numpy(), y_train_p)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    best = g_search.best_estimator_
    print(f'Best parameters {best.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train_p.to_numpy(), y_train_p, best,
                                                         metric_callback=utils.calculate_ml_classification_metrics)

    best.fit(x_train_p.to_numpy(), y_train_p)
    y_pred = best.predict(x_train_p.to_numpy())

    results['train'] = utils.calculate_classification_metrics(y_train_p, y_pred, best.predict_proba(x_train_p.to_numpy()))

    y_pred = best.predict(x_test_p.to_numpy())
    results['test'] = utils.calculate_classification_metrics(y_test_p, y_pred, best.predict_proba(x_test_p))

    print(results)
    results['feature_importances'] = utils.ml_feature_importance(best, x_train_p.columns, classification_predictor,
                                                                    10, 'Variable', 'Importance',
                                                                    'Decision Tree features', args.save_figures,
                                                                    name_str)
    utils.save_dict_as_json(args.json_output, name_str, results)
    utils.plot_multi_label_confusion_matrix(y_test_p, y_pred.toarray(), [1, 0], classification_predictor, 'RF',
                                            save=args.save_figures, extra=name_str)
