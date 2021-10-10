"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
from os.path import join
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer



import tfg_utils.utils as utils
from tfg_utils.manual_preprocessing import get_columns_type, one_hot_encoder, get_missing_values, classification_predictor
from tfg_utils.MultipleLabelClassification.MultipleLabelCCClassifiers import SVCMultipleLabelCC

data_path = join(Path(__file__).parent.parent.parent, 'data')


def preprocessing(undersampling_thr=None, feature_selection=None):
    x_train = pd.read_csv(join(data_path, 'x_train.csv'), index_col=False)
    y_train = pd.read_csv(join(data_path, 'y_train.csv'), index_col=False)
    x_test = pd.read_csv(join(data_path, 'x_test.csv'), index_col=False)
    y_test = pd.read_csv(join(data_path, 'y_test.csv'), index_col=False)

    c_cols, n_cols = get_columns_type(x_train)

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])
    numeric_transformer = Pipeline([('imputer', KNNImputer(n_neighbors=2, weights='uniform')),
                                    ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, n_cols),
                                                   ('categorical', categorical_transformer, c_cols)])
    transformed_cols = n_cols + c_cols

    y_imputer = SimpleImputer(strategy='median')

    y_train_transformed = y_imputer.fit_transform(y_train)
    y_test_transformed = y_imputer.transform(y_test)

    preprocessor.fit(x_train)
    x_train_transformed = pd.DataFrame(preprocessor.transform(x_train), columns=transformed_cols)
    x_test_transformed = pd.DataFrame(preprocessor.transform(x_test), columns=transformed_cols)

    x_train_transformed, x_test_transformed = one_hot_encoder(x_train_transformed, x_test_transformed, c_cols)

    return x_train_transformed, y_train_transformed, x_test_transformed, y_test_transformed


if __name__ == '__main__':
    name_str = 'SVC_mlabel'
    args = utils.argument_parser().parse_args()

    param_grid = {'kernel': ['poly', 'rbf'],
                  'degree': [2, 3, 4, 5],
                  'C': [0.5, 0.75, 1]
                  }
    x_train_p, y_train_p, x_test_p, y_test_p = preprocessing(args.undersampling)
    results = {'name': name_str}
    clf = SVCMultipleLabelCC(random_state=utils.seed)

    custom_scorer = make_scorer(utils.f1_multilabel_mean, greater_is_better=True)
    g_search = GridSearchCV(clf, param_grid=param_grid, scoring=custom_scorer, n_jobs=-1)

    g_search.fit(x_train_p, y_train_p)
    print(f'Best score {g_search.best_score_} with {g_search.best_estimator_}')
    clf = g_search.best_estimator_
    print(f'Best parameters {clf.get_params()}')

    results['cross_validation'] = utils.cross_validation(x_train_p.to_numpy(), y_train_p, clf,
                                                         metric_callback=utils.calculate_ml_classification_metrics)

    clf.fit(x_train_p, y_train_p)
    y_pred = clf.predict(x_train_p)

    results['train'] = utils.calculate_ml_classification_metrics(y_train_p, y_pred, clf.predict_proba(x_train_p))
    utils.plot_multilabel_class_metrics(results['train'], False, args.save_figures, classification_predictor, name_str)

    y_pred = clf.predict(x_test_p)
    results['test'] = utils.calculate_ml_classification_metrics(y_test_p, y_pred, clf.predict_proba(x_test_p))
    utils.plot_multilabel_class_metrics(results['test'], False, args.save_figures, classification_predictor, name_str)

    utils.save_dict_as_json(args.json_output, name_str, results)
