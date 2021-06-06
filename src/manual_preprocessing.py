"""
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.
"""
# Functions that will be used for the manual preprocessing of a dataset.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ordinal_variables = ['BA10.a', 'BA10.b', 'BA10.c', 'BA10.d', 'BA10.f', 'BA10.g', 'BA10.h']


data_path = os.path.join(Path(__file__).parent.parent, "data")
image_path = os.path.join(Path(__file__).parent.parent, "img")

# Columns that after manually checking, are going to be removed, as they are redundant or useless
columns_to_delete = ['ID', 'País', 'CF2', 'CF3', 'CF4', 'CF5', 'CF6', 'CF7', 'CF8', 'CF9',
                     'Los dividendos son parte de lo que una empresa paga al banco para devolver un préstamo.',
                     'Cuando una empresa obtiene capital de un inversor, le está dando al inversor parte de la propiedad de la compañía.',
                     'BA2', 'Año', 'Nacimiento']

# Name of the columns of the dataset that are going to be the predictors.
predictors_name = ['IE1', 'IE2', 'IE3', 'IE4', 'IE5', 'IE6', 'IEMedia']
# Name of the columns of the dataset that are going to be used as classification predictors.
classification_predictor = ['IE1', 'IE2', 'IE3', 'IE4', 'IE5', 'IE6']

# Rename long columns
rename_dict = {'El balance de situación es:': 'Question1',
               '¿Cuál de las siguientes opciones describe mejor el ratio de rentabilidad sobre los activos (ROA)?':
               'Question2'}


def compare_columns(data: pd.DataFrame, l1: list, l2: list, key: str, value, convert_to=None):
    """
    Compare two set of columns that match the value of a key across all the rows. If both columns are equal,
    the columns specified in l2 will be removed from the result dataset.
    Args:
        Data (DataFrame): Pandas DataFrame that contains the data
        l1 (list): First set of columns names that will be selected from Data.
        l2 (list): Second set of columns names that will be selected from Data.
        key (str): Name of the column that will be used to filter the data.
        value: Value of the key used to filter
        convert_to: If not None, both sets will be converted to the specified type
    Returns:
        A DataFrame with the columns specified in l2 removed if both columns are the same
    """
    mapper_dict = dict()
    for c1, c2 in zip(l1, l2):
        mapper_dict[c1] = c2

    # Find the data to look
    d1 = data.loc[data[key] == value, l1]
    d2 = data.loc[data[key] == value, l2]

    # Apply mapper
    d1.rename(columns=mapper_dict, inplace=True)

    if convert_to is not None:
        d1 = d1.astype(convert_to)
        d2 = d2.astype(convert_to)
    if d1.equals(d2):
        ret_data = data.drop(columns=d2.columns)
        ret_data.rename(mapper=mapper_dict, inplace=True)
        return ret_data

    return None


def get_missing_values(data: pd.DataFrame, percentage: float):
    """Look for those columns with high lost values.
       Args:
        data (DataFrame): dataset.
        percentage (float): Minimum percentage to consider that a columns has too many lost values.
       Returns:
        A list with the name of the columns
    """
    n_missing_value = round((data.isna()).sum() / len(data), 3).sort_values(ascending=False)
    return n_missing_value.index[n_missing_value > percentage]


def get_highly_correlated_columns(data: pd.DataFrame, perc: float):
    """Looks for highly correlated columns.
       Args:
        data (DataFrame): Dataset.
        perc (Float): Minimum percentage to consider that two columns are highly correlated.
       Returns:
        A list with the name of the highly correlated columns.
    """
    correlation_matrix = data.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)).stack()
    upper = upper.sort_values(ascending=False)

    to_drop = [upper.index.get_loc(index) for index in upper.index if upper.values[upper.index.get_loc(index)] > perc]
    print(f"{len(to_drop)} features are highly correlated. They will be removed: \n{upper[to_drop]}")

    return upper.index[to_drop]


def get_columns_type(data: pd.DataFrame):
    """Look for the categorical and numerical columns of a dataset.
       Args:
        data (DataFrame): Dataset
       Returns:
        Tuple with a list of the categorical columns and a list with the numeric columns.
    """
    categorical_cols = list(data.select_dtypes('object').astype(str))
    numeric_cols = list(data.select_dtypes(['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).astype(str))
    return categorical_cols, numeric_cols


def print_value_occurrences(data: pd.DataFrame):
    """Get a figure with the distribution class for each column specified in data.
       Args:
        data (DataFrame): Dataset.
       Returns:
        A matplotlib Figure
    """
    fig, axs = plt.subplots(len(data.columns), 1, figsize=(10, 10))
    plot = 0
    for col in data.columns:
        count = np.unique(data[col].dropna(), return_counts=True)
        axs[plot].bar(count[0], count[1], align='center')
        axs[plot].set_xticks(count[0])

        for index, value in enumerate(count[1]):
            axs[plot].text(x=index+0.80, y=value+1, s=f"{value}")

        axs[plot].set_xticklabels(count[0])
        axs[plot].set_xlabel(f'Value occurrence for {col}')
        axs[plot].set_ylabel('Total')
        plot = plot + 1
    return fig, count


def one_hot_encoder(x_train, x_test, categorical_cols):
    """Function to perform a sklearn one hot encoder saving the names of the columns.
    Args:
        x_train (DataFrame): train data to encode.
        x_test (DataFrame): test data to encode.
        categorical_cols (DataFrame): Columns names to encode
    Returns:
        Two DataFrames with the encoded data.
    """
    # OneHotEncode for each category separately
    onhe = OneHotEncoder(sparse=False)
    for cat_col in categorical_cols:
        # Encode the data
        train_enc = onhe.fit_transform(x_train[cat_col].to_numpy().reshape(-1, 1))
        test_enc = onhe.transform(x_test[cat_col].to_numpy().reshape(-1, 1))
        df_train_enc = pd.DataFrame(data=train_enc, columns=onhe.get_feature_names(), dtype=np.int).astype(int)
        df_test_enc = pd.DataFrame(data=test_enc, columns=onhe.get_feature_names(), dtype=np.int).astype(int)
        # Drop the column to encode
        x_train.drop(columns=[cat_col], axis=1, inplace=True)
        x_test.drop(columns=[cat_col], axis=1, inplace=True)

        # Insert the encoded data with the new columns names.
        x_train = pd.concat([x_train, df_train_enc], axis=1)
        x_test = pd.concat([x_test, df_test_enc], axis=1)

    return x_train, x_test


if __name__ == "__main__":
    args = utils.argument_parser()

    dataset = pd.read_csv(os.path.join(data_path, "data.csv"))
    res_dataset = compare_columns(dataset, ['BF1Adaptada', 'BF2Adaptada', 'BF3Adaptada', 'BF4Adaptada'],
                                  ['BF1', 'BF2', 'BF3', 'BF4'], 'Año', '19_20', int)
    print(f"Dataset shape {dataset.shape}")
    if res_dataset is not None:
        print(f"Las columnas son iguales, shape actual = {res_dataset.shape}")
        dataset = res_dataset
    else:
        print("Las columnas son distintas")

    dataset.drop(columns=columns_to_delete, inplace=True)
    dataset.rename(mapper=rename_dict, inplace=True)

    print(dataset['Género'].unique())
    dataset['Género'].replace(to_replace='Sí', value=np.nan, inplace=True)
    dataset['Género'].replace(to_replace='No', value=np.nan, inplace=True)

    dataset.replace(to_replace=' ', value=np.nan, inplace=True)
    dataset.replace(to_replace='No contesta', value=np.nan, inplace=True)
    dataset.replace(to_replace='No', value=0, inplace=True)
    dataset.replace(to_replace='Sí', value=1, inplace=True)

    # Looks for missing values

    missing_values_col = get_missing_values(dataset, 0.5)
    print(f"{len(missing_values_col)} columns have more than 50% of missing values.\nRemoving: {missing_values_col}")

    dataset.drop(columns=missing_values_col, inplace=True)
    # Get the predictors data
    predictor_data = dataset[predictors_name]
    dataset.drop(columns=predictors_name, inplace=True)

    corr_cols = get_highly_correlated_columns(dataset, 0.9)
    for pair in corr_cols:
        dataset.drop(columns=pair[1], inplace=True)

    # Print the class occurrence for the classification predictors.
    fig, count = print_value_occurrences(predictor_data[classification_predictor])
    fig.tight_layout()
    if args.save_figures:
        plt.savefig(f"{args.save_figures}/value_occurrences.png")
    else:
        plt.show()

    predictor_data = predictor_data.drop(columns=['IEMedia'])
    x_train, x_test, y_train, y_test = train_test_split(dataset, predictor_data, random_state=2342)
    x_train.to_csv(os.path.join(data_path, 'x_train.csv'), index=False)
    x_test.to_csv(os.path.join(data_path, 'x_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_path, 'y_test.csv'), index=False)
    print(f"train size {x_train.shape} test_size {x_test.shape}")
