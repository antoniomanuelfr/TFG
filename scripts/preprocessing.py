import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from tfg_utils import utils
from tfg_utils import manual_preprocessing as mp

data_path = os.path.join(Path(__file__).parent.parent, 'data')

if __name__ == '__main__':
    args = utils.argument_parser().parse_args()

    dataset = pd.read_csv(os.path.join(data_path, 'data.csv'))
    res_dataset = mp.compare_columns(dataset, ['BF1Adaptada', 'BF2Adaptada', 'BF3Adaptada', 'BF4Adaptada'],
                                     ['BF1', 'BF2', 'BF3', 'BF4'], 'Año', '19_20', int)
    print(f'Dataset shape {dataset.shape}')
    if res_dataset is not None:
        print(f'Las columnas son iguales, shape actual = {res_dataset.shape}')
        dataset = res_dataset
    else:
        print('Las columnas son distintas')

    dataset.drop(columns=mp.columns_to_delete, inplace=True)
    dataset.rename(mapper=mp.rename_dict, inplace=True)

    print(dataset['Género'].unique())
    dataset['Género'].replace(to_replace='Sí', value=np.nan, inplace=True)
    dataset['Género'].replace(to_replace='No', value=np.nan, inplace=True)

    dataset.replace(to_replace=' ', value=np.nan, inplace=True)
    dataset.replace(to_replace='No contesta', value=np.nan, inplace=True)
    dataset.replace(to_replace='No', value=0, inplace=True)
    dataset.replace(to_replace='Sí', value=1, inplace=True)

    # Looks for missing values

    missing_values_col = mp.get_missing_values(dataset, 0.5)
    print(f'{len(missing_values_col)} columns have more than 50% of missing values.\nRemoving: {missing_values_col}')

    dataset.drop(columns=missing_values_col, inplace=True)
    # Get the predictors data
    predictor_data = dataset[mp.predictors_name]
    dataset.drop(columns=mp.predictors_name, inplace=True)

    corr_cols = mp.get_highly_correlated_columns(dataset, 0.9)
    for pair in corr_cols:
        dataset.drop(columns=pair[1], inplace=True)

    # Print the class occurrence for the classification predictors.
    count = mp.print_value_occurrences(predictor_data[mp.classification_predictor])
    if args.save_figures:
        plt.savefig(f'{args.save_figures}/value_occurrences.png')
    else:
        plt.show()

    predictor_data = predictor_data.drop(columns=['IEMedia'])
    x_train, x_test, y_train, y_test = train_test_split(dataset, predictor_data, random_state=2342)
    x_train.to_csv(os.path.join(data_path, 'x_train.csv'), index=False)
    x_test.to_csv(os.path.join(data_path, 'x_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_path, 'y_test.csv'), index=False)

    print(f'train size {x_train.shape} test_size {x_test.shape}')
