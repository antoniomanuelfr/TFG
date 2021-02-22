

import pandas as pd
import os
from pathlib import Path


def read_data(data_name: str):
    """Reads a dataset from the `data` directory using pandas.
    Args
        data_name(str): Name of the file.
    Returns
    """
    d = pd.read_csv(os.path.join(Path(__file__).parent.parent, "data", data_name))

    # Remove some useless columns
    d.drop(columns=['ID', 'SEMedia', 'AcMedia', 'NSMedia', 'IEMedia'], inplace=True)

    return d


if __name__ == "__main__":
    dataset = read_data("data.csv")
