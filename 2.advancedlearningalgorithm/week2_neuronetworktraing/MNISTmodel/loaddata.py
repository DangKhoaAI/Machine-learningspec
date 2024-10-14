import pandas as pd
import os
#? file này load MNIST csv và chuyển thành dataframe panda
def load_data(filepath):
    """
    Loads the coffee dataset from a CSV file.
    """"""
    Args:
        filepath (str): The path to the CSV file containing the data.

    Returns:
        X (pd.DataFrame): The feature matrix.
        Y (pd.Series): The labels (target variable).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

    # Load the dataset
    data = pd.read_csv(filepath)

    # Assuming that all columns except the last one are features, and the last column is the label
    X = data.iloc[:, :-1]  # All rows, all columns except the last (features)
    Y = data.iloc[:, -1]   # All rows, only the last column (labels)
    
    return X, Y
