import pandas as pd
import os
#? file này load csv và chuyển thành dạng datafrane pandas
def load_coffee_data(filepath="coffee_data.csv"):
    """
    Loads the coffee dataset from a CSV file.
    """"""
    Args:
        filepath (str): The path to the CSV file containing the data.

    Returns:
        X (pd.DataFrame): The feature matrix.
        Y (pd.Series): The labels (target variable).
    """
     # Lấy đường dẫn đến thư mục hiện tại (thư mục chứa script)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Tạo đường dẫn đầy đủ tới file
    full_path = os.path.join(current_dir, filepath)
    #doi sang folder hientai
    os.chdir(current_dir)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file '{full_path}' does not exist.")
    # Load the datasets
    data = pd.read_csv(filepath)
    # Assuming that all columns except the last one are features, and the last column is the label
    X = data.iloc[:, :-1]  # All rows, all columns except the last (features)
    Y = data.iloc[:, -1]   # All rows, only the last column (labels)
    
    return X, Y
