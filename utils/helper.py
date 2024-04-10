import pandas as pd 
import numpy as np
import scipy.stats as stats
from sklearn.metrics import matthews_corrcoef

import time
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from typing import (
    Tuple,
    Optional,
    Union,
    Any,
    Dict,
    List,
)


# 1. Data Processing

def get_null_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the count and percentage of null values for each column in a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The Pandas DataFrame for which to calculate null information.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing columns for 'Column', 'Null Counts', and 'Null Percentage'.
    """
    null_counts = df.isnull().sum()
    null_counts_df = null_counts.to_frame("Null Counts")

    total_rows = len(df)
    null_counts_df["As Percentage"] = (null_counts_df["Null Counts"] / total_rows) * 100

    null_counts_df = null_counts_df.transpose()
    null_info = null_counts_df.reset_index()
    null_info.rename(columns={"index": "Column"}, inplace=True)

    return null_info


def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes instances that are outside the 1.5 x Interquartile range based on the column inputted.

    Parameters:
    - df: pandas DataFrame containing the data.
    - col: String name of the column to check for outliers.

    Returns:
    - df: pandas DataFrame with outliers removed.
    """

    rows_start = df.shape[0]

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Bounds: {lower_bound} - {upper_bound}")

    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    rows_end = df_filtered.shape[0]
    print(f"{rows_start - rows_end} rows removed.")

    return df_filtered

    
def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits a DataFrame into training, validation, and test sets along with their corresponding target variables.

    This function is tailored for a specific project structure where the dataset includes a 'set' column
    indicating whether each row belongs to the 'train', 'valid', or 'test' set. It also assumes that 'SK_ID_CURR'
    is an identifier column that should be dropped, and 'TARGET' is the target variable to predict.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to split.

    Returns:
    - Tuple containing six elements:
      - X_train (pd.DataFrame): Training features.
      - X_valid (pd.DataFrame): Validation features.
      - X_test (pd.DataFrame): Test features.
      - y_train (pd.Series): Training target variable.
      - y_valid (pd.Series): Validation target variable.
      - y_test (pd.Series): Test target variable.
    """

    X_train = df[df["set"]=="train"].drop(columns=["SK_ID_CURR", "set"])
    X_valid = df[df["set"]=="valid"].drop(columns=["SK_ID_CURR", "set"])
    X_test = df[df["set"]=="test"].drop(columns=["SK_ID_CURR", "set"])

    y_train = X_train.pop("TARGET")
    y_valid = X_valid.pop("TARGET")
    y_test = X_test.pop("TARGET")


    return X_train, X_valid, X_test, y_train, y_valid, y_test



def is_binary(series: pd.Series) -> bool:
    """Determines whether a Pandas Series contains only binary values."""
    unique_values = series.dropna().unique()
    return sorted(unique_values) == [0, 1] or sorted(unique_values) == [False, True]




# 2. Machine Learning related functions:

def evaluate_model(model: Any, name: str, X: pd.DataFrame, y_true: pd.Series, 
                   df: Optional[pd.DataFrame] = None, path: str = "") -> pd.DataFrame:
    """
    Evaluates a machine learning model's performance on provided data and returns a DataFrame with the metrics.

    This function calculates the accuracy, precision, recall, F1 score, and AUC-ROC score of the model.
    It measures the inference time for predictions, stores the model temporarily to measure its disk space,
    and compiles these metrics into a DataFrame. If an existing DataFrame is provided, it appends the results.

    Parameters:
    - model: The machine learning model to evaluate.
    - name (str): The name identifier for the model.
    - X (pd.DataFrame): The features of the dataset used for evaluation.
    - y_true (pd.Series): The true labels of the dataset.
    - df (Optional[pd.DataFrame]): An optional DataFrame to append the results to. Defaults to None.
    - path (str): The directory path for temporarily storing the model to measure disk space. 

    Returns:
    - pd.DataFrame: A DataFrame with the evaluation metrics of the model. If `df` is provided, the metrics
      are appended to it and returned.
    """
    start_time = time.time()
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    end_time = time.time()

    inference_time = end_time - start_time

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    joblib.dump(model, path + "temp_model.joblib")
    disk_space = os.path.getsize(path + "temp_model.joblib")

    row = {
        "name": name,
        "features_count": X.shape[1],
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "inference_time": inference_time,
        "disk_space": disk_space
        }

    new_row = pd.DataFrame([row])
    
    if df is None:
        return new_row
    
    else:
        new_df = pd.concat([df, new_row])
        return new_df
        

def show_results(model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_valid: pd.DataFrame, y_valid: pd.Series) -> None:
    """
    Displays the classification report and AUC scores for both training and validation sets.

    This function predicts labels and probabilities for the training and validation sets, then calculates
    and prints the classification report for the validation set along with the AUC scores for both sets.

    Parameters:
    - model: The trained model to evaluate.
    - X_train (pd.DataFrame): The features of the training set.
    - y_train (pd.Series): The true labels of the training set.
    - X_valid (pd.DataFrame): The features of the validation set.
    - y_valid (pd.Series): The true labels of the validation set.

    Returns:
    None. The function prints the classification report for the validation data and AUC scores for both
    training and validation datasets directly.
    """
    yhat_train = model.predict(X_train)
    yhat_valid = model.predict(X_valid)

    yhat_train_prob = model.predict_proba(X_train)[:, 1]
    yhat_valid_prob = model.predict_proba(X_valid)[:, 1]

    print(classification_report(y_valid, yhat_valid))
    print("Train AUC:", round(roc_auc_score(y_train, yhat_train_prob), 4))
    print("Valid AUC:", round(roc_auc_score(y_valid, yhat_valid_prob), 4))


def calculate_profit(X: pd.DataFrame, 
                     y_true: Union[np.ndarray, pd.Series, pd.DataFrame], 
                     y_predict: Union[np.ndarray, pd.Series, pd.DataFrame], 
                     ror: float = 0.1, lgd: float = 0.3, name: str = "") -> Dict[str, Any]:
    """
    Calculates financial metrics based on loan data, taking into account actual and predicted loan repayments.

    Parameters:
    - X (pd.DataFrame): Pandas DataFrame containing loan data. Must include an "AMT_CREDIT" column.
    - y_true (Union[np.ndarray, pd.Series, pd.DataFrame]): Actual target values indicating loan repayment
      (True = Default, False = Repaid).
    - y_predict (Union[np.ndarray, pd.Series, pd.DataFrame]): Predicted target values.
    - ror (float): Rate Of Return - estimated margin that lenders make on credit loans. Defaults to 0.1.
    - lgd (float): Loss Given Default - 1 minus the recovery rate, should be between 0 and 1. Defaults to 0.3.
    - name (str): Optional name of the row for identification.

    Returns:
    - Dict[str, Any]: A dictionary containing financial metrics such as total revenue, total loss, total profit,
      opportunity cost, avoided loss, value add, and average unit metrics.

    Note: This function assumes binary classification (True/False) for `y_true` and `y_predict`.
          It handles input for `y_true` and `y_predict` as either Pandas Series, Pandas DataFrame, or NumPy arrays.
    """

    X = X[["AMT_CREDIT"]]
    X["true"] = y_true
    X["predicted"] = y_predict

    true_neg = X[(X["true"]==False) & (X["predicted"]==False)]["AMT_CREDIT"]
    false_pos = X[(X["true"]==False) & (X["predicted"]==True)]["AMT_CREDIT"]
    false_neg = X[(X["true"]==True) & (X["predicted"]==False)]["AMT_CREDIT"]
    true_pos = X[(X["true"]==True) & (X["predicted"]==True)]["AMT_CREDIT"]

    total_revenue = true_neg.sum() * ror
    opportunity_cost = false_pos.sum() * ror
    total_loss = false_neg.sum() * lgd
    avoided_loss = true_pos.sum() * lgd

    total_profit = total_revenue - total_loss
    delta_profit = avoided_loss - opportunity_cost

    avg_unit_revenue = total_revenue / len(true_neg)
    avg_unit_profit = total_revenue / len(X)
    avg_unit_loss = total_loss / len(false_neg)

    row = {
    "name": name,
    "total_revenue": total_revenue,
    "total_loss": total_loss,
    "total_profit": total_profit,
    "opportunity_cost": opportunity_cost,
    "avoided_loss": avoided_loss,
    "value_add": delta_profit,
    "avg_unit_value_saved": delta_profit / len(X),
    "avg_unit_revenue": avg_unit_revenue,
    "avg_unit_profit": avg_unit_profit,
    "avg_unit_loss": avg_unit_loss,
    }

    return row

def find_optimal_threshold(X: pd.DataFrame, 
                           y_true: Union[np.ndarray, pd.Series], 
                           y_proba: np.ndarray) -> Tuple[float, float, np.ndarray, List[float]]:
    """
    Calculates the optimal threshold for maximizing the financial value add of a machine learning model predictions.

    Iterates over a range of possible thresholds to determine the threshold that maximizes the financial value
    add, using `calculate_profit`.

    Parameters:
    - X (pd.DataFrame): Feature data used for calculating profits within the `calculate_profit` function.
    - y_true (Union[np.ndarray, pd.Series]): True binary labels of the data.
    - y_proba (np.ndarray): Probability estimates for the positive class from the model.

    Returns:
    - optimal_threshold (float): The threshold that results in the highest financial value add.
    - max_value_add (float): The maximum financial value add achieved at the optimal threshold.
    - thresholds (np.ndarray): The array of evaluated thresholds.
    - value_adds (List[float]): Financial value adds corresponding to each threshold in the evaluated range.
    """

    thresholds = np.linspace(0, 1, 101)
    value_adds = []
    for threshold in thresholds:
        y_pred_threshold = y_proba > threshold
        value_add = calculate_profit(X, y_true, y_pred_threshold)["value_add"]
        value_adds.append(value_add)

    optimal_threshold = thresholds[np.argmax(value_adds)]
    max_value_add = max(value_adds)
    return optimal_threshold, max_value_add, thresholds, value_adds