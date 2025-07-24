import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, \
    root_mean_squared_error


def create_result_data(true_values, prediction_values, input_values) -> pd.DataFrame:
    compare_dataframe = pd.DataFrame()
    compare_dataframe["True"] = np.round(true_values, 1)
    compare_dataframe["Predictions"] = np.round(prediction_values, 1)
    compare_dataframe.index = true_values.index
    compare_dataframe = pd.concat([compare_dataframe, input_values], axis=1)
    return compare_dataframe


def calculate_evaluation(dataframe: pd.DataFrame) -> dict[str, float]:
    if not {"True", "Predictions"}.issubset(dataframe.columns):
        raise ValueError("DataFrame must contain 'True' and 'Predictions' columns.")

    return {
        "MAE": round(mean_absolute_error(dataframe["True"], dataframe["Predictions"]), 2),
        "MSE": round(mean_squared_error(dataframe["True"], dataframe["Predictions"]), 2),
        "RMSE": round(root_mean_squared_error(dataframe["True"], dataframe["Predictions"]), 2),
        "MAPE": round((mean_absolute_percentage_error(dataframe["True"], dataframe["Predictions"])) * 100, 2),
        "R-squared": round(r2_score(dataframe["True"], dataframe["Predictions"]), 2)
    }