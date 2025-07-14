import numpy as np
import pandas as pd
import logging
import yaml
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, \
    root_mean_squared_error
import inspect
from pathlib import Path





def convert_to_pytorch_tensors(inputs_train, inputs_test, targets_train, targets_test):
    inputs_train_tensor = map_to_tensor(inputs_train)
    inputs_test_tensor = map_to_tensor(inputs_test)
    targets_train_tensor = map_to_tensor(targets_train)
    targets_test_tensor = map_to_tensor(targets_test)

    targets_train_tensor = add_dimension(targets_train_tensor)
    targets_test_tensor = add_dimension(targets_test_tensor)

    return inputs_train_tensor, inputs_test_tensor, targets_train_tensor, targets_test_tensor


def add_dimension(targets_train_tensor):
    targets_train_tensor = targets_train_tensor.unsqueeze(1)
    return targets_train_tensor


def map_to_tensor(inputs_train):
    inputs_train_tensor = torch.tensor(inputs_train.values, dtype=torch.float32)
    return inputs_train_tensor


def create_result_data(true_values, prediction_values, input_values) -> pd.DataFrame:
    if type(prediction_values) == torch.Tensor:
        prediction_values = prediction_values.detach().numpy().flatten()

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


def get_config(file: str) -> dict:
    os_independent_path = _get_caller_directory(2) / Path(file)
    try:
        with open(os_independent_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"No config found in directory")
    except IOError:
        logging.error(f"IOError: An I/O error occurred")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def _get_caller_directory(stack_position: int) -> Path:
    caller_file = inspect.stack()[stack_position].filename
    return Path(caller_file).parent
