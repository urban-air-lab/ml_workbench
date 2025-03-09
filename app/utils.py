import pandas as pd
import logging
import yaml
import torch
from sklearn.model_selection import train_test_split
import inspect
from pathlib import Path
import os


def calculate_w_a_difference(dataframe, gases):
    for gas in gases:
        w_column = f"RAW_ADC_{gas}_W"
        a_column = f"RAW_ADC_{gas}_A"
        if w_column in dataframe.columns and a_column in dataframe.columns:
            dataframe[f"{gas}_W_A"] = dataframe[w_column] - dataframe[a_column]
            dataframe.drop([w_column, a_column], inplace=True, axis=1)
        else:
            print(f"Warning: Columns for {gas} not found in the dataframe.")
    return dataframe


def align_dataframes_by_time(df1, df2):
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)

    common_times = df1.index.intersection(df2.index)

    df1_aligned = df1.loc[common_times]
    df2_aligned = df2.loc[common_times]
    return df1_aligned, df2_aligned


def train_test_split_pytorch(inputs, targets, test_size, shuffle):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                                targets,
                                                                                test_size=test_size,
                                                                                shuffle=shuffle)

    inputs_train_tensor = torch.tensor(inputs_train.values, dtype=torch.float32)
    inputs_test_tensor = torch.tensor(inputs_test.values, dtype=torch.float32)
    targets_train_tensor = torch.tensor(targets_train.values, dtype=torch.float32)
    targets_test_tensor = torch.tensor(targets_test.values, dtype=torch.float32)

    targets_train_tensor = targets_train_tensor.unsqueeze(1)
    targets_test_tensor = targets_test_tensor.unsqueeze(1)

    return inputs_train_tensor, inputs_test_tensor, targets_train_tensor, targets_test_tensor


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


def create_run_directory() -> Path:
    os_independent_path = _get_caller_directory(2)
    results_path = os_independent_path / Path("results")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    number_of_files = len(os.listdir(results_path))
    run_path = os_independent_path / Path("results" / Path(f"run_{number_of_files + 1}"))
    os.makedirs(run_path)
    return run_path


def _get_caller_directory(stack_position: int) -> Path:
    caller_file = inspect.stack()[stack_position].filename
    return Path(caller_file).parent