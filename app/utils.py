import pandas as pd
import logging
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
from pathlib import Path
import os


def calculate_w_a_difference(df, gases):
    calculated_differences = pd.DataFrame()
    for gas in gases:
        w_column = f"RAW_ADC_{gas}_W"
        a_column = f"RAW_ADC_{gas}_A"
        if w_column in df.columns and a_column in df.columns:
            calculated_differences[f"{gas}_W_A"] = df[w_column] - df[a_column]
        else:
            print(f"Warning: Columns for {gas} not found in the dataframe.")
    return calculated_differences


def align_dataframes_by_time(df1, df2):
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)

    common_times = df1.index.intersection(df2.index)

    df1_aligned = df1.loc[common_times]
    df2_aligned = df2.loc[common_times]
    return df1_aligned, df2_aligned


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


def check_and_create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def plot_results(plot_path: str, *args) -> None:
    figure, axes = plt.subplots(nrows=len(args), figsize=(10, 10))
    for index, (values, name) in enumerate(args):
        sns.lineplot(x=values.index.tolist(), y=values.values.flatten(), ax=axes[index])
        axes[index].set_title(name)
        axes[index].grid()
    plt.savefig(plot_path)
    plt.show()
