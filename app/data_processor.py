import pandas as pd
import numpy as np
import torch
from scipy import stats


class DataProcessor:
    def __init__(self, inputs, targets):
        if not isinstance(inputs.index, pd.DatetimeIndex):
            raise ValueError("The inputs index must be a DatetimeIndex.")
        if not isinstance(targets.index, pd.DatetimeIndex):
            raise ValueError("The targets index must be a DatetimeIndex.")

        self.inputs = inputs
        self.targets = targets

    def to_hourly(self):
        self.inputs = self.inputs.resample("h").mean()
        self.targets = self.targets.resample("h").mean()
        return self

    def to_daily(self):
        self.inputs = self.inputs.resample("d").mean()
        self.targets = self.targets.resample("d").mean()
        return self

    def remove_outliers(self, outlier_range=3):
        z_scores = np.abs(stats.zscore(self.inputs, nan_policy='omit'))
        mask = (z_scores < outlier_range).all(axis=1)
        self.inputs = self.inputs[mask]
        return self

    def remove_nan(self):
        self.inputs = self.inputs.dropna()
        self.targets = self.targets.dropna()
        return self

    def align_dataframes_by_time(self):
        self.inputs, self.targets = align_dataframes_by_time(self.inputs, self.targets)
        return self

    def calculate_w_a_difference(self):
        gases = ["NO", "NO2", "O3"]
        self.inputs = calculate_w_a_difference(self.inputs, gases)
        return self

    def get_inputs(self):
        return self.inputs

    def get_target(self, target):
        return self.targets[target]


def calculate_w_a_difference(dataframe: pd.DataFrame, gases: list) -> pd.DataFrame:
    if not gases:
        raise ValueError("No gases in list")
    if dataframe.empty:
        raise ValueError("dataframe is empty")
    for gas in gases:
        w_column = f"RAW_ADC_{gas}_W"
        a_column = f"RAW_ADC_{gas}_A"
        if w_column in dataframe.columns and a_column in dataframe.columns:
            dataframe[f"{gas}_W_A"] = dataframe[w_column] - dataframe[a_column]
            dataframe.drop([w_column, a_column], inplace=True, axis=1)
        else:
            print(f"Warning: Columns for {gas} not found in the dataframe.")
    return dataframe


def align_dataframes_by_time(df1: pd.DataFrame, df2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)

    common_times = df1.index.intersection(df2.index)

    df1_aligned = df1.loc[common_times]
    df2_aligned = df2.loc[common_times]
    return df1_aligned, df2_aligned
