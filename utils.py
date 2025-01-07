import pandas as pd
import re
import keras
from keras import Model
import logging
import yaml
import matplotlib.pyplot as plt


class SensorData:
    def __init__(self, file_path_lubw: str, file_path_aqsn: str, in_hour: bool = False):
        self.dataLUBW = self.__read_lubw(file_path_lubw, )

        self.dataAQSN = self.__read_aqsn(file_path_aqsn)
        self.dataAQSN["electrode_difference_NO2"] = self.dataAQSN["data_RAW_ADC_NO2_W"] - self.dataAQSN[
            "data_RAW_ADC_NO2_A"]
        self.dataAQSN = self.dataAQSN[["electrode_difference_NO2"]]

        if in_hour:  # checken
            self.dataAQSN.resample("h").mean()

        self.all_data = self.dataLUBW.join(self.dataAQSN, how="inner", lsuffix="_LUBW", rsuffix="_AQSN")
        self.all_data = self.all_data[self.all_data["data_NO2"] != -999]

    def __read_lubw(self, file_path_lubw) -> pd.DataFrame:
        dataLUBW = pd.read_csv(file_path_lubw)
        time_column = dataLUBW.iloc[:, 0]

        if not self.has_timezone_in_string(time_column[0]):
            dates = pd.to_datetime(time_column).dt.tz_localize('Europe/Berlin')
        else:
            dates = pd.to_datetime(time_column)

        dataLUBW.set_index(dates, inplace=True, drop=True)
        return dataLUBW[["data_NO2"]]  # variable machen

    def __read_aqsn(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0])
        data.set_index("_time", inplace=True)
        return data

    def has_timezone_in_string(self, datetime_str: str) -> bool:
        tz_pattern = re.compile(r'(\+|-)\d{2}:\d{2}|Z$')
        return bool(tz_pattern.search(datetime_str))

    @property
    def get_NO2(self) -> pd.Series:
        return self.all_data["data_NO2"]

    @property
    def get_difference_electrodes_no2(self) -> pd.Series:
        return self.all_data["electrode_difference_NO2"]


def train_model(model: Model, config: dict, inputs: pd.DataFrame, targets: pd.DataFrame, save: bool = False) -> None:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss="mean_squared_error")
    model.fit(x=inputs, y=targets, epochs=config["epochs"], batch_size=config["batch"])
    if save:
        model.save(f"../../models/{config['name']}.keras")


def load_model(model_path: str) -> Model:
    return keras.saving.load_model(model_path)


def get_config() -> dict:
    try:
        with open("./workflow_config.yaml", 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"No workflow config found in directory")
    except IOError:
        logging.error(f"IOError: An I/O error occurred")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


def plot_results(plot_path: str, *args) -> None:
    figure, axes = plt.subplots(nrows=len(args), figsize=(10, 10))
    for index, (values, name) in enumerate(args):
        axes[index].plot(values)
        axes[index].set_title(name)
    plt.savefig(plot_path)
    plt.show()
