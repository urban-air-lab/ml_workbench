import pandas as pd


class SensorData:
    def __init__(self):
        self.dataLUBW = self.__read_lubw()
        self.datasonta = self.__read_custom_data("./data/sont_c_data.csv")
        self.futuredata = self.__read_custom_data("./data/sont_c_20241211-202241228.csv")

        self.datasonta["current_diff"] = self.datasonta["data_RAW_ADC_NO2_W"] - self.datasonta["data_RAW_ADC_NO2_A"]
        self.datasonta = self.datasonta[["current_diff"]]

        self.futuredata["current_diff"] = self.futuredata["data_RAW_ADC_NO2_W"] - self.futuredata["data_RAW_ADC_NO2_A"]
        self.futuredata = self.futuredata[["current_diff"]]

        self.all_data = self.dataLUBW.join(self.datasonta, how="inner", lsuffix="_LUBW", rsuffix="sonta")
        self.all_data = self.all_data[self.all_data["NO2"] != -999]

    def __read_lubw(self) -> pd.DataFrame:
        dataLUBW = pd.read_csv("./data/minute_data_lubw.csv")
        dataLUBW.set_index("Zeitstempel", inplace=True)
        dataLUBW.index = pd.to_datetime(
            dataLUBW.index, format="%Y-%m-%d %H:%M:%S")
        return dataLUBW[["NO2"]]

    def __read_custom_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data.set_index("_time", inplace=True)
        data.index = pd.to_datetime(
           data.index, format="%Y-%m-%d %H:%M:%S+00:00")
        return data

    @property
    def get_NO2(self) -> pd.Series:
        return self.all_data["NO2"]

    @property
    def get_difference_electrodes_no2(self) -> pd.Series:
        return self.all_data["current_diff"]

