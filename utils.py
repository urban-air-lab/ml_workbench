import pandas as pd

class SensorData:
    def __init__(self):
        self.dataLUBW = self.__read_lubw()
        self.datasonta = self.__read_sonta()

        self.datasonta["current_diff"] = self.datasonta["data_RAW_ADC_NO2_W"] - self.datasonta["data_RAW_ADC_NO2_A"]
        self.datasonta = self.datasonta[["current_diff"]]

        self.all_data = self.dataLUBW.join(self.datasonta, how="inner", lsuffix="_LUBW", rsuffix="sonta")
        self.all_data = self.all_data[self.all_data["NO2"] != -999]

    def __read_lubw(self):
        dataLUBW = pd.read_csv("./data/minute_data_lubw.csv")
        dataLUBW.set_index("Zeitstempel", inplace=True)
        dataLUBW.index = pd.to_datetime(
            dataLUBW.index, format="%Y-%m-%d %H:%M:%S")
        return dataLUBW[["NO2"]]

    def __read_sonta(self):
        datasonta = pd.read_csv("./data/sont_a_data.csv")
        datasonta.set_index("_time", inplace=True)
        datasonta.index = pd.to_datetime(
        datasonta.index, format="%Y-%m-%d %H:%M:%S+00:00")
        return datasonta

    @property
    def get_NO2(self):
        return self.all_data["NO2"]

    @property
    def get_difference_electrodes_no2(self):
        return self.all_data["current_diff"]

data = SensorData()
print(data.get_NO2)