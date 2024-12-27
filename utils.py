import pandas as pd

class LUBWData:

    def __init__(self):
        self.dataLUBW = pd.read_csv("./data/minute_data_lubw.csv")
        self.dataLUBW.set_index("Zeitstempel", inplace=True)
        self.dataLUBW.index = pd.to_datetime(
            self.dataLUBW.index, format = "%Y-%m-%d %H:%M:%S")
        self.dataLUBW = self.dataLUBW[["NO2"]]

        self.datasonta = pd.read_csv("./data/sont_a_data.csv")
        self.datasonta.set_index("_time", inplace=True)
        self.datasonta.index = pd.to_datetime(
            self.datasonta.index, format = "%Y-%m-%d %H:%M:%S+00:00")
        self.datasonta = self.datasonta[["data_RAW_ADC_NO2_A","data_RAW_ADC_NO2_W"]]

        self.all_data = self.dataLUBW.join(self.datasonta, how="inner", lsuffix="_LUBW", rsuffix="sonta")


data = LUBWData()
print(data.all_data)