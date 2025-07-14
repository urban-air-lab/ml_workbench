import pandas as pd


class CSVDataLoader:
    def __init__(self, file_path: str):
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            raise FileNotFoundError
        try:
            self.data.drop_duplicates(subset=['datetime'], inplace=True)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data.set_index('datetime', inplace=True, drop=True)
        except Exception as e:
            print("No Datetime Column in Dataframe?: ", e)
        self._drop_999_values()

    def _drop_999_values(self):
        self.data = self.data[~(self.data == -999).any(axis=1)]

    def set_timespan(self, start_time, end_time):
        mask = (self.data.index > start_time) & (self.data.index <= end_time)
        self.data = self.data.loc[mask]
        return self

    def get_data(self, parameter: list) -> pd.DataFrame:
        return self.data.loc[:, parameter]
