from pathlib import Path
from typing import Self

import pandas as pd
from ual.get_config import _get_caller_directory


class CSVDataLoader:
    def __init__(self, file_path: str) -> None:
        os_independent_path = _get_caller_directory(2) / Path(file_path)
        try:
            self.data = pd.read_csv(os_independent_path, sep=";")
        except Exception as e:
            raise FileNotFoundError
        try:
            self.data.drop_duplicates(subset=['datetime'], inplace=True)
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data.set_index('datetime', inplace=True, drop=True)
        except Exception as e:
            print("No Datetime Column in Dataframe?: ", e)
        self._drop_999_values()

    def _drop_999_values(self) -> None:
        self.data = self.data[~(self.data == -999).any(axis=1)]

    def set_timespan(self, start_time, end_time) -> Self:
        mask = (self.data.index > start_time) & (self.data.index <= end_time)
        self.data = self.data.loc[mask]
        return self

    def get_data(self, parameter: list) -> pd.DataFrame:
        return self.data.loc[:, parameter]
