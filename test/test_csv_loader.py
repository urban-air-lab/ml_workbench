import pandas as pd
import pytest
from app.csv_loader import CSVDataLoader


def test_load_cleans_duplicates_and_999_rows():
    loader = CSVDataLoader("ressources/test.csv")
    df = loader.data

    assert df.index.nunique() == 4
    assert df.iloc[0]['param1'] == 1
    assert df.iloc[0]['param2'] == 10


def test_set_timespan_filters_correctly():
    loader = CSVDataLoader("ressources/test.csv")

    start = pd.to_datetime('01.01.2021 02:00')
    end = pd.to_datetime('01.01.2021 04:00')
    loader.set_timespan(start, end)
    assert loader.data.size == 4
    assert len(loader.data.index) == 2


def test_get_data_selects_columns():
    loader = CSVDataLoader("ressources/test.csv")
    sub = loader.get_data(['param1'])

    assert list(sub.columns) == ['param1']
    assert sub.iloc[0,0] == 1


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        CSVDataLoader("this_file_does_not_exist.csv")