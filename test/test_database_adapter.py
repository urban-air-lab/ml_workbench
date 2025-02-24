import pandas as pd
import pytest
from database.database_adapter import InfluxDBAdapter


@pytest.fixture
def database_connection():
    connection = InfluxDBAdapter("TestBucket")
    yield connection


def test_database_connection(database_connection):
    # TestBucket data range is october 2024
    query_result = database_connection.query('''from(bucket: "TestBucket")
      |> range(start: -10m)''')
    assert isinstance(query_result, list)


def test_get_measurement(database_connection):
    bucket = "TestBucket"
    start_time = "2024-10-22T00:00:00Z"
    stop_time = "2024-10-22T23:00:00Z"
    measurement = "sont_c"
    attribute = "CO"
    query_result = database_connection.query_dataframe(f'''from(bucket: "{bucket}")
    |> range(start: {start_time}, stop: {stop_time})
    |> filter(fn: (r) => r._measurement == "{measurement}")
    |> filter(fn: (r) => r._field == "{attribute}")''')
    query_result.drop(["result", "table", "_field", "_measurement"], axis=1, inplace=True)
    print(query_result)
