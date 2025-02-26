import pandas as pd
import pytest
from database.database_connector import InfluxDBConnector


@pytest.fixture
def database_connection():
    connection = InfluxDBConnector("TestBucket")
    yield connection


def test_database_connection(database_connection):
    # TestBucket data range is october 2024
    query_result = database_connection.query('''from(bucket: "TestBucket")
      |> range(start: -10m)''')
    assert isinstance(query_result, list)


def test_get_complete_query_as_dataframe(database_connection):
    bucket = "TestBucket"
    start_time = "2024-10-22T00:00:00Z"
    stop_time = "2024-10-22T23:00:00Z"
    measurement = "sont_c"
    attributes = ["CO", "NO"]

    query_result = database_connection.query_dataframe(f'''from(bucket: "{bucket}")
    |> range(start: {start_time}, stop: {stop_time})
    |> filter(fn: (r) => r._measurement == "{measurement}")
    |> filter(fn: (r) => r._field == "{attributes[0]}" or r._field == "{attributes[1]}")''')
    query_result.drop(["result", "table", "_measurement"], axis=1, inplace=True)

    df_result = pd.DataFrame()

    for attribute in attributes:
        df = query_result[query_result["_field"] == attribute]
        df.reset_index(inplace=True)
        df_result[attribute] = df.loc[:, "_value"]

    print(query_result)
