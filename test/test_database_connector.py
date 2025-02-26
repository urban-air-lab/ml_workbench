import pandas as pd
import pytest
from database.database_connector import InfluxDBConnector
from database.influx_buckets import InfluxBuckets
from database.influx_query_builder import InfluxQueryBuilder


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
    attributes = ["CO", "NO"]

    query = InfluxQueryBuilder()\
        .set_bucket(InfluxBuckets.test_bucket.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_measurement("sont_c") \
        .set_fields(["CO", "NO"]) \
        .build()

    query_result = database_connection.query_dataframe(query)
    query_result.drop(["result", "table", "_measurement"], axis=1, inplace=True)

    df_result = pd.DataFrame()

    for attribute in attributes:
        df = query_result[query_result["_field"] == attribute]
        df.reset_index(inplace=True)
        df_result[attribute] = df.loc[:, "_value"]

    print(query_result)
