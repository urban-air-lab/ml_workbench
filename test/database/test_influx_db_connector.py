import os
import pytest

from app.database.Influx_db_connector import InfluxDBConnector
from app.database.influx_buckets import InfluxBuckets
from app.database.influx_query_builder import InfluxQueryBuilder

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def database_connection():
    connection = InfluxDBConnector()
    yield connection


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_database_connection(database_connection):
    # TestBucket data range is october 2024
    query_result = database_connection.query('''from(bucket: "test-data")
      |> range(start: -10m)''')
    assert isinstance(query_result, list)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_get_complete_query_as_dataframe(database_connection):
    query = InfluxQueryBuilder() \
        .set_bucket(InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value) \
        .set_range("2024-10-22T00:00:00Z", "2024-10-22T23:00:00Z") \
        .set_topic("ual-1") \
        .set_fields(["CO", "NO"]) \
        .build()

    query_result = database_connection.query_dataframe(query)
    print(query_result)  # TODO: extend testing
