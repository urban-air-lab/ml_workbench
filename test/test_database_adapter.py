import pytest
from database.database_adapter import InfluxDBAdapter


@pytest.fixture
def database_connection():
    connection = InfluxDBAdapter("TestBucket")
    yield connection


def test_database_connection(database_connection):
    query_result = database_connection.query('''from(bucket: "TestBucket")
      |> range(start: -inf)''')
    assert len(query_result) > 0

