import unittest
from database.database_adapter import InfluxDBAdapter


class DatabaseAdapterIntegrationTests(unittest.TestCase):
    def test_database_connection(self):
        database_adapter = InfluxDBAdapter("TestBucket")
        query_result = database_adapter.query('''from(bucket: "TestBucket")
          |> range(start: -inf)''')
        self.assertTrue(query_result)


if __name__ == '__main__':
    unittest.main()
