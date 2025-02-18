import unittest
from database.database_adapter import InfluxDBAdapter


class DatabaseAdapterIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.database_adapter = InfluxDBAdapter("TestBucket")

    def test_database_connection(self):
        query_result = self.database_adapter.query('''from(bucket: "TestBucket")
          |> range(start: -inf)''')
        self.assertTrue(query_result)




if __name__ == '__main__':
    unittest.main()
