from influxdb_client import InfluxDBClient
import pandas as pd
from utils import get_config

# TODO: remove bucket parameter
class InfluxDBConnector:
    def __init__(self, bucket: str):
        """
        Initializes the InfluxDBConnector class.

        :param url: InfluxDB server URL
        :param token: Authentication token for InfluxDB
        :param org: InfluxDB organization name
        :param bucket: InfluxDB bucket name
        """

        config = get_config("database_config.yaml")
        self.url = config["url"]
        self.token = config["token"]
        self.org = config["org"]
        self.bucket = bucket
        self.timeout = 60000

        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()

    def query(self, query: str):
        return self.query_api.query(query)

    def query_dataframe(self, query: str) -> pd.DataFrame:
        query_result = self.query_api.query_data_frame(query)
        query_result.drop(["result", "table", "_start", "_stop", "_measurement"], inplace=True, axis=1)
        query_result.set_index("_time", inplace=True, drop=True)
        return query_result
