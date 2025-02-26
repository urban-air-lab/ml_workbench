from influxdb_client import InfluxDBClient
import pandas as pd
from utils import get_config


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
        df_result = self.__get_fields_from_query(query_result)
        df_result["time"] = self._get_timestamps_from_query(query_result)
        return df_result

    def __get_fields_from_query(self, query_result):
        df_result = pd.DataFrame()
        for attribute in query_result.loc[:, "_field"].unique():
            df = query_result[query_result["_field"] == attribute]
            df.reset_index(inplace=True)
            df_result[attribute] = df.loc[:, "_value"]
        return df_result

    def _get_timestamps_from_query(self, query_result):
        return query_result.loc[:, "_time"].unique()

