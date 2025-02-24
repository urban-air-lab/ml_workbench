from influxdb_client import InfluxDBClient
import pandas as pd

from utils import get_config


class InfluxDBAdapter:
    def __init__(self, bucket: str):
        """
        Initializes the InfluxDBDataFetcher class.

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

        # Initialize InfluxDB client
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.query_api = self.client.query_api()

    def query(self, query: str):
        return self.query_api.query(query)

    def query_dataframe(self, query: str) -> pd.DataFrame:
        return self.query_api.query_data_frame(query)

    def get_all_data_between(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Fetches all data of all devices between a given start and end time. (would currenty return testdataHHN/airup_sont_b)

        :param start_time: Start time in RFC3339 format (e.g., '2024-06-17T00:00:00Z')
        :param end_time: End time in RFC3339 format (e.g., '2024-06-18T00:00:00Z')
        :return: Pandas DataFrame containing the queried data
        """
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_time}, stop: {end_time})
          |> filter(fn: (r) =>
            r._measurement == "mqtt_consumer"
            )
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        tables = self.query_api.query(org=self.org, query=query)

        rows = []
        for table in tables:
            for record in table.records:
                rows.append(record.values)

        df = pd.DataFrame(rows)
        df.columns = pd.Series(df.columns).apply(
            lambda x: f"{x}_{df.columns.tolist().count(x)}" if df.columns.tolist().count(x) > 1 else x
        )
        return self._clean_dataframe(df)

    def get_device_between(self, start_time: str, end_time: str, device: str) -> pd.DataFrame:
        """
        Fetches all data of one device between a given start and end time.

        :param start_time: Start time in RFC3339 format (e.g., '2024-06-17T00:00:00Z')
        :param end_time: End time in RFC3339 format (e.g., '2024-06-18T00:00:00Z')
        :return: Pandas DataFrame containing the queried data
        """
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {start_time}, stop: {end_time})
          |> filter(fn: (r) =>
            r.topic == "airdataHHN/{device}"
            )
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''

        tables = self.query_api.query(org=self.org, query=query)

        rows = []
        for table in tables:
            for record in table.records:
                rows.append(record.values)

        df = pd.DataFrame(rows)
        df.columns = pd.Series(df.columns).apply(
            lambda x: f"{x}_{df.columns.tolist().count(x)}" if df.columns.tolist().count(x) > 1 else x
        )
        return self._clean_dataframe(df)

    def get_devices_between_in_single_dataframes(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Fetches all data of all devices between a given start and end time.
        In specific format

        :param start_time: Start time in RFC3339 format (e.g., '2024-06-17T00:00:00Z')
        :param end_time: End time in RFC3339 format (e.g., '2024-06-18T00:00:00Z')
        :return: Pandas DataFrame containing the queried data
        """
        devices = ["airup_sont_a", "airup_sont_b", "airup_sont_c", "DEBW015", "DEBW152"]
        device_dataframes = {}

        for device in devices:
            query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {start_time}, stop: {end_time})
              |> filter(fn: (r) =>
                r.topic == "airdataHHN/{device}"
                )
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            tables = self.query_api.query(org=self.org, query=query)

            rows = []
            for table in tables:
                for record in table.records:
                    rows.append(record.values)

            if rows:
                df = pd.DataFrame(rows)
                df = self._clean_dataframe(df)
                df.columns = pd.Series(df.columns).apply(
                    lambda x: f"{x}_{df.columns.tolist().count(x)}" if df.columns.tolist().count(x) > 1 else x
                )
            else:
                print(f"ERROR: device {device} not found")
                df = pd.DataFrame()

            device_dataframes[device] = df

        return device_dataframes


    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by dropping specified columns if they exist and flooring the timestamp,
        so that there are no milliseconds

        :param df: The input Pandas DataFrame
        :return: Cleaned DataFrame
        """
        columns_to_drop = ["result", "table", "_start", "_stop", "_measurement", "host", "topic"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=existing_columns_to_drop, inplace=True)

        df["_time"] = df["_time"].dt.floor("s")
        df.set_index("_time", inplace=True)
        return df

