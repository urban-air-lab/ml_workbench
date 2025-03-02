import re
from datetime import datetime

class InfluxQueryBuilder:
    """
    returns string, usable as flux query, with the following structure:

    from(bucket: "{bucket}")
    |> range(start: {start_time}, stop: {stop_time})
    |> filter(fn: (r) => r._measurement == "{measurement}")
    |> filter(fn: (r) => r._field == "{attributes[0]}" or r._field == "{attributes[1]} or (...)")
    """

    def __init__(self):
        self.bucket = None
        self.range = None
        self.measurement = None
        self.fields = None
        self.pivot = '''|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''
        self.query = None

    def set_bucket(self, bucket: str):
        self.bucket = f'''from(bucket: "{bucket}")'''
        return self

    def set_range(self, start_time: str, stop_time: str):
        valid_start = self._is_valid_iso8601_utc(start_time)
        valid_end = self._is_valid_iso8601_utc(stop_time)
        if not valid_start & valid_end:
            raise ValueError("No valid date format - must be yyyy-mm-ddTHH:MM:SSZ")
        self.range = f'''|> range(start: {start_time}, stop: {stop_time})'''
        return self

    def set_measurement(self, measurement: str):
        self.measurement = f'''|> filter(fn: (r) => r._measurement == "{measurement}")'''
        return self

    def set_fields(self, fields: list):
        self.fields = "|> filter(fn: (r) =>"
        for index, field in enumerate(fields):
            if index == 0:
                self.fields = self.fields + f''' r._field == "{field}"'''
            else:
                self.fields = self.fields + f''' or r._field == "{field}"'''
        self.fields = self.fields + ")"
        return self

    def build(self):
        if self.bucket is None:
            raise ValueError("bucket must be set!")
        if self.range is None:
            raise ValueError("time range must be set!")
        if self.measurement is None:
            raise ValueError("measurement must be set!")

        self.query = self.bucket + self.range + self.measurement
        if self.fields is not None:
            self.query = self.query + self.fields
        self.query = self.query + self.pivot

        return self.query

    def _is_valid_iso8601_utc(self, date: str) -> bool:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        if not re.match(pattern, date):
            return False
        return True

