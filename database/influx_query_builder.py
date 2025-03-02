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
        self._bucket = None
        self._range = None
        self._measurement = None
        self._fields = None
        self._pivot = '''|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''
        self.query = None

    def set_bucket(self, bucket: str):
        self._bucket = f'''from(bucket: "{bucket}")'''
        return self

    def set_range(self, start_date: str, stop_date: str):
        valid_start = self._is_valid_iso8601_utc(start_date)
        valid_end = self._is_valid_iso8601_utc(stop_date)
        if not valid_start & valid_end:
            raise ValueError("No valid date format - must be yyyy-mm-ddTHH:MM:SSZ")
        self._range = f'''|> range(start: {start_date}, stop: {stop_date})'''
        return self

    def set_measurement(self, measurement: str):
        self._measurement = f'''|> filter(fn: (r) => r._measurement == "{measurement}")'''
        return self

    def set_fields(self, fields: list):
        self._fields = "|> filter(fn: (r) =>"
        for index, field in enumerate(fields):
            if index == 0:
                self._fields = self._fields + f''' r._field == "{field}"'''
            else:
                self._fields = self._fields + f''' or r._field == "{field}"'''
        self._fields = self._fields + ")"
        return self

    def build(self):
        if self._bucket is None:
            raise ValueError("bucket must be set!")
        if self._range is None:
            raise ValueError("time range must be set!")
        if self._measurement is None:
            raise ValueError("measurement must be set!")

        self.query = self._bucket + self._range + self._measurement
        if self._fields is not None:
            self.query = self.query + self._fields
        self.query = self.query + self._pivot

        return self.query

    def _is_valid_iso8601_utc(self, date: str) -> bool:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        if not re.match(pattern, date):
            return False
        return True

