import re

from app.database.influx_buckets import InfluxBuckets


class InfluxQueryBuilder:
    """
    returns string, usable as flux query, with the following structure:

    from(bucket: "{bucket}")
    |> range(start: {start_time}, stop: {stop_time})
    |> filter(fn: (r) => r.topic == "{topic}")
    |> filter(fn: (r) => r._field == "{attributes[0]}" or r._field == "{attributes[1]} or (...)")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """

    def __init__(self):
        self._bucket = None
        self._range = None
        self._topic = None
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

    def set_topic(self, sensor: str):
        self._topic = f'''|> filter(fn: (r) => r.topic == "{self._build_topic(sensor)}")'''
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
        if self._topic is None:
            raise ValueError("topic must be set!")

        self.query = self._bucket + self._range + self._topic
        if self._fields is not None:
            self.query = self.query + self._fields
        self.query = self.query + self._pivot

        return self.query

    def _build_topic(self, sensor: str) -> str:
        if InfluxBuckets.LUBW_HOUR_BUCKET.value in self._bucket:
            return "sensors/lubw-hour/" + sensor
        elif InfluxBuckets.LUBW_MINUTE_BUCKET.value in self._bucket:
            return "sensors/lubw-minute/" + sensor
        elif InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value in self._bucket:
            return "sensors/calibration/" + sensor
        elif InfluxBuckets.UAL_MINUTE_MEASUREMENT_BUCKET.value in self._bucket:
            return "sensors/measurement/" + sensor
        else:
            raise ValueError("No correct sensor name found, please check")

    def _is_valid_iso8601_utc(self, date: str) -> bool:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        if not re.match(pattern, date):
            return False
        return True
