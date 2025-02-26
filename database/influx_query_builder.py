
class InfluxQueryBuilder:
    """
    returns string usable as flux query, with the following structure:

    from(bucket: "{bucket}")
    |> range(start: {start_time}, stop: {stop_time})
    |> filter(fn: (r) => r._measurement == "{measurement}")
    |> filter(fn: (r) => r._field == "{attributes[0]}" or r._field == "{attributes[1]}")
    """

    def __init__(self):
        self.bucket = None
        self.range = None
        self.measurement = None
        self.fields = None
        self.query = None

    def set_bucket(self, bucket: str):
        self.bucket = f'''from(bucket: "{bucket}")'''
        return self

    def set_range(self, start_time: str, stop_time: str):
        # TODO: date format validation
        self.range = f'''|> range(start: {start_time}, stop: {stop_time})'''
        return self

    def set_measurement(self, measurement: str):
        self.measurement = f'''|> filter(fn: (r) => r._measurement == "{measurement}")'''
        return self

    def set_fields(self, fields: str):
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

        return self.query
